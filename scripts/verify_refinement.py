"""Real-checkpoint integration checks; short and isolated from production training."""
import argparse
import json
from pathlib import Path
import tempfile
import time
from dataclasses import replace

import torch
from torch.utils.data import default_collate
from lerobot.policies.smolvla.modeling_smolvla import make_att_2d_masks

from smolvla_baseline.data import ROOT
from tactile_refinement.data import TactileEpisodes, prepare_refinement_batch
from tactile_refinement.model import encode_prefix
from tactile_refinement.train import build_refiner, active_stages


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=ROOT/'logs/refinement_verification.json')
    args = parser.parse_args()
    torch.set_num_threads(2); torch.manual_seed(7)
    config = json.loads((ROOT/'configs/tactile_refinement.json').read_text())
    model, pre, post = build_refiner(config)
    ds = TactileEpisodes(ROOT/config['manifest'])
    raw = default_collate([ds[0]])
    batch, tactile = prepare_refinement_batch(raw, pre, 'cuda')
    core = model.baseline.model
    report = {'baseline_checkpoint': config['baseline_checkpoint'], 'expert_initialization': config['expert_initialization'],
              'batch_size': 1, 'checks': {}}
    shared = model.expert.implementation == 'shared_ffn_lora_v1'
    for name in ('action_in_proj', 'action_out_proj', 'action_time_mlp_in', 'action_time_mlp_out', 'state_proj'):
        fresh, source = getattr(model.expert, name).weight, getattr(core, name).weight
        assert (fresh.data_ptr() == source.data_ptr()) == shared
        torch.testing.assert_close(fresh, source, atol=0, rtol=0)
    source_parameters = dict(core.vlm_with_expert.lm_expert.named_parameters())
    matrices = 0
    for name, parameter in model.expert.lm_expert.named_parameters():
        assert (parameter.data_ptr() == source_parameters[name].data_ptr()) == shared
        if parameter.ndim >= 2:
            torch.testing.assert_close(parameter, source_parameters[name], atol=0, rtol=0)
            matrices += 1
    report['checks']['shared_transformer_matrices' if shared else 'copied_transformer_matrices'] = matrices
    if shared:
        count = sum(p.numel() for p in model.expert.ffn_lora.parameters())
        assert count == 4_251_648
        assert len(model.expert.ffn_lora) == 16
        assert not any(k.startswith(('expert.lm_expert.', 'expert.action_', 'expert.state_proj.'))
                       for k in model.added_state_dict())
        baseline_snapshot = {k: v.detach().cpu().clone() for k, v in model.baseline.state_dict().items()}
        report['checks']['ffn_lora_parameters'] = count
        report['checks']['no_duplicate_base_weights_in_checkpoint'] = True
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        prefix = encode_prefix(model.baseline, batch)
        x = torch.randn(1, 50, core.config.max_action_dim, device='cuda')
        tau = torch.tensor([.4], device='cuda')
        original = core.denoise_step(prefix.pad_mask, prefix.kv, x, tau)
        condition = model.resampler(torch.randn(1, model.predictor.queries.shape[1], 256, device='cuda'))
        report['checks']['action_tactile_tokens'] = condition.shape[1]
        fresh = model.expert(x, tau, prefix, condition)
        torch.testing.assert_close(original, fresh, atol=1e-5, rtol=1e-5)
        changed = model.expert(x, tau, prefix, -condition)
        torch.testing.assert_close(fresh, changed, atol=0, rtol=0)
        report['checks']['initial_expert_max_abs_error'] = float((original-fresh).abs().max())
        report['checks']['tactile_route_zero_output_at_initialization'] = True
        suffix, pad, att = core.embed_suffix(x, tau)
        all_pad = torch.cat((prefix.pad_mask, pad), 1)
        all_att = torch.cat((prefix.attention_mask, att), 1)
        (_, uncached), _ = core.vlm_with_expert(
            attention_mask=make_att_2d_masks(all_pad, all_att), position_ids=all_pad.cumsum(1)-1,
            inputs_embeds=[prefix.embeddings, suffix], use_cache=False, fill_kv_cache=False)
        uncached = core.action_out_proj(uncached.float())
        torch.testing.assert_close(original, uncached, atol=0.04, rtol=0.04)
        report['checks']['cached_vs_uncached_max_abs_error_bf16'] = float((original-uncached).abs().max())
        noise = torch.randn_like(x)
        baseline_chunk = core.sample_actions(*model.baseline.prepare_images(batch),
            batch['observation.language.tokens'], batch['observation.language.attention_mask'],
            model.baseline.prepare_state(batch), noise=noise)
        refined = model.predict_action_chunk(batch, tactile['depth'], noise=noise)
        torch.testing.assert_close(refined, baseline_chunk[..., :9], atol=1e-5, rtol=1e-5)
        report['checks']['initial_full_chunk_max_abs_error'] = float((refined-baseline_chunk[..., :9]).abs().max())
        changed_prefix = replace(prefix, kv={i: {k: torch.randn_like(v)*10 for k, v in layer.items()}
                                             for i, layer in prefix.kv.items()})
        masked = model.expert(x, tau, prefix, condition, mask_prefix=True)
        masked_changed = model.expert(x, tau, changed_prefix, condition, mask_prefix=True)
        torch.testing.assert_close(masked, masked_changed, atol=0, rtol=0)
        assert torch.isfinite(masked).all()
        restored = model.expert(x, tau, changed_prefix, condition)
        assert not torch.allclose(fresh, restored)
        report['checks']['stage2_prefix_perturbation_max_abs_error'] = float((masked-masked_changed).abs().max())
        report['checks']['unmasked_prefix_access_restored'] = True
    routed_masks = []
    hook = model.expert.register_forward_pre_hook(
        lambda module, args, kwargs: routed_masks.append(kwargs.get('mask_prefix', False)), with_kwargs=True)
    for stage in active_stages(config):
        model.set_stage(stage); model.train(); model.zero_grad(set_to_none=True)
        # One update per stage checks wiring, not scientific convergence.
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-5)
        if stage == 'expert':
            # The zero tactile output projections open on the first action update.
            # Check downstream resampler gradients on the following update.
            with torch.autocast('cuda', dtype=torch.bfloat16):
                opening_loss, opening_parts = model.loss(batch, tactile)
            assert set(opening_parts) == {'refinement'}
            opening_loss.backward()
            assert all(p.grad is None or p.grad.count_nonzero() == 0 for p in model.resampler.parameters())
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        start = time.monotonic()
        with torch.autocast('cuda', dtype=torch.bfloat16):
            loss, parts = model.loss(batch, tactile)
        if stage in ('expert', 'joint'):
            assert routed_masks[-1] == (stage == 'expert')
        assert torch.isfinite(loss)
        loss.backward()
        groups = {}
        for name in ('baseline', 'representation', 'decoder', 'predictor', 'expert', 'resampler'):
            module = getattr(model, name)
            grads = [p.grad for p in module.parameters() if p.grad is not None]
            groups[name] = {'trainable': sum(p.numel() for p in module.parameters() if p.requires_grad),
                           'gradient_tensors': len(grads),
                           'nonzero_gradient_tensors': sum(bool(g.abs().sum()) for g in grads)}
            assert all(torch.isfinite(g).all() for g in grads)
        assert groups['baseline']['gradient_tensors'] == 0
        if model.trex_n0_full_grid:
            assert all(p.grad is None for p in model.representation.encoder.parameters())
            assert not model.representation.encoder.training
        if stage != 'encoder' and not model.trex_n0_full_grid:
            assert groups['representation']['gradient_tensors'] == 0
        if stage == 'expert':
            assert groups['predictor']['gradient_tensors'] == 0
        if hasattr(model.resampler, 'tokens_per_finger'):
            if stage in ('expert', 'joint'):
                assert groups['resampler']['nonzero_gradient_tensors'] > 0
                assert set(parts) == {'refinement'}
            else:
                assert groups['resampler']['gradient_tensors'] == 0
        if stage == 'joint':
            assert groups['predictor']['nonzero_gradient_tensors'] > 0
            assert groups['decoder']['gradient_tensors'] == 0
        optimizer.step()
        if shared and stage in ('expert', 'joint'):
            assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.expert.ffn_lora.parameters())
        if shared:
            assert not any(p.requires_grad for p in model.baseline.parameters())
            for name, parameter in model.baseline.state_dict().items():
                torch.testing.assert_close(parameter.cpu(), baseline_snapshot[name], atol=0, rtol=0)
        report['checks'][stage] = {'loss': float(loss), 'seconds': time.monotonic()-start, 'groups': groups}
        del optimizer, loss
    model.eval()
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        unchanged = core.denoise_step(prefix.pad_mask, prefix.kv, x, tau)
        torch.testing.assert_close(original, unchanged, atol=0, rtol=0)
        report['checks']['baseline_forward_unchanged_after_adapter_training'] = True
        noise = torch.randn_like(x)
        before = model.predict_action_chunk(batch, tactile['depth'], noise=noise)
        assert before.shape == (1, 50, 9) and torch.isfinite(post(before)).all()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'added.pt'
            torch.save(model.added_state_dict(), path)
            model.load_added_state_dict(torch.load(path, weights_only=True))
        after = model.predict_action_chunk(batch, tactile['depth'], noise=noise)
        assert not any(routed_masks[-4:]), 'Inference must restore the prefix'
        torch.testing.assert_close(before, after, atol=0, rtol=0)
        report['checks']['checkpoint_roundtrip'] = True
    hook.remove()
    report['peak_vram_gib'] = torch.cuda.max_memory_allocated()/2**30
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
