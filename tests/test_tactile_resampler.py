"""Spatial compression, action integration and safe grounding checkpoint reuse."""
import copy

import pytest
import torch
from torch import nn

from tactile_refinement.model import Prefix
from tactile_refinement.representation import SpatialTactileResampler
from tactile_refinement.train import validate_resume_config
from test_ffn_lora_refinement import tiny_refiner


def test_spatial_sensitivity_finger_isolation_and_downstream_gradients():
    torch.set_num_threads(2)
    torch.manual_seed(7)
    module = SpatialTactileResampler(width=32)
    dense = torch.randn(2, 600, 32, requires_grad=True)
    compact = module(dense)
    assert compact.shape == (2, 32, 32)
    changed = dense.detach().clone()
    changed[:, 300:] = torch.randn_like(changed[:, 300:])*3
    altered = module(changed)
    torch.testing.assert_close(compact[:, :16], altered[:, :16], atol=0, rtol=0)
    assert not torch.allclose(compact[:, 16:], altered[:, 16:])
    # Identical content moved between locations must change its encoding.
    moved = dense.detach().clone()
    moved[:, :300] = moved[:, :300].flip(1)
    assert not torch.allclose(compact[:, :16], module(moved)[:, :16])
    assert not hasattr(module, 'reconstruct')
    assert not any('reconstruction' in name for name, _ in module.named_parameters())
    (compact*torch.randn_like(compact)).mean().backward()
    assert dense.grad.abs().sum() > 0
    assert module.queries.grad.abs().sum() > 0
    assert module.row.grad.abs().sum() > 0
    with pytest.raises(ValueError):
        module(torch.randn(1, 32, 32))


@pytest.mark.parametrize('implementation', ['copied_expert_v1', 'shared_ffn_lora_v1'])
def test_grounding_migration_preserves_weights_and_optimizer(implementation):
    old = tiny_refiner(implementation)
    old.set_stage('grounding')
    optimizer = torch.optim.AdamW([p for p in old.parameters() if p.requires_grad])
    sum(p.square().mean() for p in old.parameters() if p.requires_grad).backward()
    optimizer.step()
    state = old.added_state_dict()
    new = tiny_refiner(resampler_tokens_per_finger=16)
    with pytest.raises(ValueError, match='validated encoder/grounding'):
        new.load_added_state_dict(state)
    new.load_added_state_dict(state, allow_legacy_grounding=True)
    for key, value in state.items():
        if not key.startswith('expert.'):
            torch.testing.assert_close(new.added_state_dict()[key], value, atol=0, rtol=0)
    restored_optimizer = torch.optim.AdamW([p for p in new.parameters() if p.requires_grad])
    restored_optimizer.load_state_dict(optimizer.state_dict())
    for before, after in zip(optimizer.state.values(), restored_optimizer.state.values(), strict=True):
        torch.testing.assert_close(before['exp_avg'], after['exp_avg'], atol=0, rtol=0)
    assert not any(p.requires_grad for p in new.resampler.parameters())
    broken = dict(state)
    broken.pop('predictor.queries')
    with pytest.raises(ValueError, match='keys mismatch'):
        new.load_added_state_dict(broken, allow_legacy_grounding=True)
    roundtrip = tiny_refiner(resampler_tokens_per_finger=16)
    roundtrip.load_added_state_dict(new.added_state_dict())
    dense = torch.randn(1, 600, 256)
    torch.testing.assert_close(new.resampler(dense), roundtrip.resampler(dense), atol=0, rtol=0)


def test_config_migration_is_limited_to_unused_grounding_modules():
    old = {'training_recipe': 'trex_fullgrid_n0_v1', 'batch_size': 16, 'stage2_mask_prefix': True}
    new = dict(old, expert_implementation='shared_ffn_lora_v1', lora_rank=32, lora_alpha=32,
               resampler_tokens_per_finger=16)
    validate_resume_config(old, new, 'grounding')
    for stage in ('expert', 'joint'):
        with pytest.raises(ValueError):
            validate_resume_config(old, new, stage)
    with pytest.raises(ValueError):
        validate_resume_config(old, dict(new, batch_size=8), 'grounding')
    with pytest.raises(ValueError):
        validate_resume_config(new, dict(new, resampler_tokens_per_finger=8), 'expert')


def test_auxiliary_head_removal_only_migrates_unused_grounding_state():
    model = tiny_refiner(resampler_tokens_per_finger=16)
    state = model.added_state_dict()
    old_state = dict(state, **{'resampler.reconstruction_head.1.weight': torch.randn(256, 256)})
    with pytest.raises(ValueError, match='validated encoder/grounding'):
        model.load_added_state_dict(old_state)
    model.load_added_state_dict(old_state, allow_legacy_grounding=True)
    new_config = {'resampler_tokens_per_finger': 16, 'batch_size': 16}
    old_config = dict(new_config, lambda_compact_rec=1.0)
    validate_resume_config(old_config, new_config, 'grounding')
    for stage in ('expert', 'joint'):
        with pytest.raises(ValueError):
            validate_resume_config(old_config, new_config, stage)


def test_training_and_inference_use_compact_tokens_without_future_inputs(monkeypatch):
    torch.set_num_threads(2)
    model = tiny_refiner(resampler_tokens_per_finger=16)
    # Match the flattened cached prefix K/V width of the real cross-attention path.
    model.expert.lm_expert.layers[1].self_attn.k_proj = nn.Linear(16, 16, bias=False)
    model.expert.lm_expert.layers[1].self_attn.v_proj = nn.Linear(16, 16, bias=False)
    pad = torch.ones(1, 3, dtype=torch.bool)
    kv = {i: {key: torch.randn(1, 3, 2, 8) for key in ('key_states', 'value_states')} for i in range(2)}
    prefix = Prefix(torch.randn(1, 3, 64), pad, pad, kv, torch.randn(1, 32), torch.randn(1, 3, 64), pad)
    monkeypatch.setattr('tactile_refinement.model.encode_prefix', lambda *args: prefix)
    monkeypatch.setattr(model, 'handoff', lambda prefix, noise=None: torch.zeros(1, 50, 32))
    model.baseline.prepare_action = lambda batch: batch['action']
    tactile = {'depth': torch.rand(1, 2, 240, 320)}  # no future/change fields
    batch = {'action': torch.randn(1, 50, 32), 'action_is_pad': torch.zeros(1, 50, dtype=torch.bool)}
    conditions = []
    handle = model.expert.register_forward_pre_hook(lambda module, args: conditions.append(args[3].detach().clone()))
    base_before = {k: v.clone() for k, v in model.baseline.state_dict().items()}
    encoder_before = copy.deepcopy(model.representation.encoder.state_dict())
    # Zero-initialized tactile output projections block upstream action gradients
    # on the first update. Verify that an action-only update opens that path.
    model.set_stage('expert')
    first_loss, first_metrics = model.loss(batch, tactile)
    first_loss.backward()
    assert set(first_metrics) == {'refinement'}
    assert all(p.grad is None or p.grad.count_nonzero() == 0 for p in model.resampler.parameters())
    assert model.expert.tactile[0].attention.out_proj.weight.grad.abs().sum() > 0
    torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4).step()
    for stage in ('expert', 'joint'):
        model.set_stage(stage)
        model.train()
        model.zero_grad(set_to_none=True)
        loss, metrics = model.loss(batch, tactile)
        loss.backward()
        assert torch.isfinite(loss) and set(metrics) == {'refinement'}
        torch.testing.assert_close(loss.detach(), metrics['refinement'], atol=0, rtol=0)
        assert conditions[-1].shape == (1, 32, 256)
        assert model.resampler.queries.grad.abs().sum() > 0
        assert all(p.grad is None for p in model.representation.encoder.parameters())
        assert all(p.grad is None for p in model.baseline.parameters())
        if stage == 'expert':
            assert all(p.grad is None for p in model.predictor.parameters())
            assert all(p.grad is None for p in model.representation.projector.parameters())
        else:
            assert model.predictor.queries.grad.abs().sum() > 0
            assert model.representation.projector.weight.grad.abs().sum() > 0
        torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4).step()
    for k, value in model.baseline.state_dict().items():
        torch.testing.assert_close(value, base_before[k], atol=0, rtol=0)
    for k, value in model.representation.encoder.state_dict().items():
        torch.testing.assert_close(value, encoder_before[k], atol=0, rtol=0)
    calls = []
    resampler_hook = model.resampler.register_forward_hook(lambda *args: calls.append(1))
    def forbidden(*args):
        raise AssertionError('Grounding reconstruction ran during inference')
    monkeypatch.setattr(model.decoder, 'forward', forbidden)
    for condition in ('predicted', 'current', 'zero'):
        conditions.clear()
        calls.clear()
        result = model.predict_action_chunk(batch, tactile['depth'], condition=condition)
        assert result.shape == (1, 50, 9) and torch.isfinite(result).all()
        assert len(calls) == 1 and len(conditions) == 4
        assert all(z.shape == (1, 32, 256) for z in conditions)
        if condition == 'zero':
            assert all(z.count_nonzero() == 0 for z in conditions)
    handle.remove()
    resampler_hook.remove()
