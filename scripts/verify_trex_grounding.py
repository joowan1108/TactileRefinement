"""Verify real pretrained weights and the shared-projector gradient paths."""
import json
import torch
from torch.utils.data import default_collate

from smolvla_baseline.data import ROOT
from tactile_refinement.train import build_refiner
from tactile_refinement.data import TactileEpisodes, prepare_refinement_batch


def main():
    torch.set_num_threads(4)
    torch.manual_seed(42)
    config = json.loads((ROOT/'configs/tactile_refinement.json').read_text())
    model, pre, _ = build_refiner(config)
    source = torch.load(ROOT/config['tactile_encoder_checkpoint'], weights_only=True, map_location='cpu')
    encoder_before = {k: v.detach().cpu().clone() for k, v in model.representation.encoder.state_dict().items()}
    for name, value in encoder_before.items():
        torch.testing.assert_close(value, source['encoder'][name].to(value.dtype), atol=0, rtol=0)
    dataset = TactileEpisodes(ROOT/config['manifest'])
    indices = [0, dataset.ends[99], dataset.ends[299], dataset.ends[599]]
    raw = default_collate([dataset[i] for i in indices])
    batch, tactile = prepare_refinement_batch(raw, pre, config['device'])
    assert tactile['future_valid'].all()
    model.set_stage('grounding'); model.train()
    calls = []
    hook = model.representation.projector.register_forward_hook(
        lambda m, args, output: calls.append((torch.is_grad_enabled(), output.requires_grad, list(output.shape))))
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        loss, parts = model.loss(batch, tactile)
    loss.backward()
    assert [item[:2] for item in calls] == [(True, True), (False, False)]
    groups = {}
    for name, module in [('encoder', model.representation.encoder), ('projector', model.representation.projector),
                         ('predictor', model.predictor), ('reconstruction_head', model.decoder),
                         ('baseline', model.baseline), ('expert', model.expert)]:
        grads = [p.grad for p in module.parameters() if p.grad is not None]
        assert all(torch.isfinite(g).all() for g in grads)
        groups[name] = sum(bool(g.abs().sum()) for g in grads)
        if name in ('encoder', 'baseline', 'expert'):
            assert not grads
        else:
            assert groups[name] > 0
    optimizer.step()
    for name, value in model.representation.encoder.state_dict().items():
        torch.testing.assert_close(value.cpu(), encoder_before[name], atol=0, rtol=0)
    hook.remove()
    report = {'pretrained_state_entries_exact': len(encoder_before), 'encoder_parameters_and_bn_unchanged': True,
              'projector_calls_grad_enabled_requires_grad_shape': calls, 'nonzero_gradient_tensors': groups,
              'loss': float(loss.detach()), 'parts': {k: float(v) for k, v in parts.items()}}
    (ROOT/'logs/trex_grounding_gradient_verification.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
