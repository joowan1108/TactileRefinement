"""Shared-weight ownership, gradient isolation and checkpoint migration contracts."""
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from transformers import AutoModel, LlamaConfig

from tactile_refinement.model import RefinementModel
from tactile_refinement.train import validate_resume_config


def tiny_refiner(implementation='shared_ffn_lora_v1', **kwargs):
    config = LlamaConfig(hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                         num_attention_heads=8, num_key_value_heads=2, head_dim=8, vocab_size=8)
    wrapper = nn.Module()
    wrapper.lm_expert = AutoModel.from_config(config)
    wrapper.lm_expert.embed_tokens = None
    for name, value in dict(num_vlm_layers=2, num_expert_layers=2, self_attn_every_n_layers=2,
                            attention_mode='cross_attn', expert_hidden_size=64,
                            num_attention_heads=8, num_key_value_heads=2,
                            config=SimpleNamespace(text_config=config)).items():
        setattr(wrapper, name, value)
    core = nn.Module()
    core.vlm_with_expert = wrapper
    core.config = SimpleNamespace(min_period=.004, max_period=4.)
    core.action_in_proj = nn.Linear(32, 64)
    core.action_out_proj = nn.Linear(64, 32)
    core.action_time_mlp_in = nn.Linear(128, 64)
    core.action_time_mlp_out = nn.Linear(64, 64)
    core.state_proj = nn.Linear(32, 64)
    baseline = nn.Module()
    baseline.model = core
    baseline.config = SimpleNamespace(use_tactile=False, adapt_to_pi_aloha=False, chunk_size=50, use_cache=True)
    return RefinementModel(baseline, trex_n0_full_grid=True, expert_implementation=implementation, **kwargs)


def test_shared_ownership_zero_init_and_adapter_only_updates():
    model = tiny_refiner()
    expert = model.expert
    core = model.baseline.model
    assert expert.lm_expert is core.vlm_with_expert.lm_expert
    assert expert.action_in_proj is core.action_in_proj
    baseline_ids = {id(p) for p in model.baseline.parameters()}
    assert not baseline_ids.intersection(id(p) for p in expert.parameters())
    assert not any(k.startswith(('expert.lm_expert.', 'expert.action_', 'expert.state_proj.'))
                   for k in model.added_state_dict())
    # Conversion of the parent must move the original parameters and aliases together.
    model.double()
    assert expert.lm_expert is core.vlm_with_expert.lm_expert
    assert expert.action_in_proj.weight.dtype == torch.float64
    model.float()
    mlp = expert.lm_expert.layers[0].mlp
    x = torch.randn(2, 3, 64)
    original = mlp(x).detach()
    torch.testing.assert_close(expert.feed_forward(0, x), original, atol=0, rtol=0)
    for stage in ('grounding', 'expert', 'joint'):
        model.set_stage(stage)
        model.train()
        assert not any(p.requires_grad for p in model.baseline.parameters())
        assert not core.vlm_with_expert.lm_expert.training
        assert all(p.requires_grad == (stage != 'grounding') for p in expert.ffn_lora.parameters())
    optimizer = torch.optim.AdamW(expert.parameters(), lr=.01)
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        expert.feed_forward(0, x).square().mean().backward()
        optimizer.step()
    assert all(p.grad is None for p in model.baseline.parameters())
    assert expert.ffn_lora[0]['gate_proj'].a.weight.grad.abs().sum() > 0
    assert expert.ffn_lora[0]['gate_proj'].b.weight.grad.abs().sum() > 0
    torch.testing.assert_close(mlp(x), original, atol=0, rtol=0)
    assert not torch.allclose(expert.feed_forward(0, x), original)


def test_checkpoint_roundtrip_and_explicit_grounding_migration():
    legacy = tiny_refiner('copied_expert_v1')
    state = legacy.added_state_dict()
    model = tiny_refiner()
    with pytest.raises(ValueError, match='validated encoder/grounding'):
        model.load_added_state_dict(state)
    model.load_added_state_dict(state, allow_legacy_grounding=True)
    for key, value in model.added_state_dict().items():
        if not key.startswith('expert.'):
            torch.testing.assert_close(value, state[key], atol=0, rtol=0)
    assert all(delta.b.weight.count_nonzero() == 0 for layer in model.expert.ffn_lora for delta in layer.values())
    broken = {k: v for k, v in state.items() if k != 'representation.projector.weight'}
    with pytest.raises(ValueError, match='keys mismatch'):
        model.load_added_state_dict(broken, allow_legacy_grounding=True)
    restored = tiny_refiner()
    restored.load_added_state_dict(model.added_state_dict())
    for key, value in restored.added_state_dict().items():
        torch.testing.assert_close(value, model.added_state_dict()[key], atol=0, rtol=0)


def test_migration_cannot_discard_a_trained_dense_expert_or_change_objective():
    old = {'batch_size': 16, 'stage2_mask_prefix': True,
           'expert_initialization': 'trained_smolvla_baseline_v1'}
    new = dict(old, expert_implementation='shared_ffn_lora_v1', lora_rank=32, lora_alpha=32)
    validate_resume_config(old, new, 'grounding')
    for stage in ('expert', 'joint'):
        with pytest.raises(ValueError):
            validate_resume_config(old, new, stage)
    for key, value in [('batch_size', 8), ('lora_rank', 16), ('lora_alpha', 64), ('stage2_mask_prefix', False)]:
        with pytest.raises(ValueError):
            validate_resume_config(old, dict(new, **{key: value}), 'grounding')
