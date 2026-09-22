"""Cached proprioception equivalence and receding-horizon execution contracts."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn
from transformers import AutoModel, LlamaConfig

from tactile_refinement import ablation_rollout as rollout
from tactile_refinement.model import Prefix, make_att_2d_masks, refresh_prefix_state
from lerobot.policies.smolvla.smolvlm_with_expert import SmolVLMWithExpertModel


def small_prefix_model():
    config = LlamaConfig(hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                         num_attention_heads=8, num_key_value_heads=2, head_dim=8, vocab_size=8)
    text = AutoModel.from_config(config).eval()
    wrapper = SmolVLMWithExpertModel.__new__(SmolVLMWithExpertModel)
    nn.Module.__init__(wrapper)
    wrapper.lm_expert = AutoModel.from_config(config).eval()
    wrapper.get_vlm_model = lambda: SimpleNamespace(text_model=text)
    wrapper.vlm = SimpleNamespace(config=SimpleNamespace(text_config=config))
    wrapper.num_vlm_layers = wrapper.num_expert_layers = 2
    wrapper.num_attention_heads, wrapper.num_key_value_heads = 8, 2
    wrapper.attention_mode, wrapper.self_attn_every_n_layers = 'cross_attn', 2
    core = SimpleNamespace(vlm_with_expert=wrapper, state_proj=nn.Linear(9, 64).eval())
    return SimpleNamespace(model=core, prepare_state=lambda batch: batch['state'])


@pytest.mark.parametrize('trailing_padding', [0, 3])
@torch.no_grad()
def test_refreshed_state_cache_matches_full_prefix_recompute(trailing_padding):
    """Exercise the real upstream transformer/attention with small random weights."""
    torch.manual_seed(42)
    baseline = small_prefix_model()
    core = baseline.model
    embeddings = torch.randn(2, 6+trailing_padding, 64)
    pad = torch.ones(2, embeddings.shape[1], dtype=torch.bool)
    pad[1, 3:5] = False  # Batched language padding must keep RoPE positions correct.
    if trailing_padding:
        pad[:, 6:] = False
    att = torch.zeros_like(pad)
    att[:, 5] = True

    def full(state):
        emb = embeddings.clone()
        emb[:, 5] = core.state_proj(state)
        (features, _), cache = core.vlm_with_expert(
            attention_mask=make_att_2d_masks(pad, att), position_ids=pad.cumsum(1)-1,
            inputs_embeds=[emb, None], use_cache=True, fill_kv_cache=True)
        return Prefix(features, pad & ~att, pad, cache, state, emb, att)

    cached = full(torch.randn(2, 9))
    original_cache = {i: {k: v.clone() for k, v in layer.items()} for i, layer in cached.kv.items()}
    for _ in range(3):
        state = torch.randn(2, 9)
        refreshed = refresh_prefix_state(baseline, cached, {'state': state})
        reference = full(state)
        torch.testing.assert_close(refreshed.state, state)
        torch.testing.assert_close(refreshed.features[pad], reference.features[pad], atol=2e-6, rtol=2e-5)
        for i in refreshed.kv:
            for name in refreshed.kv[i]:
                torch.testing.assert_close(refreshed.kv[i][name][pad], reference.kv[i][name][pad],
                                           atol=2e-6, rtol=2e-5)
                # Input cache is immutable, length never grows, and VL stays exact.
                assert refreshed.kv[i][name].shape == cached.kv[i][name].shape
                torch.testing.assert_close(refreshed.kv[i][name][:, :5], cached.kv[i][name][:, :5],
                                           atol=0, rtol=0)
        cached = refreshed
    # A different state actually changes the state KV, rather than merely metadata.
    assert not torch.allclose(cached.kv[1]['key_states'][:, 5], original_cache[1]['key_states'][:, 5])


@pytest.mark.parametrize('episode_steps', [1, 10, 23, 55])
def test_tactile10_vl50_cadence_new_horizon_and_episode_reset(tmp_path, monkeypatch, episode_steps):
    fresh_vl, fresh_state, predictions = [], [], []

    class Env:
        def observation(self):
            return {'state': np.full(9, self.t),
                    'tactile_depth_mm': np.full((2, 240, 320), 34-self.t/10), 't': self.t}

        def reset(self, seed):
            self.t = 0
            return self.observation(), {}

        def step(self, action):
            # Each replan's first ten slots are used, rather than the old tail.
            expected = (self.t//10)*100+self.t % 10
            np.testing.assert_array_equal(action, np.full(9, expected))
            self.t += 1
            command = dict(position=np.full(8, expected), velocity=np.zeros(8),
                           observed_position=np.full(8, expected))
            return self.observation(), 0, self.t == episode_steps, False, dict(
                physics_steps=2, simulated_seconds=2/120, success=False,
                controller_commands=[command, command])

    def encode(baseline, batch):
        fresh_vl.append(batch['t'])
        return SimpleNamespace(vl_time=batch['t'], state_time=batch['t'])

    def state(baseline, prefix, batch):
        fresh_state.append(batch['t'])
        return SimpleNamespace(vl_time=prefix.vl_time, state_time=batch['t'])

    class Model:
        baseline = SimpleNamespace(reset=lambda: None)

        def predict_action_chunk_from_prefix(self, prefix, depth, *, condition, noise):
            t = prefix.state_time
            assert prefix.vl_time == (t//50)*50
            assert condition == 'predicted'
            torch.testing.assert_close(depth.mean(), torch.tensor(t/100), atol=1e-6, rtol=1e-6)
            predictions.append(t)
            return (torch.arange(50)+t*10).float()[None, :, None].expand(1, 50, 9)

    monkeypatch.setattr(rollout, 'encode_prefix', encode)
    monkeypatch.setattr(rollout, 'refresh_prefix_state', state)
    monkeypatch.setattr(rollout, 'model_input', lambda obs: {'t': obs['t']})
    monkeypatch.setattr(rollout, 'prepare_batch', lambda obs, pre: obs)
    monkeypatch.setattr(rollout, 'seeded_noise', lambda model, seed: seed)
    replans = list(range(0, episode_steps, 10))
    vl_steps = list(range(0, episode_steps, 50))
    for seed in [1, 2]:
        row = rollout.run_episode(Env(), Model(), None, lambda x: x, None,
            task='insert_HDMI', seed=seed, condition='correct', schedule='tactile10_vl50',
            device='cpu', output=tmp_path)
        assert row['replan_steps'] == row['state_refresh_steps'] == row['tactile_refresh_steps'] == replans
        assert row['vl_refresh_steps'] == vl_steps
        assert row['model_calls'] == len(replans)
        assert row['physics_steps'] == 2*episode_steps
        trace = np.load(tmp_path/row['action_trace'])
        assert trace['predicted_chunks'].shape == (len(replans), 50, 9)
    assert predictions == replans*2
    assert fresh_vl == vl_steps*2
    assert fresh_state == [t for t in replans if t % 50]*2
