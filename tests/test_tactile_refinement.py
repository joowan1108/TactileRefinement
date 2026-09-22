import json
from types import SimpleNamespace

import cv2
import h5py
import numpy as np
import pytest
import torch

from tactile_refinement.data import TactileEpisodes, normalize_depth
from tactile_refinement.execution import held_position_step
from tactile_refinement.losses import contrastive_loss, refinement_bridge, velocity_loss
from tactile_refinement.representation import TactileRepresentation, TokenDecoder, Predictor
from tactile_refinement.sim_bridge import convert_tactile_observation


def test_fixed_signed_depth_and_invalid_pixels():
    depth, valid = normalize_depth(torch.tensor([34., 24., 29., 35., 23., float('nan'), 0., float('inf')]))
    torch.testing.assert_close(depth, torch.tensor([0., 1., .5, 0., 1., 0., 0., 0.]))
    assert valid.tolist() == [True]*5+[False]*3
    current, _ = normalize_depth(torch.tensor([25., 33.]))
    future, _ = normalize_depth(torch.tensor([33., 25.]))
    torch.testing.assert_close(future-current, torch.tensor([-.8, .8]))


@pytest.fixture
def tactile_manifest(tmp_path):
    path = tmp_path/'episode.hdf5'
    n = 55
    with h5py.File(path, 'w') as f:
        f['embodiment/joint'] = np.arange(n*9, dtype=np.float32).reshape(n, 9)
        # Frame 52 is the start of a new continuous recording segment.
        f['step'] = np.r_[np.arange(52)*2, np.arange(3)*2]
        encoded = cv2.imencode('.jpg', np.zeros((8, 8, 3), dtype=np.uint8))[1]
        for camera in ('head', 'wrist'):
            dataset = f.create_dataset(f'observation/{camera}/rgb', (n,), dtype=h5py.vlen_dtype(np.uint8))
            for i in range(n):
                dataset[i] = encoded
        for side in ('left', 'right'):
            depth = np.full((n, 240, 320), 30., dtype=np.float32)
            depth[50:] = 32.
            f[f'tactile/{side}_gsmini/depth'] = depth
    common = dict(path=str(path), instruction='test', task='test', episode_id=0, recorded_step_stride=[2])
    manifest = tmp_path/'manifest.json'
    manifest.write_text(json.dumps({'training_protocol': 'all_episodes_no_holdout', 'episodes': [
        dict(common, frame_offset=0, frames=52), dict(common, frame_offset=52, frames=3)]}))
    return manifest


def test_endpoint_and_action_alignment_without_clamping(tactile_manifest):
    ds = TactileEpisodes(tactile_manifest)
    first = ds[0]
    assert first['future_valid']
    torch.testing.assert_close(first['change'], torch.full((2, 240, 320), -.2))
    assert first['baseline']['action'][0, 0] == 9
    assert first['baseline']['action'][-1, 0] == 50*9
    assert not ds[2]['future_valid']  # t+50 crosses the segment boundary
    assert not ds[2]['change_mask'].any()
    end = ds[50]
    assert (~end['baseline']['action_is_pad']).sum() == 1
    assert ds[51]['frame'] == 52  # next segment starts here


def test_action_stages_never_read_future_depth(tactile_manifest):
    ds = TactileEpisodes(tactile_manifest, future=False)
    item = ds[0]
    assert 'change' not in item and 'future_valid' not in item
    # Remove every future tactile frame; current-only retrieval must still work.
    for handle in ds.handles.values():
        handle.close()
    ds.handles.clear()
    episode = json.loads(tactile_manifest.read_text())['episodes'][0]['path']
    with h5py.File(episode, 'a') as f:
        for side in ('left', 'right'):
            key = f'tactile/{side}_gsmini/depth'
            first = f[key][0:1]
            del f[key]
            f[key] = first
    assert ds[0]['depth'].shape == (2, 240, 320)


def test_bridge_integration_and_padded_loss():
    demo, handoff = torch.randn(2, 50, 32), torch.randn(2, 50, 32, requires_grad=True)
    x, target = refinement_bridge(demo, handoff, torch.full((2,), .4))
    torch.testing.assert_close(x, handoff)
    torch.testing.assert_close(x-.4*target, demo, atol=1e-6, rtol=1e-5)
    assert not target.requires_grad
    pred = target.clone().requires_grad_()
    mask = torch.zeros(2, 50, dtype=torch.bool); mask[:, 10:] = True
    shifted = pred.clone(); shifted[:, 10:] += 1e6; shifted[:, :, 9:] += 1e6
    assert velocity_loss(shifted, target, mask) == 0


def test_contrastive_no_invalid_negatives():
    pred = torch.randn(3, 32, 8, requires_grad=True); target = torch.randn_like(pred)
    valid = torch.ones(3, dtype=torch.bool)
    change = torch.rand(3, 2, 4, 4)
    loss = contrastive_loss(pred, target, valid, torch.zeros(3, dtype=torch.long), torch.arange(3), change)
    assert loss == 0  # overlapping windows cannot be negatives
    loss = contrastive_loss(pred, target, valid, torch.arange(3), torch.arange(3), torch.zeros_like(change))
    assert loss == 0  # zero change targets cannot be negatives
    loss = contrastive_loss(pred, target, valid, torch.arange(3), torch.arange(3), change)
    assert torch.isfinite(loss) and loss > 0
    loss.backward(); assert pred.grad.abs().sum() > 0


def test_representation_shapes_and_predictor_padding():
    torch.set_num_threads(2)
    encoder = TactileRepresentation().eval()
    x = torch.rand(1, 2, 240, 320)
    assert encoder.encoder(x[:, :1]).shape == (1, 128, 15, 20)
    z = encoder(x)
    assert z.shape == (1, 32, 256)
    assert TokenDecoder()(z).shape == x.shape
    predictor = Predictor(64).eval()
    context = torch.randn(1, 5, 64)
    mask = torch.tensor([[True, True, False, False, False]])
    before = predictor(z, context, mask)
    context[:, 2:] += 1000
    torch.testing.assert_close(before, predictor(z, context, mask))


class FakeTask:
    def __init__(self, terminate=None):
        self.cfg = SimpleNamespace(decimation=1, step_lim=600)
        self.step_count = self.take_action_cnt = 0
        self.eval_success = False
        self.terminate = terminate
    def take_action(self, qpos, action_type):
        self.take_action_cnt += 1; self._step()
        return True, self.check_success()
    def _step(self):
        self.step_count += 1
    def check_success(self):
        return self.step_count == self.terminate
    def check_early_stop(self):
        return False


def test_two_step_holds_count_once_and_stop_early():
    task = FakeTask()
    for _ in range(50):
        assert held_position_step(task, np.zeros(8))[-1] == 2
    assert task.step_count == 100 and task.take_action_cnt == 50
    task = FakeTask(terminate=1)
    result = held_position_step(task, np.zeros(8))
    assert result[1] and result[-1] == 1


def test_pretrained_expert_copies_weights_without_sharing_storage():
    from transformers import AutoModel, LlamaConfig
    from tactile_refinement.model import CachedTactileExpert
    config = LlamaConfig(hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                         num_attention_heads=8, num_key_value_heads=2, head_dim=8, vocab_size=8)
    source = AutoModel.from_config(config)
    source.embed_tokens = None
    # Include a cross-attention projection with a nonstandard K/V input width.
    source.layers[1].self_attn.k_proj = torch.nn.Linear(32, 16, bias=False)
    source.layers[1].self_attn.v_proj = torch.nn.Linear(32, 16, bias=False)
    wrapper = SimpleNamespace(lm_expert=source, num_vlm_layers=2, num_expert_layers=2,
        self_attn_every_n_layers=2, attention_mode='cross_attn', expert_hidden_size=64,
        num_attention_heads=8, num_key_value_heads=2, config=SimpleNamespace(text_config=config))
    core = SimpleNamespace(vlm_with_expert=wrapper, config=SimpleNamespace(),
        action_in_proj=torch.nn.Linear(32, 64), action_out_proj=torch.nn.Linear(64, 32),
        action_time_mlp_in=torch.nn.Linear(128, 64), action_time_mlp_out=torch.nn.Linear(64, 64),
        state_proj=torch.nn.Linear(32, 64))
    originals = {name: parameter.detach().clone() for name, parameter in source.named_parameters()}
    torch.manual_seed(31)
    first = CachedTactileExpert(core, implementation='copied_expert_v1')
    for name, parameter in source.named_parameters():
        torch.testing.assert_close(parameter, originals[name], atol=0, rtol=0)
        fresh = dict(first.lm_expert.named_parameters())[name]
        assert parameter.data_ptr() != fresh.data_ptr()
        torch.testing.assert_close(parameter, fresh, atol=0, rtol=0)
    for name in ('action_in_proj', 'action_out_proj', 'action_time_mlp_in', 'action_time_mlp_out', 'state_proj'):
        copied, original = getattr(first, name), getattr(core, name)
        torch.testing.assert_close(copied.weight, original.weight, atol=0, rtol=0)
        assert copied.weight.data_ptr() != original.weight.data_ptr()
    # Source changes do not mutate an existing copy; new copies use new weights.
    with torch.no_grad():
        for parameter in source.parameters():
            parameter.fill_(17.0)
        for name in ('action_in_proj', 'action_out_proj', 'action_time_mlp_in', 'action_time_mlp_out', 'state_proj'):
            getattr(core, name).weight.fill_(17.0)
    torch.manual_seed(31)
    second = CachedTactileExpert(core, implementation='copied_expert_v1')
    for name, value in first.lm_expert.named_parameters():
        torch.testing.assert_close(value, originals[name], atol=0, rtol=0)
    assert (second.action_in_proj.weight == 17).all()
    assert not torch.equal(first.action_in_proj.weight, second.action_in_proj.weight)
    assert first.tactile[0].attention.out_proj.weight.count_nonzero() == 0
    assert first.state_adapter.weight.count_nonzero() == 0
    assert first.smolvla_initialization_version.item() == 1


def test_stage2_mask_configuration_migration_is_limited_to_early_stages():
    from tactile_refinement.train import validate_resume_config
    before = {'batch_size': 16, 'seed': 42}
    after = dict(before, stage2_mask_prefix=True)
    for stage in ('encoder', 'grounding'):
        validate_resume_config(before, after, stage)
    for stage in ('expert', 'joint'):
        with pytest.raises(ValueError):
            validate_resume_config(before, after, stage)
    with pytest.raises(ValueError):
        validate_resume_config(before, dict(after, batch_size=8), 'encoder')
    validate_resume_config(after, after, 'expert')


def test_encoder_budget_can_only_be_extended_with_other_settings_unchanged():
    from tactile_refinement.train import validate_resume_config
    before = {'batch_size': 16, 'stages': {'encoder': {'updates': 10, 'learning_rate': 1e-4},
                                           'grounding': {'updates': 20, 'learning_rate': 1e-4}}}
    extended = {'batch_size': 16, 'stages': {'encoder': {'updates': 110, 'learning_rate': 1e-4},
                                             'grounding': {'updates': 20, 'learning_rate': 1e-4}}}
    validate_resume_config(before, extended, 'encoder')
    with pytest.raises(ValueError):
        validate_resume_config(before, extended, 'grounding')
    reduced = {'batch_size': 16, 'stages': {'encoder': {'updates': 9, 'learning_rate': 1e-4},
                                            'grounding': {'updates': 20, 'learning_rate': 1e-4}}}
    with pytest.raises(ValueError):
        validate_resume_config(before, reduced, 'encoder')
    changed_lr = {'batch_size': 16, 'stages': {'encoder': {'updates': 110, 'learning_rate': 2e-4},
                                               'grounding': {'updates': 20, 'learning_rate': 1e-4}}}
    with pytest.raises(ValueError):
        validate_resume_config(before, changed_lr, 'encoder')


@pytest.mark.parametrize('implementation', ['shared_ffn_lora_v1', 'copied_expert_v1'])
def test_prefix_mask_blocks_self_and_cross_attention_without_nan(implementation):
    from transformers import AutoModel, LlamaConfig
    from tactile_refinement.model import CachedTactileExpert, Prefix
    from dataclasses import replace
    config = LlamaConfig(hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                         num_attention_heads=8, num_key_value_heads=2, head_dim=8, vocab_size=8)
    source = AutoModel.from_config(config)
    source.embed_tokens = None
    wrapper = SimpleNamespace(lm_expert=source, num_vlm_layers=2, num_expert_layers=2,
        self_attn_every_n_layers=2, attention_mode='cross_attn', expert_hidden_size=64,
        num_attention_heads=8, num_key_value_heads=2, config=SimpleNamespace(text_config=config))
    # Cross-attention consumes flattened prefix K/V (2 heads * 8 values).
    source.layers[1].self_attn.k_proj = torch.nn.Linear(16, 16, bias=False)
    source.layers[1].self_attn.v_proj = torch.nn.Linear(16, 16, bias=False)
    core = SimpleNamespace(vlm_with_expert=wrapper, config=SimpleNamespace(min_period=.004, max_period=4.),
        action_in_proj=torch.nn.Linear(32, 64), action_out_proj=torch.nn.Linear(64, 32),
        action_time_mlp_in=torch.nn.Linear(128, 64), action_time_mlp_out=torch.nn.Linear(64, 64),
        state_proj=torch.nn.Linear(32, 64))
    expert = CachedTactileExpert(core, implementation=implementation).eval()
    # Open added routes to test useful tactile/state gradients, not just zero-init equivalence.
    for block in expert.tactile:
        torch.nn.init.normal_(block.attention.out_proj.weight, std=.02)
    torch.nn.init.normal_(expert.state_adapter.weight, std=.02)
    kv = {i: {k: torch.randn(1, 5, 2, 8, requires_grad=True) for k in ('key_states', 'value_states')}
          for i in range(2)}
    pad = torch.ones(1, 5, dtype=torch.bool)
    state = torch.randn(1, 32, requires_grad=True)
    prefix = Prefix(torch.randn(1, 5, 64), pad, pad, kv, state, torch.randn(1, 5, 64), pad)
    other = replace(prefix, kv={i: {k: v.detach()*-12+50 for k, v in layer.items()} for i, layer in kv.items()})
    x, tau = torch.randn(1, 4, 32), torch.tensor([.2])
    z = torch.randn(1, 32, 256, requires_grad=True)
    result = expert(x, tau, prefix, z, mask_prefix=True)
    torch.testing.assert_close(result, expert(x, tau, other, z, mask_prefix=True), atol=0, rtol=0)
    assert not torch.allclose(expert(x, tau, prefix, z), expert(x, tau, other, z))
    result.square().mean().backward()
    assert torch.isfinite(result).all() and torch.isfinite(z.grad).all()
    assert z.grad.abs().sum() > 0 and state.grad.abs().sum() > 0
    assert all(v.grad is None or v.grad.count_nonzero() == 0 for layer in kv.values() for v in layer.values())
