from types import SimpleNamespace

import h5py
import numpy as np
import pytest
import torch

from tactile_refinement import ablation as a
from tactile_refinement import ablation_rollout as r
from tactile_refinement.model import RefinementModel


def test_continuation_preserves_stage2_mask_and_handoff():
    seen = []
    model = SimpleNamespace(stage='expert', stage2_mask_prefix=True)
    def expert(x, tau, prefix, compact, *, mask_prefix):
        seen.append(mask_prefix)
        return torch.ones_like(x)
    model.expert = expert
    handoff = torch.zeros(1, 50, 32)
    out = RefinementModel.continue_action_chunk(model, None, None, handoff)
    assert seen == [True]*4
    assert not handoff.any()
    torch.testing.assert_close(out, torch.full((1, 50, 9), -.4))
    model.stage = 'joint'
    RefinementModel.continue_action_chunk(model, None, None, handoff)
    assert seen[-4:] == [False]*4


def test_sensor_interventions_keep_context_and_zero_after_resampler():
    contexts = []
    prefix = object()
    def predictor(depth, context):
        contexts.append(context)
        return depth+7
    model = SimpleNamespace(predicted_tokens=predictor, resampler=lambda x: x+3)
    out = a.compact_conditions(model, torch.ones(1), torch.full((1,), 2.), prefix)
    assert contexts == [prefix, prefix]
    assert out['correct'].item() == 11
    assert out['mismatched'].item() == 12
    assert out['disabled'].item() == 0


def test_offline_pairs_share_handoff_bridge_and_ignore_padded_actions(monkeypatch):
    prefix = object()
    monkeypatch.setattr(a, 'encode_prefix', lambda *args: prefix)
    monkeypatch.setattr(a, 'seeded_noise', lambda *args: torch.ones(1, 50, 32))
    demo = torch.zeros(1, 50, 32)
    demo[:, 1:] = 1000  # Padding must have no effect on scores.
    handoffs, bridge_states, masks = [], [], []
    def continuation(ctx, compact, y):
        assert ctx is prefix
        handoffs.append(y.clone())
        return compact.expand(1, 50, 9)
    def expert(x, tau, ctx, compact, *, mask_prefix):
        masks.append(mask_prefix)
        bridge_states.append(x.clone())
        return compact.expand_as(x)
    model = SimpleNamespace(baseline=SimpleNamespace(prepare_action=lambda batch: demo),
        handoff=lambda ctx, noise: noise, predicted_tokens=lambda depth, ctx: depth,
        resampler=lambda x: x, continue_action_chunk=continuation, expert=expert, stage2_mask_prefix=True)
    pad = torch.ones(1, 50, dtype=torch.bool)
    pad[:, 0] = False
    metrics = a.offline_trial(model, {'action_is_pad': pad}, torch.zeros(1, 1, 1), torch.ones(1, 1, 1), 123)
    assert metrics['correct']['action_mse'] == 0
    assert metrics['mismatched']['action_mse'] == 1
    assert metrics['mismatched']['action_rms_difference_from_correct'] == 1
    assert masks == [True]*12
    for h in handoffs:
        torch.testing.assert_close(h, handoffs[0])
    for j in range(4):
        torch.testing.assert_close(bridge_states[j], bridge_states[j+4])
        torch.testing.assert_close(bridge_states[j], bridge_states[j+8])


def test_donors_are_same_task_other_episode_and_keep_fingers_ordered(tmp_path, monkeypatch):
    monkeypatch.setattr(a, 'ROOT', tmp_path)
    episodes = []
    for i, task in enumerate(('one', 'one', 'two')):
        path = f'{i}.h5'
        with h5py.File(tmp_path/path, 'w') as f:
            f['embodiment/joint'] = np.full((3, 9), i, np.float32)
            f['tactile/left_gsmini/depth'] = np.full((3, 240, 320), 30-i, np.float32)
            f['tactile/right_gsmini/depth'] = np.full((3, 240, 320), 29-i, np.float32)
        episodes.append(dict(task=task, path=path, frames=3))
    bank = a.DonorBank(SimpleNamespace(episodes=episodes), ['one', 'two'], per_task=10)
    depth, info = bank.select('one', np.zeros(9), exclude_path='0.h5')
    assert info['path'] == '1.h5'
    assert depth.shape == (2, 240, 320)
    torch.testing.assert_close(depth[:, 0, 0], torch.tensor([.5, .6]))


def test_bootstrap_clusters_by_episode_and_single_unit_has_no_ci():
    assert a.mean_ci([1, 2], ['a', 'a'])['ci95'] is None
    assert a.mean_ci([1, 1, 1], ['a', 'b', 'c'])['ci95'] == [1., 1.]
    assert a.mean_ci([])['mean'] is None


def test_summary_does_not_claim_exact_pairs_or_count_smokes():
    rows = [dict(task='insert_HDMI', seed=1, condition=c, smoke=False, success=c == 'correct',
                 early_stop=False, simulated_seconds=2., initial_info={'initial_state_id': str(i)})
            for i, c in enumerate(a.CONDITIONS)]
    summary = r.summarize(rows, ['insert_HDMI'], 1)
    comparison = summary['tasks']['insert_HDMI']['comparisons']['mismatched']
    assert comparison['matching_recorded_reset_ids'] == 0
    assert comparison['correct_minus_control_success_matching_recorded_reset_ids']['mean'] is None
    assert comparison['correct_minus_control_success_same_seed_not_exactly_paired']['mean'] == 1
    smoke = [{**row, 'smoke': True, 'success': None} for row in rows]
    assert r.summarize(smoke, ['insert_HDMI'], 1)['tasks']['insert_HDMI']['conditions']['correct']['episodes'] == 0
    with pytest.raises(ValueError):
        r.validate_rows(rows+rows, ['insert_HDMI'], 1, 1, 0)


def test_binary_intervals_do_not_claim_certainty_with_no_successes():
    rate = r.success_rate([0]*20)
    assert rate['mean'] == 0
    assert rate['ci95'][1] > .1
    effect = r.success_difference([({'success': False}, {'success': False})]*20)
    assert effect['mean'] == 0
    assert effect['ci95'][0] < 0 < effect['ci95'][1]


@pytest.mark.parametrize('schedule,expected_calls', [('official', 3), ('matched60', 3), ('full50', 1)])
def test_rollout_schedule_and_depth_only_substitution(tmp_path, monkeypatch, schedule, expected_calls):
    class Env:
        def observation(self):
            return {'state': np.zeros(9), 'tactile_depth_mm': np.full((2, 240, 320), 34.), 't': self.t}
        def reset(self, seed):
            self.t = 0
            return self.observation(), {'initial_state_id': 'a'}
        def step(self, action):
            self.t += 1
            n = 1 if schedule == 'official' else 2
            commands = ([{'position': np.zeros(8), 'velocity': np.zeros(8),
                          'observed_position': np.zeros(8)} for _ in range(n)]
                        if schedule == 'matched60' else [])
            return self.observation(), 0, False, self.t == 3, {
                'physics_steps': n, 'simulated_seconds': n/120, 'success': False,
                'controller_commands': commands}
    seen = []
    class Model:
        baseline = SimpleNamespace(reset=lambda: None)
        def predict_action_chunk(self, batch, depth, *, condition, noise):
            seen.append(batch['t'])
            assert depth.mean() == 1  # Donor only; original context retains timestep.
            assert condition == 'predicted'
            return torch.zeros(1, 50, 9)
    monkeypatch.setattr(r, 'model_input', lambda obs: {'t': obs['t']})
    monkeypatch.setattr(r, 'prepare_batch', lambda obs, pre: obs)
    monkeypatch.setattr(r, 'seeded_noise', lambda model, seed: seed)
    bank = SimpleNamespace(select=lambda *args: (torch.ones(2, 240, 320), {'path': 'donor'}))
    row = r.run_episode(Env(), Model(), None, lambda x: x, bank, task='insert_HDMI', seed=7,
        condition='mismatched', schedule=schedule, device='cpu', output=tmp_path)
    assert row['steps'] == 3
    assert row['model_calls'] == expected_calls
    assert seen == list(range(expected_calls))
    assert row['physics_steps'] == (3 if schedule == 'official' else 6)
    assert len(row['donors']) == expected_calls
    trace = np.load(tmp_path/row['action_trace'])
    assert ('substep_positions' in trace) == (schedule == 'matched60')
