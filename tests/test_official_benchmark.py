from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from smolvla_baseline import official_benchmark as benchmark
from smolvla_baseline.official_schedule import OfficialTemporalAggregator


def test_resume_rejects_duplicates_foreign_seeds_and_wrong_schedule():
    row = dict(task='insert_HDMI', seed=10, steps=3, model_calls=3, physics_steps=3, success=False)
    assert benchmark.validate_completed_rows([row], ['insert_HDMI'], 2, 10) == {('insert_HDMI', 10)}
    for rows in ([row, row], [{**row, 'seed': 12}], [{**row, 'physics_steps': 6}]):
        with pytest.raises(ValueError):
            benchmark.validate_completed_rows(rows, ['insert_HDMI'], 2, 10)
    summary = benchmark.summarize(dict(tasks=['insert_HDMI', 'lift_can'], episodes_per_task=2), [row])
    assert not summary['complete'] and summary['episodes'] == 1
    assert summary['tasks']['lift_can']['success_rate'] is None


def test_rollout_replans_from_every_new_observation_and_preserves_step_counts(tmp_path, monkeypatch):
    class Env:
        def reset(self, seed):
            self.t = 0
            return {'state': np.zeros(9), 't': 0}, {'initial_state_id': 'test'}
        def step(self, action):
            self.t += 1
            return {'state': action, 't': self.t}, 0, False, self.t == 3, {
                'physics_steps': 1, 'policy_action_count': self.t, 'simulated_seconds': 1/120, 'success': False}
    seen = []
    class Model:
        def predict_action_chunk(self, batch):
            seen.append(batch['t'])
            return torch.full((1, 50, 9), float(batch['t']))
    monkeypatch.setattr(benchmark, 'model_input', lambda obs: obs)
    monkeypatch.setattr(benchmark, 'prepare_batch', lambda obs, pre: obs)
    monkeypatch.setattr(benchmark.torch, 'autocast', lambda *args, **kwargs: nullcontext())
    row = benchmark.run_episode(Env(), Model(), None, lambda x: x, OfficialTemporalAggregator('cpu'),
                                tmp_path, 'insert_HDMI', 0, 10, record_video=False)
    assert seen == [0, 1, 2]
    assert row['steps'] == row['model_calls'] == row['physics_steps'] == 3
    assert row['simulated_seconds'] == pytest.approx(3/120)
    assert 'video' not in row and 'action_trace' not in row
    assert row['max_aggregation_reconstruction_error'] < 1e-6
