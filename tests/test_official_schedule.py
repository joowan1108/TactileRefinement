import numpy as np
import pytest
from smolvla_baseline.official_schedule import OfficialTemporalAggregator


def test_official_aggregation_aligns_future_indices_keeps_zeros_and_resets():
    aggregator = OfficialTemporalAggregator('cpu')
    rng = np.random.default_rng(314)
    chunks = rng.normal(size=(65, 50, 9)).astype(np.float32)
    chunks[0] = 0  # Zero actions must count as populated predictions.
    for t, chunk in enumerate(chunks):
        result = aggregator(chunk[None])
        starts = np.arange(max(0, t-49), t+1)
        candidates = chunks[starts, t-starts]
        weights = np.exp(-0.01*np.arange(len(starts), dtype=np.float64))
        expected = (candidates*weights[:, None]).sum(axis=0)/weights.sum()
        np.testing.assert_allclose(result, expected, atol=2e-7, rtol=2e-6)
    aggregator.reset()
    np.testing.assert_array_equal(aggregator(chunks[-1]), chunks[-1, 0])


def test_oldest_prediction_has_higher_weight_and_capacity_is_enforced():
    aggregator = OfficialTemporalAggregator('cpu')
    aggregator(np.zeros((50, 9), dtype=np.float32))
    action = aggregator(np.ones((50, 9), dtype=np.float32))
    assert np.all(action < 0.5)  # Positive k favors older, not newer, predictions.
    aggregator.reset()
    aggregator.state.t = aggregator.state.temporal_agg_max_timesteps-1
    np.testing.assert_array_equal(aggregator(np.ones((50, 9))), np.ones(9))
    with pytest.raises(RuntimeError, match='exhausted'):
        aggregator(np.ones((50, 9)))
