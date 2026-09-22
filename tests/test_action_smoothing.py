import numpy as np
import pytest

from tactile_refinement.smoothing import ArmEMA


def test_ema_carries_across_chunk_boundary_and_preserves_gripper():
    initial = np.zeros(9, dtype=np.float32)
    initial[:7] = 0.25
    smooth = ArmEMA(0.5, initial)
    chunks = [np.zeros((50, 9), dtype=np.float32), np.ones((50, 9), dtype=np.float32)]
    chunks[0][:, 7:] = [0.02, 0.03]
    chunks[1][:, 7:] = [0.01, 0.04]
    previous = initial[:7].copy()
    for chunk in chunks:
        original = chunk.copy()
        for action in chunk:
            filtered = smooth(action)
            expected = 0.5 * action[:7] + 0.5 * previous
            np.testing.assert_allclose(filtered[:7], expected)
            np.testing.assert_array_equal(filtered[7:], action[7:])
            previous = expected
        np.testing.assert_array_equal(chunk, original)
    reset = ArmEMA(0.5, initial)
    np.testing.assert_allclose(reset(chunks[1][0])[:7], 0.625)


def test_alpha_one_is_identity_and_invalid_alpha_rejected():
    action = np.linspace(0, 1, 9, dtype=np.float32)
    np.testing.assert_array_equal(ArmEMA(1, np.zeros(9))(action), action)
    for alpha in (0, -0.5, 1.1, float('nan')):
        with pytest.raises(ValueError):
            ArmEMA(alpha, np.zeros(9))
