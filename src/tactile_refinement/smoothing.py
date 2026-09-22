"""Optional evaluation-only smoothing of denormalized absolute joint targets."""
import numpy as np


class ArmEMA:
    def __init__(self, alpha, initial_state):
        if not 0 < alpha <= 1:
            raise ValueError('EMA alpha must be in (0, 1]')
        self.alpha = float(alpha)
        self.previous = np.asarray(initial_state, dtype=np.float32)[:7].copy()
        if self.previous.shape != (7,) or not np.isfinite(self.previous).all():
            raise ValueError('Expected seven finite initial arm positions')

    def __call__(self, action):
        target = np.asarray(action, dtype=np.float32).copy()
        if target.shape != (9,) or not np.isfinite(target).all():
            raise ValueError('Expected nine finite denormalized joint targets')
        target[:7] = self.alpha * target[:7] + (1 - self.alpha) * self.previous
        self.previous = target[:7].copy()
        return target
