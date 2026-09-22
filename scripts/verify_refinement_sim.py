"""Verify live depth transfer and a 50-target/100-physics-step stationary hold."""
import json

import numpy as np

from smolvla_baseline.data import ROOT
from tactile_refinement.data import normalize_depth
from tactile_refinement.sim_bridge import TactileEnvironment


def main():
    env = TactileEnvironment(task='insert_HDMI')
    try:
        obs, initial = env.reset(seed=1000999)
        depth, mask = normalize_depth(obs['tactile_depth_mm'])
        assert depth.shape == (2, 240, 320) and mask.all()
        action = obs['state'].copy()
        physics = actions = 0
        seconds = 0.0
        for _ in range(50):
            obs, _, done, truncated, info = env.step(action)
            actions += 1; physics += info['physics_steps']; seconds += info['simulated_seconds']
            assert info['policy_action_count'] == actions
            assert np.isfinite(obs['tactile_depth_mm']).all()
            if done or truncated:
                break
        report = {'task': 'insert_HDMI', 'seed': 1000999, 'actions': actions, 'physics_steps': physics,
                  'simulated_seconds': seconds, 'depth_shape': list(depth.shape),
                  'finite_depth': True, 'initial_info': initial, 'final_info': info,
                  'complete_50_action_timing_check': actions == 50 and physics == 100}
        (ROOT/'logs/refinement_sim_verification.json').write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps(report, indent=2))
        assert report['complete_50_action_timing_check'], 'Early termination prevented the full cadence check'
        assert abs(seconds-100/120) < 1e-6
    finally:
        env.close()


if __name__ == '__main__':
    main()
