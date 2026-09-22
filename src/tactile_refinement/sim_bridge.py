"""Opt-in tactile IPC used only by the new refinement benchmark."""
import numpy as np
from smolvla_baseline.sim_bridge import UniVTACEnvironment, convert_observation, as_numpy


def convert_tactile_observation(raw, instruction):
    result = convert_observation(raw, instruction)
    sensors = raw['tactile']
    arrays = []
    for side in ('left', 'right'):
        names = [name for name in sensors if name.startswith(side+'_')]
        if len(names) != 1:
            raise ValueError(f'Ambiguous {side} sensor: {list(sensors)}')
        depth = as_numpy(sensors[names[0]]['depth']).astype(np.float32)
        if depth.shape != (240, 320):
            raise ValueError(f'Unexpected live depth shape: {depth.shape}')
        arrays.append(depth)
    result['tactile_depth_mm'] = np.stack(arrays)
    return result


class TactileEnvironment(UniVTACEnvironment):
    def __init__(self, *, task, controller='hold'):
        super().__init__(task=task, worker_module='tactile_refinement.sim_worker',
                         worker_args=('--controller', controller))
