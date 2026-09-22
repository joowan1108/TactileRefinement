"""Use all 800 episodes for training and compute full-dataset normalization statistics."""
import json
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    download = json.loads((ROOT / 'data/download_manifest.json').read_text())
    records = []
    discontinuities = []
    states, actions = [], []
    for task_dir in sorted((ROOT / 'data/UniVTAC/isaac45').iterdir()):
        files = sorted(task_dir.glob('hdf5/*.hdf5'), key=lambda p: int(p.stem))
        assert len(files) == 100, (task_dir, len(files))
        instruction = json.loads((ROOT / 'UniVTAC/instructions' / f'{task_dir.name}.json').read_text())['seen'][0]
        for path in files:
            with h5py.File(path) as f:
                q = f['embodiment/joint'][:]
                assert q.ndim == 2 and q.shape[1] == 9 and len(q) > 1
                assert np.isfinite(q).all()
                assert len(f['observation/head/rgb']) == len(f['observation/wrist/rgb']) == len(q)
                step = f['step'][:]
                cuts = (np.flatnonzero(np.diff(step) <= 0) + 1).tolist()
                if cuts:
                    discontinuities.append({'path': str(path.relative_to(ROOT)), 'segment_boundary_frames': cuts, 'handling': 'Use all segments for training; exclude invalid transition and prevent action chunks crossing discontinuity'})
                boundaries = [0, *cuts, len(q)]
                for begin, end in zip(boundaries[:-1], boundaries[1:]):
                    if end-begin < 2:
                        continue
                    states.append(q[begin:end-1])
                    actions.append(q[begin+1:end])
                    records.append({'task': task_dir.name, 'episode_id': int(path.stem), 'path': str(path.relative_to(ROOT)), 'frame_offset': begin, 'frames': end-begin, 'instruction': instruction, 'recorded_step_stride': np.unique(np.diff(step[begin:end])).tolist()})
        print(f'{task_dir.name}: 100 training episodes', flush=True)
    stats = {}
    for key, arrays in [('observation.state', states), ('action', actions)]:
        x = np.concatenate(arrays).astype(np.float64)
        stats[key] = {'mean': x.mean(0).tolist(), 'std': np.maximum(x.std(0), 1e-6).tolist(), 'min': x.min(0).tolist(), 'max': x.max(0).tolist()}
    manifest = {'dataset_revision': download['revision'], 'simulator_version': '4.5.0', 'seed': 42, 'training_protocol': 'all_episodes_no_holdout', 'normalization_scope': 'all training episodes', 'state': 'absolute joint[9] at t (7 arm radians + 2 finger positions in metres)', 'action': 'absolute joint[9] at t+1, matching upstream shift', 'input_keys': ['observation.images.head', 'observation.images.wrist', 'observation.state', 'task'], 'excluded_inputs': ['tactile', 'actor'], 'episodes': records, 'stats': stats}
    (ROOT / 'data/training_manifest.json').write_text(json.dumps(manifest, indent=2))
    summary = {'source_episodes': len({r['path'] for r in records}), 'temporal_segments': len(records), 'train_episodes': len({r['path'] for r in records}), 'train_frames': sum(r['frames']-1 for r in records), 'training_protocol': 'all_episodes_no_holdout', 'discontinuities': discontinuities}
    (ROOT / 'data/summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary))


if __name__ == '__main__':
    main()
