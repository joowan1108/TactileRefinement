from collections import OrderedDict
from pathlib import Path
import bisect
import json

import cv2
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

ROOT = Path(__file__).resolve().parents[2]
IMAGE_KEYS = ('observation.images.head', 'observation.images.wrist')
INPUT_KEYS = ('observation.state', *IMAGE_KEYS, 'task')


class Episodes(Dataset):
    """Read only allowlisted RGB/joint datasets, never tactile or privileged actor state."""
    def __init__(self, manifest, chunk_size=50):
        self.manifest = json.loads(Path(manifest).read_text())
        if self.manifest.get('training_protocol') != 'all_episodes_no_holdout':
            raise ValueError('Expected the full-dataset training manifest; regenerate with prepare_data.py')
        self.episodes = self.manifest['episodes']
        if any('split' in e for e in self.episodes):
            raise ValueError('Dataset partitions are not supported for this run')
        self.ends = np.cumsum([e['frames'] - 1 for e in self.episodes]).tolist()
        self.chunk_size = chunk_size
        self.handles = OrderedDict()

    def __len__(self):
        return self.ends[-1]

    def __getitem__(self, index):
        episode_id = bisect.bisect_right(self.ends, index)
        ep = self.episodes[episode_id]
        local_frame = index - (self.ends[episode_id-1] if episode_id else 0)
        frame = local_frame + ep.get('frame_offset', 0)
        path = str(ROOT / ep['path'])
        if path not in self.handles:
            self.handles[path] = h5py.File(path, 'r')
            if len(self.handles) > 16:
                _, old = self.handles.popitem(last=False)
                old.close()
        f = self.handles[path]
        self.handles.move_to_end(path)
        joint = f['embodiment/joint']
        stop = min(frame + 1 + self.chunk_size, ep.get('frame_offset', 0) + ep['frames'])
        action = joint[frame+1:stop].astype(np.float32)
        valid = len(action)
        action = np.pad(action, ((0, self.chunk_size-valid), (0, 0)), mode='edge')
        item = {
            'observation.state': torch.from_numpy(joint[frame].astype(np.float32)),
            'action': torch.from_numpy(action),
            'action_is_pad': torch.arange(self.chunk_size) >= valid,
            'task': ep['instruction'],
        }
        for camera, key in zip(('head', 'wrist'), IMAGE_KEYS):
            encoded = f[f'observation/{camera}/rgb'][frame]
            rgb = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
            if rgb is None:
                raise ValueError(f'Invalid JPEG: {path}/{camera}/{frame}')
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
            rgb = cv2.resize(rgb, (256, 256), interpolation=cv2.INTER_LINEAR)
            item[key] = torch.from_numpy(rgb.transpose(2, 0, 1).copy())
        return item


class UpdateBatchSampler(Sampler):
    """Sample frames with replacement; each optimizer update has a reproducible global batch."""
    def __init__(self, size, batch_size, accumulation, seed, start_step=0):
        self.size, self.batch_size, self.accumulation = size, batch_size, accumulation
        self.seed, self.start_step = seed, start_step

    def __iter__(self):
        update = self.start_step
        while True:
            rng = torch.Generator().manual_seed(self.seed + update)
            indices = torch.randint(self.size, (self.accumulation, self.batch_size), generator=rng)
            for batch in indices.tolist():
                yield batch
            update += 1


def prepare_batch(batch, preprocessor):
    if set(batch) - set(INPUT_KEYS) - {'action', 'action_is_pad'}:
        raise ValueError('Unexpected input key: strict no-tactile baseline')
    batch = dict(batch)
    for key in IMAGE_KEYS:
        batch[key] = batch[key].float().div(255)
    return preprocessor(batch)
