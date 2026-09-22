"""Aligned depth supervision, separate from the strictly tactile-free baseline."""
import bisect

import numpy as np
import torch

from smolvla_baseline.data import Episodes, prepare_batch


def normalize_depth(depth):
    """Millimetres -> fixed [0,1] depth, full rectangular sensor ROI.

    Nonfinite/nonpositive pixels are invalid. Finite positive out-of-range
    depths are clipped as specified, never rescaled per image or episode.
    """
    depth = torch.as_tensor(depth, dtype=torch.float32)
    valid = torch.isfinite(depth) & (depth > 0)
    depth = torch.where(valid, depth, torch.full_like(depth, 34.0))
    return ((34.0 - depth) / 10.0).clamp(0, 1), valid


class TactileEpisodes(Episodes):
    def __init__(self, manifest, chunk_size=50, *, future=True):
        super().__init__(manifest, chunk_size)
        self.future = future
        if chunk_size != 50:
            raise ValueError('This design uses a 50-recorded-step horizon')
        if any(ep.get('recorded_step_stride') != [2] for ep in self.episodes):
            raise ValueError('Expected continuous segments at two physics steps per frame')

    def __getitem__(self, index):
        baseline = super().__getitem__(index)
        segment = bisect.bisect_right(self.ends, index)
        ep = self.episodes[segment]
        local = index - (self.ends[segment-1] if segment else 0)
        frame = local + ep.get('frame_offset', 0)
        # The parent opens the handle and enforces action segment boundaries.
        from smolvla_baseline.data import ROOT
        f = self.handles[str(ROOT / ep['path'])]
        endpoint = frame + self.chunk_size
        future_valid = local + self.chunk_size < ep['frames']
        stop = min(endpoint, ep.get('frame_offset', 0) + ep['frames'] - 1)
        if not np.all(np.diff(f['step'][frame:stop+1]) == 2):
            raise ValueError(f'Recording discontinuity inside manifest segment: {ep["path"]}:{frame}')
        current, masks = [], []
        changes, change_masks = [], []
        for finger in ('left_gsmini', 'right_gsmini'):
            depth, mask = normalize_depth(f[f'tactile/{finger}/depth'][frame])
            if depth.shape != (240, 320):
                raise ValueError(f'Unexpected tactile resolution: {depth.shape}')
            current.append(depth); masks.append(mask)
            if self.future:
                if future_valid:
                    later, later_mask = normalize_depth(f[f'tactile/{finger}/depth'][endpoint])
                    valid = mask & later_mask
                    changes.append(torch.where(valid, later-depth, 0.0))
                    change_masks.append(valid)
                else:
                    changes.append(torch.zeros_like(depth))
                    change_masks.append(torch.zeros_like(mask))
        result = {'baseline': baseline, 'depth': torch.stack(current),
                  'depth_mask': torch.stack(masks), 'segment': segment, 'frame': frame}
        if self.future:
            result.update(change=torch.stack(changes), change_mask=torch.stack(change_masks),
                          future_valid=future_valid)
        return result


def prepare_refinement_batch(raw, preprocessor, device):
    # The baseline's strict allowlist remains the only way to invoke its preprocessor.
    baseline = prepare_batch(raw['baseline'], preprocessor)
    tactile = {k: v.to(device) if torch.is_tensor(v) else v for k, v in raw.items() if k != 'baseline'}
    return baseline, tactile
