"""Reuse the pinned UniVTAC SmolVLA temporal aggregation implementation."""
import importlib
import sys
from types import SimpleNamespace

import yaml

from .data import ROOT

UPSTREAM_COMMIT = '0dafa10262e22f486f160a55d6f11aeab12d8e7b'
UPSTREAM_ROOT = ROOT/'UniVTAC'


class OfficialTemporalAggregator:
    def __init__(self, device='cpu'):
        # Importing the module does not load a checkpoint or start its server.
        # Only its buffer reset and aggregation methods are reused.
        if str(UPSTREAM_ROOT) not in sys.path:
            sys.path.append(str(UPSTREAM_ROOT))
        module = importlib.import_module('policy.smolvla.deploy_policy')
        cls = module._SmolVLALocalPolicy
        cfg = yaml.safe_load((UPSTREAM_ROOT/'policy/smolvla/deploy.yml').read_text())
        self.state = SimpleNamespace(device=device,
            temporal_agg_max_timesteps=cfg['smolvla_temporal_agg_max_timesteps'],
            temporal_agg_action_dim=cfg['smolvla_temporal_agg_action_dim'],
            temporal_agg_k=cfg['smolvla_temporal_agg_k'])
        self._reset = cls._reset_temporal_aggregation
        self._aggregate = cls._aggregate_action_chunk
        self.reset()

    def reset(self):
        self._reset(self.state)

    def __call__(self, chunk):
        return self._aggregate(self.state, chunk)
