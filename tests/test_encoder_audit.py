"""Check that readiness metrics do not hide erased contacts or reversed changes."""
import importlib.util
from pathlib import Path

import pytest
import torch

spec = importlib.util.spec_from_file_location('encoder_audit', Path(__file__).resolve().parents[1]/'scripts/audit_tactile_encoder.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def test_signed_reconstruction_metrics_and_background_shortcut():
    target = torch.zeros(2, 10, 10)
    target[0, 0, 0] = .5
    target[1, 0, 0] = -.5
    mask = torch.ones_like(target, dtype=torch.bool)
    exact = audit.metrics(target, target, mask)
    assert exact['mae'] == 0 and exact['active_mae'] == 0
    assert exact['positive_recall_0.01'] == exact['negative_recall_0.01'] == 1
    erased = audit.metrics(torch.zeros_like(target), target, mask)
    assert erased['mae'] == pytest.approx(.005)
    assert erased['active_mae'] == .5 and erased['recall_0.01'] == 0
    reversed_map = audit.metrics(-target, target, mask)
    assert reversed_map['iou_0.01'] == 1  # same spatial extent alone is insufficient
    assert reversed_map['positive_recall_0.01'] == reversed_map['negative_recall_0.01'] == 0
    mask[:, 0, 0] = False
    masked = audit.metrics(torch.zeros_like(target), target, mask)
    assert masked['mae'] == 0 and masked['active_mae'] is None


def test_bootstrap_clusters_overlapping_samples_by_episode():
    previous = [{'sample_index': i, 'kind': 'change', 'path': str(i//3), 'mae': .1} for i in range(6)]
    current = [dict(row, mae=.08) for row in previous]
    result = audit.paired_improvement(previous, current, 'change')
    assert result['episode_clusters'] == 2
    assert result['relative_improvement'] == pytest.approx(.2)
    assert result['bootstrap_95_percent_interval'] == pytest.approx([.2, .2])
