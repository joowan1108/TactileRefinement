import itertools
import json
import socket
from pathlib import Path

import h5py
import numpy as np
import torch

from smolvla_baseline.data import ROOT, Episodes, UpdateBatchSampler
from smolvla_baseline.sim_bridge import send_message, receive_message, action_to_qpos
from smolvla_baseline.train import optimizer_update, parameter_report, save_checkpoint
from smolvla_baseline.model import load_checkpoint


def test_all_800_episodes_used_and_no_discontinuous_targets():
    train = Episodes(ROOT/'data/training_manifest.json')
    assert len({e['path'] for e in train.episodes}) == 800
    assert len(train) == 152079
    assert all('split' not in e for e in train.episodes)
    for task in {e['task'] for e in train.episodes}:
        assert len({e['path'] for e in train.episodes if e['task']==task}) == 100
    for i, ep in enumerate(train.episodes):
        if ep['task'] == 'lift_can' and ep['episode_id'] == 62:
            last = train.ends[i]-1
            item = train[last]
            assert (~item['action_is_pad']).sum() == 1
            with h5py.File(ROOT/ep['path']) as f:
                np.testing.assert_array_equal(item['action'][0], f['embodiment/joint'][ep['frame_offset']+ep['frames']-1])
            assert set(item) == {'observation.state','observation.images.head','observation.images.wrist','action','action_is_pad','task'}


def test_action_is_next_state_and_no_tactile_reads(monkeypatch):
    dataset = Episodes(ROOT/'data/training_manifest.json')
    original = h5py.Group.__getitem__
    accessed = []
    def read(self, key):
        accessed.append(str(key))
        assert not str(key).startswith(('tactile', 'actor'))
        return original(self, key)
    monkeypatch.setattr(h5py.Group, '__getitem__', read)
    sample = dataset[0]
    with h5py.File(ROOT/dataset.episodes[0]['path']) as f:
        np.testing.assert_array_equal(sample['observation.state'], f['embodiment/joint'][0])
        np.testing.assert_array_equal(sample['action'][0], f['embodiment/joint'][1])
    assert not any('tactile' in key for key in accessed)


def test_normalization_uses_every_source_episode():
    manifest = json.loads((ROOT/'data/training_manifest.json').read_text())
    state_rows, action_rows = [], []
    for ep in manifest['episodes']:
        with h5py.File(ROOT/ep['path']) as f:
            begin = ep['frame_offset']
            end = begin + ep['frames']
            q = f['embodiment/joint'][begin:end]
            state_rows.append(q[:-1])
            action_rows.append(q[1:])
    for key, rows in [('observation.state',state_rows), ('action',action_rows)]:
        x = np.concatenate(rows).astype(np.float64)
        assert len(x) == 152079
        np.testing.assert_allclose(manifest['stats'][key]['mean'], x.mean(0), rtol=1e-10)
        np.testing.assert_allclose(manifest['stats'][key]['std'], x.std(0), rtol=1e-10)


def test_accumulation_equals_64_sample_update():
    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.weight = torch.nn.Parameter(torch.tensor(.3))
        def forward(self, batch, reduction):
            return (batch['observation.state'][:, 0]*self.weight-1).square(), {}
    model = Tiny(); reference = Tiny()
    optimizer = torch.optim.SGD(model.parameters(), lr=.01)
    reference_optimizer = torch.optim.SGD(reference.parameters(), lr=.01)
    values = torch.linspace(-1, 1, 64).reshape(64, 1)
    batches = [{'observation.state': x, 'observation.images.head': torch.zeros(8,3,2,2,dtype=torch.uint8), 'observation.images.wrist': torch.zeros(8,3,2,2,dtype=torch.uint8)} for x in values.split(8)]
    optimizer_update(model, lambda b: b, optimizer, iter(batches), {'gradient_accumulation_steps':8, 'grad_clip_norm':10.})
    (reference.weight*values-1).square().mean().backward(); reference_optimizer.step()
    torch.testing.assert_close(model.weight, reference.weight)


def test_sampler_resume_and_ipc():
    full = iter(UpdateBatchSampler(100,8,8,42))
    batches = list(itertools.islice(full,24))
    resumed = list(itertools.islice(iter(UpdateBatchSampler(100,8,8,42,start_step=2)),8))
    assert batches[16:] == resumed
    a,b = socket.socketpair()
    try:
        expected = {'state': np.arange(9,dtype=np.float32), 'info': {'success':False}}
        send_message(a, expected); result = receive_message(b)
        np.testing.assert_array_equal(result['state'],expected['state'])
        assert result['info'] == expected['info']
    finally:
        a.close();b.close()
    np.testing.assert_array_equal(action_to_qpos(np.arange(9)),np.arange(8))


def test_checkpoint_roundtrip(tmp_path):
    model = torch.nn.Linear(2,1)
    optimizer = torch.optim.AdamW(model.parameters())
    model(torch.ones(1,2)).sum().backward(); optimizer.step()
    recipe = {'tactile_inputs':False,'tactile_auxiliary_loss':False}
    manifest = {'dataset_revision':'test'}
    path = tmp_path/'final.pt'
    save_checkpoint(path, model, optimizer, recipe, manifest, 1, 5., ['final.pt'])
    saved = load_checkpoint(path)
    restored = torch.nn.Linear(2,1);restored.load_state_dict(saved['model'])
    torch.testing.assert_close(model.weight, restored.weight)
    assert saved['step']==1 and saved['completed_checkpoints']==['final.pt']


def test_parameter_report_categories():
    class Fake(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.vision_model = torch.nn.Linear(2, 2)
            self.text_model = torch.nn.Module()
            self.text_model.lora_A = torch.nn.Linear(2, 1, bias=False)
            self.text_model.base = torch.nn.Linear(2, 2)
            self.text_model.base.requires_grad_(False)
            self.lm_expert = torch.nn.Linear(2, 2)
    report = parameter_report(Fake())
    assert report['vision_encoder_dense']['trainable_parameters'] > 0
    assert report['language_lora']['trainable_parameters'] > 0
    assert report['language_base']['trainable_parameters'] == 0
    assert report['action_expert_dense']['trainable_parameters'] > 0
