"""Paired Stage 2 sensor interventions on training-corpus observations.

No fitting or dataset split. This measures adaptation, not generalization.
"""
import argparse
from contextlib import nullcontext
import json
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import default_collate

from smolvla_baseline.data import ROOT
from smolvla_baseline.instructions import DEFAULT_TASKS
from .data import TactileEpisodes, normalize_depth, prepare_refinement_batch
from .losses import refinement_bridge, velocity_loss
from .model import encode_prefix
from .train import build_refiner, file_sha256, training_provenance

CONDITIONS = ('correct', 'mismatched', 'disabled')


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def autocast(device):
    return torch.autocast('cuda', dtype=torch.bfloat16) if str(device).startswith('cuda') else nullcontext()


def seeded_noise(model, seed):
    device = model.baseline.model.state_proj.weight.device
    generator = torch.Generator(device=device).manual_seed(seed)
    return torch.randn((1, 50, model.baseline.model.config.max_action_dim),
                       device=device, generator=generator)


def load_action_checkpoint(path, device=None, allowed_stages=('expert', 'joint')):
    """Load an action-trained refinement checkpoint without changing stage semantics."""
    saved = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
    if saved['stage'] not in allowed_stages:
        raise ValueError(f'These diagnostics require stage in {allowed_stages}, got {saved["stage"]!r}')
    config = dict(saved['config'])
    if device:
        config['device'] = device
    if saved['provenance'] != training_provenance(config):
        raise ValueError('Baseline/data/encoder provenance mismatch')
    model, pre, post = build_refiner(config)
    model.load_added_state_dict(saved['model'])
    model.set_stage(saved['stage'])
    model.requires_grad_(False).eval()
    metadata = {'checkpoint': str(Path(path).resolve()), 'sha256': file_sha256(path),
                'step': saved['step'], 'stage': saved['stage'], 'provenance': saved['provenance'],
                'mask_prefix': saved['stage'] == 'expert' and model.stage2_mask_prefix}
    return model, pre, post, config, metadata


def load_stage2(path, device=None):
    """Backward-compatible loader that continues to require a Stage 2 checkpoint."""
    return load_action_checkpoint(path, device, allowed_stages=('expert',))


class DonorBank:
    """Deterministic same-task donor frames; nearest standardized robot pose.

    Retains both fingers together. No future information or model outputs enter
    donor selection. Excludes the recipient's entire source file offline.
    """
    def __init__(self, dataset, tasks, seed=123, per_task=256):
        rng = np.random.default_rng(seed)
        self.entries = {}
        for task in tasks:
            candidates = []
            for segment, ep in enumerate(dataset.episodes):
                if ep['task'] != task:
                    continue
                for local in np.unique(np.linspace(0, ep['frames']-2, 8, dtype=int)):
                    candidates.append({'path': ep['path'], 'frame': int(local)+ep.get('frame_offset', 0),
                                       'segment': segment, 'task': task})
            if not candidates:
                raise ValueError(f'No donor frames for {task}')
            chosen = rng.choice(len(candidates), min(per_task, len(candidates)), replace=False)
            entries = [candidates[i] for i in chosen]
            for entry in entries:
                with h5py.File(ROOT/entry['path'], 'r') as f:
                    entry['state'] = f['embodiment/joint'][entry['frame']].astype(np.float32)
            states = np.stack([e['state'] for e in entries])
            scale = np.maximum(states.std(0), np.array([.01]*7+[.001]*2))
            self.entries[task] = (entries, states, scale)

    def select(self, task, state, exclude_path=None):
        entries, states, scale = self.entries[task]
        distance = np.sqrt(np.mean(((states-np.asarray(state))/scale)**2, axis=1))
        for i, entry in enumerate(entries):
            if entry['path'] == exclude_path:
                distance[i] = np.inf
        index = int(distance.argmin())
        if not np.isfinite(distance[index]):
            raise ValueError('No donor from another episode; increase donor bank size')
        entry = entries[index]
        maps = []
        with h5py.File(ROOT/entry['path'], 'r') as f:
            for finger in ('left_gsmini', 'right_gsmini'):
                maps.append(normalize_depth(f[f'tactile/{finger}/depth'][entry['frame']])[0])
        return torch.stack(maps), {k: v for k, v in entry.items() if k != 'state'} | {
            'standardized_pose_distance': float(distance[index])}


def sample_indices(dataset, tasks, samples, seed):
    """Balanced, unique frame sample reused verbatim for all checkpoints."""
    rng = np.random.default_rng(seed)
    pools = []
    for task in tasks:
        parts = []
        for i, ep in enumerate(dataset.episodes):
            if ep['task'] == task:
                start = dataset.ends[i-1] if i else 0
                parts.append(np.arange(start, dataset.ends[i]))
        if not parts:
            raise ValueError(f'No observations for {task}')
        pool = np.concatenate(parts)
        rng.shuffle(pool)
        pools.append(pool.tolist())
    result = []
    while len(result) < samples and any(pools):
        for pool in pools:
            if pool and len(result) < samples:
                result.append(pool.pop())
    if len(result) != samples:
        raise ValueError('Requested more unique observations than available')
    return result


@torch.inference_mode()
def compact_conditions(model, depth, donor_depth, prefix):
    # Never substitute context or shuffle predicted tokens across examples.
    correct = model.resampler(model.predicted_tokens(depth, prefix))
    mismatched = model.resampler(model.predicted_tokens(donor_depth, prefix))
    return {'correct': correct, 'mismatched': mismatched, 'disabled': torch.zeros_like(correct)}


@torch.inference_mode()
def offline_trial(model, batch, depth, donor_depth, seed):
    prefix = encode_prefix(model.baseline, batch)
    y = model.handoff(prefix, seeded_noise(model, seed))
    demo = model.baseline.prepare_action(batch)
    conditions = compact_conditions(model, depth, donor_depth, prefix)
    # Fixed interior quadrature points, identical for all conditions/checkpoints.
    taus = (.05, .15, .25, .35)
    predictions, metrics = {}, {}
    for name, compact in conditions.items():
        action = model.continue_action_chunk(prefix, compact, y)
        predictions[name] = action
        flows = []
        for value in taus:
            tau = torch.full((len(y),), value, device=y.device)
            x, velocity = refinement_bridge(demo, y, tau)
            output = model.expert(x, tau, prefix, compact, mask_prefix=model.stage2_mask_prefix)
            flows.append(float(velocity_loss(output, velocity, batch['action_is_pad'])))
        valid = (~batch['action_is_pad']).unsqueeze(-1)
        error = (action.float()-demo[..., :9].float()).square()
        metrics[name] = {'action_mse': float((error*valid).sum()/(valid.sum()*9)),
                         'arm_mse': float((error[..., :7]*valid).sum()/(valid.sum()*7)),
                         'gripper_mse': float((error[..., 7:]*valid).sum()/(valid.sum()*2)),
                         'flow_mse': float(np.mean(flows))}
    for name in CONDITIONS:
        valid = (~batch['action_is_pad']).unsqueeze(-1)
        delta = (predictions[name].float()-predictions['correct'].float()).square()
        metrics[name]['action_rms_difference_from_correct'] = float(
            ((delta*valid).sum()/(valid.sum()*9)).sqrt())
    return metrics


def mean_ci(values, clusters=None, seed=123):
    """95% percentile bootstrap; offline sampling unit is the source episode."""
    values = np.asarray(values, dtype=float)
    if not len(values):
        return {'n': 0, 'mean': None, 'ci95': None}
    if clusters is None:
        clusters = list(range(len(values)))
    grouped = {}
    for value, key in zip(values, clusters):
        grouped.setdefault(key, []).append(value)
    blocks = list(grouped.values())
    if len(blocks) < 2:
        return {'n': len(values), 'independent_units': len(blocks), 'mean': float(values.mean()), 'ci95': None}
    sums = np.array([sum(x) for x in blocks])
    counts = np.array([len(x) for x in blocks])
    rng = np.random.default_rng(seed)
    choices = rng.integers(len(blocks), size=(2000, len(blocks)))
    means = sums[choices].sum(1)/counts[choices].sum(1)
    return {'n': len(values), 'independent_units': len(blocks), 'mean': float(values.mean()),
            'ci95': np.quantile(means, [.025, .975]).tolist()}


def offline_summary(rows):
    def summarize(subset):
        clusters = [r['path'] for r in subset]
        result = {'samples': len(subset), 'conditions': {}, 'paired_control_minus_correct': {}}
        for name in CONDITIONS:
            result['conditions'][name] = {key: float(np.mean([r['metrics'][name][key] for r in subset]))
                for key in subset[0]['metrics'][name]} if subset else {}
        for name in CONDITIONS[1:]:
            result['paired_control_minus_correct'][name] = {
                key: mean_ci([r['metrics'][name][key]-r['metrics']['correct'][key] for r in subset], clusters)
                for key in ('action_mse', 'flow_mse')}
        return result
    threshold = float(np.median([r['depth_activity'] for r in rows]))
    return {'all': summarize(rows),
            'tasks': {t: summarize([r for r in rows if r['task'] == t]) for t in sorted({r['task'] for r in rows})},
            'activity_median': threshold,
            'higher_depth_activity': summarize([r for r in rows if r['depth_activity'] > threshold]),
            'lower_depth_activity': summarize([r for r in rows if r['depth_activity'] <= threshold]),
            'activity_definition': 'Mean normalized current depth; a proxy, not a physical contact label',
            'positive_gap_meaning': 'Correct tactile has lower imitation error; CI clusters frames by episode'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoints', nargs='+', type=Path, required=True)
    parser.add_argument('--tasks', nargs='+', choices=DEFAULT_TASKS, default=list(DEFAULT_TASKS))
    parser.add_argument('--samples', type=int, default=256)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--donors-per-task', type=int, default=256)
    parser.add_argument('--device', default=None)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if (args.samples < 1 or args.donors_per_task < 2
            or len(args.tasks) != len(set(args.tasks))):
        parser.error('Positive sample count, unique tasks and at least two donors required')
    args.output.mkdir(parents=True, exist_ok=True)
    if any(args.output.iterdir()):
        parser.error('Use an empty output directory')
    torch.set_num_threads(4)
    results, reference = [], None
    for number, path in enumerate(args.checkpoints):
        model, pre, _, config, metadata = load_stage2(path, args.device)
        # Identical data and donor selection across checkpoint comparisons.
        signature = {k: v for k, v in config.items() if k not in ('device', 'output_dir', 'stages')}
        if reference is not None and signature != reference:
            raise ValueError('Checkpoint comparison requires matching architecture/data/settings')
        reference = signature
        dataset = TactileEpisodes(ROOT/config['manifest'], future=False)
        indices = sample_indices(dataset, args.tasks, args.samples, args.seed)
        bank = DonorBank(dataset, args.tasks, args.seed, args.donors_per_task)
        rows = []
        destination = args.output/f'checkpoint_{number:02d}_step_{metadata["step"]}'
        destination.mkdir()
        for index in indices:
            raw = dataset[index]
            ep = dataset.episodes[raw['segment']]
            donor, donor_info = bank.select(ep['task'], raw['baseline']['observation.state'].numpy(), ep['path'])
            batch, tactile = prepare_refinement_batch(default_collate([raw]), pre, config['device'])
            with autocast(config['device']):
                metrics = offline_trial(model, batch, tactile['depth'], donor[None].to(config['device']), args.seed+index)
            row = {'index': index, 'task': ep['task'], 'path': ep['path'], 'frame': raw['frame'],
                   'donor': donor_info, 'depth_activity': float(raw['depth'].mean()),
                   'donor_depth_mae': float((raw['depth']-donor).abs().mean()), 'metrics': metrics}
            rows.append(row)
            with (destination/'samples.jsonl').open('a') as stream:
                stream.write(json.dumps(row)+'\n')
            print(json.dumps({'checkpoint_step': metadata['step'], 'samples': len(rows), 'total': len(indices)}), flush=True)
        summary = {**metadata, 'scope': 'training-corpus adaptation diagnostic; no held-out/generalization claim',
                   'seed': args.seed, 'donors_per_task': args.donors_per_task, 'tasks': args.tasks,
                   'sample_indices': indices, 'metrics': offline_summary(rows)}
        atomic_json(destination/'summary.json', summary)
        results.append(summary)
        atomic_json(args.output/'comparison.json', {'checkpoints': results})
        for handle in dataset.handles.values():
            handle.close()
        del model, pre, dataset
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
