"""Paired encoder-checkpoint diagnostics on every training episode; no holdout.

This evaluates reconstruction, not future prediction or policy success. Fixed
early/middle/late examples are reused across checkpoints. Task-mean baselines
are deliberately fitted to this same diagnostic corpus (an optimistic control).
"""
import argparse
from collections import OrderedDict, defaultdict
import csv
import hashlib
import html
import json
from pathlib import Path
import time

import cv2
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from smolvla_baseline.data import ROOT
from tactile_refinement.data import normalize_depth
from tactile_refinement.representation import TactileRepresentation, TokenDecoder


class AuditMaps(Dataset):
    def __init__(self, manifest):
        self.manifest = json.loads(Path(manifest).read_text())
        self.samples = []
        self.handles = OrderedDict()
        for segment, ep in enumerate(self.manifest['episodes']):
            for kind in ('current', 'change'):
                count = ep['frames']-1 if kind == 'current' else ep['frames']-50
                if count <= 0:
                    continue
                for local in sorted(set((0, (count-1)//2, count-1))):
                    self.samples.append({'path': ep['path'], 'task': ep['task'], 'segment': segment,
                                         'kind': kind, 'frame': ep.get('frame_offset', 0)+local})

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        path = str(ROOT/sample['path'])
        if path not in self.handles:
            self.handles[path] = h5py.File(path, 'r')
            if len(self.handles) > 4:
                self.handles.popitem(last=False)[1].close()
        self.handles.move_to_end(path)
        handle = self.handles[path]
        frame = sample['frame']
        if sample['kind'] == 'change' and not np.all(np.diff(handle['step'][frame:frame+51]) == 2):
            raise ValueError('Discontinuous future target')
        maps, masks = [], []
        for finger in ('left_gsmini', 'right_gsmini'):
            dataset = handle[f'tactile/{finger}/depth']
            current, valid = normalize_depth(dataset[frame])
            if sample['kind'] == 'change':
                future, future_valid = normalize_depth(dataset[frame+50])
                valid &= future_valid
                current = torch.where(valid, future-current, 0.0)
            maps.append(current); masks.append(valid)
        return torch.stack(maps), torch.stack(masks), index


def metrics(pred, target, mask):
    """Per-observation metrics, preserving rare active regions and signed changes."""
    diff = (pred-target).abs()
    valid = mask.sum().item()
    active = mask & (target.abs() > .01)
    background = mask & ~active
    weights = mask.float()*(1+4*(target.abs() > .001))
    row = {'mae': float(diff[mask].mean()), 'rmse': float((diff[mask].square().mean()).sqrt()),
           'weighted_mae': float((weights*diff).sum()/weights.sum().clamp_min(1)),
           'zero_mae': float(target.abs()[mask].mean()), 'active_pixels': int(active.sum()),
           'valid_pixels': valid, 'active_fraction': float(active.sum()/max(valid, 1)),
           'active_mae': float(diff[active].mean()) if active.any() else None,
           'background_mae': float(diff[background].mean()) if background.any() else None}
    for threshold in (.01, .05):
        key = f'{threshold:g}'
        true_active = mask & (target.abs() > threshold)
        pred_active = mask & (pred.abs() > threshold)
        intersection = int((true_active & pred_active).sum())
        union = int((true_active | pred_active).sum())
        row['iou_'+key] = intersection/union if union else None
        row['recall_'+key] = intersection/int(true_active.sum()) if true_active.any() else None
        for sign, label in ((1, 'positive'), (-1, 'negative')):
            truth = mask & (sign*target > threshold)
            guessed = mask & (sign*pred > threshold)
            row[label+'_recall_'+key] = float((truth & guessed).sum()/truth.sum()) if truth.any() else None
    return row


def summarize(rows):
    result = {'samples': len(rows), 'source_episodes': len({r['path'] for r in rows})}
    for key in ('mae', 'weighted_mae', 'rmse', 'zero_mae', 'task_mean_mae', 'active_mae',
                'background_mae', 'active_fraction', 'iou_0.01', 'iou_0.05', 'recall_0.01',
                'recall_0.05', 'positive_recall_0.01', 'negative_recall_0.01'):
        values = [r[key] for r in rows if r.get(key) is not None]
        result[key] = float(np.mean(values)) if values else None
    for key in ('mae', 'active_mae'):
        values = [r[key] for r in rows if r.get(key) is not None]
        result[key+'_p90'] = float(np.quantile(values, .9)) if values else None
    return result


def paired_improvement(previous, current, kind):
    """Episode-cluster bootstrap: overlapping windows are not independent units."""
    episodes = defaultdict(list)
    old = {r['sample_index']: r for r in previous if r['kind'] == kind}
    for row in current:
        if row['kind'] == kind and row['sample_index'] in old:
            episodes[row['path']].append((old[row['sample_index']]['mae'], row['mae']))
    values = np.array([np.mean(x, axis=0) for x in episodes.values()])
    rng = np.random.default_rng(42)
    estimates = []
    for _ in range(2000):
        sample = values[rng.integers(len(values), size=len(values))].mean(0)
        estimates.append((sample[0]-sample[1])/max(sample[0], 1e-12))
    before, after = values.mean(0)
    return {'episode_clusters': len(values), 'previous_mae': float(before), 'current_mae': float(after),
            'relative_improvement': float((before-after)/max(before, 1e-12)),
            'bootstrap_95_percent_interval': np.quantile(estimates, [.025, .975]).tolist(),
            'scope': 'paired training-corpus change, not a generalization confidence interval'}


def save_panel(path, target, pred, title):
    """Scientific heatmaps with fixed color scales shared across checkpoints."""
    def signed(x):
        x = np.clip(x, -1, 1)
        positive, negative = np.maximum(x, 0), np.maximum(-x, 0)
        # RGB: blue negative, white zero, red positive.
        return (np.stack((1-negative, 1-np.abs(x), 1-positive), -1)*255).astype(np.uint8)[..., ::-1].copy()
    columns = []
    for label, values in (('target [-1,1]', target), ('reconstruction [-1,1]', pred),
                           ('absolute error [0,0.2]', np.abs(pred-target))):
        if label.startswith('absolute'):
            panel = cv2.applyColorMap((np.clip(np.concatenate(list(values), 0)/.2, 0, 1)*255).astype(np.uint8), cv2.COLORMAP_INFERNO)
        else:
            panel = signed(np.concatenate(list(values), 0))
        top = np.full((32, 320, 3), 255, np.uint8)
        cv2.putText(top, label, (5, 22), cv2.FONT_HERSHEY_SIMPLEX, .45, (0, 0, 0), 1)
        columns.append(np.concatenate((top, panel), 0))
    body = np.concatenate(columns, 1)
    top = np.full((34, body.shape[1], 3), 255, np.uint8)
    cv2.putText(top, title, (5, 23), cv2.FONT_HERSHEY_SIMPLEX, .45, (0, 0, 0), 1)
    cv2.imwrite(str(path), np.concatenate((top, body), 0))


def evaluate(checkpoint_path, dataset, loader, means, output, device):
    saved = torch.load(checkpoint_path, map_location='cpu', weights_only=True, mmap=True)
    if saved['stage'] != 'encoder':
        raise ValueError('Use encoder-preparation checkpoints for this audit')
    width = saved['config']['latent_width']
    encoder, decoder = TactileRepresentation(width).to(device), TokenDecoder(width).to(device)
    for name, module in (('representation', encoder), ('decoder', decoder)):
        module.load_state_dict({k[len(name)+1:]: v for k, v in saved['model'].items() if k.startswith(name+'.')}, strict=True)
        module.eval()
    step = saved['step']
    del saved
    rows, features = [], defaultdict(list)
    with torch.inference_mode():
        for maps, masks, indices in loader:
            maps, masks = maps.to(device), masks.to(device)
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=device == 'cuda'):
                tokens = encoder(maps)
                pred = decoder(tokens).float()
            for j, index in enumerate(indices.tolist()):
                meta = dataset.samples[index]
                mean = means[(meta['task'], meta['kind'])].to(device)
                row = dict(meta, sample_index=index, **metrics(pred[j], maps[j], masks[j]))
                row['task_mean_mae'] = float((mean-maps[j]).abs()[masks[j]].mean())
                rows.append(row)
                features[meta['kind']].append(tokens[j].float().flatten().cpu().numpy())
            if len(rows) % 800 < len(indices):
                print(json.dumps({'step': step, 'evaluated': len(rows), 'total': len(dataset)}), flush=True)
    result = {'step': step, 'checkpoint': str(checkpoint_path), 'overall': summarize(rows),
              'by_kind': {}, 'by_task': {}, 'small_patches': {}, 'latents': {}}
    for kind in ('current', 'change'):
        subset = [r for r in rows if r['kind'] == kind]
        result['by_kind'][kind] = summarize(subset)
        small = [r for r in subset if 0 < r['active_fraction'] <= .05]
        result['small_patches'][kind] = summarize(small) if small else {'samples': 0}
        z = np.stack(features[kind])
        norms = np.linalg.norm(z, axis=1)
        centered_energy = np.mean((z-z.mean(0))**2)
        result['latents'][kind] = {'mean_feature_std': float(z.std(0).mean()),
            'centered_energy_fraction': float(centered_energy/max(np.mean(z*z), 1e-12)),
            'near_zero_norm_samples': int((norms < 1e-6).sum()),
            'note': 'Nonzero variance excludes complete constant collapse, not loss of fine contact information.'}
    for task in sorted({r['task'] for r in rows}):
        result['by_task'][task] = {kind: summarize([r for r in rows if r['task'] == task and r['kind'] == kind])
                                  for kind in ('current', 'change')}
    csv_path = output/f'checkpoint_{step}_samples.csv'
    with csv_path.open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    # Deterministic cases and worst reconstructions per task/type.
    selected = set()
    for task in result['by_task']:
        for kind in ('current', 'change'):
            subset = [r for r in rows if r['task'] == task and r['kind'] == kind]
            selected.add(subset[len(subset)//2]['sample_index'])
            selected.add(max(subset, key=lambda r: r['mae'])['sample_index'])
    folder = output/f'checkpoint_{step}_maps'; folder.mkdir(exist_ok=True)
    with torch.inference_mode():
        for index in sorted(selected):
            maps, _, _ = dataset[index]
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=device == 'cuda'):
                pred = decoder(encoder(maps[None].to(device)))[0].float().cpu().numpy()
            meta = dataset.samples[index]
            name = f'{meta["task"]}_{meta["kind"]}_{index}.png'
            save_panel(folder/name, maps.numpy(), pred, f'{meta["task"]} | {meta["kind"]} | step {step} | sample {index} | left above right')
    (output/f'checkpoint_{step}.json').write_text(json.dumps(result, indent=2)+'\n')
    del encoder, decoder
    if device == 'cuda':
        torch.cuda.empty_cache()
    return result, rows


def write_report(output, results, comparisons, dataset):
    report = {'scope': 'All-800-episode training-corpus reconstruction diagnostics; no split and no generalization claim',
        'sampling': 'fixed early/middle/late maps per continuous segment, separately for current maps and valid t+50 changes',
        'samples': len(dataset), 'episodes': len({s['path'] for s in dataset.samples}),
        'units': 'normalized tactile camera depth; 0.01 = 0.1 mm on unclipped range, not indentation',
        'baselines': 'zero map; per-task/type pixelwise mean fitted to the SAME diagnostic sample (optimistic control)',
        'thresholds': '0.01 and 0.05 are diagnostic thresholds, not validated task-performance requirements; small patch <=5% active pixels',
        'checkpoints': results, 'paired_comparisons': comparisons}
    (output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    lines = ['# Tactile encoder readiness audit', '', report['scope'], '',
             f'{report["episodes"]} episodes; {len(dataset)} fixed two-finger maps. No optimizer updates performed by this audit.', '',
             '| Encoder step | Input | MAE | Active-region MAE | Zero-map MAE | Task-mean MAE | Active recall (>0.01) |',
             '|---|---|---:|---:|---:|---:|---:|']
    for result in results:
        for kind, m in result['by_kind'].items():
            lines.append(f'| {result["step"]} | {kind} | {m["mae"]:.6f} | {m["active_mae"]:.6f} | {m["zero_mae"]:.6f} | {m["task_mean_mae"]:.6f} | {m["recall_0.01"]:.3f} |')
    lines += ['', '## Checkpoint improvement', '']
    for comparison in comparisons:
        for kind, values in comparison['by_kind'].items():
            lo, hi = values['bootstrap_95_percent_interval']
            lines.append(f'- {comparison["previous_step"]} → {comparison["step"]}, {kind}: {100*values["relative_improvement"]:.2f}% relative MAE reduction; episode-bootstrap 95% interval [{100*lo:.2f}%, {100*hi:.2f}%].')
    lines += ['', '## Interpretation limits', '',
              '- A small aggregate loss is insufficient: inspect active pixels, signed change recall, worst tasks, and small patches.',
              '- Improvement between snapshots tests convergence on this corpus; it cannot prove that additional training would not help.',
              '- Noncollapsed tokens and good reconstruction are prerequisites, not proof of usefulness for action refinement.',
              '- Independent simulator observations and comparative rollout success are needed for generalization and downstream usefulness.',
              '- When 10,000 steps is available, compare it with 6,666 using these exact sample identities. No automatic pass/fail cutoff is claimed.',
              '', 'Detailed per-task/latent metrics: [report.json](report.json). Per-sample errors: `checkpoint_*_samples.csv`. Heatmaps: `checkpoint_*_maps/`.', '']
    (output/'REPORT.md').write_text('\n'.join(lines))
    images = ''.join(f'<details><summary>{html.escape(p.parent.name+"/"+p.name)}</summary><img loading="lazy" width="960" src="{p.relative_to(output)}"></details>' for p in sorted(output.glob('checkpoint_*_maps/*.png')))
    (output/'report.html').write_text('<!doctype html><meta charset="utf-8"><title>Encoder readiness audit</title><style>body{font:16px system-ui;margin:30px;max-width:1100px}pre{white-space:pre-wrap}summary{padding:8px;cursor:pointer}</style><h1>Encoder readiness audit</h1><pre>'+html.escape('\n'.join(lines))+'</pre><h2>Reconstruction heatmaps</h2>'+images)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--encoder-dir', type=Path, default=ROOT/'runs/tactile_refinement/encoder')
    parser.add_argument('--output', type=Path, default=ROOT/'runs/diagnostics/encoder_readiness')
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--wait-for-final', action='store_true', help='After interim results, await final.pt and automatically audit it too')
    args = parser.parse_args()
    torch.set_num_threads(2)
    args.output.mkdir(parents=True, exist_ok=True)
    config = json.loads((args.encoder_dir/'recipe.json').read_text())
    manifest = ROOT/config['manifest']
    dataset = AuditMaps(manifest)
    if len({s['path'] for s in dataset.samples}) != 800:
        raise ValueError('Audit must include all 800 episodes')
    (args.output/'sample_indices.json').write_text(json.dumps(dataset.samples, indent=2)+'\n')
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=0)
    sums, counts = {}, {}
    print(json.dumps({'phase': 'fit_task_mean_controls', 'samples': len(dataset)}), flush=True)
    for maps, masks, indices in loader:
        for j, index in enumerate(indices.tolist()):
            sample = dataset.samples[index]; key = (sample['task'], sample['kind'])
            if key not in sums:
                sums[key] = torch.zeros_like(maps[j]); counts[key] = torch.zeros_like(maps[j])
            sums[key] += maps[j]*masks[j]; counts[key] += masks[j]
    means = {key: sums[key]/counts[key].clamp_min(1) for key in sums}
    results, comparisons, previous_rows = [], [], None
    paths = [args.encoder_dir/name for name in ('checkpoint_1.pt', 'checkpoint_2.pt', 'final.pt')]
    for path in paths:
        if not path.exists() and not args.wait_for_final:
            continue
        while not path.exists():
            (args.output/'status.json').write_text(json.dumps({'phase': 'waiting_for_checkpoint',
                'checkpoint': path.name, 'completed_steps': [r['step'] for r in results]})+'\n')
            time.sleep(30)
        print(json.dumps({'phase': 'evaluate', 'checkpoint': str(path)}), flush=True)
        result, rows = evaluate(path, dataset, loader, means, args.output, args.device)
        if previous_rows is not None:
            comparisons.append({'previous_step': results[-1]['step'], 'step': result['step'],
                                'by_kind': {k: paired_improvement(previous_rows, rows, k) for k in ('current', 'change')}})
        results.append(result); previous_rows = rows
        write_report(args.output, results, comparisons, dataset)
        print(json.dumps({'phase': 'checkpoint_complete', 'step': result['step'], 'metrics': result['by_kind']}), flush=True)
    (args.output/'status.json').write_text(json.dumps({'phase': 'complete', 'completed_steps': [r['step'] for r in results]})+'\n')


if __name__ == '__main__':
    main()
