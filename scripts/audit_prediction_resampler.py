"""Read-only checkpoint audit: low-change prediction and trained compression.

Uses the training corpus without fitting or splitting it. Action comparisons
hold observation, noise, baseline handoff and expert weights fixed. Replacing
a co-adapted resampler is an intervention, not a retrained architecture ablation.
"""
import argparse
import json
from pathlib import Path
import subprocess

import h5py
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import default_collate

from smolvla_baseline.data import ROOT
from tactile_refinement.ablation import atomic_json, load_stage2, mean_ci, seeded_noise
from tactile_refinement.data import TactileEpisodes, prepare_refinement_batch
from tactile_refinement.losses import coarse_change_target
from tactile_refinement.model import encode_prefix
from tactile_refinement.representation import SpatialTactileResampler


def scan_windows(manifest, seed=123, episodes_per_task=32):
    rng = np.random.default_rng(seed)
    episodes = manifest['episodes']
    offsets = np.r_[0, np.cumsum([ep['frames']-1 for ep in episodes])]
    rows = []
    for task in sorted({ep['task'] for ep in episodes}):
        eligible = [i for i, ep in enumerate(episodes) if ep['task'] == task and ep['frames'] > 50]
        # Unique source files, including manifests with multiple continuous segments.
        unique = {}
        for i in eligible:
            unique.setdefault(episodes[i]['path'], i)
        chosen = rng.choice(list(unique.values()), min(episodes_per_task, len(unique)), replace=False)
        for i in chosen:
            ep = episodes[i]
            starts = np.unique(rng.integers(ep['frames']-50, size=2))
            with h5py.File(ROOT/ep['path'], 'r') as handle:
                for local in starts:
                    frame = int(local)+ep.get('frame_offset', 0)
                    times = frame+np.arange(0, 51, 10)
                    if not np.all(np.diff(handle['step'][frame:frame+51]) == 2):
                        continue
                    depth, masks = [], []
                    for side in ('left_gsmini', 'right_gsmini'):
                        raw = handle[f'tactile/{side}/depth'][times].astype(np.float32)
                        valid = np.isfinite(raw) & (raw > 0)
                        depth.append(np.clip((34-np.where(valid, raw, 34))/10, 0, 1))
                        masks.append(valid)
                    maps = np.stack(depth, 1)
                    valid = np.stack(masks, 1).all(0)
                    if valid.mean() < .95:
                        continue
                    amplitude = np.abs(maps[1:]-maps[0])[:, valid].mean(1)
                    rows.append(dict(task=task, path=ep['path'], frame=frame,
                        dataset_index=int(offsets[i]+local), endpoint_mae=float(amplitude[-1]),
                        sampled_peak_mae=float(amplitude.max()), valid_fraction=float(valid.mean())))
        print(json.dumps(dict(phase='scan', task=task, windows=sum(r['task']==task for r in rows))), flush=True)
    return rows


def select_cohorts(rows, count):
    tasks = {}
    for task in sorted({r['task'] for r in rows}):
        rr = [r for r in rows if r['task'] == task]
        tasks[task] = dict(windows=len(rr),
            median_sampled_peak_mae=float(np.median([r['sampled_peak_mae'] for r in rr])),
            median_endpoint_mae=float(np.median([r['endpoint_mae'] for r in rr])))
    task = min(tasks, key=lambda t: tasks[t]['median_sampled_peak_mae'])
    candidates = sorted([r for r in rows if r['task']==task], key=lambda r:r['sampled_peak_mae'])
    chosen, used = [], set()
    for group, pool in [('low_change', candidates), ('higher_change', candidates[::-1])]:
        selected = 0
        for row in pool:
            if row['path'] in used:
                continue
            chosen.append(dict(row, group=group))
            used.add(row['path'])
            selected += 1
            if selected == count:
                break
        if selected != count:
            raise ValueError('Insufficient independent episodes for both cohorts')
    return task, tasks, chosen


def cosine(a, b):
    return float(F.cosine_similarity(a.float().mean(1), b.float().mean(1)).mean())


def rms(a):
    return float(a.float().square().mean().sqrt())


@torch.inference_mode()
def measure(model, pre, dataset, selected, device, random_resamplers, output):
    rows, panels = [], []
    use_amp = lambda: torch.autocast(str(device).split(':')[0], dtype=torch.bfloat16)
    with use_amp():
        zero_latent = model.representation(torch.zeros(1, 2, 240, 320, device=device))
        zero_compact = model.resampler(zero_latent)
    for i, selection in enumerate(selected):
        raw = default_collate([dataset[selection['dataset_index']]])
        batch, tactile = prepare_refinement_batch(raw, pre, device)
        assert tactile['future_valid'].all()
        with use_amp():
            prefix = encode_prefix(model.baseline, batch)
            pred = model.predicted_tokens(tactile['depth'], prefix)
            target = model.representation(tactile['change'])
            decoded = model.decoder(pred)
            compact = model.resampler(pred)
            assert compact.shape == (1, 32, 256)
            assert torch.isfinite(compact).all()
            # Reverse locations independently within each finger, preserving all
            # feature vectors but breaking their association with spatial positions.
            reversed_grid = pred.reshape(1, 2, 300, -1).flip(2).reshape_as(pred)
            conditions = {'trained': compact, 'zero_compact': torch.zeros_like(compact),
                'spatial_reversed': model.resampler(reversed_grid)}
            for j, resampler in enumerate(random_resamplers):
                conditions[f'random_resampler_{j}'] = resampler(pred)
            # Only one finger is changed, to test sensitivity and exact isolation.
            finger_metrics = {}
            for finger in (0, 1):
                changed = pred.clone()
                changed[:, finger*300:(finger+1)*300] = zero_latent[:, finger*300:(finger+1)*300]
                alternate = model.resampler(changed)
                own = slice(finger*16, (finger+1)*16)
                other = slice((1-finger)*16, (2-finger)*16)
                finger_metrics[f'finger_{finger}_changed_output_rms'] = rms(alternate[:, own]-compact[:, own])
                finger_metrics[f'finger_{finger}_opposite_output_rms'] = rms(alternate[:, other]-compact[:, other])
                torch.testing.assert_close(alternate[:, other], compact[:, other], atol=0, rtol=0)
            y = model.handoff(prefix, seeded_noise(model, 123+i))
            demo = model.baseline.prepare_action(batch)[..., :9].float()
            valid_actions = ~batch['action_is_pad']
            actions = {}
            for name, condition in conditions.items():
                result = model.continue_action_chunk(prefix, condition, y).float()
                actions[name] = dict(mse=float((result-demo).square()[valid_actions].mean()),
                    arm_mse=float((result-demo)[..., :7].square()[valid_actions].mean()),
                    gripper_mse=float((result-demo)[..., 7:].square()[valid_actions].mean()))
        coarse, mask = coarse_change_target(tactile['change'], tactile['change_mask'],
                                           tactile['future_valid'])
        predicted = decoded.float()
        metrics = dict(
            target_change_mae=float(coarse.abs()[mask].mean()),
            prediction_change_mae=float(predicted.abs()[mask].mean()),
            prediction_map_mae=float((predicted-coarse).abs()[mask].mean()),
            zero_map_baseline_mae=float(coarse.abs()[mask].mean()),
            prediction_pooled_cosine_target=cosine(pred, target),
            prediction_pooled_cosine_zero_change=cosine(pred, zero_latent),
            target_pooled_cosine_zero_change=cosine(target, zero_latent),
            zero_change_latent_rms=rms(zero_latent), prediction_latent_rms=rms(pred),
            prediction_to_zero_latent_rms=rms(pred-zero_latent),
            trained_compact_rms=rms(compact), compact_zero_reference_rms=rms(zero_compact),
            compact_to_zero_reference_rms=rms(compact-zero_compact),
            compact_token_std=float(compact.float().reshape(1,2,16,-1).std(2,correction=0).mean()),
            spatial_reversal_compact_rms=rms(conditions['spatial_reversed']-compact), **finger_metrics)
        row = dict(selection, metrics=metrics, action=actions)
        rows.append(row)
        with (output/'samples.jsonl').open('a') as stream:
            stream.write(json.dumps(row)+'\n')
        if sum(p['group']==selection['group'] for p in panels) < 3:
            panels.append(dict(group=selection['group'], target=coarse.cpu().numpy()[0],
                               prediction=predicted.cpu().numpy()[0]))
        print(json.dumps(dict(phase='inference', completed=i+1, total=len(selected),
            group=selection['group'], target_change_mae=metrics['target_change_mae'],
            prediction_change_mae=metrics['prediction_change_mae'], actions=actions)), flush=True)
    return rows, panels


def summarize(rows):
    groups = {}
    for group in ['low_change', 'higher_change', 'all']:
        rr = rows if group == 'all' else [r for r in rows if r['group']==group]
        groups[group] = dict(n=len(rr), metrics={
            k:float(np.mean([r['metrics'][k] for r in rr])) for k in rr[0]['metrics']},
            actions={name:float(np.mean([r['action'][name]['mse'] for r in rr])) for name in rr[0]['action']},
            paired_action_mse_control_minus_trained={name:mean_ci(
                [r['action'][name]['mse']-r['action']['trained']['mse'] for r in rr],
                [r['path'] for r in rr]) for name in rr[0]['action'] if name!='trained'})
    return groups


def plot(panels, output):
    np.savez_compressed(output/'prediction_panels.npz',
        groups=np.array([p['group'] for p in panels]),
        target=np.stack([p['target'] for p in panels]),
        prediction=np.stack([p['prediction'] for p in panels]))
    # Matplotlib is already installed in the simulator environment; no package
    # changes to the concurrently running training environment are needed.
    subprocess.run([str(ROOT/'.venv-benchmark/bin/python'),
                    str(ROOT/'scripts/plot_prediction_audit.py'),str(output)],check=True)


@torch.inference_mode()
def compact_content_probe(model, pre, dataset, selected, device, output):
    """Matched conditions that are less drastic than replacing trained weights."""
    rows = []
    for i in (0, 3, 6, 9, 12, 15, 18, 21):
        if i >= len(selected):
            continue
        selection = selected[i]
        donor = selected[(i+len(selected)//2) % len(selected)]
        raw = default_collate([dataset[selection['dataset_index']]])
        batch, tactile = prepare_refinement_batch(raw, pre, device)
        donor_depth = dataset[donor['dataset_index']]['depth'][None].to(device)
        with torch.autocast(str(device).split(':')[0], dtype=torch.bfloat16):
            prefix = encode_prefix(model.baseline, batch)
            pred = model.predicted_tokens(tactile['depth'], prefix)
            compact = model.resampler(pred)
            donor_compact = model.resampler(model.predicted_tokens(donor_depth, prefix))
            per_finger = compact.reshape(1, 2, 16, -1)
            repeated = per_finger.mean(2, keepdim=True).expand_as(per_finger).reshape_as(compact)
            y = model.handoff(prefix, seeded_noise(model, 123+i))
            demo = model.baseline.prepare_action(batch)[..., :9].float()
            valid = ~batch['action_is_pad']
            actions = {}
            reference = None
            for name, tokens in [('trained',compact), ('mean_per_finger_repeated',repeated),
                                 ('same_task_donor_same_context',donor_compact)]:
                action = model.continue_action_chunk(prefix,tokens,y).float()
                if reference is None:
                    reference = action
                actions[name] = dict(mse=float((action-demo).square()[valid].mean()),
                    rms_action_delta=float((action-reference).square()[valid].mean().sqrt()))
        rows.append(dict(selection,donor_path=donor['path'],action=actions))
        print(json.dumps(dict(phase='content_probe',completed=len(rows),actions=actions)),flush=True)
    result = dict(samples=rows,conditions={name:dict(
        mean_mse=float(np.mean([r['action'][name]['mse'] for r in rows])),
        mean_action_rms_delta=float(np.mean([r['action'][name]['rms_action_delta'] for r in rows])),
        paired_mse_gap=mean_ci([r['action'][name]['mse']-r['action']['trained']['mse'] for r in rows],
                              [r['path'] for r in rows])) for name in rows[0]['action']})
    atomic_json(output/'compact_content_probe.json',result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=ROOT/'runs/tactile_refinement_trex_fullgrid_n0/expert/checkpoint_2.pt')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--samples-per-group', type=int, default=12)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--content-only', action='store_true', help='Use existing selection for eight matched content probes')
    parser.add_argument('--plots-only', action='store_true', help='Regenerate six prediction panels from existing selection')
    args = parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    if any(args.output.iterdir()) and not (args.content_only or args.plots_only):
        raise ValueError('Use an empty diagnostic output directory')
    torch.set_num_threads(4)
    if args.content_only or args.plots_only:
        selection = json.loads((args.output/'selection.json').read_text())
        model,pre,_,config,_=load_stage2(args.checkpoint,args.device)
        dataset=TactileEpisodes(ROOT/config['manifest'],future=args.plots_only)
        if args.plots_only:
            panels=[]
            for item in selection['samples']:
                if sum(p['group']==item['group'] for p in panels)>=3:
                    continue
                batch,tactile=prepare_refinement_batch(default_collate([dataset[item['dataset_index']]]),pre,args.device)
                with torch.inference_mode(),torch.autocast(str(args.device).split(':')[0],dtype=torch.bfloat16):
                    prefix=encode_prefix(model.baseline,batch)
                    prediction=model.decoder(model.predicted_tokens(tactile['depth'],prefix)).float()
                    target,_=coarse_change_target(tactile['change'],tactile['change_mask'],tactile['future_valid'])
                panels.append(dict(group=item['group'],target=target.cpu().numpy()[0],prediction=prediction.cpu().numpy()[0]))
            plot(panels,args.output)
            return
        compact_content_probe(model,pre,dataset,selection['samples'],args.device,args.output)
        return
    manifest = json.loads((ROOT/'data/training_manifest.json').read_text())
    scanned = scan_windows(manifest)
    task, scan_summary, selected = select_cohorts(scanned,args.samples_per_group)
    atomic_json(args.output/'selection.json',dict(task=task,task_scan=scan_summary,samples=selected,
        caveat='Low-change selection samples times 0,10,...50; brief events between samples may be missed.',
        scanned_windows=scanned))
    print(json.dumps(dict(phase='selected',task=task,task_scan=scan_summary)),flush=True)
    model,pre,post,config,checkpoint=load_stage2(args.checkpoint,args.device)
    dataset=TactileEpisodes(ROOT/config['manifest'],future=True)
    randoms=[]
    for seed in (101,202,303):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            randoms.append(SpatialTactileResampler(256,16).to(args.device).eval().requires_grad_(False))
    rows,panels=measure(model,pre,dataset,selected,args.device,randoms,args.output)
    summary=dict(task=task,checkpoint=checkpoint,device=args.device,groups=summarize(rows),
        caveats=['Training-corpus diagnostic, not held-out generalization or simulator success.',
                 'Random-resampler replacement measures dependence on learned compatible weights, not superiority over retrained alternatives.',
                 'No-change latent need not be zero; predictor and resampler include normalization and learned biases.',
                 'Spatial reversal is a distribution intervention; insensitive results do not prove spatial information is absent.',
                 'Coarse reconstruction head measures decoded prediction, not exact invertibility of the latent.'])
    atomic_json(args.output/'summary.json',summary)
    plot(panels,args.output)
    print(json.dumps(dict(phase='complete',task=task,summary=str(args.output/'summary.json'))),flush=True)


if __name__=='__main__':
    main()
