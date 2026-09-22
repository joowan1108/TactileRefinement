"""Measure resampler rank and per-token action influence without fitting.

The audit uses one frame from each independently selected training episode.
All action comparisons hold the observation, visual-language prefix, baseline
handoff noise, and policy weights fixed. Mean replacement is the conservative
token ablation; zero replacement is reported separately because it is OOD.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import default_collate

from smolvla_baseline.data import ROOT
from smolvla_baseline.instructions import DEFAULT_TASKS
from tactile_refinement.ablation import (atomic_json, autocast,
                                          load_action_checkpoint, seeded_noise)
from tactile_refinement.attention_audit import ResamplerAttentionRecorder
from tactile_refinement.data import TactileEpisodes, prepare_refinement_batch
from tactile_refinement.model import encode_prefix


def spectrum_metrics(matrix):
    """Scale-independent rank summaries from squared singular values."""
    singular = torch.linalg.svdvals(matrix.float())
    energy = singular.square()
    total = energy.sum()
    if not torch.isfinite(total) or total <= 0:
        return dict(effective_rank=0.0, participation_rank=0.0,
                    stable_rank=0.0, top1_energy=0.0)
    probability = energy / total
    nonzero = probability > 0
    effective = torch.exp(-(probability[nonzero] * probability[nonzero].log()).sum())
    return dict(
        effective_rank=float(effective),
        participation_rank=float(1.0 / probability.square().sum()),
        stable_rank=float(total / energy.max()),
        top1_energy=float(probability.max()))


def choose_indices(dataset, samples_per_task, seed):
    """Choose distinct source episodes, then one uniformly random frame each."""
    rng = np.random.default_rng(seed)
    selected = []
    for task in DEFAULT_TASKS:
        candidates = [i for i, episode in enumerate(dataset.episodes)
                      if episode['task'] == task]
        if len(candidates) < samples_per_task:
            raise ValueError(f'{task} has only {len(candidates)} source episodes')
        for segment in rng.choice(candidates, samples_per_task, replace=False):
            episode = dataset.episodes[int(segment)]
            start = dataset.ends[int(segment)-1] if segment else 0
            # Episodes exposes frames-1 starting observations because each item
            # must also contain the following action target.
            local = int(rng.integers(episode['frames']-1))
            selected.append(dict(index=int(start+local), task=task,
                                 path=episode['path'], frame=local))
    return selected


def masked_rms(delta, valid):
    mask = valid.unsqueeze(-1).expand_as(delta)
    return float(delta.float().square()[mask].mean().sqrt())


def masked_mse(action, demo, valid):
    mask = valid.unsqueeze(-1).expand_as(action)
    return float((action.float()-demo.float()).square()[mask].mean())


def summarize(rows):
    result = {'samples': len(rows), 'tasks': {}, 'fingers': {}, 'tokens': []}
    for task in DEFAULT_TASKS:
        subset = [row for row in rows if row['task'] == task]
        result['tasks'][task] = dict(
            samples=len(subset),
            reference_action_mse=float(np.mean([row['reference_action_mse'] for row in subset])),
            mean_replacement_action_rms=float(np.mean([
                value for row in subset for value in row['mean_replacement_action_rms']])),
            zero_replacement_action_rms=float(np.mean([
                value for row in subset for value in row['zero_replacement_action_rms']])))
    for finger, name in enumerate(('left', 'right')):
        rank_rows = [row['rank'][name] for row in rows]
        result['fingers'][name] = {
            group: {key: float(np.mean([entry[group][key] for entry in rank_rows]))
                    for key in rank_rows[0][group]}
            for group in ('attention', 'output', 'output_centered')}
    for token in range(32):
        mean_rms = np.asarray([row['mean_replacement_action_rms'][token] for row in rows])
        zero_rms = np.asarray([row['zero_replacement_action_rms'][token] for row in rows])
        mean_gap = np.asarray([row['mean_replacement_demo_mse_gap'][token] for row in rows])
        result['tokens'].append(dict(
            token=token, finger=('left' if token < 16 else 'right'),
            local_token=token % 16,
            mean_replacement_action_rms=float(mean_rms.mean()),
            mean_replacement_action_rms_p90=float(np.quantile(mean_rms, .9)),
            zero_replacement_action_rms=float(zero_rms.mean()),
            mean_replacement_demo_mse_gap=float(mean_gap.mean()),
            samples_above_action_rms_1e_3=int((mean_rms > 1e-3).sum())))
    values = np.asarray([token['mean_replacement_action_rms'] for token in result['tokens']])
    probability = values / values.sum() if values.sum() else np.zeros_like(values)
    nonzero = probability > 0
    result['token_influence'] = dict(
        mean_token_action_rms=float(values.mean()),
        maximum_token_action_rms=float(values.max()),
        effective_token_count=float(np.exp(-(probability[nonzero]*np.log(probability[nonzero])).sum()))
            if nonzero.any() else 0.0,
        top_8_influence_fraction=float(np.sort(probability)[-8:].sum()),
        left_influence_fraction=float(probability[:16].sum()),
        right_influence_fraction=float(probability[16:].sum()))
    return result


def report(summary, checkpoint, output):
    lines = [
        '# Stage 3 final resampler token-usage audit', '',
        f"Checkpoint: joint step {checkpoint['step']:,}", '',
        f"Independent source episodes: {summary['samples']}", '',
        '## Effective rank', '',
        '| Finger | Attention effective rank / 16 | Output effective rank / 16 | Centered output effective rank / 15 |',
        '|---|---:|---:|---:|']
    for finger in ('left', 'right'):
        item = summary['fingers'][finger]
        lines.append(f"| {finger} | {item['attention']['effective_rank']:.3f} | "
                     f"{item['output']['effective_rank']:.3f} | "
                     f"{item['output_centered']['effective_rank']:.3f} |")
    influence = summary['token_influence']
    lines += ['', '## Action influence', '',
        f"Mean/max single-token mean-replacement action RMS: "
        f"**{influence['mean_token_action_rms']:.6f} / {influence['maximum_token_action_rms']:.6f}**.",
        f"Influence-weighted effective token count: **{influence['effective_token_count']:.2f} / 32**.",
        f"Top eight tokens account for **{100*influence['top_8_influence_fraction']:.1f}%** of mean-replacement action influence.",
        f"Left/right influence fractions: **{100*influence['left_influence_fraction']:.1f}% / "
        f"{100*influence['right_influence_fraction']:.1f}%**.", '',
        'Mean replacement substitutes one token with the mean of the other 15 tokens from the same finger. '
        'It tests unique token information while preserving scale. Zero replacement is a stronger OOD intervention.', '',
        'This is a training-corpus sensitivity audit. Attention rank and action sensitivity do not by themselves establish causal task success.', '']
    (output/'REPORT.md').write_text('\n'.join(lines))


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--samples-per-task', type=int, default=8)
    parser.add_argument('--seed', type=int, default=321)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    if args.samples_per_task < 1:
        parser.error('--samples-per-task must be positive')
    args.output.mkdir(parents=True, exist_ok=True)
    if any(args.output.iterdir()):
        parser.error('Use an empty output directory')
    torch.set_num_threads(4)
    model, pre, _, config, checkpoint = load_action_checkpoint(args.checkpoint, args.device)
    if checkpoint['stage'] != 'joint':
        raise ValueError('Expected a Stage 3 joint checkpoint')
    dataset = TactileEpisodes(ROOT/config['manifest'], future=False)
    selections = choose_indices(dataset, args.samples_per_task, args.seed)
    atomic_json(args.output/'selection.json', dict(seed=args.seed, samples=selections))
    recorder = ResamplerAttentionRecorder(model.resampler)
    rows = []
    try:
        for number, selection in enumerate(selections):
            raw = dataset[selection['index']]
            batch, tactile = prepare_refinement_batch(default_collate([raw]), pre, args.device)
            with autocast(args.device):
                prefix = encode_prefix(model.baseline, batch)
                predicted = model.predicted_tokens(tactile['depth'], prefix)
                compact = model.resampler(predicted)
                attention = recorder.probabilities()[-1].mean(1).to(args.device)
                handoff = model.handoff(prefix, seeded_noise(model, args.seed+selection['index']))
                reference = model.continue_action_chunk(prefix, compact, handoff).float()
                demo = model.baseline.prepare_action(batch)[..., :9].float()
                valid = ~batch['action_is_pad']
                mean_rms, zero_rms, mean_gap = [], [], []
                reference_mse = masked_mse(reference, demo, valid)
                shaped = compact.reshape(1, 2, 16, -1)
                for token in range(32):
                    finger, local = divmod(token, 16)
                    conservative = shaped.clone()
                    others = torch.cat((shaped[:, finger, :local],
                                        shaped[:, finger, local+1:]), dim=1)
                    conservative[:, finger, local] = others.mean(1)
                    changed = model.continue_action_chunk(
                        prefix, conservative.reshape_as(compact), handoff).float()
                    mean_rms.append(masked_rms(changed-reference, valid))
                    mean_gap.append(masked_mse(changed, demo, valid)-reference_mse)
                    zeroed = shaped.clone()
                    zeroed[:, finger, local] = 0
                    changed_zero = model.continue_action_chunk(
                        prefix, zeroed.reshape_as(compact), handoff).float()
                    zero_rms.append(masked_rms(changed_zero-reference, valid))
            rank = {}
            shaped_float = shaped.float()[0]
            for finger, name in enumerate(('left', 'right')):
                output = shaped_float[finger]
                rank[name] = dict(
                    attention=spectrum_metrics(attention[finger]),
                    output=spectrum_metrics(output),
                    output_centered=spectrum_metrics(output-output.mean(0, keepdim=True)))
            row = dict(**selection, reference_action_mse=reference_mse, rank=rank,
                       mean_replacement_action_rms=mean_rms,
                       zero_replacement_action_rms=zero_rms,
                       mean_replacement_demo_mse_gap=mean_gap)
            rows.append(row)
            with (args.output/'samples.jsonl').open('a') as stream:
                stream.write(json.dumps(row)+'\n')
            print(json.dumps(dict(phase='inference', completed=number+1,
                                  total=len(selections), task=selection['task'])), flush=True)
    finally:
        recorder.close()
        for handle in dataset.handles.values():
            handle.close()
    summary = summarize(rows)
    result = dict(checkpoint=checkpoint,
                  scope='One frame from each independently sampled training episode; no fitting or held-out claim',
                  samples_per_task=args.samples_per_task, seed=args.seed, **summary)
    atomic_json(args.output/'summary.json', result)
    report(result, checkpoint, args.output)
    print(json.dumps(dict(phase='complete', output=str(args.output/'summary.json'))), flush=True)


if __name__ == '__main__':
    main()
