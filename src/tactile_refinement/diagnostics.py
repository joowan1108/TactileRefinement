"""Training-corpus forecast diagnostics; explicitly not generalization evaluation."""
import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import default_collate

from smolvla_baseline.data import ROOT
from .data import TactileEpisodes, prepare_refinement_batch
from .model import encode_prefix
from .train import build_refiner, training_provenance
from .losses import coarse_change_target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--samples', type=int, default=128)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.samples < 1:
        parser.error('--samples must be positive')
    saved = torch.load(args.checkpoint, weights_only=True, map_location='cpu', mmap=True)
    config = saved['config']
    if saved['stage'] == 'encoder':
        parser.error('Predictor diagnostics require grounding, expert, or joint checkpoints')
    provenance = training_provenance(config)
    if provenance != saved['provenance']:
        raise ValueError('Baseline/data mismatch')
    torch.set_num_threads(4)
    model, pre, _ = build_refiner(config)
    model.load_added_state_dict(saved['model']); model.eval()
    dataset = TactileEpisodes(ROOT/config['manifest'])
    indices = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(args.seed)).tolist()
    rows = []
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        for index in indices:
            raw = dataset[index]
            if not raw['future_valid'] or not raw['change_mask'].any():
                continue
            batch, tactile = prepare_refinement_batch(default_collate([raw]), pre, config['device'])
            prefix = encode_prefix(model.baseline, batch)
            pred = model.predicted_tokens(tactile['depth'], prefix)
            target = model.representation(tactile['change'])
            zero_target = model.representation(torch.zeros_like(tactile['change']))
            decoded = model.decoder(pred).float()
            valid = tactile['change_mask']
            change = tactile['change']
            current = tactile['depth']
            if model.trex_n0_full_grid:
                change, valid = coarse_change_target(change, valid, tactile['future_valid'], decoded.shape[-2:])
                current = torch.nn.functional.adaptive_avg_pool2d(current, decoded.shape[-2:])
            pixel_error = (decoded-change).abs()
            row = {'dataset_index': index, 'segment': raw['segment'], 'frame': raw['frame'],
                   'prediction_latent_mae': float((pred.float()-target.float()).abs().mean()),
                   'zero_change_latent_mae': float((zero_target.float()-target.float()).abs().mean()),
                   'reconstruction_mae': float(pixel_error[valid].mean()),
                   'zero_change_map_mae': float(change.abs()[valid].mean())}
            future = current+change
            contact, later_contact = current > 1e-3, future > 1e-3
            regions = {'non_contact': ~contact & ~later_contact, 'onset': ~contact & later_contact,
                       'sustained': contact & later_contact, 'unloading': contact & (change < -1e-3)}
            for name, region in regions.items():
                mask = valid & region
                row[name] = {'pixels': int(mask.sum()), 'absolute_error_sum': float(pixel_error[mask].sum())}
            rows.append(row)
            if len(rows) >= args.samples:
                break
    if not rows:
        raise ValueError('No valid future supervision found')
    metrics = {key: sum(row[key] for row in rows)/len(rows) for key in (
        'prediction_latent_mae', 'zero_change_latent_mae', 'reconstruction_mae', 'zero_change_map_mae')}
    for name in ('non_contact', 'onset', 'sustained', 'unloading'):
        pixels = sum(row[name]['pixels'] for row in rows)
        metrics[name+'_mae'] = sum(row[name]['absolute_error_sum'] for row in rows)/pixels if pixels else None
    result = {'scope': 'training-corpus diagnostics, no holdout or generalization claim',
              'checkpoint': str(args.checkpoint), 'stage': saved['stage'], 'seed': args.seed,
              'samples': len(rows), 'units': 'fixed normalized camera depth, not physical indentation',
              'metrics': metrics, 'rows': rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()
