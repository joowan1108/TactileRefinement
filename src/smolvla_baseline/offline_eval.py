"""Report training-set reconstruction error; this is not held-out evaluation."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from .data import ROOT, Episodes, prepare_batch
from .model import build_model, load_checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--max-samples', type=int, default=1024, help='0 evaluates all training frames')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    ckpt = load_checkpoint(args.checkpoint)
    recipe = ckpt['recipe']
    dataset = Episodes(ROOT / recipe['manifest'], recipe['chunk_size'])
    assert dataset.manifest == ckpt['manifest'], 'Evaluation data/statistics do not match checkpoint'
    if args.max_samples > 0:
        ids = np.random.default_rng(43).choice(len(dataset), min(args.max_samples, len(dataset)), replace=False)
        selected = Subset(dataset, ids.tolist())
    else:
        selected = dataset
    loader = DataLoader(selected, batch_size=recipe['batch_size'], num_workers=0)
    model, pre, post = build_model(recipe, ckpt['manifest'], ckpt)
    model.eval(); torch.manual_seed(44); torch.set_num_threads(8)
    sums = np.zeros(9); squares = np.zeros(9); count = 0
    first_sums = np.zeros(9); first_count = 0
    with torch.inference_mode():
        for batch in loader:
            target = batch['action'].numpy()
            valid = ~batch['action_is_pad'].numpy()
            observations = {key: value for key, value in batch.items() if key not in {'action', 'action_is_pad'}}
            with torch.autocast('cuda', dtype=torch.bfloat16):
                # No queued actions from a different training observation.
                predicted = post(model.predict_action_chunk(prepare_batch(observations, pre))).float().cpu().numpy()
            error = predicted - target
            sums += np.abs(error)[valid].sum(0); squares += np.square(error)[valid].sum(0)
            count += int(valid.sum())
            first_sums += np.abs(error[:, 0]).sum(0); first_count += len(error)
    result = {'checkpoint': str(args.checkpoint), 'dataset_revision': ckpt['dataset_revision'], 'evaluation_scope': 'training_set_diagnostic_not_held_out', 'observations': first_count, 'valid_action_targets': count, 'joint_order': ['arm_1', 'arm_2', 'arm_3', 'arm_4', 'arm_5', 'arm_6', 'arm_7', 'left_finger', 'right_finger'], 'joint_units': ['rad']*7+['m']*2, 'chunk_mae_per_joint': (sums/count).tolist(), 'chunk_rmse_per_joint': np.sqrt(squares/count).tolist(), 'next_action_mae_per_joint': (first_sums/first_count).tolist(), 'note': 'Training-set imitation diagnostic only; closed-loop success is measured separately by benchmark.sh'}
    output = args.output or args.checkpoint.with_suffix('.offline_eval.json')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
