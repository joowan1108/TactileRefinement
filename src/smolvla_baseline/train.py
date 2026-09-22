"""Wall-time-bounded SmolVLA fine-tuning with exactly three production checkpoints."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import ROOT, Episodes, UpdateBatchSampler, prepare_batch
from .model import build_model, load_checkpoint


def parameter_report(model):
    groups = {
        'vision_encoder_dense': [0, 0],
        'vision_connector_dense': [0, 0],
        'language_base': [0, 0],
        'language_lora': [0, 0],
        'action_expert_dense': [0, 0],
        'state_action_projections_dense': [0, 0],
        'other': [0, 0],
    }
    for name, parameter in model.named_parameters():
        if 'vision_model.' in name:
            group = 'vision_encoder_dense'
        elif 'connector.' in name:
            group = 'vision_connector_dense'
        elif 'text_model.' in name and 'lora_' in name:
            group = 'language_lora'
        elif 'text_model.' in name:
            group = 'language_base'
        elif 'lm_expert.' in name:
            group = 'action_expert_dense'
        elif any(value in name for value in (
            'state_proj.', 'action_in_proj.', 'action_out_proj.', 'action_time_mlp_'
        )):
            group = 'state_action_projections_dense'
        else:
            group = 'other'
        groups[group][0] += parameter.numel()
        groups[group][1] += parameter.numel() if parameter.requires_grad else 0
    return {
        group: {'total_parameters': values[0], 'trainable_parameters': values[1]}
        for group, values in groups.items()
    }


def lr_at(recipe, elapsed):
    progress = min(elapsed / (recipe['duration_hours'] * 3600), 1.)
    warmup = recipe['warmup_fraction']
    if progress < warmup:
        return recipe['learning_rate'] * max(progress / warmup, 1e-3)
    cosine = .5 * (1 + math.cos(math.pi * (progress-warmup)/(1-warmup)))
    return recipe['minimum_learning_rate'] + cosine * (recipe['learning_rate']-recipe['minimum_learning_rate'])


def optimizer_update(model, pre, optimizer, iterator, recipe):
    optimizer.zero_grad(set_to_none=True)
    loss_sum = 0.
    for _ in range(recipe['gradient_accumulation_steps']):
        batch = prepare_batch(next(iterator), pre)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            # Per-sample means give every frame equal weight, including episode-end chunks.
            sample_losses, _ = model(batch, reduction='none')
            loss = sample_losses.mean()
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite action loss')
        (loss / recipe['gradient_accumulation_steps']).backward()
        loss_sum += loss.detach().item()
    norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], recipe['grad_clip_norm'], error_if_nonfinite=True)
    optimizer.step()
    return loss_sum / recipe['gradient_accumulation_steps'], float(norm)


def save_checkpoint(path, model, optimizer, recipe, manifest, step, elapsed, completed):
    payload = {'format_version': 2, 'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'recipe': recipe, 'manifest': manifest, 'step': step, 'elapsed_seconds': elapsed, 'completed_checkpoints': completed, 'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all(), 'dataset_revision': manifest['dataset_revision']}
    temp = path.with_suffix('.pt.tmp')
    torch.save(payload, temp)
    os.replace(temp, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=str(ROOT / 'configs/smolvla_3090_72h.json'))
    parser.add_argument('--profile', type=int, default=0, help='Run this many optimizer updates; do not save production checkpoints')
    parser.add_argument('--resume', type=Path)
    args = parser.parse_args()
    recipe = json.loads(Path(args.config).read_text())
    assert recipe['batch_size'] == 16 and recipe['gradient_accumulation_steps'] == 4
    assert recipe['effective_batch_size'] == recipe['batch_size'] * recipe['gradient_accumulation_steps'] == 64
    assert recipe['training_protocol'] == 'all_episodes_no_holdout'
    assert recipe['loss'] == 'action_flow_matching_only'
    assert not recipe['tactile_inputs'] and not recipe['tactile_auxiliary_loss']
    assert recipe['vision_training'] == 'full' and not recipe['freeze_vision_encoder']
    assert recipe['language_training'] == 'full_dense' and not recipe['language_lora']
    assert recipe['action_expert_training'] == 'full'
    assert len(recipe['checkpoint_hours']) == len(recipe['checkpoint_names']) == 3
    assert recipe['checkpoint_hours'][-1] == recipe['duration_hours']
    assert recipe['checkpoint_hours'] == sorted(set(recipe['checkpoint_hours']))
    torch.set_num_threads(8)
    torch.manual_seed(recipe['seed']); np.random.seed(recipe['seed']); random.seed(recipe['seed'])
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    manifest_path = ROOT / recipe['manifest']
    manifest = json.loads(manifest_path.read_text())
    assert manifest['training_protocol'] == 'all_episodes_no_holdout'
    assert len({e['path'] for e in manifest['episodes']}) == 800, 'All 800 episodes must be used'
    assert not any('split' in e for e in manifest['episodes'])
    checkpoint = load_checkpoint(args.resume) if args.resume else None
    if checkpoint:
        assert checkpoint['recipe'] == recipe, 'Resume must use the identical recipe'
        assert checkpoint['manifest'] == manifest, 'Dataset/statistics changed since checkpoint'
    output = ROOT / recipe['output_dir']
    output.mkdir(parents=True, exist_ok=True)
    if not args.profile and not checkpoint and (any(output.glob('*.pt')) or (output / 'train.jsonl').exists()):
        raise FileExistsError('Existing run: use --resume or a fresh output_dir')
    model, pre, _ = build_model(recipe, manifest, checkpoint)
    model.train()
    parameters = parameter_report(model)
    assert parameters['vision_encoder_dense']['trainable_parameters'] > 0
    assert parameters['vision_connector_dense']['trainable_parameters'] > 0
    assert parameters['language_lora']['total_parameters'] == 0
    assert parameters['language_base']['trainable_parameters'] > 0
    assert parameters['action_expert_dense']['trainable_parameters'] > 0
    assert not any('lora_' in name for name, _ in model.named_parameters())
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=recipe['learning_rate'], betas=tuple(recipe['adam_betas']),
        weight_decay=recipe['weight_decay'], fused=recipe['fused_adamw'],
    )
    step = 0; elapsed_before = 0.; completed = []
    if checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
        step = checkpoint['step']; elapsed_before = checkpoint['elapsed_seconds']; completed = checkpoint['completed_checkpoints']
        torch.set_rng_state(checkpoint['torch_rng'])
        torch.cuda.set_rng_state_all(checkpoint['cuda_rng'])
    training = Episodes(manifest_path, recipe['chunk_size'])
    loader = DataLoader(training, batch_sampler=UpdateBatchSampler(
        len(training), recipe['batch_size'], recipe['gradient_accumulation_steps'], recipe['seed'], step
    ), generator=torch.Generator().manual_seed(recipe['seed']), num_workers=recipe['num_workers'], pin_memory=True, persistent_workers=recipe['num_workers'] > 0, **({'prefetch_factor': recipe['prefetch_factor']} if recipe['num_workers'] else {}))
    iterator = iter(loader)
    torch.cuda.reset_peak_memory_stats()
    gpu = torch.cuda.get_device_name()
    print(json.dumps({'gpu': gpu, 'parameters': sum(p.numel() for p in model.parameters()), 'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad), 'train_frames': len(training), 'training_episodes': len({e['path'] for e in training.episodes}), 'batch_size': recipe['batch_size'], 'gradient_accumulation_steps': recipe['gradient_accumulation_steps'], 'effective_batch_size': recipe['effective_batch_size'], 'duration_hours': recipe['duration_hours']}), flush=True)
    start = time.monotonic(); timings = []
    if not args.profile:
        (output / 'recipe.json').write_text(json.dumps(recipe, indent=2))
        (output / 'dataset_sha256.txt').write_text(hashlib.sha256(manifest_path.read_bytes()).hexdigest() + '\n')
        (output / 'parameter_training.json').write_text(json.dumps(parameters, indent=2) + '\n')
    log_path = ROOT / 'logs/profile_steps.jsonl' if args.profile else output / 'train.jsonl'
    with log_path.open('a') as log:
        while elapsed_before + time.monotonic() - start < recipe['duration_hours'] * 3600:
            elapsed = elapsed_before + time.monotonic() - start
            for group in optimizer.param_groups:
                group['lr'] = lr_at(recipe, elapsed)
            t0 = time.monotonic()
            loss, grad_norm = optimizer_update(model, pre, optimizer, iterator, recipe)
            torch.cuda.synchronize()
            timings.append(time.monotonic()-t0); step += 1
            elapsed = elapsed_before + time.monotonic() - start
            row = {'step': step, 'elapsed_seconds': elapsed, 'loss': loss, 'grad_norm': grad_norm, 'learning_rate': optimizer.param_groups[0]['lr'], 'seconds_per_optimizer_step': timings[-1], 'peak_vram_gib': torch.cuda.max_memory_allocated()/2**30}
            log.write(json.dumps(row)+'\n'); log.flush()
            if args.profile or step % recipe['log_every_steps'] == 0:
                print(json.dumps(row), flush=True)
            if args.profile:
                if len(timings) >= args.profile:
                    break
                continue
            for hour, name in zip(recipe['checkpoint_hours'], recipe['checkpoint_names']):
                if elapsed >= hour * 3600 and name not in completed:
                    completed.append(name)
                    save_checkpoint(output/name, model, optimizer, recipe, manifest, step, elapsed_before+time.monotonic()-start, list(completed))
                    print(json.dumps({'checkpoint': str(output/name)}), flush=True)
        if not args.profile:
            # If a checkpoint write crosses another time boundary, finish any due saves.
            for name in recipe['checkpoint_names']:
                if name not in completed:
                    completed.append(name)
                    save_checkpoint(output/name, model, optimizer, recipe, manifest, step, elapsed_before+time.monotonic()-start, list(completed))
    if args.profile:
        measured = timings[2:] if len(timings) > 2 else timings
        seconds = float(np.median(measured))
        report = {'gpu': gpu, 'batch_size': recipe['batch_size'], 'gradient_accumulation_steps': recipe['gradient_accumulation_steps'], 'effective_batch_size': recipe['effective_batch_size'], 'warmup_updates_discarded': 2 if len(timings)>2 else 0, 'measured_updates': len(measured), 'median_seconds_per_optimizer_step': seconds, 'estimated_optimizer_steps_72h': int(72*3600/seconds), 'estimated_frame_samples_72h': int(72*3600/seconds)*recipe['effective_batch_size'], 'peak_vram_allocated_gib': torch.cuda.max_memory_allocated()/2**30, 'peak_vram_reserved_gib': torch.cuda.max_memory_reserved()/2**30, 'note': 'Short throughput estimate; production termination/checkpoints use elapsed time. Setup and model loading excluded; checkpoint writing included.'}
        (ROOT / 'configs/throughput_3090.json').write_text(json.dumps(report, indent=2))
        print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
