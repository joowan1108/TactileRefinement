"""Future-tactile grounding, expert training, and joint refinement."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from smolvla_baseline.data import ROOT, UpdateBatchSampler
from smolvla_baseline.model import build_model
from .data import TactileEpisodes, prepare_refinement_batch
from .model import RefinementModel

STAGES = ('encoder', 'grounding', 'expert', 'joint')
TREX_RECIPE = 'trex_fullgrid_n0_v1'


def active_stages(config):
    return STAGES[1:] if config.get('training_recipe') == TREX_RECIPE else STAGES


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def training_provenance(config):
    result = {'baseline_sha256': file_sha256(ROOT/config['baseline_checkpoint']),
              'manifest_sha256': file_sha256(ROOT/config['manifest'])}
    if config.get('training_recipe') == TREX_RECIPE:
        result['trex_encoder_sha256'] = file_sha256(ROOT/config['tactile_encoder_checkpoint'])
    return result


def build_refiner(config):
    if config.get('lambda_compact_rec', 0) != 0:
        raise ValueError('Compression reconstruction is no longer supported; use the action-only config '
                         'and migrate a grounding checkpoint explicitly')
    recipe_name = config.get('training_recipe')
    if recipe_name not in (None, TREX_RECIPE):
        raise ValueError(f'Unknown training recipe: {recipe_name}')
    full_grid = recipe_name == TREX_RECIPE
    if full_grid and config['lambda_nce'] != 1.0:
        raise ValueError('N0 objective uses InfoNCE + lambda_rec * L1; lambda_nce must be 1')
    if config.get('expert_initialization') != 'trained_smolvla_baseline_v1':
        raise ValueError('Expected expert_initialization=trained_smolvla_baseline_v1')
    checkpoint = torch.load(ROOT/config['baseline_checkpoint'], map_location='cpu', weights_only=True, mmap=True)
    if checkpoint['recipe']['tactile_inputs'] or checkpoint['recipe']['tactile_auxiliary_loss']:
        raise ValueError('The frozen baseline must be tactile-free')
    manifest = json.loads((ROOT/config['manifest']).read_text())
    if checkpoint['manifest'] != manifest:
        raise ValueError('Refinement must reuse the baseline corpus and normalization')
    if len({e['path'] for e in manifest['episodes']}) != 800:
        raise ValueError('Expected all 800 episodes, without a split')
    recipe = dict(checkpoint['recipe'], device=config['device'])
    baseline, pre, post = build_model(recipe, manifest, checkpoint)
    model = RefinementModel(baseline, config['latent_width'],
                            stage2_mask_prefix=config.get('stage2_mask_prefix', False),
                            trex_n0_full_grid=full_grid,
                            expert_implementation=config.get('expert_implementation', 'copied_expert_v1'),
                            lora_rank=config.get('lora_rank', 32),
                            lora_alpha=config.get('lora_alpha', 32),
                            resampler_tokens_per_finger=config.get('resampler_tokens_per_finger', 0)).to(config['device'])
    if full_grid:
        pretrained = torch.load(ROOT/config['tactile_encoder_checkpoint'], weights_only=True, map_location='cpu')
        if (pretrained['repo'] != config['tactile_encoder_repo'] or
                pretrained['revision'] != config['tactile_encoder_revision']):
            raise ValueError('T-Rex pretrained encoder source/revision mismatch')
        model.representation.encoder.load_state_dict(pretrained['encoder'], strict=True)
        model.representation.encoder.requires_grad_(False).eval()
    return model, pre, post


def save(path, model, optimizer, config, stage, step, provenance):
    version = (4 if not isinstance(model.resampler, torch.nn.Identity)
               else 2 if model.expert.implementation == 'shared_ffn_lora_v1' else 1)
    payload = {'format_version': version,
               'config': config, 'stage': stage, 'step': step,
               'model': model.added_state_dict(), 'optimizer': optimizer.state_dict(),
               'provenance': provenance, 'torch_rng': torch.get_rng_state(),
               'cuda_rng': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
               'python_rng': random.getstate()}
    temporary = path.with_suffix('.pt.tmp')
    torch.save(payload, temporary)
    os.replace(temporary, path)


def validate_resume_config(saved, config, stage):
    if saved == config:
        return
    if stage in ('encoder', 'grounding') and 'stages' in saved and 'stages' in config:
        # Extending stages that have not started cannot change a saved grounding
        # checkpoint. Preserve all learning rates and other configuration fields.
        aligned = json.loads(json.dumps(config))
        changed = False
        for later in STAGES[STAGES.index(stage)+1:]:
            if later in saved['stages'] and later in config['stages']:
                old_updates = saved['stages'][later]['updates']
                new_updates = config['stages'][later]['updates']
                if new_updates > old_updates:
                    aligned['stages'][later]['updates'] = old_updates
                    changed = True
        if changed:
            validate_resume_config(saved, aligned, stage)
            return
    if stage in ('encoder', 'grounding') and 'lambda_compact_rec' in saved and 'lambda_compact_rec' not in config:
        # Removed auxiliary heads were unused during grounding; its optimizer
        # and objective are unchanged. Action-stage optimizer states cannot migrate.
        validate_resume_config({k: v for k, v in saved.items() if k != 'lambda_compact_rec'}, config, stage)
        return
    if (stage in ('encoder', 'grounding') and not saved.get('resampler_tokens_per_finger', 0)
            and config.get('resampler_tokens_per_finger', 0)):
        # A new post-predictor module does not participate in grounding.
        # Compose with the existing LoRA migration.
        before_resampler = dict(config)
        for key in ('resampler_tokens_per_finger',):
            if key in saved:
                before_resampler[key] = saved[key]
            else:
                before_resampler.pop(key, None)
        validate_resume_config(saved, before_resampler, stage)
        return
    # Only early-stage checkpoints can replace their unused dense expert. No
    # tactile training, optimizer or data settings may change in this migration.
    adapter_keys = {'expert_implementation', 'lora_rank', 'lora_alpha'}
    if (stage in ('encoder', 'grounding')
            and saved.get('expert_implementation', 'copied_expert_v1') == 'copied_expert_v1'
            and config.get('expert_implementation') == 'shared_ffn_lora_v1'
            and config.get('lora_rank') == 32 and config.get('lora_alpha') == 32
            and {k: v for k, v in saved.items() if k not in adapter_keys}
                == {k: v for k, v in config.items() if k not in adapter_keys}):
        return
    # Enabling the future Stage 2 mask cannot affect already completed encoder
    # preparation or grounding updates. Preserve their optimizer and RNG states.
    before = {k: v for k, v in saved.items() if k != 'stage2_mask_prefix'}
    after = {k: v for k, v in config.items() if k != 'stage2_mask_prefix'}
    if (stage in ('encoder', 'grounding') and before == after
            and not saved.get('stage2_mask_prefix', False) and config.get('stage2_mask_prefix') is True):
        return
    # A completed or interrupted encoder/grounding run may be extended without changing
    # its optimizer, sampler, or objective. Only the terminal update count may
    # increase; all learning rates and every other stage setting must match.
    if stage in ('encoder', 'grounding') and 'stages' in saved and 'stages' in config:
        old_updates = saved['stages'][stage]['updates']
        new_updates = config['stages'][stage]['updates']
        aligned = json.loads(json.dumps(config))
        aligned['stages'][stage]['updates'] = old_updates
        if aligned == saved and new_updates > old_updates:
            return
    raise ValueError('Resume configuration mismatch; allowed early-stage migrations are an update-budget increase, '
                     'a future-stage budget increase, enabling the Stage 2 prefix mask, adding post-predictor resampling, '
                     'or replacing the unused copied expert with rank-32 FFN LoRA')


def run_stage(model, pre, config, stage, provenance, *, resume=None, profile_updates=0):
    model.set_stage(stage); model.train()
    output = ROOT/config['output_dir']/stage
    if not profile_updates:
        output.mkdir(parents=True, exist_ok=True)
        if resume is None and any(output.iterdir()):
            raise FileExistsError(f'Existing stage output: {output}; use --resume or a new output_dir')
    stage_config = config['stages'][stage]
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=stage_config['learning_rate'], weight_decay=config['weight_decay'])
    start = 0
    if resume:
        if resume['stage'] != stage or resume['provenance'] != provenance:
            raise ValueError('Resume stage, configuration, or provenance mismatch')
        validate_resume_config(resume['config'], config, stage)
        if resume['config'] != config:
            print(json.dumps({'resume_transition': 'validated_configuration_migration', 'stage': stage,
                              'step': resume['step'], 'planned_updates': config['stages'][stage]['updates']}), flush=True)
        model.load_added_state_dict(resume['model'], allow_legacy_grounding=stage in ('encoder', 'grounding'))
        optimizer.load_state_dict(resume['optimizer'])
        start = resume['step']
        torch.set_rng_state(resume['torch_rng'])
        if resume['cuda_rng']:
            torch.cuda.set_rng_state_all(resume['cuda_rng'])
        random.setstate(resume['python_rng'])
    if not profile_updates:
        (output/'recipe.json').write_text(json.dumps(config, indent=2)+'\n')
        (output/'provenance.json').write_text(json.dumps(provenance, indent=2)+'\n')
    seed = config['seed']+100000*STAGES.index(stage)
    dataset = TactileEpisodes(ROOT/config['manifest'], future=stage in ('encoder', 'grounding'))
    loader = DataLoader(dataset, batch_sampler=UpdateBatchSampler(len(dataset), config['batch_size'],
        config['gradient_accumulation_steps'], seed, start), num_workers=config['num_workers'],
        pin_memory=config['device'].startswith('cuda'),
        generator=torch.Generator().manual_seed(seed),
        **({'prefetch_factor': 1} if config['num_workers'] else {}))
    iterator = iter(loader)
    updates = start+profile_updates if profile_updates else stage_config['updates']
    if not profile_updates and updates < 3:
        raise ValueError('At least three updates are required for three distinct stage checkpoints')
    milestones = {max(1, updates//3): 'checkpoint_1.pt', max(2, 2*updates//3): 'checkpoint_2.pt', updates: 'final.pt'}
    if config['device'].startswith('cuda'):
        torch.cuda.reset_peak_memory_stats()
    print(json.dumps({'stage': stage, 'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
                      'training_episodes': 800, 'samples': len(dataset), 'profile': bool(profile_updates),
                      'expert_initialization': config['expert_initialization'], 'batch_size': config['batch_size'],
                      'expert_implementation': model.expert.implementation,
                      'lora_rank': config.get('lora_rank'), 'lora_alpha': config.get('lora_alpha'),
                      'gradient_accumulation_steps': config['gradient_accumulation_steps'],
                      'expert_prefix_masked': stage == 'expert' and model.stage2_mask_prefix,
                      'training_recipe': config.get('training_recipe', 'legacy'),
                      'tactile_tokens': model.predictor.queries.shape[1],
                      'action_tactile_tokens': (2*model.resampler.tokens_per_finger
                          if hasattr(model.resampler, 'tokens_per_finger') else model.predictor.queries.shape[1]),
                      'resampler_objective': 'action_only',
                      'encoder_frozen': not any(p.requires_grad for p in model.representation.encoder.parameters()),
                      'planned_updates': updates, 'output': str(output)}), flush=True)
    for step in range(start, updates):
        began = time.monotonic()
        optimizer.zero_grad(set_to_none=True)
        metrics = {}
        for _ in range(config['gradient_accumulation_steps']):
            batch, tactile = prepare_refinement_batch(next(iterator), pre, config['device'])
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=config['device'].startswith('cuda')):
                loss, parts = model.loss(batch, tactile, lambda_rec=config['lambda_rec'], lambda_nce=config['lambda_nce'])
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Nonfinite loss in {stage}')
            (loss/config['gradient_accumulation_steps']).backward()
            for key, value in {'loss': loss.detach(), **parts}.items():
                metrics[key] = metrics.get(key, 0.0)+float(value)/config['gradient_accumulation_steps']
        grad = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],
                                             config['grad_clip_norm'], error_if_nonfinite=True)
        optimizer.step()
        if config['device'].startswith('cuda'):
            torch.cuda.synchronize()
        row = dict(stage=stage, step=step+1, seconds=time.monotonic()-began, grad_norm=float(grad), **metrics)
        if config['device'].startswith('cuda'):
            row['peak_vram_gib'] = torch.cuda.max_memory_allocated()/2**30
        print(json.dumps(row), flush=True)
        if not profile_updates:
            with (output/'train.jsonl').open('a') as log:
                log.write(json.dumps(row)+'\n')
            if step+1 in milestones:
                save(output/milestones[step+1], model, optimizer, config, stage, step+1, provenance)
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/'configs/tactile_refinement.json')
    parser.add_argument('--stage', choices=(*STAGES, 'all'), default='all')
    parser.add_argument('--init', type=Path, help='Previous stage final checkpoint')
    parser.add_argument('--resume', type=Path, help='Continue the saved stage, then later stages when --stage all')
    parser.add_argument('--profile-updates', type=int, default=0, help='Measure one stage without writing checkpoints')
    args = parser.parse_args()
    if args.init and args.resume:
        parser.error('--init and --resume are mutually exclusive')
    if args.profile_updates and args.stage == 'all':
        parser.error('Select one --stage when profiling')
    config = json.loads(args.config.read_text())
    torch.set_num_threads(config['cpu_threads'])
    torch.manual_seed(config['seed']); np.random.seed(config['seed']); random.seed(config['seed'])
    available = active_stages(config)
    if args.stage != 'all' and args.stage not in available:
        parser.error(f'This recipe has stages {available}; no encoder preparation')
    provenance = training_provenance(config)
    model, pre, _ = build_refiner(config)
    saved = torch.load(args.resume or args.init, weights_only=True, map_location='cpu', mmap=True) if (args.resume or args.init) else None
    if saved:
        if saved['provenance'] != provenance:
            raise ValueError('Checkpoint baseline/data mismatch')
        validate_resume_config(saved['config'], config, saved['stage'])
        if args.init and saved['step'] != saved['config']['stages'][saved['stage']]['updates']:
            raise ValueError('--init must be a completed previous stage')
        model.load_added_state_dict(saved['model'], allow_legacy_grounding=saved['stage'] in ('encoder', 'grounding'))
    if args.stage == 'all':
        first = available.index(saved['stage'])+(0 if args.resume else 1) if saved else 0
        stages = available[first:]
    else:
        stages = (args.stage,)
        if args.stage != available[0] and not saved:
            parser.error('This stage requires --init from its preceding stage or --resume')
        if args.init and available.index(args.stage) != available.index(saved['stage'])+1:
            parser.error('--init must come from the immediately preceding stage')
    for stage in stages:
        run_stage(model, pre, config, stage, provenance, resume=saved if args.resume and stage == saved['stage'] else None,
                  profile_updates=args.profile_updates)
        if config['device'].startswith('cuda'):
            torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
