"""Exercise the actual CLI, stage transitions, three saves, and resumed sampling."""
import json
import os
from pathlib import Path
import subprocess
import tempfile

import torch

from smolvla_baseline.data import ROOT
from tactile_refinement.train import active_stages


def main():
    config = json.loads((ROOT/'configs/tactile_refinement.json').read_text())
    config.update(batch_size=1, gradient_accumulation_steps=1, num_workers=0)
    for stage in config['stages'].values():
        stage['updates'] = 3
    with tempfile.TemporaryDirectory(prefix='refinement-cli-', dir=ROOT/'runs') as directory:
        folder = Path(directory)
        config['output_dir'] = str(folder/'training')
        recipe = folder/'config.json'
        # Reproduce the live job: grounding saves the old, unused dense expert.
        # Resume with shared FFN LoRA, retaining tactile optimizer and RNG states.
        shared = config.get('expert_implementation') == 'shared_ffn_lora_v1'
        initial_config = {k: v for k, v in config.items()
                          if not shared or k not in ('expert_implementation', 'lora_rank', 'lora_alpha',
                                                      'resampler_tokens_per_finger')}
        recipe.write_text(json.dumps(initial_config))
        command = [str(ROOT/'.venv-train/bin/python'), '-u', '-m', 'tactile_refinement.train', '--config', str(recipe)]
        first_stage = active_stages(config)[0]
        subprocess.run([*command, '--stage', first_stage], cwd=ROOT, env=os.environ, check=True)
        initial = Path(config['output_dir'])/first_stage/'checkpoint_1.pt'
        recipe.write_text(json.dumps(config))
        # Resume from update one, replay the remaining encoder updates, then train
        # all three policy stages using the real command-line runner.
        subprocess.run([*command, '--stage', 'all', '--resume', str(initial)], cwd=ROOT, env=os.environ, check=True)
        report = {'batch_size': 1, 'gradient_accumulation_steps': 1, 'updates_per_stage': 3,
                  'legacy_grounding_to_shared_lora_migration': shared,
                  'resume_from': f'{first_stage}/checkpoint_1.pt', 'stages': {}}
        for stage in config['stages']:
            output = Path(config['output_dir'])/stage
            names = sorted(p.name for p in output.glob('*.pt'))
            assert names == ['checkpoint_1.pt', 'checkpoint_2.pt', 'final.pt']
            final = torch.load(output/'final.pt', weights_only=True, map_location='cpu', mmap=True)
            assert final['step'] == 3 and final['stage'] == stage
            assert all(not key.startswith('baseline.') for key in final['model'])
            if shared:
                assert final['format_version'] == (4 if config.get('resampler_tokens_per_finger') else 2)
                assert not any(key.startswith('expert.lm_expert.') for key in final['model'])
            report['stages'][stage] = names
            del final
        # Prove the saved final model is accepted by the offline diagnostic CLI.
        diagnostic = folder/'diagnostic.json'
        subprocess.run([str(ROOT/'.venv-train/bin/python'), '-m', 'tactile_refinement.diagnostics',
            '--checkpoint', str(Path(config['output_dir'])/'joint/final.pt'), '--samples', '2',
            '--output', str(diagnostic)], cwd=ROOT, env=os.environ, check=True)
        report['diagnostics'] = json.loads(diagnostic.read_text())['metrics']
        output = 'ffn_lora_training_verification.json' if shared else 'refinement_training_verification.json'
        (ROOT/'logs'/output).write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
