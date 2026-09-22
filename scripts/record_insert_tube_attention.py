"""One matched60 task rollout with passive resampler attention recording."""
import argparse
import json
import os
from pathlib import Path
import subprocess

import h5py
import numpy as np
import torch

from tactile_refinement.ablation import atomic_json, autocast, load_action_checkpoint
from tactile_refinement.ablation_rollout import code_provenance, run_episode
from tactile_refinement.attention_audit import ResamplerAttentionRecorder
from tactile_refinement.data import normalize_depth
from tactile_refinement.sim_bridge import TactileEnvironment
from tactile_refinement.train import file_sha256
from smolvla_baseline.instructions import DEFAULT_TASKS

ROOT = Path(__file__).resolve().parents[1]
REPO = 'sungkyunner/univtac-stage2-tactile-ablation-checkpoint-33333'


class ObservedEnvironment:
    def __init__(self, env, model, recorder, stream, device):
        self.env, self.model, self.recorder = env, model, recorder
        self.stream, self.device = stream, device
        self.obs = None
        self.steps = 0
        self.simulated = 0.

    def reset(self, **kwargs):
        self.obs, initial = self.env.reset(**kwargs)
        return self.obs, initial

    def append(self, name, value):
        value = np.asarray(value)
        if name not in self.stream:
            self.stream.create_dataset(name, shape=(0, *value.shape),
                maxshape=(None, *value.shape), chunks=(1, *value.shape),
                dtype=value.dtype, compression='gzip', compression_opts=1)
        dataset = self.stream[name]
        dataset.resize(len(dataset)+1, axis=0)
        dataset[-1] = value

    @torch.inference_mode()
    def step(self, action):
        # These tensors belong to the prediction from self.obs, before action.
        attention = self.recorder.probabilities().numpy()
        np.testing.assert_allclose(attention.sum(-1), 1., atol=1e-5)
        with autocast(self.device):
            change = self.model.decoder(self.recorder.predicted)[0].float().cpu().numpy()
        depth, valid = normalize_depth(self.obs['tactile_depth_mm'])
        for name, value in dict(depth=depth.numpy().astype('float16'),
                depth_valid=valid.numpy(), predicted_change=change.astype('float32'),
                attention=attention.astype('float16'), head=self.obs['head'],
                wrist=self.obs['wrist'], state=self.obs['state'], action=action,
                simulated_seconds=np.float64(self.simulated)).items():
            if not np.isfinite(value).all():
                raise ValueError(f'Nonfinite diagnostic: {name}')
            self.append(name, value)
        result = self.env.step(action)
        self.obs = result[0]
        self.simulated += result[4]['simulated_seconds']
        self.steps += 1
        self.stream.flush()
        if self.steps % 20 == 0:
            print(json.dumps(dict(phase='recording', steps=self.steps,
                                  simulated_seconds=self.simulated)), flush=True)
        return result


def upload(output, metadata, prefix_override=None):
    from huggingface_hub import CommitOperationAdd, HfApi
    api = HfApi()
    if not api.repo_info(REPO, repo_type='dataset').private:
        raise ValueError('Expected existing private evaluation repository')
    task = metadata['task']
    stage = metadata['checkpoint']['stage']
    stage_prefix = 'stage3/' if stage == 'joint' else ''
    prefix = (prefix_override or
              f"{stage_prefix}checkpoint_{metadata['checkpoint']['step']}/attention/{task}/seed_{metadata['seed']}")
    names = ['attention_visualization.mp4', 'attention_snapshot.png', 'left_queries.png',
             'right_queries.png', 'attention_metrics.json', 'attention_head_mean.npz',
             'REPORT.md', 'metadata.json', 'episode.json',
             f"videos/{task}/seed_{metadata['seed']}_correct.mp4"]
    if (output/'reference_comparison.json').exists():
        names.append('reference_comparison.json')
    operations = [CommitOperationAdd(path_in_repo=f'{prefix}/{name}',
                  path_or_fileobj=str(output/name)) for name in names]
    commit = api.create_commit(repo_id=REPO, repo_type='dataset', operations=operations,
        commit_message=f'Add {stage} checkpoint {metadata["checkpoint"]["step"]} {task} attention visualization')
    remote = {entry.path: entry.size for entry in api.list_repo_tree(REPO,
              repo_type='dataset', path_in_repo=prefix, recursive=True) if hasattr(entry, 'size')}
    for name in names:
        if remote.get(f'{prefix}/{name}') != (output/name).stat().st_size:
            raise RuntimeError(f'Upload verification failed: {name}')
    return dict(commit_url=commit.commit_url, prefix=prefix, repo=REPO, files=names)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path,
        default=ROOT/'runs/tactile_refinement_trex_fullgrid_n0/expert/checkpoint_2.pt',
        help='Stage-2 or Stage-3 action checkpoint to evaluate')
    parser.add_argument('--seed', type=int, default=2000000)
    parser.add_argument('--task', choices=DEFAULT_TASKS, default='insert_tube')
    parser.add_argument('--upload', action='store_true')
    parser.add_argument('--upload-prefix', help='Hugging Face folder for this diagnostic replay')
    parser.add_argument('--reference-episodes', type=Path,
                        help='Completed scored rollouts to compare with this separate replay')
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    def status(phase, **values):
        atomic_json(output/'status.json', dict(phase=phase, **values))
        print(json.dumps(dict(phase=phase, **values)), flush=True)
    try:
        torch.set_num_threads(4)
        os.environ['UNIVTAC_EVAL_SAVE_ROOT'] = str(output/'simulator_artifacts')
        status('loading_checkpoint')
        path = args.checkpoint.resolve()
        model, pre, post, config, checkpoint = load_action_checkpoint(path)
        if checkpoint.get('stage') not in ('expert', 'joint') or int(checkpoint.get('step', 0)) <= 0:
            raise ValueError(f'Expected an action-trained checkpoint, got {checkpoint}')
        metadata = dict(checkpoint=checkpoint, task=args.task, seed=args.seed,
            episodes=1, condition='correct (predicted future tactile latent)',
            schedule='matched60', controller='linear_position',
            timing=dict(policy_hz=60, physics_hz=120, replan_every_action=True,
                        temporal_aggregation=True, substeps='midpoint then endpoint'),
            attention_source='Predicted future-change tokens: 300 per finger, 16 queries per finger',
            attention_axes=['frame', 'layer', 'finger', 'head', 'query', 'position'],
            finger_order=['left_gsmini', 'right_gsmini'], grid=[15, 20],
            normalization='clip((34-H_mm)/10,0,1); invalid pixels set to zero',
            map_caveat='Normalized tactile depth, not calibrated physical indentation',
            attention_caveat='QK softmax readout, not causal attribution or direct raw-pixel attention; float32 shadow softmax may differ slightly from fused BF16 numerics',
            code_sha256=code_provenance('matched60'))
        if args.reference_episodes:
            metadata['reference_rollouts'] = str(args.reference_episodes.resolve())
            metadata['replay_caveat'] = ('Separate simulator replay with the same seed and policy; '
                'attention video is not a recording of the scored rollout.')
        for source in ('scripts/record_insert_tube_attention.py', 'src/tactile_refinement/attention_audit.py',
                       'scripts/render_insert_tube_attention.py'):
            metadata['code_sha256'][source] = file_sha256(ROOT/source)
        atomic_json(output/'metadata.json', metadata)
        recorder = ResamplerAttentionRecorder(model.resampler)
        env = None
        try:
            status('starting_simulator')
            env = TactileEnvironment(task=args.task, controller='linear_position')
            with h5py.File(output/'diagnostics.h5', 'w') as stream:
                observed = ObservedEnvironment(env, model, recorder, stream, config['device'])
                status('running_one_episode')
                row = run_episode(observed, model, pre, post, None, task=args.task,
                    seed=args.seed, condition='correct', schedule='matched60', device=config['device'],
                    output=output, record_video=True)
                if len(stream['attention']) != row['model_calls'] or row['model_calls'] != row['steps']:
                    raise RuntimeError('Attention frames are not aligned to policy decisions')
                stream.create_dataset('terminal_depth', data=normalize_depth(observed.obs['tactile_depth_mm'])[0].numpy())
                stream.attrs['complete'] = True
            atomic_json(output/'episode.json', row)
            if args.reference_episodes:
                reference = [json.loads(line) for line in
                             args.reference_episodes.read_bytes().splitlines(keepends=True)
                             if line.endswith(b'\n')]
                matching = [item for item in reference if item['task'] == args.task
                            and item['seed'] == args.seed and item['condition'] == 'correct']
                if len(matching) != 1:
                    raise ValueError('Expected exactly one scored rollout for diagnostic replay')
                original = matching[0]
                atomic_json(output/'reference_comparison.json', {
                    'reference_is_separate_rollout': True,
                    'reference_success': original['success'], 'replay_success': row['success'],
                    'reference_steps': original['steps'], 'replay_steps': row['steps'],
                    'matching_recorded_initial_state_id': (
                        original['initial_info'].get('initial_state_id') ==
                        row['initial_info'].get('initial_state_id')),
                    'same_success_and_steps': (original['success'] == row['success']
                                               and original['steps'] == row['steps'])})
        finally:
            recorder.close()
            if env is not None:
                env.close()
        del observed, model, recorder
        torch.cuda.empty_cache()
        status('rendering', success=row['success'], steps=row['steps'])
        subprocess.run([str(ROOT/'.venv-benchmark/bin/python'), str(ROOT/'scripts/render_insert_tube_attention.py'),
                        '--output', str(output)], check=True, cwd=ROOT)
        result = {}
        if args.upload:
            status('uploading')
            result = upload(output, metadata, args.upload_prefix)
        status('complete', success=row['success'], steps=row['steps'], **result)
    except Exception as exc:
        status('failed', error=repr(exc))
        raise


if __name__ == '__main__':
    main()
