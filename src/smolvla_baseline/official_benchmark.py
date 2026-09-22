"""Resumable baseline evaluation using UniVTAC's official temporal-aggregation schedule."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np
import torch

from .benchmark import model_input, open_video, video_frame
from .data import ROOT, prepare_batch
from .model import build_model, load_checkpoint
from .official_schedule import OfficialTemporalAggregator, UPSTREAM_ROOT, UPSTREAM_COMMIT
from .sim_bridge import UniVTACEnvironment
from .instructions import DEFAULT_TASKS


def validate_completed_rows(rows, tasks, episodes, seed):
    identities = set()
    for row in rows:
        identity = row['task'], row['seed']
        if identity in identities or row['task'] not in tasks or not seed <= row['seed'] < seed+episodes:
            raise ValueError('Duplicate or out-of-range completed episode')
        if row['steps'] != row['model_calls'] or row['steps'] != row['physics_steps']:
            raise ValueError('Completed episode has incompatible execution timing')
        identities.add(identity)
    return identities


def summarize(metadata, rows):
    result = {**metadata, 'episodes': len(rows),
              'complete': len(rows) == len(metadata['tasks'])*metadata['episodes_per_task'], 'tasks': {}}
    for task in metadata['tasks']:
        subset = [r for r in rows if r['task'] == task]
        successes = sum(bool(r['success']) for r in subset)
        result['tasks'][task] = {'episodes': len(subset), 'successes': successes,
                               'success_rate': successes/len(subset) if subset else None}
    result['successes'] = sum(bool(r['success']) for r in rows)
    result['success_rate'] = result['successes']/len(rows) if rows else None
    if len(metadata['tasks']) == 1 and metadata['episodes_per_task'] == 1 and rows:
        result['result'] = rows[0]  # Preserve the original single-probe publisher interface.
    return result


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=1000000)
    parser.add_argument('--tasks', nargs='+', choices=DEFAULT_TASKS, default=['insert_HDMI'])
    parser.add_argument('--episodes', type=int, default=1)
    parser.add_argument('--video-episodes', type=int, default=3)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--hf-repo', help='Private dataset repository for incremental completed results')
    parser.add_argument('--checkpoint', type=Path, default=ROOT/'runs/smolvla_full_dense_b16_ga4_72h/final.pt')
    args = parser.parse_args()
    output = args.output.resolve()
    if args.episodes < 1 or args.video_episodes < 0 or len(args.tasks) != len(set(args.tasks)):
        parser.error('Expected positive episode count, nonnegative videos, and unique tasks')
    if output.exists() and any(output.iterdir()) and not args.resume:
        raise ValueError('Use a fresh output directory for the official-schedule probe')
    revision = subprocess.check_output(['git', '-C', str(UPSTREAM_ROOT), 'rev-parse', 'HEAD'], text=True).strip()
    if revision != UPSTREAM_COMMIT:
        raise ValueError('UniVTAC revision changed; revalidate the official schedule')
    sources = ['docs/Deploy.md', 'policy/smolvla/deploy.yml', 'policy/smolvla/deploy_policy.py',
               'scripts/eval_policy.py', 'envs/_base_task.py', 'envs/robot/robot.py', 'task_config/demo.yml']
    for source in sources:
        committed = subprocess.check_output(['git', '-C', str(UPSTREAM_ROOT), 'show', f'{revision}:{source}'])
        if (UPSTREAM_ROOT/source).read_bytes() != committed:
            raise ValueError(f'Upstream schedule source has local changes: {source}')
    output.mkdir(parents=True, exist_ok=True)
    provenance = output/'provenance'; provenance.mkdir(exist_ok=True)
    for source in sources:
        dest = provenance/'UniVTAC'/source; dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(UPSTREAM_ROOT/source, dest)
    for source in ['src/smolvla_baseline/official_schedule.py', 'src/smolvla_baseline/official_benchmark.py',
                   'src/smolvla_baseline/sim_worker.py', 'src/smolvla_baseline/sim_bridge.py']:
        shutil.copy2(ROOT/source, provenance/Path(source).name)
    torch.set_num_threads(4)
    checkpoint = load_checkpoint(args.checkpoint)
    model, pre, post = build_model(checkpoint['recipe'], checkpoint['manifest'], checkpoint)
    model.eval().requires_grad_(False)
    device = checkpoint['recipe']['device']
    del checkpoint
    aggregator = OfficialTemporalAggregator(device)
    metadata = {'baseline': True, 'tasks': args.tasks, 'seed': args.seed,
        'episodes_per_task': args.episodes, 'video_episodes': min(args.video_episodes, args.episodes),
        'baseline_sha256': sha256(args.checkpoint), 'upstream_commit': revision,
        'schedule': 'univtac_smolvla_every_step_temporal_aggregation',
        'action_chunk': 50, 'inferences_per_action': 1, 'physics_steps_per_action': 1,
        'physics_dt': 1/120, 'video_fps': 120, 'ema': False, 'interpolation': False,
        'aggregation_k': aggregator.state.temporal_agg_k,
        'aggregation_capacity': aggregator.state.temporal_agg_max_timesteps,
        'aggregation_order': 'oldest_first_exp_minus_k_i',
        'controller': 'native_task.take_action(qpos), force=True default, velocity targets untouched',
        'policy_inputs': 'head RGB, wrist RGB, joint[9], instruction; no tactile',
        'qpos_mapping': 'joint[0:7] plus finger[0], broadcast by UniVTAC to both fingers',
        'inference_precision': 'bfloat16 autocast, same as preceding baseline probes',
        'scope': 'official execution schedule with our trained baseline and its original preprocessing; not reproduction of published policy scores'}
    meta_path = output/'metadata.json'
    if meta_path.exists():
        if not args.resume or json.loads(meta_path.read_text()) != metadata:
            raise ValueError('Cannot resume with different checkpoint, schedule, tasks or budget')
    else:
        meta_path.write_text(json.dumps(metadata, indent=2)+'\n')
    records = output/'episodes.jsonl'
    rows = [json.loads(line) for line in records.read_text().splitlines() if line.strip()] if records.exists() else []
    completed = validate_completed_rows(rows, args.tasks, args.episodes, args.seed)
    for row in rows:
        for key in ('video', 'action_trace'):
            if key in row and not (output/row[key]).is_file():
                raise ValueError(f'Missing completed-episode artifact: {row[key]}')
    os.environ['UNIVTAC_EVAL_SAVE_ROOT'] = str(ROOT/'runs/simulator_workspaces'/output.name)
    for task in args.tasks:
        if all((task, args.seed+i) in completed for i in range(args.episodes)):
            continue
        env = UniVTACEnvironment(task=task)
        try:
            for episode in range(args.episodes):
                seed = args.seed+episode
                if (task, seed) in completed:
                    continue
                torch.manual_seed(seed); model.reset(); aggregator.reset()
                row = run_episode(env, model, pre, post, aggregator, output, task, episode, seed,
                                  record_video=episode < args.video_episodes)
                with records.open('a') as stream:
                    stream.write(json.dumps(row)+'\n'); stream.flush(); os.fsync(stream.fileno())
                rows.append(row); completed.add((task, seed))
                summary = summarize(metadata, rows)
                (output/'summary.tmp').write_text(json.dumps(summary, indent=2)+'\n')
                (output/'summary.tmp').replace(output/'summary.json')
                if args.hf_repo and ('video' in row or len(rows) % 10 == 0):
                    from .official_publish import publish_progress
                    try:
                        publish_progress(output, args.hf_repo, summary, rows)
                    except Exception as exc:
                        print(json.dumps({'phase': 'upload_deferred', 'error': str(exc)}), flush=True)
        finally:
            env.close()
    summary = summarize(metadata, rows)
    (output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    if args.hf_repo:
        from .official_publish import publish_progress
        publish_progress(output, args.hf_repo, summary, rows)


def run_episode(env, model, pre, post, aggregator, output, task, episode, seed, record_video=True):
    writer = None
    chunks, actions, observations, step_counts = [], [], [], []
    video_path = output/'videos'/task/f'episode_{episode:03d}_seed_{seed}.mp4'
    label = 'UniVTAC schedule: replan every step; temporal aggregation k=0.01'
    try:
        obs, initial_info = env.reset(seed=seed)
        initial_state = obs['state'].copy()
        if record_video:
            writer = open_video(video_path, video_frame(obs, task=task, seed=seed, step=0, status=label), 120)
        done = truncated = False
        started = time.monotonic()
        simulated = 0.0
        while not (done or truncated):
            batch = prepare_batch(model_input(obs), pre)
            with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                chunk = post(model.predict_action_chunk(batch))
            if tuple(chunk.shape) != (1, 50, 9):
                raise ValueError(f'Unexpected baseline chunk shape: {chunk.shape}')
            action = aggregator(chunk)
            obs, _, done, truncated, info = env.step(action)
            if info['physics_steps'] != 1 or info['policy_action_count'] != len(actions)+1:
                raise RuntimeError('Official schedule did not execute one physics step per inference')
            chunks.append(chunk[0].float().cpu().numpy())
            actions.append(action.copy()); observations.append(obs['state'].copy())
            step_counts.append(info['physics_steps']); simulated += info['simulated_seconds']
            if writer is not None:
                writer.write(video_frame(obs, task=task, seed=seed, step=len(actions), status=label))
            if len(actions) % 25 == 0:
                print(json.dumps({'phase': 'rollout', 'task': task, 'seed': seed, 'actions': len(actions), 'model_calls': len(chunks),
                    'simulated_seconds': simulated, 'elapsed_seconds': time.monotonic()-started}), flush=True)
        row = {'task': task, 'seed': seed, 'success': info['success'],
            'steps': len(actions), 'model_calls': len(chunks), 'physics_steps': sum(step_counts),
            'simulated_seconds': simulated, 'seconds': time.monotonic()-started,
            'initial_info': initial_info, 'info': info}
        # Independently verify alignment and weighting before counting an episode.
        chunk_array = np.asarray(chunks)
        max_error = 0.0
        for t, action in enumerate(actions):
            starts = np.arange(max(0, t-49), t+1)
            weights = np.exp(-aggregator.state.temporal_agg_k*np.arange(len(starts), dtype=np.float64))
            expected = (chunk_array[starts, t-starts]*weights[:, None]).sum(axis=0)/weights.sum()
            np.testing.assert_allclose(action, expected, rtol=2e-6, atol=1e-6)
            max_error = max(max_error, float(np.abs(action-expected).max()))
        row['max_aggregation_reconstruction_error'] = max_error
        np.testing.assert_allclose(simulated, len(actions)/120)
        if record_video:
            row['video'] = str(video_path.relative_to(output))
            row['action_trace'] = f'action_traces/{task}/episode_{episode:03d}_seed_{seed}.npz'
            trace_path = output/row['action_trace']; trace_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(trace_path, initial_state=initial_state, predicted_chunks=chunk_array,
                executed_actions=np.asarray(actions), observed_states=np.asarray(observations),
                physics_steps=np.asarray(step_counts))
        print(json.dumps({'phase': 'rollout_complete', **row}), flush=True)
        return row
    finally:
        if writer is not None:
            writer.release()


if __name__ == '__main__':
    main()
