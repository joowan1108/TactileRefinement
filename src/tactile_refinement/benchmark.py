"""Matched full-50-action/two-physics-step baseline and refinement rollouts."""
import argparse
from collections import deque
import json
from pathlib import Path
import time

import numpy as np
import torch

from smolvla_baseline.benchmark import model_input, video_frame, open_video
from smolvla_baseline.data import ROOT, prepare_batch
from smolvla_baseline.instructions import DEFAULT_TASKS
from .data import normalize_depth
from .sim_bridge import TactileEnvironment
from .smoothing import ArmEMA
from .train import build_refiner, file_sha256, training_provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/'configs/tactile_refinement.json')
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--baseline', action='store_true', help='Matched frozen baseline, full 50 actions, two-step holds')
    parser.add_argument('--condition', choices=('predicted', 'zero', 'current'), default='predicted',
                        help='zero/current are inference perturbations, not retrained controls')
    parser.add_argument('--tasks', nargs='+', choices=DEFAULT_TASKS, default=list(DEFAULT_TASKS))
    parser.add_argument('--episodes', type=int, default=100)
    parser.add_argument('--start-seed', type=int, default=1000000)
    parser.add_argument('--video-episodes', type=int, default=3)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--arm-ema-alpha', type=float,
                        help='Evaluation-only arm EMA after denormalization; gripper unchanged')
    parser.add_argument('--controller', choices=('hold', 'linear', 'linear_position'),
                        help='New runs default to linear_position (midpoint/end, zero velocity targets); resumes retain their controller')
    args = parser.parse_args()
    if args.episodes < 1 or args.video_episodes < 0:
        parser.error('--episodes must be positive and --video-episodes nonnegative')
    if bool(args.baseline) == bool(args.checkpoint):
        parser.error('Select exactly one of --baseline and --checkpoint')
    if args.arm_ema_alpha is not None and not 0 < args.arm_ema_alpha <= 1:
        parser.error('--arm-ema-alpha must be in (0, 1]')
    meta_path = args.output/'metadata.json'
    if args.controller is None:
        old_meta = json.loads(meta_path.read_text()) if args.resume and meta_path.exists() else None
        args.controller = old_meta.get('controller', {}).get('mode', 'hold') if old_meta is not None else 'linear_position'
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=True, mmap=True) if args.checkpoint else None
    config = saved['config'] if saved else json.loads(args.config.read_text())
    provenance = training_provenance(config)
    if saved and saved['provenance'] != provenance:
        raise ValueError('Baseline/data do not match the refinement checkpoint')
    if saved and saved['stage'] not in ('expert', 'joint'):
        raise ValueError('Benchmark requires an action-trained checkpoint')
    torch.set_num_threads(4)
    model, pre, post = build_refiner(config)
    if saved:
        model.load_added_state_dict(saved['model'])
        model.set_stage(saved['stage'])
    model.eval()
    # select_action queues are not used: exactly one complete chunk per observation.
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    metadata = {'baseline': args.baseline, 'condition': args.condition, 'provenance': provenance,
                'continuation_mask_prefix': bool(saved and saved['stage'] == 'expert' and model.stage2_mask_prefix),
                'checkpoint_sha256': file_sha256(args.checkpoint) if saved else None,
                'action_chunk': 50, 'physics_steps_per_action': 2, 'seed': args.start_seed,
                'episodes_per_task': args.episodes, 'tasks': args.tasks, 'video_episodes': args.video_episodes}
    if args.arm_ema_alpha is not None:
        metadata['action_smoothing'] = {'method': 'ema', 'alpha': args.arm_ema_alpha,
            'joints': list(range(7)), 'initialization': 'observed_arm_positions',
            'reset': 'episode_only', 'space': 'denormalized_absolute_joint_positions',
            'gripper': 'unfiltered'}
    if args.controller != 'hold':
        metadata['controller'] = {'mode': args.controller, 'substeps': 2,
            'positions': 'midpoint_then_endpoint', 'velocity': ('zero' if args.controller == 'linear_position' else 'endpoint_difference_divided_by_two_physics_timesteps'),
            'start': 'observed_reset_pose_then_previous_commanded_endpoint',
            'gripper': 'interpolated_shared_finger_target', 'force': True}
    meta_path = output/'metadata.json'
    if meta_path.exists():
        if not args.resume or json.loads(meta_path.read_text()) != metadata:
            raise ValueError('Existing benchmark output/configuration mismatch')
    else:
        if any(output.iterdir()):
            raise ValueError('Use an empty output directory')
        meta_path.write_text(json.dumps(metadata, indent=2)+'\n')
    records = output/'episodes.jsonl'
    rows = [json.loads(line) for line in records.read_text().splitlines()] if records.exists() else []
    completed = {(r['task'], r['seed']) for r in rows}
    if len(completed) != len(rows):
        raise ValueError('Duplicate completed episodes')
    for task in args.tasks:
        env = TactileEnvironment(task=task, controller=args.controller)
        try:
            for episode in range(args.episodes):
                seed = args.start_seed+episode
                if (task, seed) in completed:
                    continue
                torch.manual_seed(seed)
                model.baseline.reset()
                obs, initial = env.reset(seed=seed)
                smoother = ArmEMA(args.arm_ema_alpha, obs['state']) if args.arm_ema_alpha is not None else None
                initial_state = obs['state'].copy()
                trace_raw, trace_filtered, trace_observed = [], [], []
                trace_substeps = []
                record_trace = smoother is not None or args.controller != 'hold'
                video_status = f'arm EMA alpha={args.arm_ema_alpha}' if smoother is not None else None
                if args.controller != 'hold':
                    label = 'linear pos; zero vel 120Hz' if args.controller == 'linear_position' else 'linear pos+vel 120Hz'
                    video_status = (video_status+'; ' if video_status else 'no EMA; ')+label
                queue = deque()
                done = truncated = False
                steps = physics = 0
                simulated = 0.0
                video = None
                video_path = output/'videos'/task/f'episode_{episode:03d}_seed_{seed}.mp4'
                started = time.monotonic()
                if episode < args.video_episodes:
                    video = open_video(video_path, video_frame(obs, task=task, seed=seed, step=0, status=video_status), 60.0)
                try:
                    while not (done or truncated):
                        if not queue:
                            batch = prepare_batch(model_input(obs), pre)
                            with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                                if args.baseline:
                                    chunk = model.baseline.predict_action_chunk(batch)
                                else:
                                    depth, _ = normalize_depth(obs['tactile_depth_mm'])
                                    chunk = model.predict_action_chunk(batch, depth[None].to(config['device']), condition=args.condition)
                                queue.extend(post(chunk).squeeze(0).cpu().numpy())
                        raw_action = queue.popleft()
                        action = smoother(raw_action) if smoother is not None else raw_action
                        obs, _, done, truncated, info = env.step(action)
                        if record_trace:
                            trace_raw.append(raw_action.copy())
                            trace_filtered.append(action.copy())
                            trace_observed.append(obs['state'].copy())
                            trace_substeps.extend(info.pop('controller_commands', []))
                        steps += 1; physics += info['physics_steps']; simulated += info['simulated_seconds']
                        if video:
                            video.write(video_frame(obs, task=task, seed=seed, step=steps, status=video_status))
                finally:
                    if video:
                        video.release()
                row = {'task': task, 'seed': seed, 'success': info['success'], 'steps': steps,
                       'physics_steps': physics, 'simulated_seconds': simulated,
                       'seconds': time.monotonic()-started, 'initial_info': initial, 'info': info}
                if video:
                    row['video'] = str(video_path.relative_to(output))
                if record_trace:
                    trace_path = output/'action_traces'/task/f'episode_{episode:03d}_seed_{seed}.npz'
                    trace_path.parent.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(trace_path, initial_state=initial_state,
                        raw_actions=np.asarray(trace_raw), filtered_actions=np.asarray(trace_filtered),
                        observed_states=np.asarray(trace_observed),
                        substep_positions=np.asarray([s['position'] for s in trace_substeps]),
                        substep_velocities=np.asarray([s['velocity'] for s in trace_substeps]),
                        substep_observed_positions=np.asarray([s['observed_position'] for s in trace_substeps]))
                    row['action_trace'] = str(trace_path.relative_to(output))
                with records.open('a') as stream:
                    stream.write(json.dumps(row)+'\n')
                rows.append(row)
                print(json.dumps(row), flush=True)
        finally:
            env.close()
    summary = {**metadata, 'episodes': len(rows), 'tasks': {}}
    for task in args.tasks:
        subset = [r for r in rows if r['task'] == task]
        successes = sum(r['success'] for r in subset)
        summary['tasks'][task] = {'episodes': len(subset), 'successes': successes,
                                  'success_rate': successes/len(subset)}
    (output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')


if __name__ == '__main__':
    main()
