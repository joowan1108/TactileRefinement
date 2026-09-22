"""Matched action-stage correct/mismatched/disabled tactile rollouts.

Default: training-aligned 60 Hz replanning/aggregation with two 120 Hz substeps.
Opt-in tactile10_vl50: execute 10 of each fresh 50-action chunk, caching VL for 50 actions.
The official one-step and original full-chunk schedules remain explicit probes.
"""
import argparse
from collections import deque
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
import torch

from smolvla_baseline.benchmark import model_input, open_video, video_frame
from smolvla_baseline.data import ROOT, prepare_batch
from smolvla_baseline.instructions import DEFAULT_TASKS
from smolvla_baseline.official_schedule import OfficialTemporalAggregator, UPSTREAM_COMMIT, UPSTREAM_ROOT
from .ablation import (CONDITIONS, DonorBank, atomic_json, autocast, load_action_checkpoint,
                       mean_ci, seeded_noise)
from .data import TactileEpisodes, normalize_depth
from .model import encode_prefix, refresh_prefix_state
from .sim_bridge import TactileEnvironment
from .train import file_sha256


TOKEN_PRUNING_CONDITIONS = ('full16', 'top8', 'uniform8')


def code_provenance(schedule):
    sources = ('src/tactile_refinement/ablation.py', 'src/tactile_refinement/ablation_rollout.py',
               'src/tactile_refinement/model.py', 'src/tactile_refinement/execution.py',
               'src/smolvla_baseline/sim_worker.py', 'src/smolvla_baseline/official_schedule.py')
    hashes = {s: file_sha256(ROOT/s) for s in sources}
    if schedule in ('official', 'matched60', 'tactile10_vl50'):
        revision = subprocess.check_output(['git', '-C', str(UPSTREAM_ROOT), 'rev-parse', 'HEAD'], text=True).strip()
        if revision != UPSTREAM_COMMIT:
            raise ValueError('UniVTAC revision changed; revalidate the official schedule')
        for source in ('policy/smolvla/deploy.yml', 'policy/smolvla/deploy_policy.py',
                       'envs/_base_task.py', 'envs/robot/robot.py', 'task_config/demo.yml'):
            expected = subprocess.check_output(['git', '-C', str(UPSTREAM_ROOT), 'show', f'{revision}:{source}'])
            if (UPSTREAM_ROOT/source).read_bytes() != expected:
                raise ValueError(f'Upstream schedule source has local changes: {source}')
            hashes['UniVTAC/'+source] = file_sha256(UPSTREAM_ROOT/source)
    return hashes


def validate_rows(rows, tasks, episodes, start_seed, smoke_steps, conditions=CONDITIONS):
    completed = set()
    for row in rows:
        key = (row['task'], row['seed'], row['condition'])
        if (key in completed or row['task'] not in tasks or row['condition'] not in conditions
                or not start_seed <= row['seed'] < start_seed+episodes
                or row['smoke'] != bool(smoke_steps)):
            raise ValueError('Duplicate, foreign, or incompatible completed rollout')
        completed.add(key)
    return completed


def success_rate(values, z=1.959963984540054):
    """Wilson interval stays nondegenerate for all-success/all-failure samples."""
    n = len(values)
    if not n:
        return {'n': 0, 'mean': None, 'ci95': None}
    p = float(np.mean(values))
    denominator = 1+z*z/n
    center = (p+z*z/(2*n))/denominator
    half = z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/denominator
    return {'n': n, 'mean': p, 'ci95': [max(0., center-half), min(1., center+half)]}


def success_difference(pairs):
    """Seed-block bootstrap plus conservative marginal score bounds.

    The latter combine two 97.5% Wilson intervals to avoid claiming zero
    uncertainty when every observed pair has the same binary outcomes.
    """
    correct = [int(a['success']) for a, _ in pairs]
    control = [int(b['success']) for _, b in pairs]
    result = mean_ci([a-b for a, b in zip(correct, control)])
    result['paired_bootstrap_ci95'] = result.pop('ci95')
    result['ci95'] = None
    result['interval_method'] = 'Conservative difference of marginal 97.5% Wilson bounds'
    if pairs:
        ca = success_rate(correct, z=2.241402727604947)['ci95']
        cb = success_rate(control, z=2.241402727604947)['ci95']
        result['ci95'] = [max(-1., ca[0]-cb[1]), min(1., ca[1]-cb[0])]
    return result


def summarize(rows, tasks, episodes, selected_conditions=CONDITIONS):
    expected = len(tasks)*episodes*len(selected_conditions)
    result = {'completed_rollouts': len(rows), 'expected_rollouts': expected,
              'complete': len(rows) == expected, 'tasks': {},
              'reset_caveat': 'Matching IDs cover rounded robot/actor poses, not full simulator/UIPC state',
              'controls_caveat': 'Sensor substitutions and zeros can be out of distribution; these are not retrained controls'}
    for task in tasks:
        subset = [r for r in rows if r['task'] == task and not r['smoke']]
        condition_stats = {}
        for name in selected_conditions:
            group = [r for r in subset if r['condition'] == name]
            successes = [r for r in group if r['success']]
            condition_stats[name] = {'episodes': len(group), 'success_rate': success_rate([int(r['success']) for r in group]),
                'early_stop_rate': success_rate([int(r['early_stop']) for r in group]),
                'successful_completion_seconds': mean_ci([r['simulated_seconds'] for r in successes]),
                'failed_duration_seconds': mean_ci([r['simulated_seconds'] for r in group if not r['success']])}
        grouped = {}
        for row in subset:
            grouped.setdefault(row['seed'], {})[row['condition']] = row
        paired = {}
        for control in (name for name in selected_conditions if name != 'correct'):
            pairs = [(g['correct'], g[control]) for g in grouped.values() if 'correct' in g and control in g]
            matched = [(a, b) for a, b in pairs if a['initial_info'].get('initial_state_id') is not None
                       and a['initial_info']['initial_state_id'] == b['initial_info'].get('initial_state_id')]
            paired[control] = {
                'seed_blocks': len(pairs), 'matching_recorded_reset_ids': len(matched),
                'mismatching_or_missing_reset_ids': len(pairs)-len(matched),
                'correct_minus_control_success_same_seed_not_exactly_paired': success_difference(pairs),
                'correct_minus_control_success_matching_recorded_reset_ids': success_difference(matched)}
        result['tasks'][task] = {'conditions': condition_stats, 'comparisons': paired}
    return result


@torch.inference_mode()
def run_episode(env, model, pre, post, bank, *, task, seed, condition, schedule,
                device, output, record_video=False, smoke_steps=0):
    if condition not in CONDITIONS + TOKEN_PRUNING_CONDITIONS:
        raise ValueError(condition)
    obs, initial = env.reset(seed=seed)
    model.baseline.reset()
    queue = deque()
    # The same upstream aggregation rule is used, but matched60 advances its
    # timestep once per recorded-data interval rather than once per physics tick.
    aggregator = OfficialTemporalAggregator(device) if schedule in ('official', 'matched60') else None
    steps = calls = physics = 0
    simulated = 0.0
    actions, observed, donors, substep_commands = [], [], [], []
    prefix = None
    replan_steps, vl_refresh_steps, replan_states, predicted_chunks = [], [], [], []
    initial_state = np.asarray(obs['state']).copy()
    initial_activity = float(normalize_depth(obs['tactile_depth_mm'])[0].mean())
    video_path = output/'videos'/task/f'seed_{seed}_{condition}.mp4'
    video_status = f'{condition} {schedule}' if schedule == 'tactile10_vl50' else condition
    writer = None
    started = time.monotonic()
    done = truncated = False
    try:
        if record_video:
            writer = open_video(video_path, video_frame(obs, task=task, seed=seed, step=0, status=video_status),
                                120 if schedule == 'official' else 60)
        while not (done or truncated):
            if not queue:
                depth, _ = normalize_depth(obs['tactile_depth_mm'])
                if condition == 'mismatched':
                    depth, donor = bank.select(task, obs['state'])
                    donors.append({'model_call': calls, **donor})
                batch = prepare_batch(model_input(obs), pre)
                with autocast(device):
                    kwargs = dict(condition='zero' if condition == 'disabled' else 'predicted',
                                  noise=seeded_noise(model, seed+1000003*calls))
                    if schedule == 'tactile10_vl50':
                        if prefix is None or steps % 50 == 0:
                            prefix = encode_prefix(model.baseline, batch)
                            vl_refresh_steps.append(steps)
                        else:
                            prefix = refresh_prefix_state(model.baseline, prefix, batch)
                        chunk = model.predict_action_chunk_from_prefix(
                            prefix, depth[None].to(device), **kwargs)
                    else:
                        chunk = model.predict_action_chunk(batch, depth[None].to(device), **kwargs)
                    physical = post(chunk)
                calls += 1
                if aggregator is not None:
                    queue.append(aggregator(physical))
                elif schedule == 'tactile10_vl50':
                    plan = physical.squeeze(0).float().cpu().numpy()
                    if plan.shape != (50, 9):
                        raise ValueError(f'Expected a fresh 50x9 action chunk, got {plan.shape}')
                    # Discard the unexecuted 40 actions at the next replan. No
                    # temporal averaging with an older horizon or older tactile.
                    queue.extend(plan[:10])
                    replan_steps.append(steps)
                    replan_states.append(np.asarray(obs['state']).copy())
                    predicted_chunks.append(plan.copy())
                else:
                    queue.extend(physical.squeeze(0).float().cpu().numpy())
            action = queue.popleft()
            obs, _, done, truncated, info = env.step(action)
            steps += 1
            physics += info['physics_steps']
            simulated += info['simulated_seconds']
            commands = info.pop('controller_commands', [])
            if schedule == 'official' and (info['physics_steps'] != 1 or calls != steps):
                raise RuntimeError('Official schedule must replan and execute exactly one physics step')
            if schedule in ('matched60', 'tactile10_vl50'):
                expected_calls = steps if schedule == 'matched60' else (steps+9)//10
                if calls != expected_calls or info['physics_steps'] not in (0, 1, 2):
                    raise RuntimeError('Replan cadence or two-physics-substep execution mismatch')
                if len(commands) != info['physics_steps']:
                    raise RuntimeError('Matched60 controller telemetry does not match physics advancement')
                if info['physics_steps'] == 2:
                    expected = np.asarray(action, dtype=np.float32)[:8]
                    np.testing.assert_allclose(commands[-1]['position'], expected, rtol=0, atol=1e-7)
            if schedule == 'full50' and info['physics_steps'] not in (1, 2):
                raise RuntimeError('Full50 schedule must execute at most two physics steps')
            actions.append(np.asarray(action).copy())
            observed.append(np.asarray(obs['state']).copy())
            if commands:
                substep_commands.extend(commands)
            if writer:
                writer.write(video_frame(obs, task=task, seed=seed, step=steps, status=video_status))
            if smoke_steps and steps >= smoke_steps:
                break
            if steps % 50 == 0:
                print(json.dumps({'task': task, 'seed': seed, 'condition': condition, 'steps': steps}), flush=True)
    finally:
        if writer:
            writer.release()
    trace = output/'action_traces'/task/f'seed_{seed}_{condition}.npz'
    trace.parent.mkdir(parents=True, exist_ok=True)
    trace_values = dict(initial_state=initial_state, actions=np.asarray(actions), observed=np.asarray(observed))
    if schedule == 'tactile10_vl50':
        trace_values.update(replan_steps=np.asarray(replan_steps),
                            vl_refresh_steps=np.asarray(vl_refresh_steps),
                            replan_states=np.asarray(replan_states),
                            predicted_chunks=np.asarray(predicted_chunks))
    if substep_commands:
        trace_values.update(
            substep_positions=np.asarray([x['position'] for x in substep_commands]),
            substep_velocities=np.asarray([x['velocity'] for x in substep_commands]),
            substep_observed_positions=np.asarray([x['observed_position'] for x in substep_commands]))
    np.savez_compressed(trace, **trace_values)
    schedule_info = ({'replan_steps': replan_steps, 'vl_refresh_steps': vl_refresh_steps,
                      'vl_context_age_at_replan': [s % 50 for s in replan_steps],
                      'state_refresh_steps': replan_steps, 'tactile_refresh_steps': replan_steps}
                     if schedule == 'tactile10_vl50' else {})
    return {'task': task, 'seed': seed, 'condition': condition, 'smoke': bool(smoke_steps),
            'success': bool(info['success']) if not smoke_steps else None,
            'early_stop': bool(info.get('early_stop', False)), 'terminated': bool(done), 'truncated': bool(truncated),
            'steps': steps, 'model_calls': calls, 'physics_steps': physics, 'simulated_seconds': simulated,
            'wall_seconds': time.monotonic()-started, 'initial_info': initial,
            'initial_depth_activity': initial_activity, 'donors': donors,
            'video': str(video_path.relative_to(output)) if record_video else None,
            'action_trace': str(trace.relative_to(output)), **schedule_info}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--tasks', nargs='+', choices=DEFAULT_TASKS, default=list(DEFAULT_TASKS))
    parser.add_argument('--episodes', type=int, default=20)
    parser.add_argument('--start-seed', type=int, default=2000000)
    parser.add_argument('--donor-seed', type=int, default=123)
    parser.add_argument('--donors-per-task', type=int, default=256)
    parser.add_argument('--video-episodes', type=int, default=3)
    parser.add_argument('--conditions', nargs='+', choices=CONDITIONS, default=list(CONDITIONS))
    parser.add_argument('--schedule', choices=('matched60', 'tactile10_vl50', 'official', 'full50'), default='matched60')
    parser.add_argument('--smoke-steps', type=int, default=0, help='Integration check only; excluded from success statistics')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if (min(args.episodes, args.donors_per_task) < 1 or min(args.video_episodes, args.smoke_steps) < 0
            or len(args.tasks) != len(set(args.tasks)) or len(args.conditions) != len(set(args.conditions))
            or 'correct' not in args.conditions):
        parser.error('Invalid counts')
    torch.set_num_threads(4)
    source_hashes = code_provenance(args.schedule)
    model, pre, post, config, checkpoint = load_action_checkpoint(args.checkpoint)
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    # Isolate simulator side products from concurrent baseline evaluations.
    os.environ['UNIVTAC_EVAL_SAVE_ROOT'] = str(output.resolve()/'simulator_artifacts')
    controller = {'matched60': 'linear_position', 'tactile10_vl50': 'linear_position',
                  'official': 'official', 'full50': 'hold'}[args.schedule]
    timing = ({'policy_hz': 60, 'physics_hz': 120, 'physics_substeps_per_policy_step': 2,
               'replan_every_policy_step': False, 'predicted_chunk_actions': 50,
               'execute_actions_per_prediction': 10, 'tactile_refresh_every_actions': 10,
               'proprioception_refresh_every_actions': 10, 'vl_refresh_every_actions': 50,
               'state_cache': 'recompute_state_token_per_layer_using_cached_VL_KV',
               'temporal_aggregation': False, 'substep_positions': 'midpoint_then_endpoint',
               'velocity_targets': 'zero',
               'training_caveat': 'Inference probe: grounding/action training used same-time VL and tactile'}
              if args.schedule == 'tactile10_vl50' else
              {'policy_hz': 60, 'physics_hz': 120, 'physics_substeps_per_policy_step': 2,
               'replan_every_policy_step': True, 'aggregation_index_hz': 60,
               'substep_positions': 'midpoint_then_endpoint', 'velocity_targets': 'zero'}
              if args.schedule == 'matched60' else
              {'policy_hz': 120, 'physics_hz': 120, 'physics_substeps_per_policy_step': 1,
               'timing_caveat': '120 Hz chunk indexing for actions trained from 60 Hz recordings'}
              if args.schedule == 'official' else
              {'policy_hz': 60, 'physics_hz': 120, 'physics_substeps_per_policy_step': 2,
               'replan_every_policy_step': False, 'open_loop_chunk_actions': 50})
    metadata = {'checkpoint': checkpoint, 'code_sha256': source_hashes, 'tasks': args.tasks, 'episodes_per_task': args.episodes,
                'start_seed': args.start_seed, 'conditions': args.conditions, 'schedule': args.schedule,
                'controller': controller, 'timing': timing,
                'upstream_commit': UPSTREAM_COMMIT, 'donor_seed': args.donor_seed,
                'donors_per_task': args.donors_per_task, 'video_episodes': args.video_episodes,
                'smoke_steps': args.smoke_steps, 'noise': 'fixed seed per episode and model call across conditions'}
    meta_path = output/'metadata.json'
    if meta_path.exists():
        if not args.resume or json.loads(meta_path.read_text()) != metadata:
            raise ValueError('Existing output/configuration mismatch')
    else:
        if any(output.iterdir()):
            raise ValueError('Use an empty output directory')
        atomic_json(meta_path, metadata)
    records = output/'episodes.jsonl'
    rows = [json.loads(line) for line in records.read_text().splitlines()] if records.exists() else []
    completed = validate_rows(rows, args.tasks, args.episodes, args.start_seed, args.smoke_steps, args.conditions)
    bank = None
    if 'mismatched' in args.conditions:
        dataset = TactileEpisodes(ROOT/config['manifest'], future=False)
        bank = DonorBank(dataset, args.tasks, args.donor_seed, args.donors_per_task)
    for task in args.tasks:
        if all((task, args.start_seed+i, c) in completed for i in range(args.episodes) for c in args.conditions):
            continue
        env = TactileEnvironment(task=task, controller=metadata['controller'])
        try:
            for episode in range(args.episodes):
                seed = args.start_seed+episode
                # Counterbalance ordering; persistent simulator state may affect resets.
                order = np.random.default_rng(seed).permutation(args.conditions).tolist()
                for condition in order:
                    if (task, seed, condition) in completed:
                        continue
                    row = run_episode(env, model, pre, post, bank, task=task, seed=seed,
                        condition=condition, schedule=args.schedule, device=config['device'], output=output,
                        record_video=episode < args.video_episodes, smoke_steps=args.smoke_steps)
                    with records.open('a') as stream:
                        stream.write(json.dumps(row)+'\n')
                        stream.flush()
                        os.fsync(stream.fileno())
                    rows.append(row)
                    completed.add((task, seed, condition))
                    atomic_json(output/'summary.json', summarize(rows, args.tasks, args.episodes, args.conditions))
                    print(json.dumps({k: row[k] for k in ('task', 'seed', 'condition', 'success', 'steps', 'smoke')}), flush=True)
        finally:
            env.close()
    atomic_json(output/'summary.json', summarize(rows, args.tasks, args.episodes, args.conditions))


if __name__ == '__main__':
    main()
