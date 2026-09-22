#!/usr/bin/env python
"""Run and publish one HDMI controller probe, optionally with legacy EMA."""
import hashlib
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import cv2
from huggingface_hub import HfApi
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REPO = 'sungkyunner/univtac-smolvla-baseline-full50-two-step-eval'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--controller', choices=('hold', 'linear', 'linear_position'), default='hold')
    parser.add_argument('--no-ema', action='store_true', help='Pass raw denormalized model endpoints to the controller unchanged')
    args = parser.parse_args()
    if args.no_ema and args.controller == 'hold':
        parser.error('--no-ema probe requires an interpolated controller for its action trace')
    NAME = 'baseline_ema_arm_alpha0p5_insert_HDMI_seed1000000'
    if args.no_ema:
        NAME = 'baseline_no_ema_insert_HDMI_seed1000000'
    if args.controller != 'hold':
        NAME += '_'+args.controller+'_controller'
    RUN = ROOT/'runs/benchmark'/NAME
    REMOTE = 'experiments/'+NAME
    env = dict(os.environ, UNIVTAC_EVAL_SAVE_ROOT=str(ROOT/'runs/simulator_workspaces'/NAME))
    command = ['bash', 'scripts/benchmark_refinement.sh', '--baseline', '--tasks', 'insert_HDMI',
               '--episodes', '1', '--start-seed', '1000000', '--video-episodes', '1',
               '--controller', args.controller, '--resume', '--output', str(RUN)]
    if not args.no_ema:
        command.extend(['--arm-ema-alpha', '0.5'])
    print(json.dumps({'phase': 'evaluation', 'command': command}), flush=True)
    subprocess.run(command, cwd=ROOT, env=env, check=True)
    rows = [json.loads(line) for line in (RUN/'episodes.jsonl').read_text().splitlines()]
    if len(rows) != 1 or rows[0]['task'] != 'insert_HDMI' or rows[0]['seed'] != 1000000:
        raise ValueError('Expected exactly one HDMI episode with seed 1000000')
    row = rows[0]
    video = RUN/row['video']
    capture = cv2.VideoCapture(str(video))
    frames = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames += 1
    fps = capture.get(cv2.CAP_PROP_FPS)
    capture.release()
    if frames != row['steps']+1 or fps != 60:
        raise ValueError(f'Video does not match rollout: {frames=} {fps=} steps={row["steps"]}')
    trace = np.load(RUN/row['action_trace'])
    raw, filtered = trace['raw_actions'], trace['filtered_actions']
    if args.no_ema:
        np.testing.assert_array_equal(raw, filtered)
        metadata = json.loads((RUN/'metadata.json').read_text())
        if 'action_smoothing' in metadata:
            raise ValueError('Unexpected endpoint smoothing in no-EMA experiment')
    else:
        previous = trace['initial_state'][:7].copy()
        for action, target in zip(raw, filtered):
            previous = 0.5*action[:7]+0.5*previous
            np.testing.assert_allclose(target[:7], previous, rtol=1e-6, atol=1e-7)
    np.testing.assert_array_equal(raw[:, 7:], filtered[:, 7:])
    if args.controller != 'hold':
        dt = row['simulated_seconds']/row['physics_steps']
        np.testing.assert_allclose(dt, 1/120)
        expected_pos, expected_vel = [], []
        start = trace['initial_state'][:8].copy()
        for endpoint in filtered[:, :8]:
            velocity = (endpoint-start)/(2*dt) if args.controller == 'linear' else np.zeros_like(endpoint)
            expected_pos.extend([(start+endpoint)/2, endpoint.copy()])
            expected_vel.extend([velocity.copy(), velocity.copy()])
            start = endpoint.copy()
        count = row['physics_steps']
        if count not in (2*row['steps']-1, 2*row['steps']):
            raise ValueError('Unexpected physical step count')
        np.testing.assert_allclose(trace['substep_positions'], np.asarray(expected_pos)[:count], rtol=1e-5, atol=1e-7)
        np.testing.assert_allclose(trace['substep_velocities'], np.asarray(expected_vel)[:count], rtol=1e-5, atol=1e-7)
    metrics = {'video_frames': frames, 'video_fps': fps, 'success': row['success'],
               'steps': row['steps'], 'physics_steps': row['physics_steps'], 'controller': args.controller,
               'ema_alpha': None if args.no_ema else 0.5}
    for label, array in [('raw_targets', raw), ('filtered_targets', filtered), ('observed_joints', trace['observed_states'])]:
        metrics[label] = {'arm_mean_abs_first_difference': float(np.abs(np.diff(array[:, :7], axis=0)).mean()),
                          'arm_mean_abs_second_difference': float(np.abs(np.diff(array[:, :7], n=2, axis=0)).mean())}
    error = trace['observed_states'][:, :7]-filtered[:, :7]
    delta = np.diff(np.vstack([trace['initial_state'][:7], filtered[:, :7]]), axis=0)
    metrics['arm_tracking'] = {'mean_absolute_error_rad': float(np.abs(error).mean()),
        'maximum_absolute_error_rad': float(np.abs(error).max()),
        'error_vs_target_delta_slope': float((error*delta).sum()/max(float((delta**2).sum()), 1e-20))}
    if 'substep_observed_positions' in trace and trace['substep_observed_positions'].size:
        sub_error = trace['substep_observed_positions']-trace['substep_positions']
        metrics['physics_step_tracking'] = {
            'arm_mean_absolute_error_rad': float(np.abs(sub_error[:, :7]).mean()),
            'arm_maximum_absolute_error_rad': float(np.abs(sub_error[:, :7]).max()),
            'gripper_mean_absolute_error_m': float(np.abs(sub_error[:, 7]).mean()),
            'gripper_maximum_absolute_error_m': float(np.abs(sub_error[:, 7]).max())}
    reference = ROOT/'runs/benchmark/baseline_full50_two_step_100eps_seed1000000/episodes.jsonl'
    for line in reference.read_text().splitlines():
        original = json.loads(line)
        if original['task'] == row['task'] and original['seed'] == row['seed']:
            metrics['unfiltered_reference'] = {'success': original['success'], 'steps': original['steps'],
                'initial_state_signature_matches': original['initial_info']['initial_state_id'] == row['initial_info']['initial_state_id']}
            break
    (RUN/'diagnostics.json').write_text(json.dumps(metrics, indent=2)+'\n')
    schedule = ('Each target is held for two physics steps.' if args.controller == 'hold' else
                'Each target is reached via midpoint then endpoint at 120 Hz; arm and shared gripper velocity targets are explicitly set on both substeps. This is linear reconstruction, not an exact replay of demonstration planner commands.')
    if args.controller == 'linear_position':
        schedule += ' Velocity targets are explicitly ZERO, to test removing velocity-induced endpoint overshoot while retaining position interpolation.'
    filtering = ('No EMA or endpoint averaging is applied. The denormalized model endpoints are passed through unchanged, verified by exact array equality.' if args.no_ema else
                 'After action denormalization, the first seven joint targets use EMA with alpha=0.5. The filter starts from observed initial arm positions and retains state across chunk boundaries. Both gripper outputs remain unfiltered by EMA.')
    title = 'no EMA' if args.no_ema else 'EMA alpha=0.5'
    (RUN/'README.md').write_text(f'''# SmolVLA HDMI single-episode probe ({title}; {args.controller} controller)

One insert_HDMI episode, seed 1000000, using the same frozen final SmolVLA baseline as the main benchmark.
This experiment does not modify model weights, training, or the main evaluation results.

{filtering}
All 50 actions execute before replanning. {schedule}
Video is recorded at 60 fps with head and wrist views. No tactile input is passed to the baseline.

Success: {row['success']}. Actions executed: {row['steps']}. This is one diagnostic episode, not an estimate of task success rate.
The matching unfiltered video is in the repository's main videos/insert_HDMI/ directory.
Identical seeds do not guarantee identical GPU simulator initial states; see diagnostics.json for the observed-state signature comparison.

action_traces/ contains raw predicted targets, EMA targets, observed post-action joint positions and the initial joint state.
metadata.json records the checkpoint hash and execution settings. diagnostics.json reports command smoothness and observed tracking error. When available, substep_observed_positions records actual joints after each physics step.
''')
    provenance = RUN/'provenance'
    provenance.mkdir(exist_ok=True)
    for source in ['src/tactile_refinement/benchmark.py', 'src/tactile_refinement/smoothing.py',
                   'src/tactile_refinement/execution.py', 'src/smolvla_baseline/sim_worker.py',
                   'scripts/run_ema_hdmi_probe.py']:
        shutil.copy2(ROOT/source, provenance/Path(source).name)
    api = HfApi()
    print(json.dumps({'phase': 'upload', 'metrics': metrics}), flush=True)
    commit = api.upload_folder(repo_id=REPO, repo_type='dataset', folder_path=str(RUN),
        path_in_repo=REMOTE, commit_message=f'Add separate HDMI experiment: {title}, {args.controller} controller')
    video_remote = REMOTE+'/'+row['video']
    entry = api.get_paths_info(REPO, [video_remote], repo_type='dataset', revision=commit.oid)[0]
    digest = hashlib.sha256(video.read_bytes()).hexdigest()
    if entry.size != video.stat().st_size or entry.lfs is None or entry.lfs.sha256 != digest:
        raise RuntimeError('Remote video size or SHA-256 mismatch')
    result = {'phase': 'complete', 'commit_url': commit.commit_url,
              'video_url': f'https://huggingface.co/datasets/{REPO}/blob/main/{video_remote}',
              'video_sha256': digest, 'metrics': metrics}
    report = 'ema_hdmi_probe_upload.json' if args.controller == 'hold' else 'ema_hdmi_'+args.controller+'_probe_upload.json'
    if args.no_ema:
        report = 'no_ema_hdmi_'+args.controller+'_probe_upload.json'
    (ROOT/'logs'/report).write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
