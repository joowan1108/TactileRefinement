#!/usr/bin/env python
"""Detached-friendly one-episode official-schedule rollout, validation and upload."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

import cv2
from huggingface_hub import HfApi
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
NAME = 'baseline_official_schedule_insert_HDMI_seed1000000'
RUN = ROOT/'runs/benchmark'/NAME
REPO = 'sungkyunner/univtac-smolvla-baseline-full50-two-step-eval'
REMOTE = 'experiments/'+NAME


def main():
    lock = (ROOT/'logs/official_hdmi_probe.lock').open('a')
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    env = dict(os.environ, PYTHONPATH=str(ROOT/'src'), OMNI_KIT_ACCEPT_EULA='YES',
               OMNI_KIT_ALLOW_ROOT='1', TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='4')
    command = [str(ROOT/'.venv-train/bin/python'), '-u', '-m', 'smolvla_baseline.official_benchmark',
               '--output', str(RUN), '--seed', '1000000']
    if not (RUN/'summary.json').exists():
        print(json.dumps({'phase': 'evaluation', 'command': command}), flush=True)
        subprocess.run(command, cwd=ROOT, env=env, check=True)
    summary = json.loads((RUN/'summary.json').read_text())
    row = summary['result']
    assert row['task'] == 'insert_HDMI' and row['seed'] == 1000000
    assert row['model_calls'] == row['steps'] == row['physics_steps']
    np.testing.assert_allclose(row['simulated_seconds'], row['steps']/120)
    assert summary['ema'] is False and summary['interpolation'] is False
    trace = np.load(RUN/row['action_trace'])
    chunks, actions = trace['predicted_chunks'], trace['executed_actions']
    assert chunks.shape == (row['steps'], 50, 9) and actions.shape == (row['steps'], 9)
    assert np.all(trace['physics_steps'] == 1)
    max_aggregation_error = 0.0
    # Independent offline reconstruction, checking actual logged chunks/actions.
    for t, action in enumerate(actions):
        starts = np.arange(max(0, t-49), t+1)
        candidates = chunks[starts, t-starts]
        weights = np.exp(-summary['aggregation_k']*np.arange(len(starts), dtype=np.float64))
        expected = (candidates*weights[:, None]).sum(axis=0)/weights.sum()
        np.testing.assert_allclose(action, expected, rtol=2e-6, atol=1e-6)
        max_aggregation_error = max(max_aggregation_error, float(np.abs(action-expected).max()))
    video = RUN/row['video']; capture = cv2.VideoCapture(str(video))
    frames = 0
    while True:
        ok, _ = capture.read()
        if not ok:
            break
        frames += 1
    fps = capture.get(cv2.CAP_PROP_FPS); capture.release()
    if frames != row['steps']+1 or fps != 120:
        raise ValueError(f'Invalid video timing: {frames=} {fps=}')
    metrics = {'success': row['success'], 'steps': row['steps'], 'model_calls': row['model_calls'],
        'physics_steps': row['physics_steps'], 'simulated_seconds': row['simulated_seconds'],
        'video_frames': frames, 'video_fps': fps, 'max_aggregation_reconstruction_error': max_aggregation_error}
    for label, a in [('new_chunk_first_targets', chunks[:, 0]), ('aggregated_targets', actions), ('observed_joints', trace['observed_states'])]:
        metrics[label] = {'arm_mean_abs_first_difference_rad': float(np.abs(np.diff(a[:, :7], axis=0)).mean()),
            'arm_mean_abs_second_difference_rad': float(np.abs(np.diff(a[:, :7], n=2, axis=0)).mean())}
    metrics['arm_tracking_mae_rad'] = float(np.abs(trace['observed_states'][:, :7]-actions[:, :7]).mean())
    reference = ROOT/'runs/benchmark/baseline_full50_two_step_100eps_seed1000000/episodes.jsonl'
    for line in reference.read_text().splitlines():
        old = json.loads(line)
        if old['task'] == row['task'] and old['seed'] == row['seed']:
            metrics['full50_reference'] = {'success': old['success'], 'steps': old['steps'],
                'initial_state_signature_matches': old['initial_info']['initial_state_id'] == row['initial_info']['initial_state_id']}
            break
    (RUN/'diagnostics.json').write_text(json.dumps(metrics, indent=2)+'\n')
    (RUN/'README.md').write_text(f'''# SmolVLA HDMI: official UniVTAC execution schedule

One episode, seed 1000000; success: {row['success']}. This is a diagnostic, not a task success-rate estimate.

The same final tactile-free SmolVLA checkpoint and its original preprocessing are used.
UniVTAC commit `{summary['upstream_commit']}` supplies the unmodified temporal-buffer reset and aggregation methods.
Every control step predicts 50 actions from the newest observation, then aggregates overlapping predictions for the current timestep using exp(-0.01*i), oldest first.
The resulting qpos target is executed through the native task.take_action API once: one 1/120-second physics step. There is no added EMA, interpolation, two-step hold, or velocity override.
Nine predicted joint values map to seven arm targets plus the first gripper target, following the documented eight-value qpos API.
The HDMI limit is 600 actions (five simulated seconds), not the ten-second limit of our earlier two-step schedule.

This matches the upstream execution schedule, not a reproduction of published policy scores. Training remains based on recorded frames separated by two physics steps; this official schedule directly follows upstream indexing without rescaling that horizon. The checkpoint, input modalities and bfloat16 inference match our prior probes. Exact initial states may differ between same-seed GPU resets.

The video has 120 fps to preserve simulation time. Raw predicted chunks, aggregated commands and observed joints are in action_traces/. diagnostics.json independently reconstructs aggregation and verifies one model call and one physics step per executed action. Do not compare unscaled finite differences directly across 60 Hz and 120 Hz runs.

Official sources: [deployment guide](https://github.com/univtac/UniVTAC/blob/{summary['upstream_commit']}/docs/Deploy.md), [SmolVLA policy](https://github.com/univtac/UniVTAC/blob/{summary['upstream_commit']}/policy/smolvla/deploy_policy.py), [configuration](https://github.com/univtac/UniVTAC/blob/{summary['upstream_commit']}/policy/smolvla/deploy.yml).
''')
    shutil.copy2(__file__, RUN/'provenance/run_official_hdmi_probe.py')
    print(json.dumps({'phase': 'upload', 'metrics': metrics}), flush=True)
    api = HfApi()
    commit = api.upload_folder(repo_id=REPO, repo_type='dataset', folder_path=str(RUN), path_in_repo=REMOTE,
        commit_message='Add isolated HDMI probe using official UniVTAC temporal aggregation and single-step execution')
    remote_video = REMOTE+'/'+row['video']
    entry = api.get_paths_info(REPO, [remote_video], repo_type='dataset', revision=commit.oid)[0]
    digest = hashlib.sha256(video.read_bytes()).hexdigest()
    if entry.size != video.stat().st_size or entry.lfs is None or entry.lfs.sha256 != digest:
        raise RuntimeError('Uploaded video hash or size mismatch')
    result = {'phase': 'complete', 'commit_url': commit.commit_url,
        'video_url': f'https://huggingface.co/datasets/{REPO}/blob/main/{remote_video}',
        'video_sha256': digest, 'metrics': metrics}
    (ROOT/'logs/official_hdmi_probe_upload.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
