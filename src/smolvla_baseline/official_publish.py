"""Publish completed official-schedule episodes without exposing in-progress videos."""
import csv
import hashlib
import json

import cv2
from huggingface_hub import HfApi


def publish_progress(output, repo, summary, rows):
    artifacts = []
    for row in rows:
        if 'video' not in row:
            continue
        video = output/row['video']
        cap = cv2.VideoCapture(str(video))
        frames, fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        if frames != row['steps']+1 or fps != 120:
            raise ValueError(f'Invalid completed video: {video}, {frames=} {fps=}')
        artifacts.extend([row['video'], row['action_trace']])
    expected = len(summary['tasks'])*summary['episodes_per_task']
    if summary['complete']:
        for task, result in summary['tasks'].items():
            if result['episodes'] != summary['episodes_per_task']:
                raise ValueError('Incomplete task in final summary')
            videos = [r for r in rows if r['task'] == task and 'video' in r]
            if len(videos) != summary['video_episodes']:
                raise ValueError(f'Missing requested videos for {task}')
    with (output/'results.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['task', 'successes', 'episodes', 'success_rate'])
        for task, result in summary['tasks'].items():
            writer.writerow([task, result['successes'], result['episodes'], result['success_rate']])
    table = '\n'.join(f'| {task} | {r["successes"]}/{r["episodes"]} | '+
                      (f'{100*r["success_rate"]:.1f}%' if r['episodes'] else 'Not started')+' |'
                      for task, r in summary['tasks'].items())
    state = 'Complete' if summary['complete'] else 'In progress — partial results'
    (output/'README.md').write_text(f'''---
pretty_name: UniVTAC SmolVLA Baseline — Official Execution Schedule
tags:
- robotics
- univtac
- smolvla
---

# SmolVLA baseline with the official UniVTAC execution schedule

{state}: {len(rows)}/{expected} episodes. Target: {summary['episodes_per_task']} per task.

| Task | Successes / completed episodes | Success rate |
|---|---:|---:|
{table}

Results are uploaded as episodes finish; partial totals are not final benchmark scores.
All evaluation seeds start at {summary['seed']}. The first {summary['video_episodes']} episodes of each task have videos and action traces, regardless of success.

The frozen baseline checkpoint is `{summary['baseline_sha256']}` (SHA-256).
Policy inputs are head and wrist RGB, joint state, and instruction; no tactile input.
UniVTAC commit `{summary['upstream_commit']}` provides the unmodified temporal aggregation code.
Each step predicts a new 50-action chunk from the latest observation and aggregates predictions for that timestep with exp(-0.01*i), oldest first.
The native qpos API executes one physics step per action at 120 Hz; no added EMA, interpolation, velocity override or two-step hold is used.
The video frame rate is 120 fps to preserve simulated time. Every completed episode is checked for one inference/action/physics step and independently reconstructed aggregation.

This is the official execution schedule applied to our checkpoint and its original preprocessing, not a reproduction of published policy scores. The model was trained on recordings separated by two physics steps; the official schedule's indexing is used directly without horizon rescaling.
These results are separate from the earlier full-50/two-step benchmark. That run's success totals are not merged here.
''')
    api = HfApi()
    api.create_repo(repo, repo_type='dataset', private=True, exist_ok=True)
    commit = api.upload_folder(repo_id=repo, repo_type='dataset', folder_path=str(output),
        allow_patterns=['README.md', 'metadata.json', 'summary.json', 'episodes.jsonl', 'results.csv', 'provenance/**', *artifacts],
        commit_message=f'Official schedule: {len(rows)}/{expected} episodes, {summary["successes"]} successes')
    # Verify every new video against the remote LFS object, retaining receipts.
    receipt_path = output/'upload_receipt.json'
    previous = json.loads(receipt_path.read_text()) if receipt_path.exists() else {}
    verified = previous.get('verified_videos', {})
    pending = [r['video'] for r in rows if 'video' in r and r['video'] not in verified]
    if pending:
        entries = api.get_paths_info(repo, pending, repo_type='dataset', revision=commit.oid)
        by_path = {entry.path: entry for entry in entries}
        for path in pending:
            local = output/path; digest = hashlib.sha256(local.read_bytes()).hexdigest()
            entry = by_path[path]
            if entry.size != local.stat().st_size or entry.lfs is None or entry.lfs.sha256 != digest:
                raise RuntimeError(f'Remote video verification failed: {path}')
            verified[path] = digest
    receipt = {'episodes': len(rows), 'repo': repo, 'commit_url': commit.commit_url, 'verified_videos': verified}
    receipt_path.write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps({'phase': 'uploaded', **receipt}), flush=True)
