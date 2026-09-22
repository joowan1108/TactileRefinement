"""Closed-loop evaluation using UniVTAC's official task execution and success predicates."""
import argparse
import json
from pathlib import Path
import time

import cv2
import numpy as np
import torch

from .data import ROOT, IMAGE_KEYS, prepare_batch
from .instructions import DEFAULT_TASKS
from .model import build_model, load_checkpoint
from .sim_bridge import UniVTACEnvironment


def model_input(obs):
    result = {'observation.state': torch.from_numpy(obs['state']), 'task': obs['instruction']}
    for camera, key in zip(('head', 'wrist'), IMAGE_KEYS):
        rgb = cv2.resize(obs[camera], (256, 256), interpolation=cv2.INTER_LINEAR)
        result[key] = torch.from_numpy(rgb.transpose(2, 0, 1).copy())
    return result


def video_frame(obs, *, task, seed, step, status=None):
    """Build a side-by-side RGB rollout frame without exposing tactile observations."""
    views = []
    for name in ('head', 'wrist'):
        frame = obs[name]
        if frame.shape[0] != 360 or frame.shape[1] != 640:
            frame = cv2.resize(frame, (640, 360), interpolation=cv2.INTER_AREA)
        views.append(frame)
    frame = np.concatenate(views, axis=1)
    label = f'{task}  seed={seed}  action={step}'
    if status is not None:
        label += f'  {status}'
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 34), (0, 0, 0), -1)
    cv2.putText(frame, label, (10, 24), cv2.FONT_HERSHEY_SIMPLEX,
                0.65, (255, 255, 255), 1, cv2.LINE_AA)
    return frame


def open_video(path, frame, fps):
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), fps,
                             (frame.shape[1], frame.shape[0]))
    if not writer.isOpened():
        raise RuntimeError(f'Could not open rollout video writer: {path}')
    writer.write(frame)
    return writer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--tasks', nargs='+', choices=DEFAULT_TASKS, default=list(DEFAULT_TASKS))
    parser.add_argument('--episodes', type=int, default=50)
    parser.add_argument('--start-seed', type=int, default=1000000)
    parser.add_argument('--video-episodes', type=int, default=3,
                        help='Record the first N episodes of each task (0 disables video).')
    parser.add_argument('--video-fps', type=float, default=30.0)
    parser.add_argument('--resume', action='store_true',
                        help='Resume a partially completed output directory by task and seed.')
    parser.add_argument('--smoke', action='store_true', help='Reset and execute one action (stationary unless --checkpoint supplied); no policy quality score')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if not args.smoke and args.checkpoint is None:
        parser.error('--checkpoint is required except with --smoke')
    torch.set_num_threads(4)
    output = args.output or ROOT / 'runs/benchmark' / ('smoke' if args.smoke else args.checkpoint.stem)
    output.mkdir(parents=True, exist_ok=True)
    episodes_path = output / 'episodes.jsonl'
    if episodes_path.exists() and not args.resume:
        raise FileExistsError(f'Benchmark results already exist in {output}; choose a fresh --output directory')
    model = pre = post = None
    if args.checkpoint is not None:
        checkpoint = load_checkpoint(args.checkpoint)
        model, pre, post = build_model(checkpoint['recipe'], checkpoint['manifest'], checkpoint)
        model.eval()
    rows = []
    if episodes_path.exists():
        rows = [json.loads(line) for line in episodes_path.read_text().splitlines() if line.strip()]
        identities = [(row['task'], row['seed']) for row in rows]
        if len(identities) != len(set(identities)):
            raise ValueError(f'Duplicate task/seed records in {episodes_path}')
        print(json.dumps({'resume_records': len(rows), 'output': str(output)}), flush=True)
    for task in args.tasks:
        env = UniVTACEnvironment(task=task)
        try:
            for episode_index, seed in enumerate(range(args.start_seed, args.start_seed + (1 if args.smoke else args.episodes))):
                if any(row['task'] == task and row['seed'] == seed for row in rows):
                    continue
                started = time.monotonic()
                obs, initial_info = env.reset(seed=seed)
                video_path = None
                video = None
                if not args.smoke and episode_index < args.video_episodes:
                    video_path = output / 'videos' / task / f'episode_{episode_index:03d}_seed_{seed}.mp4'
                    initial_frame = video_frame(obs, task=task, seed=seed, step=0)
                    video = open_video(video_path, initial_frame, args.video_fps)
                torch.manual_seed(seed)
                if model is not None:
                    model.reset()
                done = truncated = False
                step = 0; info = {}
                try:
                    while not (done or truncated):
                        if model is None:
                            action = obs['state']
                        else:
                            with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                                action = post(model.select_action(prepare_batch(model_input(obs), pre))).squeeze(0).cpu().numpy()
                        obs, reward, done, truncated, info = env.step(action)
                        step += 1
                        if video is not None:
                            status = 'success' if done and info.get('success') else ('stopped' if done else None)
                            video.write(video_frame(obs, task=task, seed=seed, step=step, status=status))
                        if args.smoke:
                            break
                finally:
                    if video is not None:
                        video.release()
                row = {'task': task, 'seed': seed, 'steps': step, 'seconds': time.monotonic()-started, 'initial_info': initial_info, 'info': info, 'smoke_only': args.smoke}
                if video_path is not None:
                    row['video'] = str(video_path.relative_to(output))
                if not args.smoke:
                    row['success'] = bool(info['success'])
                else:
                    row['smoke_passed'] = bool(info['execution_success']) and info['tactile_health']['finite']
                rows.append(row)
                with episodes_path.open('a') as f:
                    f.write(json.dumps(row)+'\n')
                print(json.dumps(row), flush=True)
        finally:
            env.close()
    summary = {'checkpoint': str(args.checkpoint), 'smoke_only': args.smoke,
               'start_seed': args.start_seed, 'episodes_per_task': args.episodes,
               'recorded_episodes_per_task': 0 if args.smoke else min(args.video_episodes, args.episodes),
               'video_fps': args.video_fps, 'episodes': len(rows), 'tasks': {}}
    for task in args.tasks:
        values = [r for r in rows if r['task'] == task]
        summary['tasks'][task] = {'episodes': len(values)}
        if not args.smoke:
            summary['tasks'][task]['success_rate'] = float(np.mean([r['success'] for r in values]))
        else:
            summary['tasks'][task]['all_smoke_passed'] = all(r['smoke_passed'] for r in values)
    (output / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
