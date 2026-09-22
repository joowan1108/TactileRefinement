"""Run one Stage-2 final-checkpoint attention rollout for every UniVTAC task.

Tasks run sequentially to bound GPU memory. Finished episodes are never replayed;
their uploads may be retried safely after a network error.
"""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
import torch
from huggingface_hub import CommitOperationAdd, HfApi

from record_insert_tube_attention import REPO, upload
from tactile_refinement.ablation import atomic_json
from smolvla_baseline.instructions import DEFAULT_TASKS

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT/'runs/tactile_refinement_trex_fullgrid_n0/expert/final.pt'
OUTPUT = ROOT/'runs/diagnostics/stage2_final_attention_100000_seed2000000'
SEED = 2000000


def status(phase, **values):
    row = dict(phase=phase, updated_at=time.time(), **values)
    atomic_json(OUTPUT/'status.json', row)
    print(json.dumps(row), flush=True)


def retry(operation):
    for attempt in range(3):
        try:
            return operation()
        except Exception:
            if attempt == 2:
                raise
            time.sleep(10)


def read_checkpoint_step():
    checkpoint = torch.load(CHECKPOINT, map_location='cpu', weights_only=False)
    if checkpoint.get('stage') != 'expert':
        raise ValueError(f'Expected Stage-2 expert checkpoint, got stage={checkpoint.get("stage")!r}')
    step = int(checkpoint['step'])
    if step != 100000:
        raise ValueError(f'Expected final Stage-2 update 100000, got {step}')
    return step


def comparison(step, errors):
    records = []
    for task in DEFAULT_TASKS:
        folder = OUTPUT/task
        if not (folder/'episode.json').exists() or not (folder/'attention_metrics.json').exists():
            continue
        episode = json.loads((folder/'episode.json').read_text())
        metrics = json.loads((folder/'attention_metrics.json').read_text())
        metadata = json.loads((folder/'metadata.json').read_text())
        if (episode['seed'] != SEED or metadata['checkpoint']['step'] != step
                or metadata['schedule'] != 'matched60' or episode['smoke']):
            raise ValueError(f'Incompatible {task} recording in cross-task comparison')
        with np.load(folder/'attention_head_mean.npz') as saved:
            attention = saved['attention'][:, -1]
        unit = attention/np.maximum(np.linalg.norm(attention, axis=-1, keepdims=True), 1e-12)
        cosine = np.einsum('tfqd,tfrd->tfqr', unit, unit)
        a, b = np.triu_indices(16, 1)
        similarity = cosine[:, :, a, b].mean(axis=(0, 2))
        records.append(dict(
            task=task, success=episode['success'], steps=episode['steps'],
            simulated_seconds=episode['simulated_seconds'],
            entropy=[metrics['fingers'][f]['query_head_mean_normalized_entropy_mean']
                     for f in ('left', 'right')],
            query_cosine=similarity.tolist(),
            mean_attention_peak_times_uniform=[
                metrics['fingers'][f]['mean_attention_peak_times_uniform']
                for f in ('left', 'right')]))
    result = dict(checkpoint=step, seed=SEED, schedule='matched60',
                  requested_tasks=list(DEFAULT_TASKS), completed_tasks=len(records),
                  records=records, errors=errors)
    atomic_json(OUTPUT/'comparison.json', result)
    report = [
        '# Stage 2 final-checkpoint resampler attention across UniVTAC tasks', '',
        f'One episode per task, checkpoint {step:,}, seed {SEED}, matched60 schedule. '
        'Each task is uploaded immediately after it finishes.', '',
        '| Task | Episode success | Decisions | Entropy L / R | Query similarity L / R | Visualization |',
        '|---|---|---:|---|---|---|']
    for row in records:
        report.append(
            f"| {row['task']} | {row['success']} | {row['steps']} | "
            f"{row['entropy'][0]:.3f} / {row['entropy'][1]:.3f} | "
            f"{row['query_cosine'][0]:.5f} / {row['query_cosine'][1]:.5f} | "
            f"[View](../{row['task']}/seed_{SEED}/REPORT.md) |")
    report += ['',
        'Entropy is normalized: 1 means uniform spatial attention; lower values mean more concentration. '
        'Query similarity is mean pairwise cosine similarity between the 16 queries after averaging heads. '
        'Attention is over predicted future-tactile token positions and is not causal attribution. '
        'One episode per task is not a task success-rate estimate.', '']
    if errors:
        report += ['## Incomplete tasks or uploads', '',
                   *[f'- {task}: {error}' for task, error in errors.items()], '']
    (OUTPUT/'COMPARISON.md').write_text('\n'.join(report))
    prefix = f'checkpoint_{step}/attention/comparison_seed{SEED}'
    operations = [CommitOperationAdd(path_in_repo=f'{prefix}/{name}',
                  path_or_fileobj=str(OUTPUT/name))
                  for name in ('COMPARISON.md', 'comparison.json')]
    return retry(lambda: HfApi().create_commit(
        repo_id=REPO, repo_type='dataset', operations=operations,
        commit_message=f'Update checkpoint {step} one-episode attention comparison')).commit_url


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    lock = (OUTPUT/'runner.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    step = read_checkpoint_step()
    env = dict(os.environ, PYTHONPATH=str(ROOT/'src'), OMNI_KIT_ACCEPT_EULA='YES',
        OMNI_KIT_ALLOW_ROOT='1', TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='4',
        HF_HUB_DISABLE_PROGRESS_BARS='1')
    errors, completed = {}, []
    for task in DEFAULT_TASKS:
        folder = OUTPUT/task
        try:
            if not folder.exists():
                status('evaluating', checkpoint=step, task=task,
                       completed=completed, pid=os.getpid())
                with (OUTPUT/f'{task}.log').open('a') as log:
                    result = subprocess.run([
                        str(ROOT/'.venv-train/bin/python'), '-u',
                        str(ROOT/'scripts/record_insert_tube_attention.py'),
                        '--checkpoint', str(CHECKPOINT), '--task', task,
                        '--seed', str(SEED), '--output', str(folder), '--upload'],
                        cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
                if result.returncode:
                    raise RuntimeError(f'evaluator exited with code {result.returncode}')
            child = json.loads((folder/'status.json').read_text())
            if child['phase'] != 'complete':
                if (folder/'REPORT.md').exists() and (folder/'episode.json').exists():
                    metadata = json.loads((folder/'metadata.json').read_text())
                    uploaded = retry(lambda: upload(folder, metadata))
                    episode = json.loads((folder/'episode.json').read_text())
                    atomic_json(folder/'status.json', dict(
                        phase='complete', success=episode['success'],
                        steps=episode['steps'], **uploaded))
                else:
                    raise RuntimeError(f'Existing attempt incomplete; not replayed: {child}')
            completed.append(task)
        except Exception as exc:
            errors[task] = repr(exc)
        atomic_json(OUTPUT/'errors.json', errors)
        try:
            comparison(step, errors)
        except Exception as exc:
            print(json.dumps(dict(phase='comparison_upload_failed', error=repr(exc))), flush=True)
    try:
        commit = comparison(step, errors)
        status('complete' if not errors else 'finished_with_errors', checkpoint=step,
               completed=completed, errors=errors, comparison_commit=commit, pid=os.getpid())
    except Exception as exc:
        status('comparison_upload_failed', checkpoint=step, completed=completed,
               errors=errors, error=repr(exc), pid=os.getpid())


if __name__ == '__main__':
    main()
