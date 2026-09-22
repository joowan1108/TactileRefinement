"""One episode per remaining task, with incremental analysis and HF uploads.

Never reruns an attempted episode. Completed rollouts may have uploads retried.
The existing Insert Tube recording is included in the comparison without replay.
"""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
from huggingface_hub import CommitOperationAdd, HfApi

from record_insert_tube_attention import REPO, upload
from tactile_refinement.ablation import atomic_json
from smolvla_baseline.instructions import DEFAULT_TASKS

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT/'runs/diagnostics/other_task_attention_66666_seed2000000'
TUBE = ROOT/'runs/diagnostics/insert_tube_attention_66666_seed2000000'
TASKS = [task for task in DEFAULT_TASKS if task != 'insert_tube']
PREFIX = 'checkpoint_66666/attention/comparison_seed2000000'


def status(phase, **values):
    atomic_json(OUTPUT/'status.json', dict(phase=phase, updated_at=time.time(), **values))
    print(json.dumps(dict(phase=phase, **values)), flush=True)


def retry_upload(operation):
    for attempt in range(3):
        try:
            return operation()
        except Exception:
            if attempt == 2:
                raise
            time.sleep(10)


def comparison():
    records = []
    for task in DEFAULT_TASKS:
        folder = TUBE if task == 'insert_tube' else OUTPUT/task
        if not (folder/'episode.json').exists() or not (folder/'attention_metrics.json').exists():
            continue
        episode = json.loads((folder/'episode.json').read_text())
        metrics = json.loads((folder/'attention_metrics.json').read_text())
        metadata = json.loads((folder/'metadata.json').read_text())
        if (episode['seed'] != 2000000 or metadata['checkpoint']['step'] != 66666
                or metadata['schedule'] != 'matched60' or episode['smoke']):
            raise ValueError('Incompatible recording in cross-task comparison')
        # Compute the same statistic for the original Tube recording too.
        with np.load(folder/'attention_head_mean.npz') as saved:
            attention = saved['attention'][:, -1]
        unit = attention/np.maximum(np.linalg.norm(attention, axis=-1, keepdims=True), 1e-12)
        cosine = np.einsum('tfqd,tfrd->tfqr', unit, unit)
        a, b = np.triu_indices(16, 1)
        similarity = cosine[:, :, a, b].mean(axis=(0, 2))
        records.append(dict(task=task, success=episode['success'], steps=episode['steps'],
            simulated_seconds=episode['simulated_seconds'],
            entropy=[metrics['fingers'][f]['query_head_mean_normalized_entropy_mean'] for f in ('left', 'right')],
            query_cosine=similarity.tolist(),
            mean_attention_peak_times_uniform=[metrics['fingers'][f]['mean_attention_peak_times_uniform'] for f in ('left', 'right')],
            source=str(folder), reused_existing_episode=task == 'insert_tube'))
    errors = json.loads((OUTPUT/'errors.json').read_text()) if (OUTPUT/'errors.json').exists() else {}
    result = dict(checkpoint=66666, seed=2000000, schedule='matched60',
                  new_tasks=TASKS, completed_tasks=len(records), records=records, errors=errors)
    atomic_json(OUTPUT/'comparison.json', result)
    report = ['# Resampler attention across UniVTAC tasks', '',
        'One episode per task, checkpoint 66,666, seed 2000000, matched60 schedule. '
        'The previously recorded Insert Tube episode is reused. Other tasks are added as they finish.', '',
        '| Task | Episode success | Decisions | Entropy L / R | Query similarity L / R | Visualization |',
        '|---|---|---:|---|---|---|']
    for r in records:
        link = f"../{r['task']}/seed_2000000/REPORT.md"
        report.append(f"| {r['task']} | {r['success']} | {r['steps']} | "
            f"{r['entropy'][0]:.3f} / {r['entropy'][1]:.3f} | "
            f"{r['query_cosine'][0]:.5f} / {r['query_cosine'][1]:.5f} | [View]({link}) |")
    report += ['', 'Entropy is normalized: 1 means uniform spatial attention; lower values mean more concentration. '
        'Query similarity is mean pairwise cosine similarity between the 16 queries after averaging attention heads; '
        'near 1 suggests overlapping spatial attention. All decisions contribute equally within each episode.', '',
        'These measures describe attention to predicted future-tactile token positions. They do not establish '
        'identical output tokens, physical pixel-level causality, or the value of tactile guidance. '
        'Head averaging may hide head-specific differences. One episode per task is not a reliable task success rate.', '',
        'Video color scales are fixed within each episode but may differ across tasks; compare the numerical metrics across tasks.', '']
    if records:
        concentrated = min(records, key=lambda r: np.mean(r['entropy']))
        redundant = max(records, key=lambda r: np.mean(r['query_cosine']))
        report += ['## Observations from completed recordings', '',
            f"- Most concentrated attention on average across fingers: **{concentrated['task']}**.",
            f"- Most similar query-attention patterns on average across fingers: **{redundant['task']}**.",
            '- These rankings apply only to the recorded episodes and can change as remaining tasks finish.', '']
    if errors:
        report += ['## Incomplete tasks or uploads', '', *[f'- {task}: {error}' for task, error in errors.items()], '']
    (OUTPUT/'COMPARISON.md').write_text('\n'.join(report))
    api = HfApi()
    operations = [CommitOperationAdd(path_in_repo=f'{PREFIX}/{name}', path_or_fileobj=str(OUTPUT/name))
                  for name in ('COMPARISON.md', 'comparison.json')]
    return retry_upload(lambda: api.create_commit(repo_id=REPO, repo_type='dataset',
        operations=operations, commit_message='Update one-episode cross-task resampler attention comparison')).commit_url


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    lock = (OUTPUT/'runner.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    env = dict(os.environ, PYTHONPATH=str(ROOT/'src'), OMNI_KIT_ACCEPT_EULA='YES',
        OMNI_KIT_ALLOW_ROOT='1', TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='4',
        HF_HUB_DISABLE_PROGRESS_BARS='1')
    errors = {}
    completed = []
    for task in TASKS:
        folder = OUTPUT/task
        try:
            if not folder.exists():
                status('evaluating', task=task, completed=completed, pid=os.getpid())
                with (OUTPUT/f'{task}.log').open('a') as log:
                    result = subprocess.run([str(ROOT/'.venv-train/bin/python'), '-u',
                        str(ROOT/'scripts/record_insert_tube_attention.py'), '--task', task,
                        '--seed', '2000000', '--output', str(folder), '--upload'],
                        cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
                print(json.dumps(dict(task=task, returncode=result.returncode)), flush=True)
            child = json.loads((folder/'status.json').read_text())
            if child['phase'] != 'complete':
                if (folder/'REPORT.md').exists() and (folder/'episode.json').exists():
                    # Retry finalized artifact upload only; never replay an episode.
                    meta = json.loads((folder/'metadata.json').read_text())
                    uploaded = retry_upload(lambda: upload(folder, meta))
                    episode = json.loads((folder/'episode.json').read_text())
                    atomic_json(folder/'status.json', dict(phase='complete',
                        success=episode['success'], steps=episode['steps'], **uploaded))
                else:
                    raise RuntimeError(f"Existing attempt incomplete; not rerun: {child}")
            completed.append(task)
        except Exception as exc:
            errors[task] = repr(exc)
        atomic_json(OUTPUT/'errors.json', errors)
        try:
            comparison()
        except Exception as exc:
            print(json.dumps(dict(phase='comparison_upload_failed', error=repr(exc))), flush=True)
    try:
        commit = comparison()
        status('complete' if not errors else 'finished_with_errors', completed=completed,
               errors=errors, comparison_commit=commit, pid=os.getpid())
    except Exception as exc:
        status('comparison_upload_failed', completed=completed, errors=errors, error=repr(exc))


if __name__ == '__main__':
    main()
