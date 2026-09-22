"""Upload attention videos for the first three seeds of every scored Stage 3 task."""

import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

from huggingface_hub import CommitOperationAdd, HfApi

from record_insert_tube_attention import REPO, upload
from smolvla_baseline.instructions import DEFAULT_TASKS


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / 'runs/tactile_refinement_trex_fullgrid_n0/joint/final.pt'
EVAL = ROOT / 'runs/benchmark/stage3_checkpoint_10000_matched60_100eps_seed2000000'
OUTPUT = ROOT / 'runs/diagnostics/stage3_final_eval_attention_replays'
STATUS = ROOT / 'logs/stage3_checkpoint_10000_attention_replays_status.json'
LOCK = ROOT / 'logs/stage3_checkpoint_10000_attention_replays.lock'
PREFIX = 'stage3/checkpoint_10000/univtac_matched60_100eps_seed2000000/attention_replays'
EXISTING = ROOT / 'runs/diagnostics/stage3_final_attention_10000_seed2000000'
SEEDS = (2000001, 2000002)


def status(phase, **details):
    value = {'phase': phase, 'checked_at': time.time(), 'completed': completed_count(),
             'expected': len(DEFAULT_TASKS) * len(SEEDS),
             'url': f'https://huggingface.co/datasets/{REPO}/tree/main/{PREFIX}', **details}
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    temporary = STATUS.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(STATUS)
    print(json.dumps(value), flush=True)


def completed_count():
    return sum((OUTPUT / task / f'seed_{seed}' / 'status.json').exists() and
               json.loads((OUTPUT / task / f'seed_{seed}' / 'status.json').read_text())['phase'] == 'complete'
               for task in DEFAULT_TASKS for seed in SEEDS)


def scored_rollout_ready(task, seed):
    path = EVAL / 'episodes.jsonl'
    if not path.exists():
        return False
    rows = [json.loads(line) for line in path.read_bytes().splitlines(keepends=True)
            if line.endswith(b'\n')]
    return any(row['task'] == task and row['seed'] == seed and row['condition'] == 'correct'
               for row in rows)


def gpu_memory_mib():
    value = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used',
                                     '--format=csv,noheader,nounits'], text=True)
    return int(value.splitlines()[0])


def archive(folder):
    number = 1
    while True:
        target = folder.with_name(folder.name + f'_failed_{number}')
        if not target.exists():
            folder.rename(target)
            return target
        number += 1


def upload_readme():
    readme = ('# Attention visualization replays\n\n'
              'Each `attention_visualization.mp4` records a separate diagnostic rollout '
              'with the same Stage 3 checkpoint, task seed, and matched60 schedule '
              'as the scored evaluation. These replays collect live tactile maps and '
              'resampler attention, which were not captured by the scored rollout. '
              'They must not be treated as the exact scored episode. '
              'The seed-2,000,000 videos were already recorded for this checkpoint '
              'before the 100-episode evaluation began; they are copied here. '
              'Seeds 2,000,001 and 2,000,002 are recorded as their scored rollouts finish. '
              '`reference_comparison.json` reports whether the recorded initial-state '
              'signature, success, and duration matched the scored rollout for the new replays.\n')
    api = HfApi()
    if not api.repo_info(REPO, repo_type='dataset').private:
        raise ValueError('Expected private Hugging Face dataset')
    operations = [CommitOperationAdd(path_in_repo=f'{PREFIX}/README.md',
                                     path_or_fileobj=readme.encode())]
    for task in DEFAULT_TASKS:
        source = EXISTING / task
        metadata = json.loads((source / 'metadata.json').read_text())
        if (metadata['checkpoint']['stage'], metadata['checkpoint']['step'],
                metadata['task'], metadata['seed'], metadata['schedule']) != (
                'joint', 10000, task, 2000000, 'matched60'):
            raise ValueError(f'Existing diagnostic does not match this evaluation: {task}')
        for name in ('attention_visualization.mp4', 'attention_snapshot.png',
                     'left_queries.png', 'right_queries.png', 'attention_metrics.json',
                     'REPORT.md', 'metadata.json', 'episode.json'):
            operations.append(CommitOperationAdd(
                path_in_repo=f'{PREFIX}/{task}/seed_2000000/{name}',
                path_or_fileobj=str(source / name)))
    for attempt in range(1, 5):
        try:
            return api.create_commit(repo_id=REPO, repo_type='dataset',
                operations=operations,
                commit_message='Add existing Stage 3 attention videos for all eight tasks').commit_url
        except Exception:
            if attempt == 4:
                raise
            time.sleep(10)


def run_one(task, seed, env):
    folder = OUTPUT / task / f'seed_{seed}'
    remote = f'{PREFIX}/{task}/seed_{seed}'
    for attempt in range(1, 4):
        if (folder / 'status.json').exists():
            child = json.loads((folder / 'status.json').read_text())
            if child['phase'] == 'complete':
                return
            # A rendering succeeded but only its remote commit failed: upload
            # existing artifacts instead of spending another simulator rollout.
            if all((folder / name).exists() for name in (
                    'attention_visualization.mp4', 'episode.json', 'metadata.json',
                    'reference_comparison.json')):
                metadata = json.loads((folder / 'metadata.json').read_text())
                receipt = upload(folder, metadata, remote)
                child.update(phase='complete', **receipt)
                (folder / 'status.json').write_text(json.dumps(child, indent=2) + '\n')
                return
            archive(folder)
        status('recording', task=task, seed=seed, attempt=attempt,
               gpu_memory_mib=gpu_memory_mib())
        command = [str(ROOT / '.venv-train/bin/python'), '-u',
                   str(ROOT / 'scripts/record_insert_tube_attention.py'),
                   '--checkpoint', str(CHECKPOINT), '--task', task, '--seed', str(seed),
                   '--output', str(folder), '--upload', '--upload-prefix', remote,
                   '--reference-episodes', str(EVAL / 'episodes.jsonl')]
        log = OUTPUT / task / f'seed_{seed}.log'
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open('a') as stream:
            result = subprocess.run(command, cwd=ROOT, env=env, stdout=stream,
                                    stderr=subprocess.STDOUT)
        if result.returncode == 0 and json.loads((folder / 'status.json').read_text())['phase'] == 'complete':
            return
        status('retry_pending', task=task, seed=seed, attempt=attempt,
               returncode=result.returncode)
        time.sleep(15)
    raise RuntimeError(f'Failed to record attention replay for {task} seed {seed}')


def main():
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with LOCK.open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        status('starting', readme_commit=upload_readme())
        env = dict(os.environ, PYTHONPATH=str(ROOT / 'src'),
                   OMNI_KIT_ACCEPT_EULA='YES', OMNI_KIT_ALLOW_ROOT='1',
                   TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='4',
                   HF_HUB_DISABLE_PROGRESS_BARS='1')
        for task in DEFAULT_TASKS:
            for seed in SEEDS:
                folder = OUTPUT / task / f'seed_{seed}'
                if (folder / 'status.json').exists() and json.loads(
                        (folder / 'status.json').read_text())['phase'] == 'complete':
                    continue
                while not scored_rollout_ready(task, seed) or gpu_memory_mib() > 13000:
                    status('waiting_for_scored_rollout_or_gpu', task=task, seed=seed,
                           scored_rollout_ready=scored_rollout_ready(task, seed),
                           gpu_memory_mib=gpu_memory_mib())
                    time.sleep(30)
                run_one(task, seed, env)
                status('uploaded', task=task, seed=seed)
        status('complete')


if __name__ == '__main__':
    main()
