"""Resume the full Stage 3 final UniVTAC evaluation after transient failures."""

import fcntl
import json
import os
from pathlib import Path
import subprocess
import time


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'runs/benchmark/stage3_checkpoint_10000_matched60_100eps_seed2000000'
STATUS = ROOT / 'logs/stage3_checkpoint_10000_eval_status.json'
LOCK = ROOT / 'logs/stage3_checkpoint_10000_eval.lock'
CHECKPOINT = ROOT / 'runs/tactile_refinement_trex_fullgrid_n0/joint/final.pt'
EXPECTED = 8 * 100


def completed_count():
    records = OUTPUT / 'episodes.jsonl'
    return sum(1 for _ in records.open()) if records.exists() else 0


def status(phase, **details):
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    value = {'phase': phase, 'checked_at': time.time(),
             'completed_rollouts': completed_count(), 'expected_rollouts': EXPECTED,
             'output': str(OUTPUT), **details}
    temporary = STATUS.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(STATUS)
    print(json.dumps(value), flush=True)


def main():
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    with LOCK.open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        command = [str(ROOT / '.venv-train/bin/python'), '-u', '-m',
                   'tactile_refinement.ablation_rollout',
                   '--checkpoint', str(CHECKPOINT),
                   '--episodes', '100', '--start-seed', '2000000',
                   '--conditions', 'correct', '--video-episodes', '3',
                   '--schedule', 'matched60', '--output', str(OUTPUT)]
        env = dict(os.environ, PYTHONPATH=str(ROOT / 'src'),
                   OMNI_KIT_ACCEPT_EULA='YES', OMNI_KIT_ALLOW_ROOT='1',
                   TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='4',
                   HF_HUB_DISABLE_PROGRESS_BARS='1')
        attempt = 0
        consecutive_no_progress = 0
        previous_completed = completed_count()
        while True:
            attempt += 1
            resume = ['--resume'] if (OUTPUT / 'metadata.json').exists() else []
            status('running', attempt=attempt, command=command + resume)
            child = subprocess.Popen(command + resume, cwd=ROOT, env=env)
            while child.poll() is None:
                status('running', attempt=attempt, child_pid=child.pid)
                time.sleep(30)
            if child.returncode == 0:
                if completed_count() != EXPECTED:
                    raise RuntimeError('Evaluator exited successfully without all rollouts')
                status('complete', attempt=attempt)
                return
            current_completed = completed_count()
            if current_completed > previous_completed:
                consecutive_no_progress = 0
            else:
                consecutive_no_progress += 1
            previous_completed = current_completed
            status('retry_pending', attempt=attempt, returncode=child.returncode,
                   consecutive_no_progress=consecutive_no_progress)
            if consecutive_no_progress >= 5:
                status('failed', attempts=attempt,
                       reason='five consecutive evaluator failures without a completed rollout')
                raise SystemExit(1)
            time.sleep(15)


if __name__ == '__main__':
    main()
