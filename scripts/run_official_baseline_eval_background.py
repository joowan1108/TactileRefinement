#!/usr/bin/env python
"""Run/resume all 800 baseline episodes with the official UniVTAC schedule."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]


def main():
    lock = (ROOT/'logs/univtac_baseline_official_eval.lock').open('a')
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    (ROOT/'logs/univtac_baseline_official_eval.pid').write_text(str(os.getpid())+'\n')
    env = dict(os.environ, PYTHONPATH=str(ROOT/'src'), OMNI_KIT_ACCEPT_EULA='YES', OMNI_KIT_ALLOW_ROOT='1',
               TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='4')
    command = [str(ROOT/'.venv-train/bin/python'), '-u', '-m', 'smolvla_baseline.official_benchmark',
        '--tasks', 'grasp_classify', 'pull_out_key', 'insert_HDMI', 'lift_bottle', 'lift_can',
        'insert_tube', 'put_bottle_in_shelf', 'insert_hole', '--episodes', '100', '--video-episodes', '3',
        '--seed', '1000000', '--resume', '--output', str(ROOT/'runs/benchmark/baseline_official_100eps_seed1000000'),
        '--hf-repo', 'sungkyunner/univtac-smolvla-baseline-official-schedule-eval']
    status_path = ROOT/'logs/univtac_baseline_official_eval_status.json'
    for attempt in range(1, 6):
        status = {'phase': 'running', 'attempt': attempt, 'command': command, 'supervisor_pid': os.getpid()}
        status_path.write_text(json.dumps(status, indent=2)+'\n')
        print(json.dumps(status), flush=True)
        result = subprocess.run(command, cwd=ROOT, env=env)
        if result.returncode == 0:
            status['phase'] = 'completed_and_uploaded'
            status_path.write_text(json.dumps(status, indent=2)+'\n')
            print(json.dumps(status), flush=True)
            return
        status.update(phase='retry_pending' if attempt < 5 else 'failed', returncode=result.returncode)
        status_path.write_text(json.dumps(status, indent=2)+'\n')
        print(json.dumps(status), flush=True)
        if attempt < 5:
            time.sleep(15)
    raise SystemExit('Five attempts failed; inspect the log. Completed episodes are preserved.')


if __name__ == '__main__':
    main()
