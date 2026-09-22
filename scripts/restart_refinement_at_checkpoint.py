"""Apply the Stage 2 masking change at an atomic checkpoint boundary.

The existing worker predates this change and cannot save on a new signal.
Wait for its next scheduled checkpoint, stop only that worker, then resume
the same optimizer/RNG state under the updated configuration.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import select
import signal
import subprocess
import time

import torch

from smolvla_baseline.data import ROOT
from tactile_refinement.train import validate_resume_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pid', type=int, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--step', type=int, required=True)
    args = parser.parse_args()
    lock = (ROOT/'logs/refinement_reconfigure.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    status_path = ROOT/'logs/refinement_reconfigure.json'
    status = {'old_pid': args.pid, 'checkpoint': str(args.checkpoint), 'expected_step': args.step}

    def record(phase, **fields):
        status.update(phase=phase, updated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), **fields)
        temporary = status_path.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(status, indent=2)+'\n')
        os.replace(temporary, status_path)
        print(json.dumps(status), flush=True)

    pidfd = None
    try:
        command = Path(f'/proc/{args.pid}/cmdline').read_bytes().split(b'\0')
        if b'tactile_refinement.train' not in command or b'--stage' not in command or b'all' not in command:
            raise ValueError('Target PID is not the expected full refinement training worker')
        pidfd = os.pidfd_open(args.pid)
        record('waiting_for_checkpoint')
        while not args.checkpoint.exists():
            if select.select([pidfd], [], [], 0.5)[0]:
                raise RuntimeError('Original worker exited before the requested checkpoint')
        saved = torch.load(args.checkpoint, weights_only=True, map_location='cpu', mmap=True)
        config = json.loads((ROOT/'configs/tactile_refinement.json').read_text())
        if saved['stage'] != 'encoder' or saved['step'] != args.step:
            raise ValueError('Unexpected checkpoint stage or update')
        validate_resume_config(saved['config'], config, saved['stage'])
        # Existing verification must have demonstrated independence from prefix K/V.
        verified = json.loads((ROOT/'logs/refinement_prefix_mask_verification.json').read_text())
        if verified['checks']['stage2_prefix_perturbation_max_abs_error'] != 0:
            raise ValueError('Stage 2 mask verification is missing or failed')
        del saved
        record('stopping_original_worker')
        signal.pidfd_send_signal(pidfd, signal.SIGTERM)
        if not select.select([pidfd], [], [], 60)[0]:
            raise RuntimeError('Original worker did not exit; refusing to start a second trainer')
        record('launching_updated_worker')
        subprocess.run(['bash', str(ROOT/'scripts/start_refinement.sh'), '--stage', 'all',
                        '--resume', str(args.checkpoint)], cwd=ROOT, check=True)
        new_pid = int((ROOT/'logs/tactile_refinement.pid').read_text())
        record('resumed', new_pid=new_pid, stage2_mask_prefix=True)
    except BaseException as error:
        record('error', error=str(error))
        raise
    finally:
        if pidfd is not None:
            os.close(pidfd)
        lock.close()


if __name__ == '__main__':
    main()
