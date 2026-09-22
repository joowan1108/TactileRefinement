"""Wait for the existing grounding run, then resume with its extended budget.

The training lock is held through the resumed child process. Earlier milestone
files are archived; the extended run writes its own three scheduled checkpoints.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--previous-updates', type=int, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    target = config['stages']['grounding']['updates']
    if target <= args.previous_updates:
        raise ValueError('Expected an increased grounding budget')
    output = ROOT/config['output_dir']/'grounding'
    status_file = ROOT/'logs/tactile_refinement_extension_status.json'

    def status(state, **fields):
        row = dict(state=state, supervisor_pid=os.getpid(), previous_updates=args.previous_updates,
                   target_updates=target, config=str(args.config.resolve()), **fields)
        temporary = status_file.with_suffix('.tmp')
        temporary.write_text(json.dumps(row, indent=2)+'\n')
        os.replace(temporary, status_file)
        print(json.dumps(row), flush=True)

    with (ROOT/'logs/tactile_refinement_extension.lock').open('a') as guard:
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        (ROOT/'logs/tactile_refinement_extension.pid').write_text(str(os.getpid())+'\n')
        try:
            with (ROOT/'logs/tactile_refinement.lock').open('a') as lock:
                status('waiting_for_existing_grounding_run')
                fcntl.flock(lock, fcntl.LOCK_EX)
                # No training process now owns the output directory. Validate the
                # completed checkpoint before moving files or starting a child.
                import torch
                from tactile_refinement.train import training_provenance, validate_resume_config
                final = output/'final.pt'
                saved = torch.load(final, map_location='cpu', weights_only=True, mmap=True)
                if saved['stage'] != 'grounding' or saved['step'] != args.previous_updates:
                    raise ValueError('Existing run did not complete its expected grounding budget')
                validate_resume_config(saved['config'], config, 'grounding')
                if saved['provenance'] != training_provenance(config):
                    raise ValueError('Checkpoint provenance mismatch')
                if json.loads(args.config.read_text()) != config:
                    raise ValueError('Configuration changed while waiting; refusing stale continuation')
                del saved
                archive = output.parent/f'grounding_{args.previous_updates}_before_extension'
                archive.mkdir(exist_ok=False)
                for name in ('checkpoint_1.pt', 'checkpoint_2.pt', 'final.pt', 'recipe.json', 'provenance.json'):
                    (output/name).rename(archive/name)
                command = ['bash', str(ROOT/'scripts/train_refinement.sh'), '--config', str(args.config.resolve()),
                           '--stage', 'grounding', '--resume', str(archive/'final.pt')]
                with (ROOT/'logs/tactile_refinement.log').open('a') as log:
                    child = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log,
                                             stderr=subprocess.STDOUT, pass_fds=(lock.fileno(),))
                    (ROOT/'logs/tactile_refinement.pid').write_text(str(child.pid)+'\n')
                    status('running_extended_grounding', training_pid=child.pid, resume=str(archive/'final.pt'))
                    result = child.wait()
                if result:
                    raise RuntimeError(f'Extended training exited with code {result}')
                status('completed_grounding_and_paused', training_pid=child.pid)
        except BaseException as error:
            status('failed', error=str(error))
            raise


if __name__ == '__main__':
    sys.path.insert(0, str(ROOT/'src'))
    main()
