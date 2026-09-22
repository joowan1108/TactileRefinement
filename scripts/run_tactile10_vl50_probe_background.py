"""Detached checkpoint-66666 cached-context probe; wait for an existing runner."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

from huggingface_hub import CommitOperationAdd, HfApi

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT/'runs/benchmark/stage2_checkpoint_66666_tactile10_vl50_probe_20260917'
STATUS = ROOT/'logs/stage2_checkpoint_66666_tactile10_vl50_status.json'
REPO = 'sungkyunner/univtac-stage2-tactile-ablation-checkpoint-33333'


def status(phase, **values):
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    temporary = STATUS.with_suffix('.tmp')
    temporary.write_text(json.dumps(dict(phase=phase, checked_at=time.time(), **values), indent=2)+'\n')
    temporary.replace(STATUS)


def process_identity(pid):
    try:
        proc = Path(f'/proc/{pid}')
        fields = (proc/'stat').read_text().rsplit(')', 1)[1].split()
        if fields[0] == 'Z':
            return None
        return fields[19], (proc/'cmdline').read_bytes()
    except FileNotFoundError:
        return None


def evaluate(name, task, smoke_steps):
    output = OUTPUT/name
    command = [str(ROOT/'.venv-train/bin/python'), '-u', '-m', 'tactile_refinement.ablation_rollout',
        '--checkpoint', str(ROOT/'runs/tactile_refinement_trex_fullgrid_n0/expert/checkpoint_2.pt'),
        '--tasks', task, '--conditions', 'correct', '--episodes', '1', '--start-seed', '2000000',
        '--video-episodes', '1', '--schedule', 'tactile10_vl50', '--smoke-steps', str(smoke_steps),
        '--output', str(output)]
    if (output/'metadata.json').exists():
        command.append('--resume')
    env = dict(os.environ, PYTHONPATH=str(ROOT/'src'), OMNI_KIT_ACCEPT_EULA='YES',
               OMNI_KIT_ALLOW_ROOT='1', TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='4',
               HF_HUB_DISABLE_PROGRESS_BARS='1')
    status('evaluating', probe=name, command=command)
    subprocess.run(command, cwd=ROOT, env=env, check=True)
    rows = [json.loads(line) for line in (output/'episodes.jsonl').read_text().splitlines()]
    if len(rows) != 1:
        raise RuntimeError(f'Expected one completed rollout in {output}')
    row = rows[0]
    expected = list(range(0, row['steps'], 10))
    if row['replan_steps'] != expected or row['vl_refresh_steps'] != list(range(0, row['steps'], 50)):
        raise RuntimeError('Saved rollout has incorrect refresh timing')
    video = output/row['video']
    # A directory entry alone is insufficient: require completed episode and a
    # decodable finalized MP4 with the initial frame plus every executed action.
    probe = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-count_frames',
        '-select_streams', 'v:0', '-show_entries', 'stream=nb_read_frames', '-of', 'json', str(video)], text=True))
    if int(probe['streams'][0]['nb_read_frames']) != row['steps']+1:
        raise RuntimeError('Finalized video frame count mismatch')
    return output, row


def upload(name, output, row):
    prefix = f'checkpoint_66666/tactile10_vl50/{name}'
    paths = ['metadata.json', 'episodes.jsonl', 'summary.json', row['video'], row['action_trace']]
    status('uploading', probe=name)
    for attempt in range(1, 4):
        try:
            operations = [CommitOperationAdd(path_in_repo=f'{prefix}/{p}', path_or_fileobj=str(output/p))
                          for p in paths]
            result = HfApi().create_commit(repo_id=REPO, repo_type='dataset', operations=operations,
                commit_message=f'Upload checkpoint 66666 tactile10_vl50 {name} probe')
            return result.commit_url
        except Exception:
            if attempt == 3:
                raise
            time.sleep(10)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wait-pid', type=int)
    args = parser.parse_args()
    try:
        identity = process_identity(args.wait_pid) if args.wait_pid else None
        if identity and b'run_checkpoint_66666_one_per_task_background.py' not in identity[1]:
            raise ValueError('Wait PID does not identify the expected existing evaluation supervisor')
        while identity and process_identity(args.wait_pid) == identity:
            status('waiting_for_existing_evaluation', wait_pid=args.wait_pid)
            time.sleep(30)
        results = []
        # The smoke crosses the 50-action VL refresh boundary if the task stays
        # nonterminal. It is excluded from success metrics by the rollout CLI.
        for name, task, smoke in [('smoke_grasp', 'grasp_classify', 55),
                                  ('pull_out_key', 'pull_out_key', 0)]:
            output, row = evaluate(name, task, smoke)
            commit = upload(name, output, row)
            results.append(dict(probe=name, steps=row['steps'], smoke=row['smoke'],
                                success=row['success'], vl_refresh_steps=row['vl_refresh_steps'],
                                commit_url=commit))
        status('complete_and_uploaded', results=results)
        print(json.dumps(dict(phase='complete_and_uploaded', results=results)), flush=True)
    except Exception as exc:
        status('failed', error=repr(exc))
        raise


if __name__ == '__main__':
    main()
