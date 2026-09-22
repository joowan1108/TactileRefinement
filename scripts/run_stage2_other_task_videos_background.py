"""Wait for the active ablation, record other-task Stage 2 videos, then upload them."""
import argparse
from io import BytesIO
import json
import os
from pathlib import Path
import subprocess
import time

from huggingface_hub import CommitOperationAdd, HfApi


ROOT = Path(__file__).resolve().parents[1]
TASKS = ('grasp_classify', 'lift_bottle', 'lift_can', 'insert_tube',
         'put_bottle_in_shelf', 'insert_hole')
REPO = 'sungkyunner/univtac-stage2-tactile-ablation-checkpoint-33333'
OUTPUT = ROOT/'runs/benchmark/stage2_tactile_ablation_checkpoint_33333_other_tasks_videos_20260917'
STATUS = ROOT/'logs/stage2_other_task_videos_status.json'


def write_status(**values):
    temporary = STATUS.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(values, indent=2)+'\n')
    os.replace(temporary, STATUS)


def is_expected_process(pid):
    command = Path(f'/proc/{pid}/cmdline')
    if not command.exists():
        return False
    text = command.read_bytes().replace(b'\0', b' ').decode(errors='replace')
    return 'tactile_refinement.ablation_rollout' in text


def upload_results():
    rows = [json.loads(line) for line in (OUTPUT/'episodes.jsonl').read_text().splitlines()]
    operations = [
        CommitOperationAdd(path_in_repo='checkpoint_33333/other_tasks/metadata.json',
                           path_or_fileobj=str(OUTPUT/'metadata.json')),
        CommitOperationAdd(path_in_repo='checkpoint_33333/other_tasks/episodes.jsonl',
                           path_or_fileobj=str(OUTPUT/'episodes.jsonl')),
        CommitOperationAdd(path_in_repo='checkpoint_33333/other_tasks/summary.json',
                           path_or_fileobj=str(OUTPUT/'summary.json')),
    ]
    uploaded = []
    for task in TASKS:
        for condition in ('correct', 'mismatched'):
            path = OUTPUT/'videos'/task/f'seed_2100000_{condition}.mp4'
            if not path.exists():
                raise FileNotFoundError(path)
            remote = f'checkpoint_33333/videos/{task}/seed_2100000_{condition}.mp4'
            operations.append(CommitOperationAdd(path_in_repo=remote, path_or_fileobj=str(path)))
            uploaded.append(remote)
    manifest = {'checkpoint_step': 33333, 'seed': 2100000, 'tasks': TASKS,
                'uploaded_conditions': ['correct', 'mismatched'],
                'completed_rollouts': len(rows), 'videos': uploaded}
    operations.append(CommitOperationAdd(
        path_in_repo='checkpoint_33333/other_tasks/video_manifest.json',
        path_or_fileobj=BytesIO((json.dumps(manifest, indent=2)+'\n').encode())))
    commit = HfApi().create_commit(repo_id=REPO, repo_type='dataset', operations=operations,
        commit_message='Upload correct and mismatched Stage 2 videos for the other six tasks')
    return commit.commit_url, uploaded


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wait-pid', type=int, required=True)
    args = parser.parse_args()
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    while is_expected_process(args.wait_pid):
        write_status(phase='waiting_for_current_ablation', wait_pid=args.wait_pid,
                     checked_at=time.time())
        time.sleep(60)

    env = dict(os.environ, PYTHONPATH=str(ROOT/'src'), OMNI_KIT_ACCEPT_EULA='YES',
               OMNI_KIT_ALLOW_ROOT='1', TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='4')
    command = [str(ROOT/'.venv-train/bin/python'), '-u', '-m', 'tactile_refinement.ablation_rollout',
        '--checkpoint', str(ROOT/'runs/tactile_refinement_trex_fullgrid_n0/expert/checkpoint_1.pt'),
        '--tasks', *TASKS, '--episodes', '1', '--start-seed', '2100000',
        '--conditions', 'correct', 'mismatched',
        '--video-episodes', '1', '--schedule', 'matched60', '--output', str(OUTPUT)]
    if (OUTPUT/'metadata.json').exists():
        command.append('--resume')
    for attempt in range(1, 6):
        write_status(phase='running', attempt=attempt, command=command, checked_at=time.time())
        result = subprocess.run(command, cwd=ROOT, env=env)
        if result.returncode == 0:
            break
        command = command if '--resume' in command else [*command, '--resume']
        write_status(phase='retry_pending', attempt=attempt, returncode=result.returncode,
                     checked_at=time.time())
        time.sleep(15)
    else:
        write_status(phase='failed', attempts=5, checked_at=time.time())
        raise SystemExit(1)

    write_status(phase='uploading', checked_at=time.time())
    commit, videos = upload_results()
    write_status(phase='complete_and_uploaded', commit_url=commit, videos=videos,
                 checked_at=time.time())
    print(json.dumps({'phase': 'complete_and_uploaded', 'commit_url': commit,
                      'videos': videos}), flush=True)


if __name__ == '__main__':
    main()
