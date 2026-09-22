"""Publish finalized Stage 3 evaluation videos and live success rates to Hugging Face."""

from io import BytesIO
import fcntl
import json
from pathlib import Path
import time

from huggingface_hub import CommitOperationAdd, HfApi


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'runs/benchmark/stage3_checkpoint_10000_matched60_100eps_seed2000000'
RUN_STATUS = ROOT / 'logs/stage3_checkpoint_10000_eval_status.json'
UPLOAD_STATUS = ROOT / 'logs/stage3_checkpoint_10000_upload_status.json'
STATE = ROOT / 'logs/stage3_checkpoint_10000_upload_state.json'
LOCK = ROOT / 'logs/stage3_checkpoint_10000_upload.lock'
REPO = 'sungkyunner/univtac-stage2-tactile-ablation-checkpoint-33333'
PREFIX = 'stage3/checkpoint_10000/univtac_matched60_100eps_seed2000000'
EXPECTED = 800


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def read_rows():
    path = OUTPUT / 'episodes.jsonl'
    if not path.exists():
        return []
    # The evaluator appends and fsyncs one row at a time. Ignore a line still
    # being written so the uploader never publishes an incomplete episode.
    lines = path.read_bytes().splitlines(keepends=True)
    return [json.loads(line) for line in lines if line.endswith(b'\n')]


def result(rows, metadata):
    tasks = {}
    for task in metadata['tasks']:
        selected = [r for r in rows if r['task'] == task and r['condition'] == 'correct'
                    and not r['smoke']]
        successes = sum(bool(r['success']) for r in selected)
        tasks[task] = {'episodes': len(selected), 'successes': successes,
                       'success_rate': successes / len(selected) if selected else None,
                       'complete': len(selected) == metadata['episodes_per_task']}
    completed = sum(value['episodes'] for value in tasks.values())
    successes = sum(value['successes'] for value in tasks.values())
    return {'checkpoint': metadata['checkpoint'], 'schedule': metadata['schedule'],
            'condition': 'correct = predicted future tactile latent',
            'start_seed': metadata['start_seed'], 'episodes_per_task': metadata['episodes_per_task'],
            'completed_rollouts': completed, 'expected_rollouts': EXPECTED,
            'successes': successes, 'pooled_success_rate': successes / completed if completed else None,
            'complete': completed == EXPECTED, 'tasks': tasks,
            'updated_at_unix': time.time()}


def item(remote, content):
    if isinstance(content, (dict, list)):
        content = json.dumps(content, indent=2).encode() + b'\n'
    if isinstance(content, str):
        content = content.encode()
    return CommitOperationAdd(path_in_repo=f'{PREFIX}/{remote}', path_or_fileobj=BytesIO(content))


def status(phase, **details):
    value = {'phase': phase, 'checked_at': time.time(),
             'repo': REPO, 'url': f'https://huggingface.co/datasets/{REPO}/tree/main/{PREFIX}',
             **details}
    write_json(UPLOAD_STATUS, value)
    print(json.dumps(value), flush=True)


def commit_with_retry(api, operations, message):
    for attempt in range(1, 6):
        try:
            return api.create_commit(repo_id=REPO, repo_type='dataset', operations=operations,
                                     commit_message=message)
        except Exception as error:
            status('upload_retry', attempt=attempt, error=repr(error))
            if attempt == 5:
                raise
            time.sleep(min(120, 15 * 2 ** (attempt - 1)))


def main():
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    with LOCK.open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        api = HfApi()
        if not api.repo_info(REPO, repo_type='dataset').private:
            raise ValueError('Expected the existing private Hugging Face evaluation dataset')
        while not (OUTPUT / 'metadata.json').exists():
            status('waiting_for_evaluation_metadata')
            time.sleep(5)
        metadata = json.loads((OUTPUT / 'metadata.json').read_text())
        checkpoint = metadata['checkpoint']
        if (checkpoint['stage'], checkpoint['step'], metadata['schedule'],
                metadata['conditions'], metadata['episodes_per_task']) != (
                'joint', 10000, 'matched60', ['correct'], 100):
            raise ValueError('Unexpected evaluation metadata')
        state = json.loads(STATE.read_text()) if STATE.exists() else {
            'published_rollouts': 0, 'published_videos': [], 'last_commit': None}
        readme = ('# Stage 3 checkpoint 10,000: UniVTAC evaluation\n\n'
                  'The policy uses its predicted future tactile latent (`correct` condition). '
                  'It replans each 60 Hz policy step with temporal aggregation; the simulator '
                  'runs at 120 Hz using two linear-position substeps. '
                  'There are 100 seeds per task, beginning at 2,000,000.\n\n'
                  '`running_results.json` gives current success counts and rates. '
                  '`episodes.jsonl` gives the completed rollouts. Videos for the first three '
                  'episodes of each task are uploaded as they finish. Rates are provisional '
                  'until each task has 100 completed episodes.\n')
        if state['last_commit'] is None:
            rows = read_rows()
            operations = [item('README.md', readme),
                          item('metadata.json', (OUTPUT / 'metadata.json').read_bytes()),
                          item('running_results.json', result(rows, metadata)),
                          item('episodes.jsonl', b''.join(
                              json.dumps(row).encode() + b'\n' for row in rows))]
            receipt = commit_with_retry(api, operations, 'Start Stage 3 checkpoint 10000 live evaluation')
            state['published_rollouts'] = len(rows)
            state['last_commit'] = receipt.commit_url
            write_json(STATE, state)
            status('publishing_live', completed_rollouts=len(rows), commit=receipt.commit_url)
        while True:
            rows = read_rows()
            if len(rows) > EXPECTED:
                raise ValueError('More rollouts than expected')
            videos = []
            published = set(state['published_videos'])
            for row in rows:
                relative = row.get('video')
                if relative and relative not in published:
                    video = OUTPUT / relative
                    if not video.is_file() or video.stat().st_size == 0:
                        raise RuntimeError(f'Finalized rollout has no video: {video}')
                    videos.append(relative)
            task_finished = any(sum(r['task'] == task for r in rows) == 100
                                and sum(r['task'] == task for r in rows[:state['published_rollouts']]) < 100
                                for task in metadata['tasks'])
            changed = len(rows) > state['published_rollouts']
            should_publish = bool(videos) or (changed and (
                len(rows) - state['published_rollouts'] >= 10 or task_finished or len(rows) == EXPECTED))
            if should_publish:
                current = result(rows, metadata)
                operations = [item('running_results.json', current),
                              item('episodes.jsonl', b''.join(
                                  json.dumps(row).encode() + b'\n' for row in rows))]
                for relative in videos:
                    operations.append(CommitOperationAdd(
                        path_in_repo=f'{PREFIX}/{relative}',
                        path_or_fileobj=str(OUTPUT / relative)))
                status('uploading', completed_rollouts=len(rows), videos=videos)
                receipt = commit_with_retry(api, operations,
                    f'Update Stage 3 evaluation: {len(rows)}/{EXPECTED} rollouts')
                state['published_rollouts'] = len(rows)
                state['published_videos'] = sorted(published.union(videos))
                state['last_commit'] = receipt.commit_url
                write_json(STATE, state)
                status('publishing_live', completed_rollouts=len(rows),
                       published_videos=len(state['published_videos']), commit=receipt.commit_url)
            if len(rows) == EXPECTED and state['published_rollouts'] == EXPECTED:
                status('complete', completed_rollouts=EXPECTED,
                       published_videos=len(state['published_videos']),
                       commit=state['last_commit'])
                return
            if RUN_STATUS.exists():
                run_status = json.loads(RUN_STATUS.read_text())
                if run_status['phase'] == 'failed':
                    status('evaluation_failed', completed_rollouts=len(rows))
                    raise RuntimeError('Evaluation supervisor failed')
            time.sleep(15)


if __name__ == '__main__':
    main()
