"""Run and upload matched final-Stage-3 rollouts with 16 or 8 tokens/finger."""
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from huggingface_hub import CommitOperationAdd, HfApi

from smolvla_baseline.data import ROOT
from smolvla_baseline.instructions import DEFAULT_TASKS
from tactile_refinement.ablation import atomic_json, load_action_checkpoint
from tactile_refinement.ablation_rollout import code_provenance, run_episode
from tactile_refinement.sim_bridge import TactileEnvironment
from tactile_refinement.train import file_sha256


CHECKPOINT = ROOT/'runs/tactile_refinement_trex_fullgrid_n0/joint/final.pt'
AUDIT = ROOT/'runs/diagnostics/stage3_final_resampler_token_audit_10000/summary.json'
OUTPUT = ROOT/'runs/benchmark/stage3_final_token_pruning_seed2000000'
LOG_STATUS = ROOT/'logs/stage3_final_token_pruning_status.json'
REPO = 'sungkyunner/univtac-stage2-tactile-ablation-checkpoint-33333'
PREFIX = 'stage3/checkpoint_10000/resampler_token_pruning'
SEED = 2_000_000
CONDITIONS = ('full16', 'top8', 'uniform8')


class TokenSelector(nn.Module):
    """Inference-only selection after the trained 16-query/finger resampler."""
    def __init__(self, trained, selections):
        super().__init__()
        self.trained = trained
        self.selections = selections
        self.condition = 'full16'

    def forward(self, predicted):
        compact = self.trained(predicted)
        if self.condition == 'full16':
            return compact
        shaped = compact.reshape(len(compact), 2, 16, compact.shape[-1])
        left, right = self.selections[self.condition]
        return torch.cat((shaped[:, 0, left], shaped[:, 1, right]), dim=1)


def status(phase, **values):
    row = dict(phase=phase, checked_at=time.time(), **values)
    atomic_json(LOG_STATUS, row)
    print(json.dumps(row), flush=True)


def load_selections():
    audit = json.loads(AUDIT.read_text())
    if audit['checkpoint']['stage'] != 'joint' or audit['checkpoint']['step'] != 10_000:
        raise ValueError('Token ranking must come from the Stage 3 final checkpoint')
    ranked = {}
    for finger in ('left', 'right'):
        values = sorted((row for row in audit['tokens'] if row['finger'] == finger),
                        key=lambda row: row['mean_replacement_action_rms'], reverse=True)
        ranked[finger] = sorted(row['local_token'] for row in values[:8])
    uniform = np.rint(np.linspace(0, 15, 8)).astype(int).tolist()
    if len(set(uniform)) != 8:
        raise ValueError('Uniform selector produced duplicate query IDs')
    return {'top8': (ranked['left'], ranked['right']),
            'uniform8': (uniform, uniform)}, audit


def read_rows():
    path = OUTPUT/'episodes.jsonl'
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def comparison(rows, metadata, errors):
    records = []
    for task in DEFAULT_TASKS:
        task_rows = [row for row in rows if row['task'] == task]
        by_condition = {row['condition']: row for row in task_rows}
        reset_ids = {row['initial_info'].get('initial_state_id') for row in task_rows}
        records.append(dict(task=task, complete=set(by_condition) == set(CONDITIONS),
            matching_initial_state_id=len(reset_ids) == 1 and None not in reset_ids,
            conditions={name: ({key: by_condition[name][key] for key in
                ('success', 'steps', 'early_stop', 'terminated', 'truncated',
                 'simulated_seconds', 'wall_seconds')} if name in by_condition else None)
                for name in CONDITIONS}))
    result = dict(checkpoint=metadata['checkpoint'], seed=SEED, schedule='matched60',
        conditions=metadata['condition_details'], completed_rollouts=len(rows),
        expected_rollouts=len(DEFAULT_TASKS)*len(CONDITIONS), records=records, errors=errors,
        interpretation='Inference-only pruning; this is not a retrained 8-query resampler.')
    atomic_json(OUTPUT/'comparison.json', result)
    report = ['# Stage 3 final: inference-time resampler token pruning', '',
        'One matched-seed rollout per task and condition. `top8` and `uniform8` retain '
        'eight of the trained 16 output tokens per finger; the model was not retrained.', '',
        '| Task | 16/finger | Top 8/finger | Uniform 8/finger | Matching reset ID |',
        '|---|---|---|---|---|']
    for item in records:
        def value(name):
            row = item['conditions'][name]
            return 'pending' if row is None else f"{'success' if row['success'] else 'failure'} ({row['steps']} steps)"
        report.append(f"| {item['task']} | {value('full16')} | {value('top8')} | "
                      f"{value('uniform8')} | {item['matching_initial_state_id']} |")
    report += ['', 'Success/failure is the simulator result. With one seed per task, these are '
        'paired case studies rather than success-rate estimates.', '']
    if errors:
        report += ['## Errors', '', *[f'- {key}: {value}' for key, value in errors.items()], '']
    (OUTPUT/'REPORT.md').write_text('\n'.join(report))
    return result


def retry(operation):
    for attempt in range(3):
        try:
            return operation()
        except Exception:
            if attempt == 2:
                raise
            time.sleep(10)


def upload_task(task, rows, metadata, errors):
    files = ['metadata.json', 'comparison.json', 'REPORT.md']
    task_rows = [row for row in rows if row['task'] == task]
    task_result = dict(task=task, seed=SEED,
        conditions={row['condition']: {key: row[key] for key in
            ('success', 'steps', 'early_stop', 'terminated', 'truncated', 'simulated_seconds')}
            for row in task_rows})
    task_file = OUTPUT/f'{task}_result.json'
    atomic_json(task_file, task_result)
    operations = [CommitOperationAdd(path_in_repo=f'{PREFIX}/{name}',
        path_or_fileobj=str(OUTPUT/name)) for name in files]
    operations.append(CommitOperationAdd(path_in_repo=f'{PREFIX}/tasks/{task}/result.json',
                                         path_or_fileobj=str(task_file)))
    remote_files = []
    for row in task_rows:
        video = OUTPUT/row['video']
        remote = f'{PREFIX}/tasks/{task}/{video.name}'
        operations.append(CommitOperationAdd(path_in_repo=remote, path_or_fileobj=str(video)))
        remote_files.append((remote, video))
    commit = retry(lambda: HfApi().create_commit(repo_id=REPO, repo_type='dataset',
        operations=operations, commit_message=f'Add Stage 3 token-pruning rollouts for {task}'))
    tree = {entry.path: entry.size for entry in HfApi().list_repo_tree(
        REPO, repo_type='dataset', path_in_repo=f'{PREFIX}/tasks/{task}', recursive=True)
        if hasattr(entry, 'size')}
    for remote, local in remote_files:
        if tree.get(remote) != local.stat().st_size:
            raise RuntimeError(f'Upload verification failed: {remote}')
    return commit.commit_url


@torch.inference_mode()
def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    selections, audit = load_selections()
    model, pre, post, config, checkpoint = load_action_checkpoint(CHECKPOINT)
    if checkpoint['stage'] != 'joint' or checkpoint['step'] != 10_000:
        raise ValueError('Expected Stage 3 final checkpoint at step 10,000')
    original = model.resampler
    model.resampler = TokenSelector(original, selections).to(config['device']).eval()
    metadata = dict(checkpoint=checkpoint, seed=SEED, tasks=list(DEFAULT_TASKS),
        conditions=list(CONDITIONS), schedule='matched60', controller='linear_position',
        condition_details={
            'full16': {'tokens_per_finger': 16, 'selection': list(range(16))},
            'top8': {'tokens_per_finger': 8, 'left': selections['top8'][0],
                     'right': selections['top8'][1],
                     'ranking': 'training-corpus mean-replacement action RMS'},
            'uniform8': {'tokens_per_finger': 8, 'left': selections['uniform8'][0],
                         'right': selections['uniform8'][1]}},
        noise='Fixed by task seed and model-call index across conditions',
        caveat='Inference-only pruning of a model trained with 16 tokens per finger',
        audit_sha256=file_sha256(AUDIT), code_sha256=code_provenance('matched60'))
    metadata['code_sha256']['scripts/run_stage3_token_pruning_rollouts.py'] = file_sha256(
        ROOT/'scripts/run_stage3_token_pruning_rollouts.py')
    meta_path = OUTPUT/'metadata.json'
    if meta_path.exists():
        if json.loads(meta_path.read_text()) != metadata:
            raise ValueError('Existing output metadata mismatch')
    else:
        if any(OUTPUT.iterdir()):
            raise ValueError('Use an empty output directory')
        atomic_json(meta_path, metadata)
    rows = read_rows()
    completed = {(row['task'], row['condition']) for row in rows}
    errors = json.loads((OUTPUT/'errors.json').read_text()) if (OUTPUT/'errors.json').exists() else {}
    os.environ['UNIVTAC_EVAL_SAVE_ROOT'] = str((OUTPUT/'simulator_artifacts').resolve())
    records = OUTPUT/'episodes.jsonl'
    api = HfApi()
    if not api.repo_info(REPO, repo_type='dataset').private:
        raise ValueError('Expected private Hugging Face evaluation repository')
    for task in DEFAULT_TASKS:
        order = np.random.default_rng(SEED).permutation(CONDITIONS).tolist()
        for condition in order:
            if (task, condition) in completed:
                continue
            key = f'{task}/{condition}'
            status('evaluating', task=task, condition=condition,
                   completed_rollouts=len(rows), expected_rollouts=24)
            try:
                model.resampler.condition = condition
                last_error = None
                for attempt in range(1, 4):
                    env = None
                    try:
                        env = TactileEnvironment(task=task, controller='linear_position')
                        row = run_episode(env, model, pre, post, None, task=task, seed=SEED,
                            condition=condition, schedule='matched60', device=config['device'],
                            output=OUTPUT, record_video=True)
                        break
                    except Exception as exc:
                        last_error = exc
                        if attempt == 3:
                            raise
                        status('retrying', task=task, condition=condition, attempt=attempt,
                               error=repr(exc), completed_rollouts=len(rows))
                        time.sleep(10)
                    finally:
                        if env is not None:
                            env.close()
                row['tokens_per_finger'] = 16 if condition == 'full16' else 8
                with records.open('a') as stream:
                    stream.write(json.dumps(row)+'\n'); stream.flush(); os.fsync(stream.fileno())
                rows.append(row); completed.add((task, condition)); errors.pop(key, None)
                comparison(rows, metadata, errors)
            except Exception as exc:
                errors[key] = repr(exc)
                atomic_json(OUTPUT/'errors.json', errors)
                comparison(rows, metadata, errors)
        if all((task, condition) in completed for condition in CONDITIONS):
            status('uploading_task', task=task, completed_rollouts=len(rows))
            commit = upload_task(task, rows, metadata, errors)
            status('task_uploaded', task=task, completed_rollouts=len(rows), commit=commit)
    final = comparison(rows, metadata, errors)
    operations = [CommitOperationAdd(path_in_repo=f'{PREFIX}/{name}', path_or_fileobj=str(OUTPUT/name))
                  for name in ('metadata.json', 'comparison.json', 'REPORT.md')]
    commit = retry(lambda: api.create_commit(repo_id=REPO, repo_type='dataset', operations=operations,
        commit_message='Finalize Stage 3 inference-time token-pruning comparison'))
    phase = 'complete' if final['completed_rollouts'] == final['expected_rollouts'] and not errors else 'finished_with_errors'
    status(phase, completed_rollouts=len(rows), expected_rollouts=24, errors=errors,
           commit=commit.commit_url,
           url=f'https://huggingface.co/datasets/{REPO}/tree/main/{PREFIX}')


if __name__ == '__main__':
    main()
