"""Compare Lift Can truth, decoded target latent, and decoded prediction."""
import argparse
import json
import os
from pathlib import Path
import subprocess

import cv2
import h5py
import numpy as np
import torch
from torch.utils.data import default_collate

from smolvla_baseline.data import ROOT
from tactile_refinement.ablation import atomic_json, autocast, load_stage2
from tactile_refinement.data import TactileEpisodes, prepare_refinement_batch
from tactile_refinement.losses import coarse_change_target
from tactile_refinement.model import encode_prefix
from tactile_refinement.train import file_sha256

REPO = 'sungkyunner/univtac-stage2-tactile-ablation-checkpoint-33333'


def append(stream, name, value):
    value = np.asarray(value)
    if name not in stream:
        stream.create_dataset(name, shape=(0, *value.shape), maxshape=(None, *value.shape),
                              chunks=(1, *value.shape), dtype=value.dtype,
                              compression='gzip', compression_opts=1)
    item = stream[name]
    item.resize(len(item)+1, axis=0)
    item[-1] = value


def upload(output):
    from huggingface_hub import CommitOperationAdd, HfApi
    names = ['comparison.mp4', 'comparison_snapshot.png', 'metrics.json', 'REPORT.md',
             'metadata.json', 'maps.npz']
    prefix = 'checkpoint_66666/diagnostics/lift_can_decoder_episode_0'
    api = HfApi()
    if not api.repo_info(REPO, repo_type='dataset').private:
        raise ValueError('Expected private repository')
    result = api.create_commit(repo_id=REPO, repo_type='dataset',
        operations=[CommitOperationAdd(path_in_repo=f'{prefix}/{name}',
                    path_or_fileobj=str(output/name)) for name in names],
        commit_message='Add Lift Can decoder versus predictor diagnostic')
    remote = {x.path: x.size for x in api.list_repo_tree(REPO, repo_type='dataset',
              path_in_repo=prefix, recursive=True) if hasattr(x, 'size')}
    for name in names:
        if remote.get(f'{prefix}/{name}') != (output/name).stat().st_size:
            raise RuntimeError(f'Upload verification failed: {name}')
    return result.commit_url


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--upload', action='store_true')
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    status = lambda phase, **kw: atomic_json(output/'status.json', dict(phase=phase, **kw))
    try:
        torch.set_num_threads(4)
        status('loading')
        checkpoint_path = ROOT/'runs/tactile_refinement_trex_fullgrid_n0/expert/checkpoint_2.pt'
        model, pre, _, config, checkpoint = load_stage2(checkpoint_path)
        if checkpoint['step'] != 66666:
            raise ValueError('Expected checkpoint 66,666')
        dataset = TactileEpisodes(ROOT/config['manifest'])
        start = 0
        selected = None
        for index, episode in enumerate(dataset.episodes):
            if episode['task'] == 'lift_can' and episode['episode_id'] == 0:
                selected = (index, start, episode)
                break
            start = dataset.ends[index]
        if selected is None:
            raise ValueError('Lift Can episode 0 is absent')
        segment, start, episode = selected
        count = episode['frames']-50
        if count <= 0:
            raise ValueError('Episode is shorter than prediction horizon')
        status('inference', frames=count)
        raw_path = ROOT/episode['path']
        with h5py.File(raw_path, 'r') as source, h5py.File(output/'maps.h5', 'w') as stream:
            for local in range(count):
                raw = dataset[start+local]
                if raw['segment'] != segment or not raw['future_valid']:
                    raise RuntimeError('Dataset episode alignment failed')
                batch, tactile = prepare_refinement_batch(default_collate([raw]), pre, config['device'])
                with torch.inference_mode(), autocast(config['device']):
                    prefix = encode_prefix(model.baseline, batch)
                    target_tokens = model.representation(tactile['change'])
                    predicted_tokens = model.predicted_tokens(tactile['depth'], prefix)
                    decoded_target = model.decoder(target_tokens).float()
                    decoded_prediction = model.decoder(predicted_tokens).float()
                truth, mask = coarse_change_target(tactile['change'], tactile['change_mask'],
                                                   tactile['future_valid'])
                if not mask.all():
                    raise ValueError('Unexpected invalid Lift Can depth pixels')
                frame = local+episode.get('frame_offset', 0)
                cameras = []
                for name in ('head', 'wrist'):
                    encoded = source[f'observation/{name}/rgb'][frame]
                    image = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
                    cameras.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                values = dict(current=tactile['depth'][0].float().cpu().numpy().astype('float16'),
                    truth=truth[0].cpu().numpy().astype('float32'),
                    decoded_target=decoded_target[0].cpu().numpy().astype('float32'),
                    decoded_prediction=decoded_prediction[0].cpu().numpy().astype('float32'),
                    target_tokens=target_tokens[0].float().cpu().numpy().astype('float16'),
                    predicted_tokens=predicted_tokens[0].float().cpu().numpy().astype('float16'),
                    head=cameras[0], wrist=cameras[1], frame=np.int32(frame))
                for name, value in values.items():
                    if not np.isfinite(value).all():
                        raise ValueError(f'Nonfinite value in {name}')
                    append(stream, name, value)
                if (local+1) % 20 == 0:
                    print(json.dumps(dict(phase='inference', completed=local+1, total=count)), flush=True)
            stream.attrs['complete'] = True
        metadata = dict(task='lift_can', episode_id=0, source=str(raw_path.relative_to(ROOT)),
            source_segment=segment, frame_offset=episode.get('frame_offset', 0), frames=count,
            horizon_recorded_steps=50, checkpoint=checkpoint,
            truth='area-mean downsample of D[t+50]-D[t] from 240x320 to 60x80',
            decoded_target='decoder(representation(D[t+50]-D[t]))',
            decoded_prediction='decoder(predictor(representation(D[t]), visual-language context))',
            decoder_training_caveat='The decoder was supervised on predicted tokens, not directly on target encoder tokens',
            code_sha256={name:file_sha256(ROOT/name) for name in
                ('scripts/analyze_lift_can_decoder.py','scripts/render_lift_can_decoder.py',
                 'src/tactile_refinement/model.py','src/tactile_refinement/representation.py')})
        atomic_json(output/'metadata.json', metadata)
        del model
        torch.cuda.empty_cache()
        status('rendering')
        subprocess.run([str(ROOT/'.venv-benchmark/bin/python'),
            str(ROOT/'scripts/render_lift_can_decoder.py'), '--output', str(output)], check=True)
        commit = upload(output) if args.upload else None
        status('complete', commit_url=commit)
        print(json.dumps(dict(phase='complete', commit_url=commit)), flush=True)
    except Exception as exc:
        status('failed', error=repr(exc))
        raise


if __name__ == '__main__':
    main()
