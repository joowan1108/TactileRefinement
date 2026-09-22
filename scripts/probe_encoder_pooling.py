"""Matched frozen-feature reconstruction probes; no policy training or split.

Four conditions: UniVTAC-trained versus released T-Rex DeformEncoder, each
with a full 15x20 grid or adaptive 4x4 pooling. Identical randomly initialized
decoders are trained on the SAME 4,754 audit examples, evaluated on those same
examples, and repeated with independent decoder seeds. This probes information
recoverability under a fixed decoder budget, not held-out generalization.
"""
import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from huggingface_hub import hf_hub_download

from audit_tactile_encoder import AuditMaps, metrics, summarize, paired_improvement, save_panel
from tactile_refinement.representation import DeformEncoder
from smolvla_baseline.data import ROOT

REPO = 'miniFranka/T-Rex_midtrain_mecka23k_ucb100_vqvae_epoch6'
REVISION = '62efb3bcb45a3df0e088c8909d759b582cfb98af'
CONDITIONS = ('ours_full', 'ours_pool4', 'trex_full', 'trex_pool4')


def digest_state(state):
    digest = hashlib.sha256()
    for key, value in sorted(state.items()):
        digest.update(key.encode()); digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def spatial_input(features, pooled):
    # Both branches enter an identically shaped/parameterized decoder. The only
    # difference within each encoder is loss of information due to 4x4 pooling.
    if pooled:
        features = F.interpolate(F.adaptive_avg_pool2d(features, (4, 4)),
                                 size=(15, 20), mode='bilinear', align_corners=False)
    return features


class ProbeDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.low = nn.Sequential(nn.Conv2d(128, 64, 3, padding=1), nn.GELU(),
                                 nn.Conv2d(64, 64, 3, padding=1), nn.GELU())
        self.mid = nn.Sequential(nn.Conv2d(64, 32, 3, padding=1), nn.GELU(),
                                 nn.Conv2d(32, 16, 3, padding=1), nn.GELU())
        self.output = nn.Conv2d(16, 1, 3, padding=1)

    def forward(self, x):
        x = self.low(x)
        x = self.mid(F.interpolate(x, size=(60, 80), mode='bilinear', align_corners=False))
        return self.output(F.interpolate(x, size=(240, 320), mode='bilinear', align_corners=False))


def weighted_loss(pred, target, valid):
    weights = valid.float()*(1+4*(target.abs() > .001))
    return ((pred-target).abs()*weights).sum()/weights.sum().clamp_min(1)


def load_encoders(output):
    ours_path = ROOT/'runs/tactile_refinement/encoder/final.pt'
    saved = torch.load(ours_path, map_location='cpu', weights_only=True, mmap=True)
    assert saved['stage'] == 'encoder' and saved['step'] == 110000
    ours = {k.removeprefix('representation.encoder.'): v for k,v in saved['model'].items()
            if k.startswith('representation.encoder.')}
    extracted = output/'trex_deform_encoder.pt'
    if extracted.exists():
        trex_path = str(extracted)
        upstream = torch.load(trex_path, map_location='cpu', weights_only=True, mmap=True)
        assert upstream['repo'] == REPO and upstream['revision'] == REVISION
        trex = upstream['encoder']
    else:
        trex_path = hf_hub_download(REPO, 'model.pt', revision=REVISION)
        upstream = torch.load(trex_path, map_location='cpu', weights_only=True, mmap=True)
        trex = {k.removeprefix('deform_encoder.'): v for k,v in upstream.items() if k.startswith('deform_encoder.')}
    if not trex:
        raise ValueError(f'No deform_encoder weights: top-level keys {list(upstream)[:15]}')
    result = {}
    provenance = {'ours_path': str(ours_path), 'ours_step': saved['step'],
                  'trex_repository': REPO, 'trex_revision': REVISION, 'trex_path': trex_path,
                  'tensor_hashes': {}, 'parameters': {}}
    for name, state in (('ours', ours), ('trex', trex)):
        model = DeformEncoder()
        model.load_state_dict(state, strict=True)
        model.eval().requires_grad_(False)
        result[name] = model.cuda()
        provenance['tensor_hashes'][name] = digest_state(model.state_dict())
        provenance['parameters'][name] = sum(p.numel() for p in model.parameters())
    assert provenance['tensor_hashes']['ours'] != provenance['tensor_hashes']['trex']
    if not extracted.exists():
        torch.save({'encoder': trex, 'repo': REPO, 'revision': REVISION}, extracted)
    (output/'provenance.json').write_text(json.dumps(provenance, indent=2)+'\n')
    return result


def cache_features(dataset, encoders, output):
    cache_path = output/'feature_cache.pt'
    if cache_path.exists():
        return torch.load(cache_path, weights_only=True, mmap=True)
    maps, masks = [], []
    features = {k: [] for k in encoders}
    for depth, valid, indices in DataLoader(dataset, batch_size=16, num_workers=0):
        maps.append(depth); masks.append(valid)
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            for name, encoder in encoders.items():
                z = encoder(depth.flatten(0, 1).unsqueeze(1).cuda())
                assert z.shape[1:] == (128, 15, 20)
                features[name].append(z.reshape(len(depth), 2, 128, 15, 20).half().cpu())
        if int(indices[-1]) % 800 < 16:
            print(json.dumps({'phase': 'cache', 'completed': int(indices[-1])+1, 'total': len(dataset)}), flush=True)
    cache = {'maps': torch.cat(maps), 'masks': torch.cat(masks),
             'features': {k: torch.cat(v) for k,v in features.items()}}
    assert torch.isfinite(cache['maps']).all()
    for value in cache['features'].values():
        assert torch.isfinite(value).all()
    torch.save(cache, cache_path)
    return cache


def evaluate(models, features, maps, masks, metadata, output, seed, step):
    all_rows, results = {}, {}
    for name, decoder in models.items():
        decoder.eval()
        rows = []
        with torch.inference_mode():
            for start in range(0, len(maps), 16):
                stop = min(start+16, len(maps))
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    pred = decoder(features[name][start:stop].flatten(0, 1)).float().reshape(-1, 2, 240, 320)
                for j in range(stop-start):
                    index = start+j
                    rows.append(dict(metadata[index], sample_index=index,
                                     **metrics(pred[j], maps[index], masks[index])))
        result = {'condition': name, 'seed': seed, 'step': step,
                  'by_kind': {k: summarize([r for r in rows if r['kind'] == k]) for k in ('current','change')},
                  'small_change': summarize([r for r in rows if r['kind'] == 'change' and 0 < r['active_fraction'] <= .05]),
                  'by_task_change': {task: summarize([r for r in rows if r['task'] == task and r['kind'] == 'change'])
                                     for task in sorted({r['task'] for r in rows})}}
        stem = f'{name}_seed{seed}_step{step}'
        (output/f'{stem}.json').write_text(json.dumps(result, indent=2)+'\n')
        with (output/f'{stem}.csv').open('w') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
        all_rows[name], results[name] = rows, result
        # Identical samples across conditions: prior HDMI example plus worst
        # small-change example selected using the target's activity, not errors.
        selected = [854]
        small = [r for r in rows if r['kind'] == 'change' and 0 < r['active_fraction'] <= .05]
        selected.append(max(small, key=lambda r: r['zero_mae'])['sample_index'])
        for index in selected:
            with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                pred = decoder(features[name][index]).float().reshape(2,240,320).cpu().numpy()
            save_panel(output/f'{stem}_sample{index}.png', maps[index].cpu().numpy(), pred, stem+f' | sample {index}')
        print(json.dumps({'phase': 'evaluated', **result}), flush=True)
        decoder.train()
    comparisons = {}
    for old, new in [('ours_pool4','ours_full'),('trex_pool4','trex_full'),
                     ('ours_pool4','trex_pool4'),('ours_full','trex_full')]:
        comparisons[f'{old}_to_{new}'] = {k: paired_improvement(all_rows[old], all_rows[new], k)
                                          for k in ('current','change')}
    return {'seed': seed, 'step': step, 'results': results, 'comparisons': comparisons}


def write_report(output, evaluations, recipe):
    (output/'report.json').write_text(json.dumps({'recipe': recipe, 'evaluations': evaluations}, indent=2)+'\n')
    lines = ['# Frozen tactile encoder and pooling comparison', '',
             'Matched reconstruction probes on the same 4,754 examples from all 800 training episodes. No split. '
             'Decoder training and diagnostics use these same examples. This is an information-recovery probe, not a generalization or policy-success test.', '',
             'Encoders and their BatchNorm statistics stay frozen. Identical decoder parameter counts, initialization, batches, optimizer and objective across all four conditions. '
             'Each channel is standardized with statistics from this diagnostic corpus before the matched decoder. '
             'Both encoders receive our normalized current depth or signed t+50 difference directly. Negative differences are outside T-Rex current-map pretraining input range.', '',
             '| Seed | Updates | Condition | Current MAE | Change MAE | Small-change active MAE | Small-change recall | Small-change IoU |',
             '|---:|---:|---|---:|---:|---:|---:|---:|']
    for item in evaluations:
        for name,r in item['results'].items():
            s=r['small_change']
            lines.append(f"| {item['seed']} | {item['step']} | {name} | {r['by_kind']['current']['mae']:.5f} | {r['by_kind']['change']['mae']:.5f} | {s['active_mae']:.5f} | {100*s['recall_0.01']:.1f}% | {s['iou_0.01']:.3f} |")
    lines += ['', 'The full-grid conditions use 15×20 features per finger; pooled conditions use 4×4, then interpolate back to the same decoder input size. '
              'The learned policy projector is excluded from every condition to isolate the pooling operation. '
              'Comparisons to the original 110k checkpoint decoder are not matched-budget comparisons.', '',
              'These finite decoder budgets do not prove maximal information capacity. The UniVTAC encoder was trained on this corpus whereas T-Rex was trained on a different sensor; '
              'a poorer T-Rex result here does not measure the quality of its original policy.', '',
              'JSON and CSV files contain per-task and per-example results; report.json includes episode-clustered paired MAE comparisons. '
              'PNG panels use the same target examples and color scales across conditions.', '']
    (output/'REPORT.md').write_text('\n'.join(lines))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'runs/diagnostics/encoder_pooling_probe')
    parser.add_argument('--steps',type=int,default=2000)
    parser.add_argument('--seeds',type=int,nargs='+',default=[42,314])
    parser.add_argument('--batch-size',type=int,default=16)
    args=parser.parse_args()
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark=False
    args.output.mkdir(parents=True,exist_ok=True)
    if (args.output/'report.json').exists():
        raise FileExistsError('Use a fresh output directory to avoid replacing probe results')
    dataset=AuditMaps(ROOT/'data/training_manifest.json')
    assert len(dataset)==4754
    (args.output/'sample_indices.json').write_text(json.dumps(dataset.samples,indent=2)+'\n')
    encoders=load_encoders(args.output)
    cache=cache_features(dataset,encoders,args.output)
    del encoders
    maps,masks=cache['maps'].cuda(),cache['masks'].cuda()
    features,stats={},{}
    for name in CONDITIONS:
        source=cache['features'][name.split('_')[0]].flatten(0,1).cuda()
        source=spatial_input(source, name.endswith('pool4'))
        # Compute float32 moments batchwise to avoid a second full FP32 cache.
        sums=torch.zeros(128,device='cuda'); squares=torch.zeros_like(sums); count=0
        for block in source.split(128):
            f=block.float(); sums+=f.sum((0,2,3)); squares+=f.square().sum((0,2,3)); count+=f.shape[0]*300
        mean=sums/count; std=(squares/count-mean.square()).clamp_min(1e-8).sqrt().clamp_min(.001)
        for block in source.split(128):
            block.copy_((block.float()-mean[None,:,None,None])/std[None,:,None,None])
        features[name]=source.reshape(len(maps),2,128,15,20)
        stats[name]={'mean':mean.cpu(),'std':std.cpu()}
    del cache
    torch.save(stats,args.output/'feature_normalization.pt')
    recipe={'steps':args.steps,'seeds':args.seeds,'batch_size':args.batch_size,'lr':.001,
            'samples':len(dataset),'encoder_grid':[15,20],'pooled_grid':[4,4],
            'target':'normalized current depth and signed D[t+50]-D[t]',
            'train_eval_same_examples':True,'no_policy_updates':True}
    (args.output/'recipe.json').write_text(json.dumps(recipe,indent=2)+'\n')
    evaluations=[]
    for seed in args.seeds:
        torch.manual_seed(seed)
        template=ProbeDecoder().cuda()
        models={k:copy.deepcopy(template) for k in CONDITIONS}; del template
        opts={k:torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.0001) for k,m in models.items()}
        rng=torch.Generator(device='cuda').manual_seed(seed)
        for step in range(1,args.steps+1):
            began=time.monotonic()
            indices=torch.randint(len(maps),(args.batch_size,),device='cuda',generator=rng)
            target,valid=maps[indices].flatten(0,1).unsqueeze(1),masks[indices].flatten(0,1).unsqueeze(1)
            losses={}
            for name,decoder in models.items():
                opts[name].zero_grad(set_to_none=True)
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    pred=decoder(features[name][indices].flatten(0,1))
                    loss=weighted_loss(pred.float(),target,valid)
                if not torch.isfinite(loss):raise FloatingPointError(name)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(decoder.parameters(),10,error_if_nonfinite=True)
                opts[name].step(); losses[name]=float(loss.detach())
            torch.cuda.synchronize()
            if step%50==0 or step==1:
                row={'phase':'train','seed':seed,'step':step,'seconds':time.monotonic()-began,'losses':losses}
                print(json.dumps(row),flush=True)
                with (args.output/'train.jsonl').open('a') as stream:stream.write(json.dumps(row)+'\n')
                (args.output/'status.json').write_text(json.dumps(row)+'\n')
            if step in (args.steps//2,args.steps):
                evaluations.append(evaluate(models,features,maps,masks,dataset.samples,args.output,seed,step))
                write_report(args.output,evaluations,recipe)
        torch.save({k:v.state_dict() for k,v in models.items()},args.output/f'decoders_seed{seed}.pt')
        del models,opts
    (args.output/'status.json').write_text(json.dumps({'phase':'complete','seeds':args.seeds,'steps':args.steps})+'\n')


if __name__=='__main__':
    main()
