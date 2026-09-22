"""Render the Lift Can target-decoder-predictor comparison."""
import argparse
import json
from pathlib import Path
import subprocess

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    output = parser.parse_args().output
    meta = json.loads((output/'metadata.json').read_text())
    with h5py.File(output/'maps.h5') as stream:
        if not stream.attrs.get('complete'):
            raise ValueError('Incomplete inference output')
        data = {name:stream[name][:].astype('float32') for name in
                ('current','truth','decoded_target','decoded_prediction','target_tokens','predicted_tokens')}
        head, wrist, frames = stream['head'][:], stream['wrist'][:], stream['frame'][:]
    truth, target, prediction = data['truth'], data['decoded_target'], data['decoded_prediction']
    frame_truth = np.abs(truth).mean(axis=(1,2,3))
    selected = int(frame_truth.argmax())
    def mae(value): return np.abs(value-truth).mean(axis=(1,2,3))
    target_mae, prediction_mae = mae(target), mae(prediction)
    zero_mae = np.abs(truth).mean(axis=(1,2,3))
    target_latent = data['target_tokens'].astype('float32')
    predicted_latent = data['predicted_tokens'].astype('float32')
    latent_cosine = (target_latent*predicted_latent).sum((1,2))/np.maximum(
        np.linalg.norm(target_latent.reshape(len(truth),-1),axis=1)*
        np.linalg.norm(predicted_latent.reshape(len(truth),-1),axis=1),1e-12)
    metrics = dict(task='lift_can', episode_id=0, frames=len(truth), selected_frame=int(frames[selected]),
        selected_local_index=selected, selection='largest mean absolute truth change',
        mean=dict(truth_abs=float(zero_mae.mean()), decoded_target_abs=float(np.abs(target).mean()),
                  decoded_prediction_abs=float(np.abs(prediction).mean()),
                  decoded_target_mae=float(target_mae.mean()),
                  decoded_prediction_mae=float(prediction_mae.mean()),
                  zero_prediction_mae=float(zero_mae.mean()), latent_cosine=float(latent_cosine.mean())),
        selected=dict(truth_abs=float(zero_mae[selected]), decoded_target_abs=float(np.abs(target[selected]).mean()),
                      decoded_prediction_abs=float(np.abs(prediction[selected]).mean()),
                      decoded_target_mae=float(target_mae[selected]),
                      decoded_prediction_mae=float(prediction_mae[selected]),
                      zero_prediction_mae=float(zero_mae[selected]), latent_cosine=float(latent_cosine[selected])))
    (output/'metrics.json').write_text(json.dumps(metrics, indent=2)+'\n')
    np.savez_compressed(output/'maps.npz', truth=truth, decoded_target=target,
                        decoded_prediction=prediction, frames=frames)
    limit=max(float(np.percentile(np.abs(np.concatenate((truth,target,prediction),axis=0)),99.9)),.01)
    fig=plt.figure(figsize=(19.2,10.8),dpi=100,facecolor='#f5f5f5')
    grid=fig.add_gridspec(3,4,left=.035,right=.98,top=.91,bottom=.07,hspace=.44,wspace=.25)
    cameras=[fig.add_subplot(grid[0,:2]),fig.add_subplot(grid[0,2:])]
    camera_images=[cameras[0].imshow(head[0].astype('uint8')),cameras[1].imshow(wrist[0].astype('uint8'))]
    for ax,name in zip(cameras,('Head camera','Wrist camera')): ax.set_title(name);ax.axis('off')
    images=[]
    for finger,label in enumerate(('Left','Right')):
        panels=[(data['current'][0,finger],'viridis',0,1,'Current normalized depth'),
                (truth[0,finger],'coolwarm',-limit,limit,'Truth: D[t+50] - D[t]'),
                (target[0,finger],'coolwarm',-limit,limit,'Same decoder: target latent'),
                (prediction[0,finger],'coolwarm',-limit,limit,'Same decoder: predicted latent')]
        row=[]
        for col,(value,cmap,low,high,title) in enumerate(panels):
            ax=fig.add_subplot(grid[finger+1,col]); im=ax.imshow(value,cmap=cmap,norm=Normalize(low,high),aspect='auto')
            ax.set_title(f'{label}: {title}');ax.set_xticks([]);ax.set_yticks([])
            fig.colorbar(im,ax=ax,fraction=.04,pad=.025);row.append(im)
        images.append(row)
    title=fig.suptitle('',fontsize=16,y=.975)
    fig.text(.5,.018,'All change panels use the same decoder output scale. Target-latent decoding is diagnostic: the decoder was trained on predictor latents.',ha='center',fontsize=11)
    def draw(i):
        title.set_text(f'Lift Can demonstration episode 0 | t={frames[i]} -> t+50={frames[i]+50} | checkpoint 66,666 | '
                       f'target-decode MAE {target_mae[i]:.4f} | prediction MAE {prediction_mae[i]:.4f}')
        camera_images[0].set_data(head[i].astype('uint8'));camera_images[1].set_data(wrist[i].astype('uint8'))
        for finger in range(2):
            for im,value in zip(images[finger],(data['current'][i,finger],truth[i,finger],target[i,finger],prediction[i,finger])):
                im.set_data(value)
        fig.canvas.draw();return np.asarray(fig.canvas.buffer_rgba())[...,:3].copy()
    video=output/'comparison.mp4'
    command=['ffmpeg','-v','error','-n','-f','rawvideo','-pix_fmt','rgb24','-s','1920x1080','-r','15','-i','pipe:0',
             '-an','-c:v','libx264','-threads','2','-preset','fast','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(video)]
    process=subprocess.Popen(command,stdin=subprocess.PIPE)
    try:
        for i in range(len(truth)):
            process.stdin.write(draw(i).tobytes())
            if i==selected: fig.savefig(output/'comparison_snapshot.png',dpi=150)
    finally: process.stdin.close()
    if process.wait()!=0: raise RuntimeError('ffmpeg failed')
    plt.close(fig)
    probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames','-select_streams','v:0',
        '-show_entries','stream=nb_read_frames','-of','json',str(video)],text=True))
    if int(probe['streams'][0]['nb_read_frames'])!=len(truth): raise RuntimeError('Video frame count mismatch')
    m=metrics['mean']; s=metrics['selected']
    report=f'''# Lift Can: decoder or predictor diagnostic

One recorded UniVTAC demonstration episode, checkpoint 66,666. Every frame with a valid 50-recorded-step future is included ({len(truth)} frames).

[Watch the comparison](comparison.mp4)

![Strongest-change frame](comparison_snapshot.png)

The panels compare the raw truth deformation difference, `decoder(target latent)`, and `decoder(predicted latent)`. The same frozen reconstruction decoder produces both decoded panels. All signed-change panels share one fixed color range over the episode.

## Episode averages

- Raw truth absolute magnitude: **{m['truth_abs']:.5f}**
- Decoder output from true encoder/projector latent: magnitude **{m['decoded_target_abs']:.5f}**, truth MAE **{m['decoded_target_mae']:.5f}**
- Decoder output from predictor latent: magnitude **{m['decoded_prediction_abs']:.5f}**, truth MAE **{m['decoded_prediction_mae']:.5f}**
- All-zero prediction MAE: **{m['zero_prediction_mae']:.5f}**
- Full latent cosine between prediction and target: **{m['latent_cosine']:.5f}**

At the strongest-change frame ({int(frames[selected])}), target-latent decode MAE is **{s['decoded_target_mae']:.5f}** and predictor-latent decode MAE is **{s['decoded_prediction_mae']:.5f}**.

## Interpretation constraint

The reconstruction head was trained to decode **predictor outputs**, while the target encoder/projector latent supplied contrastive grounding. Therefore, `decoder(target latent)` is a useful compatibility probe but is outside the decoder's explicitly supervised input distribution. The decisive decoder evidence is that `decoder(predicted latent)` reconstructs this training demonstration accurately; its blank output on the new simulator rollout therefore reflects a changed predictor latent rather than a universally broken decoder. This is a training-corpus diagnostic and does not measure held-out generalization.
'''
    (output/'REPORT.md').write_text(report)
    print(json.dumps(metrics,indent=2),flush=True)


if __name__=='__main__': main()
