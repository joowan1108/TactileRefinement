"""Render one recorded episode; no policy or simulator execution."""
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


def probe_frames(path, expected):
    result = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-count_frames',
        '-select_streams', 'v:0', '-show_entries', 'stream=nb_read_frames', '-of', 'json', str(path)], text=True))
    if int(result['streams'][0]['nb_read_frames']) != expected:
        raise ValueError(f'Incorrect frame count: {path}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    output = parser.parse_args().output
    row = json.loads((output/'episode.json').read_text())
    meta = json.loads((output/'metadata.json').read_text())
    task_title = meta['task'].replace('_', ' ').title().replace('Hdmi', 'HDMI')
    with h5py.File(output/'diagnostics.h5', 'r') as stream:
        if not stream.attrs.get('complete'):
            raise ValueError('Refuse to render an incomplete episode')
        depth = stream['depth'][:].astype('float32')
        change = stream['predicted_change'][:]
        # Head mean retains layers, fingers, queries and spatial positions.
        attention = stream['attention'][:].astype('float32').mean(axis=3)
        attention /= attention.sum(-1, keepdims=True)
        head, wrist = stream['head'][:], stream['wrist'][:]
        times = stream['simulated_seconds'][:]
    n = len(depth)
    if n != row['steps'] or attention.shape != (n, 2, 2, 16, 300):
        raise ValueError('Unexpected recorded dimensions')
    final = attention[:, -1]
    unit = final / np.maximum(np.linalg.norm(final, axis=-1, keepdims=True), 1e-12)
    similarities = np.einsum('tfqd,tfrd->tfqr', unit, unit)
    pairs = np.triu_indices(16, 1)
    query_similarity = similarities[:, :, pairs[0], pairs[1]]
    entropy = -(final * np.log(np.maximum(final, 1e-30))).sum(-1)/np.log(300)
    most_focused = entropy.argmin(-1)
    mean = final.mean(axis=2).reshape(n, 2, 15, 20)*300
    focused = np.stack([final[t, np.arange(2), most_focused[t]] for t in range(n)]).reshape(n, 2, 15, 20)*300
    # Select a frame by tactile temporal variation, independent of attention.
    variation_by_finger = np.abs(depth-depth[:1]).mean(axis=(2, 3))
    variation = variation_by_finger.mean(1)
    selected = int(variation.argmax())
    depth_limit = max(float(depth.max()), .01)
    change_limit = max(float(np.abs(change).max()), 1e-4)
    attention_limit = max(3., float(np.percentile(final*300, 99.9)))
    metrics = dict(task=meta['task'], frames=n, success=row['success'], selected_frame=selected,
        selection='Largest mean absolute normalized-depth change from the first observation; visualization selection only',
        selected_simulated_seconds=float(times[selected]),
        color_ranges=dict(normalized_depth=[0, depth_limit], predicted_change=[-change_limit, change_limit],
                          attention_times_uniform=[0, attention_limit]),
        attention_reference='Uniform attention = 1/300 per token; plotted weights multiplied by 300',
        fingers={})
    for f, name in enumerate(('left', 'right')):
        y, x = np.mgrid[:15, :20]
        weights = final[:, f].mean(1).reshape(n, 15, 20)
        metrics['fingers'][name] = dict(
            query_head_mean_normalized_entropy_mean=float(entropy[:, f].mean()),
            query_head_mean_normalized_entropy_min=float(entropy[:, f].min()),
            pairwise_query_attention_cosine_mean=float(query_similarity[:, f].mean()),
            pairwise_query_attention_cosine_frame_mean_min=float(query_similarity[:, f].mean(-1).min()),
            mean_attention_peak_times_uniform=float(mean[:, f].max()),
            any_query_peak_times_uniform=float((final[:, f]*300).max()),
            mean_attention_centroid_row=float((weights*y).sum((1, 2)).mean()),
            mean_attention_centroid_col=float((weights*x).sum((1, 2)).mean()),
            largest_depth_change_from_first=float(variation_by_finger[:, f].max()),
            predicted_change_abs_mean=float(np.abs(change[:, f]).mean()),
            depth_range=[float(depth[:, f].min()), float(depth[:, f].max())])
    (output/'attention_metrics.json').write_text(json.dumps(metrics, indent=2)+'\n')
    np.savez_compressed(output/'attention_head_mean.npz', attention=attention,
                        simulated_seconds=times, most_focused_query=most_focused)

    plt.rcParams.update({'font.size': 10, 'axes.titlesize': 11})
    fig = plt.figure(figsize=(19.2, 10.8), dpi=100, facecolor='#f5f5f5')
    grid = fig.add_gridspec(3, 4, left=.035, right=.98, top=.92, bottom=.06,
                          hspace=.46, wspace=.27)
    cameras = [fig.add_subplot(grid[0, :2]), fig.add_subplot(grid[0, 2:])]
    camera_images = [cameras[0].imshow(head[0]), cameras[1].imshow(wrist[0])]
    for ax, name in zip(cameras, ('Head camera', 'Wrist camera')):
        ax.set_title(name); ax.axis('off')
    images = []
    for f, name in enumerate(('Left', 'Right')):
        panels = [(depth[0, f], 'viridis', 0, depth_limit, 'Current normalized depth', 'normalized depth'),
                  (change[0, f], 'coolwarm', -change_limit, change_limit, 'Decoded predicted future change', 'signed normalized-depth change'),
                  (mean[0, f], 'magma', 0, attention_limit, 'Resampler attention: mean of 16 queries', 'attention / uniform'),
                  (focused[0, f], 'magma', 0, attention_limit, 'Most focused query (head mean)', 'attention / uniform')]
        finger_images = []
        for col, (data, cmap, low, high, title, label) in enumerate(panels):
            ax = fig.add_subplot(grid[f+1, col])
            im = ax.imshow(data, cmap=cmap, norm=Normalize(low, high), interpolation='nearest', aspect='auto')
            ax.set_title(f'{name}: {title}'); ax.set_xticks([]); ax.set_yticks([])
            bar = fig.colorbar(im, ax=ax, fraction=.04, pad=.025)
            bar.set_label(label, fontsize=8); bar.ax.tick_params(labelsize=8)
            finger_images.append(im)
        images.append(finger_images)
    title = fig.suptitle('', fontsize=17, y=.98)
    fig.text(.5, .018, 'Attention source: predicted future-change tokens (15 x 20 per finger). Final resampler layer. '
             'Weights show attention, not causal attribution. 4x slower playback.', ha='center', fontsize=11)
    def draw(t):
        title.set_text(f'{task_title} | checkpoint {meta["checkpoint"]["step"]:,} | seed {meta["seed"]} | decision {t} | '
                       f'simulation {times[t]:.3f} s | episode success: {row["success"]}')
        for im, data in zip(camera_images, (head[t], wrist[t])):
            im.set_data(data)
        for f in range(2):
            for im, data in zip(images[f], (depth[t, f], change[t, f], mean[t, f], focused[t, f])):
                im.set_data(data)
            images[f][3].axes.set_title(f'{("Left", "Right")[f]}: focused query #{most_focused[t, f]} '
                f'(entropy {entropy[t, f, most_focused[t, f]]:.3f})')
        fig.canvas.draw()
        return np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    video = output/'attention_visualization.mp4'
    command = ['ffmpeg', '-v', 'error', '-n', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
               '-s', '1920x1080', '-r', '15', '-i', 'pipe:0', '-an', '-c:v', 'libx264',
               '-threads', '2', '-preset', 'fast', '-crf', '20', '-pix_fmt', 'yuv420p',
               '-movflags', '+faststart', str(video)]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    try:
        for t in range(n):
            frame = draw(t)
            process.stdin.write(frame.tobytes())
            if t == selected:
                fig.savefig(output/'attention_snapshot.png', dpi=150)
            if t % 50 == 0:
                print(json.dumps(dict(phase='rendering', frame=t, total=n)), flush=True)
    finally:
        process.stdin.close()
    if process.wait() != 0:
        raise RuntimeError('ffmpeg failed')
    plt.close(fig)
    probe_frames(video, n)
    probe_frames(output/row['video'], n+1)
    for f, name in enumerate(('left', 'right')):
        fig, axes = plt.subplots(4, 4, figsize=(12, 10), layout='constrained')
        for query, ax in enumerate(axes.flat):
            im = ax.imshow(final[selected, f, query].reshape(15, 20)*300, cmap='magma',
                           vmin=0, vmax=attention_limit, interpolation='nearest')
            ax.set_title(f'Query {query}: H/Hmax={entropy[selected, f, query]:.3f}')
            ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(im, ax=axes.ravel().tolist(), shrink=.75, label='Attention / uniform (1 = uniform)')
        fig.suptitle(f'{name.capitalize()} finger: 16 learned queries, head mean, final layer\n'
                     f'{task_title} decision {selected}; attention to predicted future-change tokens')
        fig.savefig(output/f'{name}_queries.png', dpi=150)
        plt.close(fig)
    report = f'''# {task_title}: tactile depth and trained resampler attention

One complete episode, checkpoint **{meta['checkpoint']['step']:,}**, seed **{meta['seed']}**. Simulator success: **{row['success']}**.
Recorded {n} policy decisions, {row['physics_steps']} physics steps, {row['simulated_seconds']:.3f} simulated seconds.

[Watch the visualization](attention_visualization.mp4) | [Static snapshot](attention_snapshot.png)

![Episode snapshot](attention_snapshot.png)

## What is shown

The two cameras and both tactile sensors are synchronized to the observation **before** each action. Depth is clip((34-H_mm)/10,0,1), with invalid values set to zero. No episode-start subtraction enters the policy. These are normalized depth maps, not calibrated indentation measurements.

The reconstruction head decodes the predictor's 600 future-change tokens to a signed 60x80 map per finger, for visualization only. This is the predicted 50-recorded-step change, not an observed future image or a new model input.

The resampler consumes those **predicted future-change tokens**. The final layer's cross-attention is plotted on its ordered 15x20 source grid. The mean panel averages eight heads and all 16 queries; the focused panel shows the lowest-entropy query after averaging heads. Its query ID is labeled and may change between frames. All 16 query maps are saved for each finger at decision {selected}.

Attention values are divided by uniform attention (1/300): a plotted value of 1 means uniform weight. Color scales are fixed over this entire episode; attention above {attention_limit:.3f} is clipped visually, and the un-clipped weights are in attention_head_mean.npz. Entropy is normalized to [0,1], with 1 meaning uniform attention.

The attention grids refer to **predicted latent positions**, not direct attention to raw depth pixels. Encoder receptive fields and predictor mixing prevent interpreting them as exact pixel-level causes. Attention weights alone do not prove causal importance. The current depth maps are displayed alongside them to support inspection.

## Execution and verification

Existing matched60 schedule: fresh 50-action prediction and temporal aggregation every logical action; midpoint/endpoint position commands across two 120 Hz physics steps, with early termination allowed. The new tactile10_vl50 schedule was not run. The visualization plays at 15 fps (4x slower than policy time); the separate RGB rollout uses 60 fps.

Passive hooks read Q and K without changing the resampler's forward computation. Unit tests compare the shadow readout with PyTorch attention and verify unchanged resampler outputs. The diagnostic softmax is calculated in float32 and can differ slightly from fused mixed-precision numerics. No readout or decoded map is fed back to the policy. Both attention layers are retained in the downloadable head-mean arrays; full per-head weights and sensor frames remain in the local diagnostics.h5.

The snapshot is selected by the largest mean absolute tactile-depth change from the initial observation, purely for visualization. This is not a contact classifier or an episode-start normalization.

## Measured attention

'''
    for name, values in metrics['fingers'].items():
        report += (f"- **{name.capitalize()}**: query/head-mean entropy averages {values['query_head_mean_normalized_entropy_mean']:.4f}; "
                   f"lowest query entropy {values['query_head_mean_normalized_entropy_min']:.4f}. "
                   f"Maximum mean-query attention is {values['mean_attention_peak_times_uniform']:.2f}x uniform; "
                   f"maximum individual-query attention is {values['any_query_peak_times_uniform']:.2f}x uniform. "
                   f"Mean pairwise query-attention cosine similarity is {values['pairwise_query_attention_cosine_mean']:.5f}.\n")
    report += ('\nCosine similarity averages the 120 distinct query pairs within each finger and then all decision frames, '
               'using head-averaged attention distributions. Values near 1 suggest overlapping spatial attention; '
               'they do not establish identical output tokens, and head averaging can hide head-specific differences. '
               'One episode per task cannot establish a task success rate or causal benefit from tactile guidance.\n')
    report += '\n![Left queries](left_queries.png)\n\n![Right queries](right_queries.png)\n'
    (output/'REPORT.md').write_text(report)
    print(json.dumps(dict(phase='rendered', frames=n, selected_frame=selected, metrics=metrics['fingers'])), flush=True)


if __name__ == '__main__':
    main()
