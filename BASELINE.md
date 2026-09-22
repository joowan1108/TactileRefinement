# UniVTAC SmolVLA baseline

This workspace prepares one multitask SmolVLA policy across all eight Isaac Sim 4.5 tasks. The baseline takes head RGB, wrist RGB, task language, and the nine robot joint positions as inputs. Its only training objective is the action flow-matching loss. Tactile images, markers, depth, force, tactile auxiliary losses, and privileged actor state are excluded from the model inputs and the training data reader.

The three-day production run uses all 800 episodes. No demonstration is reserved for validation or testing; policy performance is evaluated with fresh simulator rollouts.

## Data

All 800 source HDF5 episodes and eight task metadata files are available in `data/UniVTAC/isaac45/`: 119,141,467,375 bytes (about 111 GiB). The 800 episodes were already on this computer and were SHA256-verified against the pinned Hugging Face release before being hardlinked here; the eight metadata files were downloaded. Treat these shared raw files as read-only. No contact-pretraining data, Isaac 5.1 data, or other policies' trained weights are needed for this baseline.

- Dataset: [byml/UniVTAC](https://huggingface.co/datasets/byml/UniVTAC), revision `e1aee7b0c95543b535e0146b2de3ee1bc6ddaabd`.
- Simulator code: [univtac/UniVTAC](https://github.com/univtac/UniVTAC), main commit `0dafa10262e22f486f160a55d6f11aeab12d8e7b`.
- Project and task descriptions: [UniVTAC website](https://univtac.github.io/).
- `data/download_manifest.json`: per-file checksums and download/reuse provenance.
- `data/training_manifest.json`: all training episodes, segment boundaries, instructions, and normalization statistics.

All 100 episodes of each of the eight tasks are used for training: 800 source episodes and 152,079 usable observation/action pairs. Normalization is recomputed from this complete training dataset. There is no train/validation partition, validation loader, validation loss, or validation-based checkpoint selection.

The action target at frame `t` is the absolute joint position at `t+1`, following upstream's `HDF5Handler` alignment. A 50-action chunk is truncated and masked at each episode/segment end. Seven arm values are radians; two finger positions are metres. For simulator deployment, the official eight-dimensional `qpos` interface takes seven arm values and the first finger value, broadcasting that gripper command to both fingers.

One source recording, `lift_can/hdf5/62.hdf5`, has a backwards step-counter jump before frame 228. The reader splits it there, excludes the discontinuous transition, and prevents future action chunks from crossing the jump. The raw recording is preserved.

Reverify/download missing files and rebuild the training manifest:

```bash
cd /tactile/new
python3 scripts/download_data.py
.venv-train/bin/python scripts/prepare_data.py
```

The reader accesses JPEGs and joint positions directly from HDF5; an additional LeRobot-format copy is unnecessary.

## Training configuration

`configs/smolvla_3090_72h.json` targets this machine's NVIDIA GeForce RTX 3090 (24 GB). It densely fine-tunes the pretrained `lerobot/smolvla_base` vision encoder, vision connector, language backbone, action expert, and state/action projections. LoRA is disabled. Upstream SmolVLA keeps architecture-unused output layers frozen because they do not participate in the action-loss forward pass. The pretrained model and VLM configuration/tokenizer revisions are pinned. “Only this data” refers to fine-tuning data; initialization is pretrained SmolVLA.

| Setting | Value |
|---|---|
| Microbatch size | 16 |
| Gradient accumulation | 4 |
| Effective batch | 64 on one GPU |
| Precision | bfloat16 autocast |
| RGB preprocessing | UniVTAC export resize to 256 × 256, then native SmolVLA 512 × 512 model input |
| Trainable model | 402,737,376 dense parameters; no LoRA |
| State / action dimensions | 9 / 9, internally padded by SmolVLA |
| Training action chunk | 50 recorded targets |
| Replanning | Execute 10 actions, then refresh policy observation |
| Optimizer | AdamW, peak learning rate 1e-4, betas 0.9/0.95 |
| Schedule | Linear warmup for first 3.33% of runtime, then cosine decay to 2.5e-6 |
| Training budget | 72 hours of elapsed run time |
| Checkpoints | At approximately 24, 48, and 72 hours |
| Data workers | 2, prefetch 1; batches carry uint8 RGB to fit the 64 MB shared-memory limit |

`configs/throughput_3090.json` records the actual GPU timing measurement and projected optimizer updates. This is an estimate; the production loop uses elapsed time rather than a guessed step count. The timer starts after model/data-loader setup and includes checkpoint I/O. Final serialization can make the process finish slightly after 72 hours. The duration is a compute budget, not a guarantee of optimal policy performance.

Exactly three production checkpoint files will be written under `runs/smolvla_full_dense_b16_ga4_72h/`:

```text
checkpoint_24h.pt
checkpoint_48h.pt
final.pt
```

Each `.pt` contains the full model, optimizer state, RNG states, completed update count, elapsed runtime, configuration, full training manifest and normalization statistics. Files are written through a temporary path and atomically renamed. There are no extra best-model checkpoints. No validation is run at checkpoints.

Start the three-day run in a persistent background process:

```bash
cd /tactile/new
bash scripts/start_training.sh
tail -f logs/training.log
```

Or run in the foreground with `bash scripts/train.sh`. Resume after interruption with:

```bash
bash scripts/start_training.sh --resume runs/smolvla_full_dense_b16_ga4_72h/checkpoint_48h.pt
```

Resume continues the saved runtime budget and skips completed checkpoint milestones. It restarts at the saved optimizer update; updates after the most recent checkpoint must be repeated. Existing runs are not overwritten by a fresh launch.

Repeat the GPU timing measurement without creating production checkpoints:

```bash
bash scripts/train.sh --profile 30
```

## Environments and benchmarking

- `.venv-train`: Python 3.12, PyTorch 2.11/CUDA 12.8, the LeRobot SmolVLA implementation vendored by the pinned UniVTAC checkout, and HDF5 loading dependencies. Exact packages are in `configs/training_freeze.txt`.
- `.venv-benchmark`: Python 3.10 using the existing compiled Isaac Sim 4.5, Isaac Lab 2.1.1, TacEx, and cuRobo installation on this computer. Its `.pth` file references `/tactile/contrast/.venv-univtac/lib/python3.10/site-packages`; compiled/editable dependencies also reference `/tactile/contrast/vendor/UniVTAC`. These directories must remain available. The setup does not modify that existing installation.
- The local benchmark runs the new pinned `UniVTAC/envs` task code, which matches the existing installation's task code. `configs/benchmark_environment.json`, `configs/benchmark_freeze.txt`, and `configs/shared_tacex_attachment.patch` record dependency provenance and the existing tactile attachment validity checks.

Training and simulation run in separate Python processes. A private Unix socket transports RGB, joint state, and task text. The simulator retains its tactile physics and task predicates, but tactile measurements never reach the policy. Array payloads use dtype/shape/bytes to bridge NumPy 1.x and 2.x. Headless scene queries are enabled so both gelpads attach correctly.

Recreate/check the environments on this computer:

```bash
bash scripts/setup_training.sh
bash scripts/setup_benchmark.sh
```

Optional training-set imitation diagnostic, reported separately for each joint in radians/metres. This uses the training data and is not a generalization score:

```bash
bash scripts/evaluate_offline.sh \
  --checkpoint runs/smolvla_full_dense_b16_ga4_72h/final.pt \
  --max-samples 0
```

Closed-loop benchmarking with 50 fresh seeds per task, 400 rollouts in total:

```bash
bash scripts/benchmark.sh --checkpoint runs/smolvla_full_dense_b16_ga4_72h/final.pt
```

Seeds default to 1,000,000–1,000,049. Each task keeps its official step limit and early-stop/success predicates, with one official `take_action(qpos)` call per predicted target. The simulator's success predicate supplies the score. Results are saved in `runs/benchmark/final/episodes.jsonl` and `summary.json`. Invalid resets or simulator exceptions stop the run instead of silently counting or discarding them. Do not run this simultaneously with throughput profiling or the full training job if you want uncontended timings.

Run reset-and-stationary-action environment checks without a trained policy:

```bash
bash scripts/benchmark.sh --smoke
```

Smoke checks demonstrate environment operation, not policy success. They check RGB/state transfer, one action, finite tactile simulator state, and valid gelpad attachments. Add `--checkpoint PATH` to test one predicted action instead of a stationary action. Existing benchmark results are preserved; use `--output runs/benchmark/ANOTHER_NAME` for another run.

## Verification

```bash
PYTHONPATH=src .venv-train/bin/python -m pytest -q tests
```

Tests check inclusion of all 800 episodes and full-dataset normalization, next-state targets, temporal-discontinuity masking, exclusion of tactile reads, accumulated-gradient equivalence, resumable sampling, IPC array transfer, and checkpoint loading. `logs/` contains installation, data validation, GPU timing, training/checkpoint verification, and simulator smoke logs. The short `configs/verification_54s.json` recipe is only for testing the save/load path and is separate from the production run. Files under `runs/verification_54s/` and its obsolete setup archive are historical checks of the earlier split configuration; the production run starts from the pretrained SmolVLA base using the new full-dataset manifest.

## Tactile refinement extension

The future tactile prediction and full-chunk SmolVLA refinement implementation is
documented in [Tactile-Refinement/IMPLEMENTATION.md](Tactile-Refinement/IMPLEMENTATION.md).
It has separate training and benchmark entry points from the baseline described above.
The active refinement architecture uses a frozen pretrained T-Rex encoder without spatial pooling, a 600-token future predictor, a learned post-predictor resampler producing 32 tokens (16 per finger), and rank-32 FFN LoRA on the existing SmolVLA action expert. The resampler learns through the action objective alone in Stages 2 and 3; there is no compression reconstruction head, reconstruction loss, or spatial-difference loss. Existing grounding checkpoints and the frozen encoder are reused. It shares the original action weights instead of copying an expert. See the [architecture diagram and stage freeze table](Tactile-Refinement/README.md#implementation-diagram).
