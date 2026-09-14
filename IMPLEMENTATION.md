# Tactile refinement implementation

The active grounding recipe is `trex_fullgrid_n0_v1`, with `expert_implementation=shared_ffn_lora_v1` in `configs/tactile_refinement.json`. Code and run paths below are relative to the GPU workspace `/tactile/new`. It starts with pretrained frozen T-Rex weights and randomly initialized projector, future predictor and reconstruction head. Action refinement uses rank-32 FFN LoRA on the **same frozen SmolVLA action expert**, plus tactile residuals and a direct-state adapter. Legacy copied experts and 4×4/preparation code remain loadable for historical configurations; the active recipe never selects them. See [the implementation diagram](README.md#implementation-diagram).

## Implementation path

1. `src/tactile_refinement/train.py`: load the baseline and original all-800-episode manifest, enforce tactile-free baseline inputs, strictly load the pinned T-Rex encoder, and select grounding → expert → joint. Record SHA-256 fingerprints for the baseline, dataset manifest and encoder checkpoint. The new recipe rejects encoder preparation.
2. `src/tactile_refinement/data.py`: read ordered left/right depth maps, apply fixed `(34-H)/10` clipping, construct signed `D[t+50]-D[t]` when valid, and keep tactile data outside the baseline preprocessor. Action stages read current depth only.
3. `src/tactile_refinement/representation.py`: `DeformEncoder` returns 128×15×20 per finger. `TactileRepresentation(full_grid=True)` flattens without spatial pooling and projects to 600×256 tokens. `Predictor` has 600 learned queries, two transformer decoder blocks, eight heads and a visual-language adapter.
4. `src/tactile_refinement/model.py`: `predicted_tokens` computes `F(P(E(D_t)), C_t)`. During grounding, `loss` applies the identical E/P modules to signed future change under `no_grad`. The predictor never reads the future endpoint. Encoder parameters and BatchNorm remain frozen in every stage.
5. `src/tactile_refinement/losses.py`: `n0_grounding_loss` combines symmetric cosine InfoNCE on mean-pooled/L2-normalized tokens with plain L1 reconstruction. `CoarseChangeHead` maps predicted tokens through a 256→64 projection, two spatial convolutions, interpolation and a final convolution to two signed 60×80 maps. Only reconstruction supervision is downsampled; the representation retains its full grid. No additional latent regression or contact weighting is active.
6. `RefinementModel.handoff` obtains the full action flow state at time 0.4 using six calls to the frozen original `core.denoise_step`, with no adapters and no gradients. `CachedTactileExpert` references that same expert and the same action/time/state projections through non-registered aliases. The baseline alone owns these weights, so adapter training and serialization cannot unfreeze or duplicate them. Its custom layer loop evaluates frozen attention, FFN plus `LowRankDelta`, and a tactile cross-attention residual. `gate_proj`, `up_proj` and `down_proj` in all 16 layers receive rank-32/alpha-32 LoRA: 4,251,648 added FFN parameters, scaling 1, no dropout, Kaiming-uniform A and zero B. Tactile/state residual outputs also start at zero. The state adapter enters before the first expert layer. Stage 2 removes all prefix K/V; Stage 3 restores it. The direct raw-state path remains available in both stages.
7. `predict_action_chunk` uses four continuation steps, returns all 50 nine-dimensional actions, and skips the reconstruction head. `src/tactile_refinement/benchmark.py` denormalizes using baseline statistics and holds each target for two physical steps.

| Stage | Trainable | Frozen | Updates / learning rate |
|---|---|---|---|
| Grounding | Projector, predictor, reconstruction head | T-Rex, original SmolVLA, LoRA/tactile/state adapters | 100,000 / 1e-4 |
| Expert | FFN LoRA, tactile residuals, direct-state adapter; prefix masked | T-Rex, projector, predictor, reconstruction head, all original SmolVLA weights | 20,000 / 5e-5 |
| Joint | Projector, predictor, FFN LoRA, tactile residuals, direct-state adapter; prefix restored | T-Rex, reconstruction head, all original SmolVLA weights | 10,000 / 1e-5 |

The FT target shares the current branch's actual encoder/projector objects. It is detached, while the projector receives gradients through current tactile input. The reconstruction head learns from the **predicted** latent, not from a future-target shortcut. AdamW, bfloat16, batch 16 and accumulation 4 are configured. The NCE negative set is a microbatch, not the accumulation window.

## Training commands

From `/tactile/new`, launch Stage 1 detached from SSH:

```bash
bash scripts/start_refinement.sh --stage grounding
tail -f logs/tactile_refinement.log
```

The process exits after grounding; later stages do not start automatically. Output is `runs/tactile_refinement_trex_fullgrid_n0/grounding/`, with `checkpoint_1.pt` at update 33,333, `checkpoint_2.pt` at 66,666 and `final.pt` at 100,000. Each checkpoint contains all added model weights, optimizer, RNG, recipe and provenance. The pretrained encoder and baseline source files must remain available for loading.

The September 13 live run began with a 10,000-update budget before the user extended it to **100,000 total updates**. `scripts/extend_grounding.py` waits on its training lock, verifies its completed 10,000-step checkpoint, archives the three earlier milestones in `grounding_10000_before_extension/`, then resumes grounding from step 10,000 using the new budget. Weights, optimizer, sampler position and RNG are preserved. The supervisor and resumed child survive SSH disconnects. Status is recorded in `logs/tactile_refinement_extension_status.json`; the training log remains `logs/tactile_refinement.log`. Thus the existing process's initial 10,000-step log header is historical, and its completion triggers continuation through 100,000 rather than Stage 2. Do not launch a second trainer while this supervisor is active.

```bash
# Resume an interrupted grounding run from its latest saved checkpoint.
bash scripts/start_refinement.sh --stage grounding \
  --resume runs/tactile_refinement_trex_fullgrid_n0/grounding/checkpoint_2.pt

# Later, explicitly start Stage 2 from the completed grounding checkpoint.
bash scripts/start_refinement.sh --stage expert \
  --init runs/tactile_refinement_trex_fullgrid_n0/grounding/final.pt

# Start Stage 3 after Stage 2 completes.
bash scripts/start_refinement.sh --stage joint \
  --init runs/tactile_refinement_trex_fullgrid_n0/expert/final.pt
```

Fresh runs refuse to overwrite existing stage outputs. The previous 110k encoder and other historical checkpoints are untouched in `runs/tactile_refinement/`. Full-grid checkpoints are incompatible with the legacy 32-token predictor, as enforced by recipe/provenance and tensor shapes.

The live grounding job is not restarted for the LoRA change. Its in-memory configuration still writes legacy grounding checkpoints. `validate_resume_config` allows the sole architecture change to rank-32/alpha-32 shared FFN LoRA before Stage 2 when every other setting matches. `load_added_state_dict(..., allow_legacy_grounding=True)` transfers all non-expert tensors and keeps the new adapters at initialization; the CLI enables this only for validated encoder/grounding checkpoints. Resuming grounding also restores its optimizer, sampler position and RNG. Dense Stage 2/3 checkpoints cannot migrate this way. New format-version-2 checkpoints contain LoRA/tactile/state additions and tactile prediction modules, with no original action weights duplicated under `expert`. Saved configuration and the baseline SHA-256 identify the exact frozen base. Old configs without `expert_implementation` explicitly select the legacy loader for diagnostics.

## Checks and diagnostics

```bash
PYTHONPATH=src .venv-train/bin/python -m pytest -q tests/test_ffn_lora_refinement.py tests/test_tactile_refinement.py tests/test_trex_grounding.py tests/test_grounding_extension.py
PYTHONPATH=src .venv-train/bin/python scripts/verify_trex_grounding.py
PYTHONPATH=src .venv-train/bin/python scripts/verify_refinement.py --output logs/ffn_lora_verification.json
PYTHONPATH=src .venv-train/bin/python scripts/verify_refinement_training.py
```

The focused tests cover spatial pooling absence, 600-token shapes, exact contrastive math and signed/invalid endpoint handling. Real-weight checks cover exact T-Rex loading, current/target gradient separation, frozen BatchNorm, all stage gradient paths, prefix masking, initial baseline/expert equivalence, checkpoint reloads and resumed CLI training. These are implementation checks, not convergence or task-success measurements.

Rank-32 verification on September 14, 2026: **21 focused tests passed**. The real checkpoint shared all 112 original transformer matrices and contained exactly 4,251,648 FFN LoRA parameters. Initial velocity and full-chunk outputs matched the baseline exactly; original weights and baseline output remained unchanged after adapter updates. Stage 2 ignored perturbed prefix K/V, while Stage 3 restored that access and gradients reached the predictor/projector. The separate three-update-per-stage CLI test migrated a legacy grounding checkpoint, restored its optimizer/RNG, completed all three stages, saved three checkpoints per stage and loaded the final checkpoint for diagnostics. Reports: `logs/ffn_lora_verification.json` and `logs/ffn_lora_training_verification.json`. These smoke checks used microbatch 1 and do not measure rank quality or convergence. Stage 2 trains 27,499,728 parameters including tactile/state adapters; Stage 3 trains 30,117,072 including projector/predictor.

Verification on September 13, 2026: 14 focused tests passed. All 104 pretrained encoder entries matched after loading (source bf16 values represented exactly in float32) and remained unchanged after an optimizer update. Real grounding gradients reached the projector, predictor and reconstruction head only. The three-stage CLI smoke test saved all three checkpoints per stage, resumed grounding from update one and loaded the final model for diagnostics. Reports are `logs/trex_grounding_gradient_verification.json`, `logs/trex_fullgrid_verification.json` and `logs/refinement_training_verification.json`. A three-update batch-16/accumulation-4 grounding probe completed with finite losses, approximately 1.29 seconds per warmed-up update and 2.03 GiB peak PyTorch allocated memory; this short probe is an initial throughput estimate, not a convergence test.

```bash
PYTHONPATH=src .venv-train/bin/python -m tactile_refinement.diagnostics \
  --checkpoint runs/tactile_refinement_trex_fullgrid_n0/grounding/final.pt \
  --output runs/tactile_refinement_trex_fullgrid_n0/grounding_diagnostics.json

bash scripts/benchmark_refinement.sh \
  --checkpoint runs/tactile_refinement_trex_fullgrid_n0/joint/final.pt \
  --output runs/benchmark/trex_fullgrid_n0
```

Forecast diagnostics use training-corpus samples without creating a holdout; reconstruction is scored at the head's 60×80 resolution for this recipe. Benchmark defaults are 100 episodes per task, three videos per task and new simulator seeds. Compare against `--baseline` with identical full-chunk/two-step execution. No upload is performed by these commands.
