# Tactile refinement implementation

The active grounding recipe is `trex_fullgrid_n0_v1`, with `expert_implementation=shared_ffn_lora_v1` and `resampler_tokens_per_finger=16` in `configs/tactile_refinement.json`. Code and run paths below are relative to the GPU workspace `/tactile/new`. It starts with pretrained frozen T-Rex weights and randomly initialized projector, future predictor and reconstruction head. A learned post-predictor resampler compresses 600 predicted tokens to 32 before action refinement, which uses rank-32 FFN LoRA on the **same frozen SmolVLA action expert**, plus tactile residuals and a direct-state adapter. Legacy copied experts and 4×4/preparation code remain loadable for historical configurations; the active recipe never selects them. See [the implementation diagram](README.md#implementation-diagram).

## Implementation path

1. `src/tactile_refinement/train.py`: load the baseline and original all-800-episode manifest, enforce tactile-free baseline inputs, strictly load the pinned T-Rex encoder, and select grounding → expert → joint. Record SHA-256 fingerprints for the baseline, dataset manifest and encoder checkpoint. The new recipe rejects encoder preparation.
2. `src/tactile_refinement/data.py`: read ordered left/right depth maps, apply fixed `(34-H)/10` clipping, construct signed `D[t+50]-D[t]` when valid, and keep tactile data outside the baseline preprocessor. Action stages read current depth only.
3. `src/tactile_refinement/representation.py`: `DeformEncoder` returns 128×15×20 per finger. `TactileRepresentation(full_grid=True)` flattens without spatial pooling and projects to 600×256 tokens. `Predictor` has 600 learned queries, two transformer decoder blocks, eight heads and a visual-language adapter.
4. `src/tactile_refinement/model.py`: `predicted_tokens` computes `F(P(E(D_t)), C_t)`. During grounding, `loss` applies the identical E/P modules to signed future change under `no_grad`. The predictor never reads the future endpoint. Encoder parameters and BatchNorm remain frozen in every stage.
5. `src/tactile_refinement/losses.py`: `n0_grounding_loss` combines symmetric cosine InfoNCE on mean-pooled/L2-normalized tokens with plain L1 reconstruction. `CoarseChangeHead` maps predicted tokens to two signed 60×80 maps for Stage 1 supervision. In action stages, `SpatialTactileResampler` in `representation.py` attends separately to each 300-token finger grid, using 16 learned queries per finger, learned row/column/finger embeddings, two width-256/eight-head transformer decoder blocks and a final LayerNorm. The same weights serve both fingers, while each finger has 16 dedicated output slots. `model.loss` uses the action velocity MSE alone in Stages 2 and 3. The compression reconstruction head and loss have been removed; no spatial-difference loss is used. The predictor/projector are frozen in Stage 2 and receive action gradients in Stage 3. Equal token allocation guarantees capacity for each finger, not equal attention from the expert.
6. `RefinementModel.handoff` obtains the full action flow state at time 0.4 using six calls to the frozen original `core.denoise_step`, with no adapters and no gradients. `CachedTactileExpert` references that same expert and the same action/time/state projections through non-registered aliases. The baseline alone owns these weights, so adapter training and serialization cannot unfreeze or duplicate them. Its custom layer loop evaluates frozen attention, FFN plus `LowRankDelta`, and a tactile cross-attention residual. `gate_proj`, `up_proj` and `down_proj` in all 16 layers receive rank-32/alpha-32 LoRA: 4,251,648 added FFN parameters, scaling 1, no dropout, Kaiming-uniform A and zero B. Tactile/state residual outputs also start at zero. The state adapter enters before the first expert layer. Stage 2 removes all prefix K/V; Stage 3 restores it. The direct raw-state path remains available in both stages.
7. `predict_action_chunk` compresses the prediction once per observation, reuses the 32 compact tokens across four continuation steps, returns all 50 nine-dimensional actions, and skips the Stage 1 reconstruction head. The `zero` ablation zeros the actual compact condition after resampling; `shuffle` and `current` also pass through the resampler. `src/tactile_refinement/benchmark.py` denormalizes using baseline statistics. New evaluations default to `linear_position`: raw model endpoints without EMA, midpoint then endpoint at physics cadence, and explicit zero velocity targets. The historical `linear` mode uses derived velocity targets and amplified observed roughness; the zero-velocity mode eliminated measured endpoint overshoot in the EMA probe. Removing EMA does not remove roughness already present in model endpoints. Existing resumed evaluations retain their recorded controller; older runs use the two-step position hold. Optional arm EMA precedes interpolation. This approximates intermediate motion without reconstructing the original demonstration planner trajectory.

| Stage | Trainable | Frozen | Updates / learning rate |
|---|---|---|---|
| Grounding | Projector, predictor, coarse-change head | T-Rex, original SmolVLA, action adapters, resampler | 100,000 / 1e-4 |
| Expert | Resampler, FFN LoRA, tactile residuals, direct-state adapter; prefix masked | T-Rex, projector, predictor, coarse-change head, all original SmolVLA weights | 100,000 / 5e-5 |
| Joint | Projector, predictor, resampler, FFN LoRA, tactile residuals, direct-state adapter; prefix restored | T-Rex, coarse-change head, all original SmolVLA weights | 10,000 / 1e-5 |

The FT target shares the current branch's actual encoder/projector objects. It is detached, while the projector receives gradients through current tactile input. The reconstruction head learns from the **predicted** latent, not from a future-target shortcut. AdamW, bfloat16, batch 16 and accumulation 4 are configured. The NCE negative set is a microbatch, not the accumulation window.

## Stage 2 checkpoint testing

The new offline entry point `src/tactile_refinement/ablation.py` accepts
`--checkpoints PATH [PATH ...] --samples 256 --output DIR`. Launch it from the
workspace with `PYTHONPATH=src .venv-train/bin/python -m tactile_refinement.ablation`.
It runs correct/mismatched/disabled tactile interventions with identical prefix,
noise, handoff and bridge timesteps, and compares normalized action/flow errors.
Donor maps retain left/right ordering and exclude the recipient's episode.

The rollout entry point is `tactile_refinement.ablation_rollout`, with
`--checkpoint PATH --tasks insert_HDMI --episodes 20 --schedule official --output DIR`.
It runs all three conditions, saves the first three videos per condition, supports
`--resume`, and separates reset-matched contrasts from same-seed-only contrasts.
`--schedule full50` selects the older whole-chunk/two-step hold schedule for all
conditions. `--smoke-steps 1` is an integration check, excluded from success rates.
See workspace `docs/stage2_tactile_ablation.md` for complete commands and limitations.

`RefinementModel.continue_action_chunk` now applies the saved stage's prefix mask
at inference, and the benchmark calls `set_stage` after checkpoint loading.
Stage 2 retains masking; Stage 3 restores access. This changes inference only;
the existing action-training loss and checkpoints are unchanged.

## Training launch

Current launch: Stage 2 only, detached from SSH, initialized from the completed 100,000-update grounding checkpoint. It runs 100,000 updates at learning rate 5e-5, batch 16 and accumulation 4, saving at 33,333, 66,666 and 100,000. It exits after Stage 2; Stage 3 is not automatic. The first two-update launch is archived at `runs/tactile_refinement_trex_fullgrid_n0/expert_initial_2_updates_before_stage2_100k_20260915/`. Extending the future Stage 2 budget is permitted when loading a grounding checkpoint, without changing its training settings or learned weights.

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

Grounding completed 100,000 updates and paused on September 15, 2026. `validate_resume_config` permits adding the unused resampler and switching the unused expert to rank-32/alpha-32 shared FFN LoRA before Stage 2, with grounding settings preserved. It also permits removing the obsolete compression-loss setting from early-stage checkpoints. `load_added_state_dict(..., allow_legacy_grounding=True)` transfers existing tactile tensors, initializes new modules, and can discard unused obsolete reconstruction-head tensors; the CLI enables this only for validated encoder/grounding checkpoints. Grounding optimizer and RNG restoration remain compatible. Already-trained action-stage checkpoints cannot silently change objective or discard modules. New format-version-4 checkpoints include the resampler without its removed reconstruction head. Configurations without `resampler_tokens_per_finger` use the historical uncompressed action route; those without `expert_implementation` select the legacy copied-expert loader. Configurations with a nonzero `lambda_compact_rec` fail explicitly and require migration from grounding. The baseline SHA-256 identifies the frozen base.

The resampler adds **2,120,960 parameters**. Stage 2 trains **29,620,688** parameters and Stage 3 **32,238,032**. Training logs distinguish `tactile_tokens=600` from `action_tactile_tokens=32`, with `resampler_objective=action_only`; action stages report only `refinement` as their loss component. The zero-initialized tactile output projections block upstream action gradients on the first update. That update opens the tactile route, after which the action loss trains the resampler. No encoder retraining or additional grounding stage is required. Production Stage 2 still requires the explicit command above.

## Checks and diagnostics

```bash
PYTHONPATH=src .venv-train/bin/python -m pytest -q tests/test_tactile_resampler.py tests/test_ffn_lora_refinement.py tests/test_tactile_refinement.py tests/test_trex_grounding.py tests/test_grounding_extension.py
PYTHONPATH=src .venv-train/bin/python scripts/verify_trex_grounding.py
PYTHONPATH=src .venv-train/bin/python scripts/verify_refinement.py --output logs/ffn_lora_verification.json
PYTHONPATH=src .venv-train/bin/python scripts/verify_refinement_training.py
```

The focused tests cover spatial pooling absence, 600-token shapes, exact contrastive math and signed/invalid endpoint handling. Real-weight checks cover exact T-Rex loading, current/target gradient separation, frozen BatchNorm, all stage gradient paths, prefix masking, initial baseline/expert equivalence, checkpoint reloads and resumed CLI training. These are implementation checks, not convergence or task-success measurements.

Action-supervised resampler checks: **27 focused tests passed**, including per-finger isolation, sensitivity to spatial rearrangement, action-only resampler/predictor gradients after the initial adapter update, old grounding weight/optimizer migration, obsolete-head migration guards, and 32-token conditioning in training and inference. The real-SmolVLA check also passed: action gradients reached the resampler after the first adapter update and reached the predictor/projector in Stage 3; baseline weights/output stayed unchanged, prefix masking worked, and checkpoint round trips matched exactly. Report: `logs/resampler_action_only_verification.json`. The earlier `logs/resampler_verification.json` describes the superseded auxiliary-loss variant. These checks establish wiring and compatibility, not lossless compression or rollout improvement.

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
