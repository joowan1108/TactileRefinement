# Tactile Refinement

Current implementation: pretrained T-Rex full-grid tactile encoder, N₀-VTLA-style future-tactile grounding, **a learned post-predictor resampler from 600 to 32 tokens**, and rank-32 FFN LoRA on one shared SmolVLA action expert (September 15, 2026). See [IMPLEMENTATION.md](IMPLEMENTATION.md) for code, configuration and commands. Earlier experiment descriptions are preserved in `archive/`; they are not the active recipe.

The hypothesis is that predicted future tactile change can guide LoRA-adapted continuation of SmolVLA's full action chunk. The baseline and tactile continuation share the same frozen original action weights; there is no second SmolVLA or copied action expert in the active recipe. Training losses alone do not establish manipulation success; matched UniVTAC rollouts are required.

## Implementation diagram

```mermaid
flowchart TB
    subgraph Inputs["Current observation at time t"]
        DEPTH["Left and right tactile depth<br/>2 × 240 × 320"]
        VL["Head and wrist RGB<br/>language and robot state"]
        STATE["Current robot state"]
    end

    subgraph TactilePath["Future-tactile prediction"]
        NORM["Fixed depth normalization<br/>clip((34 - H) / 10, 0, 1)"]
        ENC["T-Rex DeformEncoder<br/>frozen"]
        GRID["Two 128 × 15 × 20 grids<br/>no 4 × 4 pooling"]
        PROJ["Shared projector<br/>128 to 256"]
        TOKENS["600 current tactile tokens"]
        PRED["Future-tactile predictor<br/>600 learned queries"]
        ZHAT["600 predicted future-tactile tokens"]
        DEPTH --> NORM --> ENC --> GRID --> PROJ --> TOKENS --> PRED --> ZHAT
    end

    subgraph SmolVLAPath["Frozen tactile-free SmolVLA"]
        PREFIX["Vision-language prefix encoder"]
        CTX["Visual-language context"]
        KV["Prefix K/V<br/>includes contextualized state token"]
        WEIGHTS["ONE original action expert<br/>shared frozen weights"]
        BASE["Baseline forward route<br/>six Euler steps: 1.0 to 0.4<br/>LoRA and tactile adapters bypassed"]
        FLOW["Intermediate 50-action flow state"]
        VL --> PREFIX
        PREFIX --> CTX
        PREFIX --> KV
        KV --> BASE --> FLOW
        WEIGHTS --> BASE
    end

    CTX --> PRED

    subgraph Stage1["Stage 1 grounding supervision"]
        FUTURE["Normalized future depth D at t + 50"]
        DELTA["Signed change<br/>D at t + 50 minus D at t"]
        TARGET["Same frozen encoder<br/>and same projector"]
        STOP["Detached future-tactile target"]
        NCE["Symmetric InfoNCE"]
        RECON["Trainable reconstruction head"]
        COARSE["Predicted signed change<br/>2 × 60 × 80"]
        L1["L1 reconstruction loss"]
        FUTURE --> DELTA --> TARGET --> STOP --> NCE
        ZHAT --> NCE
        ZHAT --> RECON --> COARSE --> L1
        DELTA --> L1
    end
    NORM -.-> DELTA

    subgraph Refine["Stage 2, Stage 3, and inference"]
        RESAMPLE["Learned spatial resampler<br/>row, column and finger embeddings<br/>independent attention per finger"]
        COMPACT["32 compact future-tactile tokens<br/>16 per finger"]
        ADAPTERS["Trainable FFN LoRA<br/>gate, up and down projections<br/>rank 32, alpha 32"]
        TACTILE["Trainable tactile cross-attention residuals"]
        DIRECT["Frozen state projection<br/>plus trainable direct-state adapter"]
        EXPERT["Tactile continuation forward<br/>SAME original action weights<br/>predict action velocity"]
        SOLVE["Inference: four Euler steps<br/>0.4 to 0.0"]
        CHUNK["Refined full 50-action chunk"]
        EXEC["Execute each target for<br/>two physics steps"]
        FLOW --> SOLVE
        SOLVE --> EXPERT
        ZHAT --> RESAMPLE --> COMPACT --> TACTILE --> EXPERT
        ADAPTERS --> EXPERT
        STATE --> DIRECT --> EXPERT
        WEIGHTS --> EXPERT
        MASK["Prefix route<br/>masked in Stage 2<br/>restored in Stage 3 and inference"]
        KV --> MASK --> EXPERT
        EXPERT --> SOLVE
        SOLVE --> CHUNK --> EXEC
    end

    subgraph ActionTraining["Stage 2 and Stage 3 action supervision"]
        RECGRID["Training-only spatial reconstruction<br/>position queries attend to compact tokens"]
        FEATURETARGET["Detached 600-token prediction<br/>feature target; no future frames"]
        COMPACTLOSS["Compact feature preservation<br/>Smooth L1 loss"]
        DEMO["Demonstrated 50-action chunk"]
        BRIDGE["Sample flow time in 0 to 0.4<br/>interpolate demo and detached handoff<br/>construct target velocity"]
        ACTIONLOSS["Padding-masked velocity MSE<br/>nine physical action dimensions"]
        FLOW --> BRIDGE
        DEMO --> BRIDGE --> EXPERT
        BRIDGE --> ACTIONLOSS
        EXPERT --> ACTIONLOSS
        COMPACT --> RECGRID --> COMPACTLOSS
        ZHAT --> FEATURETARGET --> COMPACTLOSS
    end

    classDef frozen fill:#e8eef7,stroke:#46658a,color:#172334;
    classDef trainable fill:#e5f6ea,stroke:#39814d,color:#17351f;
    classDef supervision fill:#fff1d6,stroke:#a66b00,color:#4b3100;
    class ENC,PREFIX,WEIGHTS,BASE frozen;
    class PROJ,PRED,RECON,RESAMPLE,RECGRID,ADAPTERS,TACTILE,DIRECT trainable;
    class FUTURE,DELTA,STOP,NCE,L1,DEMO,BRIDGE,ACTIONLOSS,FEATURETARGET,COMPACTLOSS supervision;
```

Blue components are frozen, green components contain parameters trained in at least one stage, and orange components provide training supervision. Both action routes reference the **same** original weights. Action training evaluates one sampled bridge state; inference integrates four continuation steps. The direct-state adapter remains available when Stage 2 masks all prefix K/V. The shared tactile projector is trained through the current-depth branch in Stage 1; its future-target output is detached. The coarse depth-change head is used only for grounding; the separate compact-feature reconstruction head is used only in Stages 2 and 3. Neither reconstruction head runs at inference.

## Tactile representation

Use both ordered gripper depth maps, `tactile/left_gsmini/depth` and `tactile/right_gsmini/depth`, at 240×320. Normalize each map as

$$D_t=\mathrm{clip}((34-H_t)/10,0,1).$$

Depth is in millimetres. No episode-start frame is subtracted. Invalid pixels are masked. This is a normalized tactile depth representation, not a calibrated measurement of gel indentation or force.

Load the released pretrained **T-Rex DeformEncoder** weights and freeze its parameters and BatchNorm statistics. The source is [miniFranka/T-Rex_midtrain_mecka23k_ucb100_vqvae_epoch6](https://huggingface.co/miniFranka/T-Rex_midtrain_mecka23k_ucb100_vqvae_epoch6/tree/62efb3bcb45a3df0e088c8909d759b582cfb98af), pinned to revision `62efb3bcb45a3df0e088c8909d759b582cfb98af`. All 104 encoder state entries are loaded strictly. Our previous 110,000-update encoder is not used in this recipe.

Each finger produces a 128×15×20 feature grid. Flatten the **full 15×20 grid without 4×4 pooling**, then apply a shared trainable linear projector from 128 to 256 channels. Concatenate the fingers in left/right order: 600×256 tokens per observation. Learned spatial and sensor embeddings preserve their identities in the predictor.

## Stage 1: future-tactile grounding

The predictor sees current tactile tokens and frozen SmolVLA visual-language prefix features:

$$g_t=P(E(D_t)),\qquad \hat z_t=F(g_t,C_t).$$

The future-tactile target uses the **same encoder and the same projector**:

$$z_t^*=\mathrm{stopgrad}\left[P(E(D_{t+50}-D_t))\right].$$

Preserve the signed change, including negative values. Future frames are supervision only; they never enter the predictor's input. Endpoints crossing a recording discontinuity or episode boundary are masked, not clamped to the final frame. The current projector branch receives gradients; the target branch is detached. There is no separate target projector or EMA encoder.

Following the [N₀-VTLA grounding objective, equations 3–5](https://arxiv.org/html/2607.23782v1#S4.SS2), mean-pool tokens and L2-normalize those pooled vectors **only for the contrastive loss**. Use cosine similarities and symmetric in-batch cross entropy. A lightweight reconstruction head takes the predicted tokens and reconstructs a signed coarse future-change map:

$$\mathcal L_1=\mathcal L_{\mathrm{InfoNCE}}(\hat z_t,z_t^*)+\lambda_{\mathrm{rec}}\lVert R(\hat z_t)-\Delta D_{\mathrm{coarse}}\rVert_1,\qquad\lambda_{\mathrm{rec}}=1.$$

There is no additional latent regression loss, contact weighting, or autoencoder preparation stage. Train the projector, predictor and reconstruction head together; keep T-Rex and SmolVLA frozen. The head supplies a grounding loss during prediction training; it is not needed at deployment.

Our reconstruction output is two 60×80 maps, supervised using valid-pixel area means of the original signed change. This coarse supervision does **not** pool the encoder features: the predictor retains all 600 tokens. Only the subsequent action-conditioning path compresses them to 32. The depth modality, 600-token grounding layout and reconstruction resolution are adaptations to this project, not an exact reproduction of N₀-VTLA's RGB architecture. We use the published cosine logits without a temperature parameter. Negatives come from each microbatch of 16; gradient accumulation does not create 64-example contrastive batches. Grounding retrieval continues to measure the dense predictor, not the quality of the new compact representation.

## Post-predictor spatial compression

Keep the existing frozen T-Rex encoder, full-grid projector, and 600-query future predictor. Add the resampler **after** the predictor:

$$\hat z_t\in\mathbb R^{600\times256},\qquad c_t=S(\hat z_t)\in\mathbb R^{32\times256}.$$

Each finger retains its complete 15×20 predicted grid as attention memory. Add learned row, column and finger embeddings, then use 16 learned queries per finger and two transformer decoder blocks (width 256, eight heads, no dropout). The weights are shared across fingers, but attention is evaluated separately for each finger. Concatenate the 16 left and 16 right outputs in that order. This is learned attention compression, not fixed 4×4 averaging. The 16-token budget per finger is an initial experiment, not a proven optimum.

To encourage spatial recoverability, a training-only head uses row/column/finger queries to reconstruct all 600 predicted feature vectors from **only the compact tokens**. It has no dense-feature skip connection. In Stages 2 and 3 add

$$\mathcal L_{\mathrm{compact}}=\mathrm{SmoothL1}\left(U(c_t),\mathrm{stopgrad}(\hat z_t)\right),\qquad
\mathcal L_{2,3}=\mathcal L_{\mathrm{action}}+\lambda_{\mathrm{compact}}\mathcal L_{\mathrm{compact}},\quad\lambda_{\mathrm{compact}}=1.$$

This is feature preservation against the current prediction, not future-depth supervision. Action stages still read no future tactile frames. Stage 2 keeps the predictor/projector frozen; Stage 3 permits gradients through them on the compact-input branch, while the reconstruction target is detached. Consequently this auxiliary loss does not guarantee that Stage 3 retains true future-tactile semantics.

The resampler and its feature head are unused and frozen in Stage 1. Train both starting in Stage 2, reusing the existing grounding checkpoint without retraining T-Rex or restarting grounding. At inference compute the compact representation once per observation and reuse it across all four continuation steps. The expert's tactile attention has 50×32 query/key entries per head instead of 50×600; this reduces that matrix by 18.75×, not total model runtime by that factor. Compression is not guaranteed lossless: spatial reconstruction, small-contact sensitivity and matched rollout performance still need evaluation.

## Stage 2: tactile-conditioned action refinement

Freeze the encoder, projector, predictor, Stage 1 reconstruction head and all original SmolVLA parameters, including vision/language, action attention, FFNs and state/action projections. Train **the resampler and compact-feature reconstruction head, rank-32 FFN LoRA**, tactile cross-attention residuals and the direct-state adapter. The refinement route references the baseline's actual action modules; it does not copy them.

Apply LoRA to `gate_proj`, `up_proj` and `down_proj` in all 16 action-expert layers (48 matrices), with rank 32, alpha 32, scaling alpha/rank = 1 and no dropout. Each projection becomes $W x + (\alpha/r) B A x$ only in the tactile continuation route. Initialize A with Kaiming uniform and B to zero; the tactile residual output projections and direct-state adapter also start at zero. This adds **4,251,648 FFN LoRA parameters**, plus the tactile-conditioning and state-adapter parameters. Original attention projections receive no LoRA. With prefix access enabled, the initial continuation reproduces baseline behavior; Stage 2 intentionally changes that behavior by masking the prefix.

The action adapters contain 27,499,728 trainable parameters: 4,251,648 in FFN LoRA and 23,248,080 in tactile cross-attention/state adapters. The new resampler and its training-only feature head add 2,516,224 parameters, for **30,015,952 trainable parameters in Stage 2** and **32,633,296 in Stage 3**. LoRA changes how the existing FFNs process features; the cross-attention residuals introduce the **32 compact predicted tactile tokens** into those features.

The frozen SmolVLA runs six Euler steps from flow time 1 to 0.4. Let its intermediate full 50-action chunk be $Y$. Sample $\tau\in(0,0.4]$ and use the demo-to-handoff bridge

$$x_\tau=(1-\tau/0.4)A^{\mathrm{demo}}+(\tau/0.4)Y,\qquad v^*=(Y-A^{\mathrm{demo}})/0.4.$$

Train the adapted velocity with action-padding-masked MSE on the nine physical action dimensions, plus the compact-feature preservation loss above. Each layer retains the frozen original attention, applies its FFN with LoRA, then adds a trainable cross-attention residual over 32 compact future tactile tokens. A direct-state adapter adds projected robot state to the suffix embedding. **The continuation route's prefix K/V access is masked in Stage 2**, including the contextualized state-token path. The predictor and baseline handoff retain their original visual-language context. The six-step handoff is detached and always bypasses LoRA and all new tactile/state adapters.

The expert receives neither current tactile tokens directly nor an explicit predicted-minus-current latent error. Removing the direct tactile shortcut does not guarantee that the expert uses the predictor; tactile ablations are still necessary.

## Stage 3: joint refinement and deployment

Restore the continuation route's prefix K/V access. Jointly train the projector, predictor, resampler, compact-feature reconstruction head, FFN LoRA, tactile cross-attention residuals and direct-state adapter using the action and compact-feature preservation objectives. Keep the pretrained T-Rex encoder, all original SmolVLA weights and Stage 1 reconstruction head frozen. Future-tactile supervision is not used in this stage, so the learned latent may drift from its Stage 1 predictive interpretation; compare forecast diagnostics before and after joint training.

At inference, observe current RGB, language, robot state and depth, predict 600 future tactile tokens and compress them to 32, run the shared action expert for six steps with adapters bypassed, then finish denoising with four tactile-conditioned steps using FFN LoRA at 0.4, 0.3, 0.2 and 0.1. LoRA deltas remain unmerged so the baseline route stays unchanged. Each step updates the whole 50-action chunk. Execute each joint-position target for two physical steps, then observe again after the complete chunk, unless the environment terminates. There is no intermediate 10-action correction loop and neither reconstruction head runs at inference.

| Stage | Trainable parameters | Continuation prefix K/V |
|---|---|---|
| 1: grounding | Tactile projector, predictor, reconstruction head | Continuation unused |
| 2: action refinement | Resampler, compact-feature head, FFN LoRA, tactile residuals, direct-state adapter | Fully masked |
| 3: joint refinement | Stage 2 parameters plus tactile projector and predictor | Restored |

The pretrained T-Rex encoder and every original SmolVLA parameter stay frozen in all three stages. Stage 2 supplies predicted tokens, not current tactile tokens, to the action route; withholding current tokens alone does not prove that the trained policy uses tactile information.

## Data, schedule and validation

All 800 UniVTAC episodes remain available, with no training/validation split. Stage 1 masks examples without valid future supervision. Batch size is 16 with accumulation 4, effective batch 64, using bfloat16. Configured budgets are **100,000 grounding updates**, 20,000 expert updates and 10,000 joint updates. These are experiment budgets, not proven convergence thresholds. The current launch runs **grounding only and then exits**, saving checkpoints at 33,333, 66,666 and 100,000. An automatic continuation preserves the live run's first 10,000 updates and resumes through the new total; the earlier milestones are archived separately.

The pretrained encoder was trained on another tactile domain, and signed difference maps differ from its nonnegative pretraining inputs. Transfer quality remains an empirical question. Measure future-change reconstruction against zero-change predictions, latent variation and retrieval, then evaluate matched policies in UniVTAC using identical action holds, replanning intervals and seeds. Existing baseline rollouts with another execution schedule are not a matched comparison.

The live Stage 1 process continues its existing grounding objective and budget. Before Stage 2, loading its checkpoint transfers the learned tactile components and initializes the new LoRA/adapters, resampler and compact-feature head. Migration is permitted only for encoder/grounding checkpoints with matching training settings and provenance; an already-trained action-stage checkpoint cannot silently acquire random compression weights or discard its dense expert. New format-version-3 checkpoints store the resampler and added components without another copy of the original action weights. Legacy configurations without resampling remain loadable for historical diagnostics. Adding this module does not start Stage 2 automatically.
