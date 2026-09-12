# Tactile Refinement

Final research design — September 12, 2026.

**Status:** This is a technically coherent experiment, not an implemented or empirically validated method. The hypothesis is that current deformation and a prediction of its future change can improve the terminal denoising of a tactile-unaware SmolVLA. Success must be demonstrated in closed-loop UniVTAC rollouts.

## Objective and scope

Use the trained vision-language-proprioception SmolVLA as a frozen motion planner. Add a shared deformation encoder, a future-change predictor, and a separate tactile expert that completes partially denoised action chunks.

The tactile input is a single current deformation map from each gripper finger, corrected using a verified unloaded reference. No tactile history is required. The baseline retains its original observation inputs and weights throughout refinement training.

All 800 UniVTAC demonstration episodes remain available for training, with **no train/validation split**. Benchmark performance is measured through separate simulator rollouts with new evaluation seeds. Training-set reconstruction and action losses are diagnostics, not estimates of generalization.

## 1. Deformation maps and reference frames

For finger $k\in\{L,R\}$, let $H_t^k$ be the recorded sensor depth and $B_e^k$ the verified unloaded depth reference for episode $e$. Define

$$
D_t^k=\sigma_k(H_t^k-B_e^k),\qquad
\Delta D_{t,50}^k=D_{t+50}^k-D_t^k.
$$

The calibration sign $\sigma_k\in\{-1,+1\}$ makes indentation positive. Verify it through a controlled contact test. Preserve signed values, especially in $\Delta D$: increasing and decreasing indentation must remain distinguishable.

In our downloaded Isaac45 data, use the depth streams belonging to **left_gsmini** and **right_gsmini**. Their frames are $240\times320$ float arrays. These are camera depths in millimetres, rather than deformation values already referenced to an unloaded gel. The [UniVTAC sensor implementation](https://github.com/univtac/UniVTAC/blob/main/third_party/TacEx/source/tacex/tacex/gelsight_sensor.py) converts camera depth to millimetres.

Save each reference with its sensor identity, source, calibration configuration, contact-free verification, and validity mask. Capture it before object contact and before any scripted pre-grasp. **Do not automatically use the first saved demonstration frame:** UniVTAC runs a pre-move phase while recording is suppressed, so recording can begin after contact. See the [task reset and recording logic](https://github.com/univtac/UniVTAC/blob/main/envs/_base_task.py).

For existing episodes, obtain a reference from a verified unloaded frame or reproduce a validated unloaded calibration with the matching simulator sensor configuration. Record which method was used. Availability of valid references across the downloaded corpus remains an implementation prerequisite; this document does not establish that those references already exist.

Use a fixed sensor-valid region and mask invalid or clipped depth pixels. Apply the same spatial transform and a fixed physical scale to current maps and change maps. Do not normalize each frame independently, take absolute differences, or convert the training maps to unsigned images: these operations can erase deformation magnitude or unloading direction. Store the normalization and calibration with the model.

The reference cancels algebraically in the change target:

$$
\Delta D_{t,50}^k=\sigma_k(H_{t+50}^k-H_t^k).
$$

It remains essential for the predictor's current-contact input.

## 2. Shared T-Rex deformation encoder

Use the actual **DeformEncoder** architecture in [T-Rex's released code](https://github.com/ZhuoyangLiu2005/T-Rex/blob/f88e10c61da123c68bf0927cf4860bc97a0381f3/qwen_vla/DeformAE.py):

| Component | Operation |
|---|---|
| Input stem | Single-channel $7\times7$ convolution, stride 2, 64 channels; batch normalization, ReLU, max pooling |
| Residual stages | ResNet-18 layer1, layer2, and layer3 |
| After layer2 | $3\times3$ convolution, $128\rightarrow128$, batch normalization, ReLU |
| After layer3 | $3\times3$ convolution, $256\rightarrow128$, batch normalization, ReLU |
| Output at our resolution | $128\times15\times20$ spatial feature map |

The code has two added projection blocks; follow this executable architecture rather than interpreting the paper's description as a projection after every stage. T-Rex pretrains the spatial encoder with an autoencoder and freezes it for policy learning. Its temporal force encoder is a separate component. [T-Rex, Appendix C](https://arxiv.org/html/2606.17055v1#A3)

Our token adapter $P$ applies a $4\times4$ adaptive spatial pool, flattens the 16 positions, and linearly projects each 128-dimensional feature to width $d=256$. This pooling and token width are our implementation choices. Keep the two fingers separate, giving 32 ordered tokens. Add sensor and position embeddings to conditioning tokens; preserve the same ordered slots in prediction targets. Do not average the fingers, since asymmetric contact matters.

Pretrain the shared $E$ and $P$ on **both** $D_t$ and signed $\Delta D_{t,50}$. Train a decoder from the projected tokens to reconstruct each input map using a valid-pixel, contact-weighted reconstruction objective. The decoder must reconstruct through the token bottleneck, without bypassing it. This grounds the projection as well as the encoder.

Then freeze $E$ and $P$, including batch-normalization running statistics. Reuse these exact weights on the current and target branches. Architectural reuse does not establish that T-Rex's sensor-specific weights transfer directly to UniVTAC or to signed change maps. The ability of this compact bottleneck to retain small contact patches must be checked through reconstruction.

## 3. Predict the change over the next 50 steps

Define current-contact tokens and the supervised target as

$$
g_t^k=P(E(D_t^k)),\qquad
z_t^{*,k}=\operatorname{stopgrad}\!\left[P(E(\Delta D_{t,50}^k))\right].
$$

Concatenate the left and right slots to obtain $g_t$ and $z_t^*$. The predictor receives only information available at decision time:

$$
\hat z_t=F_\phi(g_t,C_t),
$$

where $C_t$ is the contextualized visual-language prefix from frozen SmolVLA. Use 32 learned output queries, two cross-attention/FFN blocks, width 256, and eight attention heads as initial settings.

Cache both the contextualized prefix features and the base expert's per-layer K/V tensors. The predictor projects the prefix features into its own attention space; the copied tactile expert can reuse the base K/V layout. Prefix features and per-layer K/V caches are different tensor interfaces. Respect padding masks and verify cache-based inference against an uncached reference.

This target construction follows the endpoint-change formulation in [N0-VTLA, Section 2.2](https://arxiv.org/html/2607.23782v1#S2.SS2), which applies the tactile encoder to the future-minus-current image. Our substitutions are depth-derived deformation, the T-Rex CNN, and separate finger targets. In general,

$$
P(E(D_{t+50}-D_t))\ne P(E(D_{t+50}))-P(E(D_t)).
$$

The left-hand expression is the selected target. It is a **latent representation of future net deformation change**, not an absolute future-contact latent. Consequently, subtracting $g_t$ from $\hat z_t$ does not produce a meaningful contact prediction error.

Ground the predictor with

$$
\mathcal L_{\rm pred}=
\mathcal L_{\rm Huber}(\hat z_t,z_t^*)
+\lambda_{\rm rec}\mathcal L_1(R(\hat z_t),\Delta D_{t,50})
+\lambda_{\rm NCE}\mathcal L_{\rm NCE}(\hat z_t,z_t^*).
$$

All terms use valid samples or pixels; reconstruction retains separate finger outputs. Initialize $R$ from the representation-pretraining decoder. N0-VTLA motivates symmetric contrastive supervision and spatial reconstruction; pointwise latent regression and a permanently frozen target projection are our stabilizing choices. Keep reconstruction in physical-scale units when reporting results.

For contrastive sampling, avoid nearby overlapping windows as negatives and mask duplicate or indistinguishable near-zero targets. Samples without distinct valid negatives still contribute regression and reconstruction. Gradient accumulation does not enlarge the simultaneous contrastive negative pool.

The [released N0-VTLA policy](https://github.com/neoteai/N0-VTLA/blob/03a0ce4d7091ca2354864796770715aa212601b7/n0vtla/models_pytorch/n0vtla_policy.py) explicitly omits the paper's supervised predictor-grounding stage. Our future-target construction and loss must therefore be implemented explicitly.

**Interpretation:** Since $F_\phi$ receives no candidate action, its output is a scene- and contact-conditioned forecast under the demonstration distribution. It is not a causal prediction of what a particular SmolVLA candidate will do. Multiple valid future contacts can share the same current observation.

## 4. Tactile expert and intermediate-action handoff

Let physical data time be $t$, and flow time be $\tau$. Use the SmolVLA convention: $\tau=1$ is Gaussian noise and $\tau=0$ is a completed action chunk.

The frozen base expert generates a 50-step intermediate action chunk

$$
Y_t=\operatorname{Integrate}\!\left(v_{\rm base},\epsilon;
\tau:1\rightarrow\tau_s\right),\qquad \tau_s=0.4.
$$

The tactile expert receives the evolving flow state, flow time, robot state $q_t$, cached prefix, current deformation tokens, and predicted change tokens:

$$
v_\theta(x_\tau,\tau;\,q_t,C_t,g_t,\hat z_t).
$$

It completes the trajectory from $\tau_s$ to zero. Current tokens provide direct measured-contact information; predicted tokens provide anticipated contact change. **Only predicted future tokens enter the expert during both training and deployment.** Future ground-truth maps and latents are supervision only.

Initialize the tactile expert by copying SmolVLA's trained action-expert weights, action/state/time projections, and output head. Add separate tactile cross-attention residual blocks with zero-initialized output projections. Initialize their inner attention weights normally. This preserves the copied expert's initial computation while allowing the tactile route to learn. Numerically verify equivalence before training. Simply appending zero-valued tokens to an existing attention softmax does not guarantee equivalence.

Retain the baseline's normalized nine-dimensional action semantics, padded model slots, action masks, and controller conversion. An intermediate flow state is not an executable joint command.

### Train on the handoff states the expert will actually receive

[FM-Steer, Section 4.2](https://openaccess.thecvf.com/content/CVPR2026/papers/Song_FM-Steer_Enhance_Generalist_Policies_with_Value-Guided_Cascaded_Denoising_CVPR_2026_paper.pdf) trains its continuation module using intermediate actions sampled from a frozen VLA. Adopt this principle so the tactile expert encounters base-policy errors during training.

For each demonstration observation, sample $Y_t$ from frozen SmolVLA. Pair it with the demonstration action target $A_t^{\rm demo}$. With $\alpha=\tau/\tau_s$, define our refinement bridge

$$
x_\tau=(1-\alpha)A_t^{\rm demo}+\alpha Y_t,\qquad
v^*=\frac{Y_t-A_t^{\rm demo}}{\tau_s},
\quad 0<\tau\le\tau_s.
$$

Train the expert with masked velocity regression:

$$
\mathcal L_{\rm refine}
=\mathbb E\left[\left\|v_\theta(x_\tau,\tau;\cdot)-v^*\right\|_{\rm valid}^2\right].
$$

The scaling by $\tau_s$ is required because the expert integrates in the original $\tau$ coordinate. Resample base noise across training visits, or cache multiple seeds with the exact checkpoint and sampler metadata. Detach the base outputs and caches.

This bridge is our explicit continuation objective. Replacing Gaussian noise with $Y_t$ while keeping the Gaussian-to-demonstration velocity target would be inconsistent. The proposal adopts cascaded refinement from FM-Steer; it does not include FM-Steer's value verifier or best-of-N selection.

For initial deployment use six base Euler steps and four refinement steps, with step size $-0.1$. Execute the first 10 actions, acquire fresh observations, and replan another 50-step chunk. The 50-step forecast is recomputed at every replan. This first version is synchronous at the replanning boundary; no unmeasured high-frequency or asynchronous-control claim is made.

## 5. Training sequence and data alignment

| Stage | Train | Freeze |
|---|---|---|
| Representation pretraining | Shared deformation encoder, token projector, reconstruction decoder | Baseline SmolVLA |
| Predictor grounding | Predictor, prefix adapter, prediction reconstruction head | Baseline; deformation encoder and target projector |
| Refinement alignment | Copied tactile expert and tactile conditioning adapters | Baseline; encoder/projector; grounded predictor |
| Joint refinement tuning | Tactile expert, predictor, their adapters and reconstruction head; retain predictive losses | Baseline and shared encoder/projector |

Keep the base vision encoder, language backbone, and original action expert frozen during these new stages. This is separate from the completed/configured dense baseline finetuning. A GPU memory and throughput measurement is required before assigning a duration or microbatch size to the new training stages.

A training example at $t$ uses current observations at $t$, action targets corresponding to the next 50 recorded states, and tactile target at $t+50$. Verify this mapping against the actual loader and timestamps. Fifty indexes mean 50 recorded transitions, not necessarily 50 simulator physics steps.

Future targets must stay inside the same episode and continuous recording segment. If $t+50$ is unavailable or crosses a recording discontinuity, mask the future losses; retain valid action training with its own horizon mask. Do not silently clamp the endpoint or cross episode boundaries. Missing or invalid unloaded references mask the affected current-contact input; they must never be labelled zero contact.

All episodes remain in the training corpus. Eligibility masks enforce valid supervision and do not create a held-out split. Overlapping action chunks are acceptable, but losses and contrastive sampling must account for correlated samples.

## 6. Strict assessment and required evidence

The architecture is valid as a research hypothesis. The combination of a deformation representation, endpoint prediction, and action continuation is an incremental synthesis of existing mechanisms. Its contribution would be demonstrated improvement from predictive deformation conditioning under a controlled frozen-baseline comparison.

Three limitations matter:

- A single depth-derived map measures surface geometry. It does not uniquely identify tangential force, friction, slip velocity, or the required joint correction.
- Endpoint subtraction loses intermediate events. Contact followed by release can give nearly zero change, as can a stable loaded grasp. Keeping current tokens helps distinguish the present state, but cannot recover the full contact trajectory.
- Demonstrations teach continuation toward demonstrated actions. They provide limited coverage of failed grasps and recovery states; low offline loss does not establish robust correction.

Evaluate on the same new simulator seeds and task settings:

| Comparison | Question |
|---|---|
| Original SmolVLA | Reference success, execution time, and contact failures |
| Copied continuation expert without tactile inputs | Does the additional training or expert alone explain the gain? |
| Continuation with current deformation only | Does touch help without future prediction? |
| Full current-plus-predicted-change model | Does the predictive objective add useful information? |

Also perturb or shuffle tactile inputs and predicted tokens to test whether the policy uses them. Report per-task success and confidence intervals, drops or failed grasps, and inference latency. Report physical reconstruction error separately for non-contact, contact onset, sustained contact, and unloading.

Check prediction against a zero-change forecast and a constant training-target forecast. Use independent evaluation rollouts for generalization measurements; label any training-corpus retrieval results accordingly. Do not count nearby overlapping windows as independent evidence.

Implementation acceptance requires verified references, preserved signed changes, non-collapsed representations, correct future/action alignment, no future-target leakage, initial copied-expert equivalence, and improved closed-loop behavior over the current-deformation-only control.
