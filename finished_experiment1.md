# Finished Experiment 1: Predicted Tactile Conditioning for X-VLA

Recorded on October 6, 2026.

**Outcome:** The trained tactile-refinement policy performed worse than the
baseline X-VLA, according to the experimenter's report. The cause has not yet
been isolated. In particular, the result does not establish whether the action
expert ignored touch or used it in a way that harmed control.

This document records the completed experiment using the corrections supplied
in our discussion. The checked-in code and main README describe an earlier
SmolVLA implementation. They explain the method's development, but are not an
exact record of the updated X-VLA run. Exact numerical results and the updated
run configuration were not available when this record was written.

## 1. Research question

Can a prediction of near-future tactile information improve the actions of an
already-trained X-VLA policy?

The proposed benefit was that current touch, interpreted together with the
visual scene, instruction, and robot state, could anticipate upcoming contact.
The action expert would then use that prediction to refine its actions during
manipulation.

The experiment compared a baseline X-VLA with a tactile-refinement version of
X-VLA. It was not a comparison between SmolVLA and X-VLA.

## 2. Confirmed experimental choices

| Item | Completed experiment |
|---|---|
| Base policy | X-VLA |
| Comparison policy | Baseline X-VLA |
| Tactile prediction endpoint | `t + 20` |
| Predictor inputs | Current tactile information, visual-language context, and proprioceptive state |
| Purpose of Stage 2 | Teach the action expert to use the tactile prediction |
| Direct visual-language access in Stage 2 | Masked for the action expert |
| Original X-VLA parameters | Kept frozen throughout training |
| Reported result | Worse performance than baseline X-VLA |

The `t + 20` endpoint replaces the earlier proposal's `t + 50` tactile target.
It specifies the tactile prediction horizon; it does not by itself specify the
action chunk length, replanning interval, or physical duration.

The exact LoRA rank, target layers, adapter parameter count, and training budget
of the X-VLA run were not supplied. The older SmolVLA configuration's rank-32
FFN LoRA settings should not be treated as confirmed X-VLA settings.

## 3. Model and information flow

Let:

- \(V_t\) be the current visual observations;
- \(L\) be the task instruction;
- \(q_t\) be the current proprioceptive state;
- \(T_t\) be the current tactile observations;
- \(C_t\) be the visual-language features;
- \(\hat z_t\) be the predicted future tactile representation.

At a functional level, the updated predictor is

\[
\hat z_t = F_\phi(E(T_t), C_t, q_t),
\]

with supervision associated with the tactile observation at \(t+20\).
The prediction is then supplied to the action expert through the added tactile
conditioning path. Proprioception helps the predictor interpret contact in the
context of the robot's current configuration.

The earlier repository design uses a frozen tactile encoder, a trainable
projector and predictor, a post-predictor resampler, and tactile cross-attention
inside the action expert. Its original action weights remain frozen while
additional modules learn the refinement. The exact token counts and injection
layers in the updated X-VLA implementation still need to be recorded from that
run's configuration.

The intended prediction represents anticipated touch. It is not a direct
measurement of future contact at deployment. Future tactile frames provide
training supervision; the deployed predictor uses current observations.

## 4. Training procedure and its intended effect

### Stage 1: Ground the tactile predictor

The method first trains a representation of future tactile information. In the
repository's grounding formulation, predicted tokens are supervised using a
contrastive objective and a reconstruction objective:

\[
\mathcal L_{\mathrm{ground}}
= \mathcal L_{\mathrm{contrastive}}
+ \lambda_{\mathrm{rec}}\mathcal L_{\mathrm{reconstruction}}.
\]

The contrastive term associates each prediction with its corresponding future
tactile target. The reconstruction term encourages the representation to retain
spatial contact information. In the updated experiment, the confirmed endpoint
is \(t+20\), and current proprioception is an additional predictor input.

The older implementation constructs targets from signed future-minus-current
depth changes. Whether the X-VLA run retained that exact target encoding and
all grounding-loss details is not independently verified here.

### Stage 2: Align the action expert with predicted touch

Stage 2 was designed to make the action expert learn to consume the grounded
prediction. Its key intervention was to mask the expert's direct
visual-language conditioning while keeping the original X-VLA weights frozen.
Learning therefore occurred through the added trainable components.

The inherited recipe freezes the grounded predictor during this stage and
trains the action-conditioning modules using the demonstrated-action objective.
For a flow-matching expert, this is schematically

\[
\mathcal L_{\mathrm{action}}
= \mathbb E\!\left[
\left\|v_\theta(x_\tau,\tau,q_t,\hat z_t)-u^*\right\|^2_M
\right],
\]

where \(x_\tau\) is a sampled flow input, \(u^*\) is its target velocity, and
\(M\) excludes padded action entries. The direct visual-language input to the
expert is omitted here to show the Stage 2 mask; visual information can still
reach the expert indirectly through the predictor.

The intended reasoning was: reducing access to the original visual-language
path would encourage the expert to learn the new tactile interface before
joint training.

The earlier refinement implementation also uses an intermediate baseline action
trajectory as a handoff to the tactile continuation. If retained in X-VLA, that
handoff provides another route for scene and task information. The exact X-VLA
flow construction and denoising schedule were not supplied.

### Stage 3: Restore direct context and jointly adapt the added path

The proposal's final stage restores direct visual-language conditioning and
allows the predictor and action-conditioning modules to adapt together under
the action objective. The original base-policy weights remain frozen in this
recipe. Thus, “joint” training refers to the added components, not full
fine-tuning of X-VLA.

The evaluated checkpoint's stage was not specified. This record therefore does
not assume that the reported result came from Stage 2 or from a completed
Stage 3 run. That distinction matters because a Stage 2 checkpoint may still
apply the visual-language mask during inference.

## 5. Observed result

| Policy | Reported performance |
|---|---|
| Baseline X-VLA | Better than the tactile-refinement policy |
| X-VLA with predicted tactile refinement | Worse than baseline X-VLA |

No per-task scores, rollout counts, uncertainty estimates, or training curves
were provided. The size and statistical reliability of the difference cannot
be reconstructed from the available record. Matching of seeds, controllers,
replanning intervals, and evaluation budgets also remains unverified.

The supported conclusion is that the completed configuration did not deliver
the expected improvement in the reported comparison. It does not establish
that tactile sensing or future-tactile prediction is generally unhelpful.

## 6. Why Stage 2 may not have achieved its purpose

These are explanations to test, not established causes of the result.

**Using the predicted tokens does not necessarily mean using touch.** The
predictor receives vision, language, and proprioception as well as tactile
observations. Its output can carry scene, task, and pose information. The action
expert could learn to use those features while remaining insensitive to the
tactile input. Masking its direct visual-language path does not eliminate that
possibility.

**The action objective did not explicitly compare correct and mismatched
tactile conditioning in the original recipe.** Demonstration error can decrease
without a useful contribution from touch. A low action-training loss therefore
does not demonstrate that the expert learned the intended tactile dependence.

**Prefix masking changes the pretrained controller's operating conditions.**
Even when added residuals start at zero, removing an existing conditioning path
can change the output. The trainable components may spend capacity compensating
for missing context. If direct context is restored later, those compensations
may need further adaptation. Frozen original weights do not guarantee preserved
behavior of the complete adapted policy.

**An accurate forecast is not automatically useful action guidance.** A
representation can predict aspects of future contact without providing the
information needed to correct a particular action. Conversely, the expert can
be sensitive to tactile tokens in ways that reduce task success. Forecast
quality, action sensitivity, and useful closed-loop control are separate claims.

The experimenter's working hypothesis is that Stage 2 failed to establish
useful tactile dependence. The reported performance alone cannot distinguish
that hypothesis from harmful tactile use, predictor errors, or evaluation
differences.

## 7. Follow-up ideas are separate from this experiment

The later discussion proposed retaining visual-language access and adding a
correct-versus-mismatched tactile adherence objective. A capped version would
also require correct-touch action predictions to improve over a frozen
reference. This was proposed after the reported failure and is not an observed
result of Experiment 1.

Similarly, LoRA on both attention and MLP projections, and full fine-tuning of
the action expert, were discussed as follow-up choices. Neither discussion
establishes the exact training scope used in the completed run.

The proposed next objective is documented separately in
[tactile_adherence_proposal.md](docs/tactile_adherence_proposal.md).

## 8. Details needed for a reproducible record

- Updated X-VLA implementation revision and complete configuration.
- Baseline and evaluated refinement checkpoint identifiers, including stage.
- Tactile target encoding, predictor architecture, and grounding diagnostics.
- Trainable parameter groups, LoRA configuration, and per-stage update budgets.
- Action horizon, flow schedule, and execution/replanning settings.
- Per-task evaluation scores, episode counts, and seeds.
- Correct-touch, touch-only substitution, and original-baseline comparisons.

The earlier architecture is described in [README.md](README.md) and
[IMPLEMENTATION.md](IMPLEMENTATION.md). Those files remain references for the
SmolVLA predecessor and should not override the confirmed X-VLA, `t + 20`, and
proprioception settings recorded above.
