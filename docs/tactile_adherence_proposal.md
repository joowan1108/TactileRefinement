# Proposed tactile adherence objective for X-VLA

This is a proposed experiment, not a validated loss or an implementation change.
It targets the user's updated X-VLA policy: a tactile predictor with a t+20
target and current proprioception. The checked-in executable implementation is
an older SmolVLA variant; its architecture and 50-step tactile horizon should
not be assumed to describe the updated experiment.

The aim is to make accurate tactile conditioning improve action prediction
while retaining the original visual-language conditioning pathway. Stage 2
prefix masking alone does not establish this: the predicted tokens and the
baseline handoff can carry visual/task information even when the expert's
direct prefix is masked.

## Connection to RA-VLA

[RA-VLA, Equations (5)–(7)](https://arxiv.org/pdf/2608.25585v1#page=5)
compares flow-field regression error with relevant and irrelevant retrieved
context. Its adherence loss is

\[
L_{adhere}=[m+L_{rel}-L_{irrel}]_+,
\qquad L=L_{rel}+\lambda L_{adhere}.
\]

We propose replacing the retrieved-context intervention with a tactile-only
intervention, and capping the negative error at a frozen reference error.
These modifications are our proposal, not results established by RA-VLA.

## Construct a paired tactile intervention

For anchor i, let C_i contain current RGB/language features, q_i be current
proprioception, and T_i contain the ordered left/right tactile observations.
Use the already-grounded predictor F to produce

\[
z_i^+=F(C_i,q_i,T_i),\qquad
z_i^-=F(C_i,q_i,T_j).
\]

Only the tactile observation changes. The donor's RGB, language, state, actions,
flow sample, and baseline handoff must not replace the anchor's values. Pass
both predictions through the same resampler, if one is present. Whole-latent
shuffling is not the primary control because the latent also contains visual
and proprioceptive information. Both branches use predicted tokens; future
ground-truth tactile maps are never substituted as action-expert inputs.

Select donors from another episode of the same task, with similar robot pose,
gripper aperture, visible scene and task phase, but meaningfully different
contact. Keep the two fingers together. Exclude duplicates and identical
no-contact samples. Prefer examples where contact differences plausibly affect
the demonstrated near-term action; different sensor readings alone do not
imply different correct actions. Skip an anchor's adherence term if no suitable
donor exists. Random cross-task donors and arbitrary Gaussian token noise make
too easy an inconsistency-detection problem.

Before training, measure whether the frozen predictor responds to these
touch-only interventions. If z+ and z- are identical, the action expert cannot
learn to distinguish them; fix the predictor or sampling first. A nonzero latent
difference is necessary for this experiment but does not establish useful touch
information.

## Score both conditions against the same action target

Draw the noisy action input x_s, flow time s, noise, and target velocity u once
for each anchor, using the existing X-VLA training objective. If retaining a
baseline-to-demo continuation bridge, also compute its handoff once and reuse
it. Do not change the flow convention or target between branches.

\[
v_i^+=v_\theta(x_s,s,C_i,q_i,z_i^+),\quad
v_i^-=v_\theta(x_s,s,C_i,q_i,z_i^-),\quad
v_i^0=v_{base}(x_s,s,C_i,q_i).
\]

All three branches have the original visual-language context available. The
reference is the frozen original X-VLA route with every added adapter bypassed;
zeroing tactile tokens while retaining learned adapters is not that reference.
Share stochastic masks, or disable action-head dropout for these paired
forwards, so stochasticity cannot create the measured gap.

Let d_W be mean squared velocity error on normalized physical action dimensions,
with padded actions/dimensions excluded. As an initial design, use the next
min(20, action_horizon) valid action slots for the adherence score; retain the
ordinary action loss over the full chunk. This temporal weighting is a design
choice to test, not a claim that a t+20 endpoint supervises every intermediate
contact event.

\[
e_i^+=d_W(v_i^+,u_i),\qquad
e_i^-=d_W(v_i^-,u_i),\qquad
b_i=\operatorname{sg}[d_W(v_i^0,u_i)].
\]

Compute these errors per sample before applying the hinge. The frozen reference
is scored on the same flow/bridge distribution. If that distribution differs
from original X-VLA training, beating this reference error also measures
adaptation to the new objective; include a separately trained no-tactile control
with the same objective to isolate tactile gains.

## Proposed capped margin

\[
m_i=\rho b_i,\qquad 0<\rho<1,
\]

\[
\boxed{L_{TA}=\frac{\sum_i w_i
 [m_i+e_i^+-\min(e_i^-,b_i)]_+}
 {\max(1,\sum_i w_i)}}
\]

For the initial experiment, w_i is a fixed binary eligibility label: valid
contact or imminent contact, and a suitable donor. Include stable contact such
as holding an object; a small future depth difference does not imply touch is
irrelevant. Eligibility may use recorded future contact during training, but
must not depend on a trainable gate that can suppress its own penalty. Keep
ordinary action supervision on all samples, including those with w_i=0.

Use

\[
L= L_{action}^{+}+\lambda_{TA} L_{TA}.
\]

A zero adherence penalty on an eligible sample requires both

\[
e_i^+\le e_i^- -m_i,\qquad e_i^+\le b_i-m_i.
\]

Thus correct tactile conditioning must outperform both the mismatched condition
and the frozen reference on this supervised metric. Inflating the negative
error beyond b_i cannot help. The adaptive margin shrinks near an already-good
baseline instead of demanding a fixed positive gap at zero error. There is no
division by a small action error.

For example, b=0.10 and rho=0.05 give m=0.005. Keeping e+=0.10 and increasing
e- to 0.30 satisfies the uncapped ranking loss, but leaves a 0.005 penalty here.
With e+=0.09 and e-=0.10 the capped penalty is zero.

This caps an incentive to degrade the negative branch; it is not a bound on
that branch's actual outputs. Nor does zero training loss guarantee improved
closed-loop success or exclude detecting implausible donor inputs. Donor
quality, held-out evaluation, and positive-branch improvement remain essential.

## Gradient and training choices

1. Restart action adaptation from original X-VLA plus the grounded t+20
   predictor for the clean experiment. Retain direct vision/language access
   throughout. An existing masked-stage checkpoint can be a separate recovery
   experiment, but cannot isolate the curriculum's effect.
2. In Stage 2, freeze the tactile encoder/predictor and original X-VLA weights;
   train the intended action adapters and resampler. The main action loss uses
   correct predicted tactile conditioning.
3. Backpropagate the adherence loss through both e+ and e-. Detach only the
   frozen reference and fixed margin/eligibility. Below the cap, the negative
   branch supplies a contrastive gradient; above the cap it receives no direct
   incentive from this term to increase its error. Detaching all of e- would
   instead make this largely a reweighted positive regression objective.
4. Start with rho=0.05 and lambda_TA=0.1 as unvalidated pilot values. Warm up
   the coefficient after the initially zero tactile residual begins learning.
   Inspect gradient magnitudes relative to the main action loss; these values
   are not tuned for X-VLA, this action normalization, or a custom bridge.
5. Retain an adherence term during Stage 3; removing it may permit a return to
   ignoring tactile. If the predictor is unfrozen, retain a grounding term
   on true future tactile targets and monitor forecast quality, since action
   ranking alone does not preserve predictive semantics. First test the frozen
   predictor version to avoid changing both representation and interface.

Pseudocode for already-computed per-sample errors:

```python
# e_pos, e_neg, e_base: [B], with the SAME masks/normalization/flow target.
# eligible: [B], fixed binary labels; action_loss covers the entire batch/chunk.
reference = e_base.detach()
margin = rho * reference
negative_score = torch.minimum(e_neg, reference)  # Keep e_neg attached.
penalty = torch.relu(margin + e_pos - negative_score)
weights = eligible.to(penalty.dtype)
adherence = (weights * penalty).sum() / weights.sum().clamp_min(1)
loss = action_loss + lambda_ta * adherence
```

This adds one mismatched expert forward and a no-gradient reference forward,
plus donor tactile prediction. Cache/reuse the anchor's vision features and
handoff. The loss adds no inference-time branch.

## What would count as success

Compare original X-VLA, the existing masked-stage recipe, a prefix-visible
action-only control, prefix-visible training with ordinary RA-VLA-style
ranking, and the capped variant. Keep initialization, data, update budget,
execution schedule, and evaluation seeds matched where applicable. Add a
separately trained no-tactile adapter control if reflow or extra adaptation
could explain a gain. Retain X-VLA's actual action dimensions and normalization;
do not inherit the older SmolVLA implementation's nine-dimensional assumption.

Log e+, e-, b, positive improvement b-e+, tactile gap e--e+, eligible count,
margin violation rate, and how often the negative score hits its cap. Evaluate
these on held-out episodes, stratified by flow time and contact phase. A larger
gap produced only by larger e- is not success. Inspect the actual sampled action
chunks as well: one-step regression error does not establish integration quality.

In rollouts, require correct-touch performance to improve over the corresponding
prefix-visible action-only control and assess whether matched touch substitutions
reduce that improvement. Report contact-task gains and pre-contact regressions
separately. Zero-touch degradation and attention weights alone are insufficient.

If matched visual/proprioceptive observations in the demonstrations never require
different actions across contact states, forcing a tactile gap may create
artificial dependence. No adherence loss can manufacture missing evidence that
touch is useful; recovery demonstrations or controlled contact variations may
then be necessary. This remains an experiment to establish useful tactile
dependence, not a guarantee that the expert will exploit it correctly.
