# Value-factorization algorithms for MATE

This change adds five CTDE algorithms and a FOCUS variant of each:
`FACMAC`, `FACMAC_FOCUS`, `RESQ`, `RESQ_FOCUS`, `RISKQ`,
`RISKQ_FOCUS`, `QFREE`, `QFREE_FOCUS`, `CONCAVEQ`, and
`CONCAVEQ_FOCUS`.  They are registered as RLlib trainers and have matched
camera-team entry points under `examples/<algorithm>/camera`.

This document distinguishes implementation validation from performance
reproduction.  Passing unit/smoke tests establishes interface, gradient, and
checkpoint correctness; it does **not** establish the paper's benchmark
results or a coverage-rate improvement.

## Sources and scope

- FACMAC: [official code](https://github.com/oxwhirl/facmac) and
  [paper](https://arxiv.org/abs/2003.06709).
- ResQ: [official code](https://github.com/xmu-rl-3dv/ResQ) and
  [paper](https://arxiv.org/abs/2205.12106).
- RiskQ: [official code](https://github.com/xmu-rl-3dv/RiskQ) and
  [paper](https://arxiv.org/abs/2306.10857).
- QFree: [paper](https://arxiv.org/abs/2311.00356); the recurrent agent and
  replay/target infrastructure starts from the repository's QPLEX path, but
  its transformation and losses are new code.
- ConcaveQ: [paper and supplementary material](https://ojs.aaai.org/index.php/AAAI/article/view/29695);
  the recurrent local utilities and replay/target infrastructure reuse the
  same MATE components as QPLEX, while the concave and unrestricted estimators
  are new code.

Only algorithm components were adapted.  No external training framework or
repository was copied into MATE.

## Shared MATE integration

`ray/rllib/agents/value_factorization` contains the shared grouped-policy,
episode replay, recurrent unroll, target-return, mixer, and FOCUS components.
All ten trainers use the same environment pipeline:

1. `MultiAgentTracking-v0`, `MATE-4v8-9.yaml`, dense cooperative reward;
2. the existing HRL-QPLEX camera interface: each camera chooses one of 256
   discrete target-subset actions and `HierarchicalCamera` deterministically
   converts that selection into continuous pan/zoom commands, frame skip 5;
3. relative/rescaled local observations and the existing centralized-training
   wrapper;
4. four homogeneous cameras grouped into one recurrent policy;
5. complete-episode replay, 5,000 environment-step warm-up, and a 1,024-step
   training sample in the comparison configuration.

The policy validates that rewards are shared across the grouped camera agents.
Online action selection uses only each camera's local observation, recurrent
hidden state, previous action, and agent identity.  Global state is consumed
only by training-time mixers/critics.  Target next observations and next state
are used only on the bootstrap side.  Tests perturb centralized state and next
observations independently to check decentralized execution and absence of
future-state leakage in current recurrent outputs.

An earlier validation pass used direct 25-action camera control.  Although it
was internally stable, its coverage was not comparable to the repository's
HRL-QPLEX results because HRL-QPLEX learns target selection and receives the
analytic low-level tracker.  Those runs are retained as debugging artifacts
only and were superseded by the HRL configuration above.

Target networks are copied every 200 **learner updates**, matching the
episode/update counter used by the original PyMARL-style implementations.  The
counter, all modules, optimizers, and exploration state are included in policy
checkpoints.  Replay does not begin until the configured sampled-step warm-up
has completed.

## Algorithms

### FACMAC

The implementation uses a recurrent decentralized categorical actor, a
centralized per-agent action critic, and a non-negative hypernetwork mixer.
Critic targets are TD(lambda) targets from target actor, critic, and mixer
networks.  The actor uses a straight-through Gumbel-Softmax joint action and
maximizes the centralized mixed value while critic/mixer parameters are frozen.
Rollout exploration follows the official discrete FACMAC policy:
`(1-epsilon) softmax(logits) + epsilon uniform(available actions)`.  Actor and
critic use separate Adam optimizers with the official epsilon.

### ResQ

ResQ has separate main, residual, and unrestricted centralized recurrent
utilities and mixers.  The main branch is monotonic.  The residual branch is
an unrestricted state-conditioned mixer, constrained non-positive using the
official v3 absolute-value parameterization, and is active only for joint
actions that differ from the current decentralized greedy joint action.  An
unrestricted centralized estimator supplies the double-Q/TD(lambda) target.
The optimized loss is the weighted sum of main-plus-residual TD error,
centralized TD error, and the no-opt residual constraint.

### RiskQ

Each recurrent agent predicts 32 QR-DQN atoms per action.  Decentralized
actions maximize the Wang-distorted expectation.  The state-conditioned
attention mixer uses the official query/key, multi-head weight, and constant
value networks; distortion weights are `g'(tau_i) / N` and negative distortion
parameters reflect quantile fractions.  Double action selection, the target
distributional mixer, TD(lambda), and pairwise quantile-Huber regression are
used for learning.  For MATE, per-camera key blocks are sliced from the known
camera section of the centralized state (`start=13`, `width=9`), with detached
local-observation keys as a guarded fallback if a different state layout is
used.

### QFree

The local recurrent dueling network produces `V_i` and raw `A_i`; subtracting
the maximum available-action advantage makes the locally greedy advantage
exactly zero and all other local advantages non-positive.  A state-conditioned
positive affine transformation is applied agent-wise, followed by separate
unrestricted feed-forward value and advantage mixers.  Learning combines the
double-Q one-step TD loss, the zero joint-advantage loss for greedy actions,
and the non-positive joint-advantage constraint for non-greedy actions.

Equation (19) prints `min(A_tot, 0)^2`, while the theorem, constraint, and
adjacent prose require penalizing `A_tot > 0`.  This implementation uses
`relu(A_tot)^2`; using the printed expression would penalize already-valid
negative advantages and leave violations unpenalized.  This interpretation is
covered by a focused loss test.

### ConcaveQ

ConcaveQ contains:

- recurrent decentralized soft policies and separate recurrent local Q
  utilities;
- a state-conditioned input-concave neural mixing network with non-negative
  hidden-to-hidden weights;
- a separately trained unrestricted joint-Q estimator;
- exact discrete coordinate ascent, evaluating every available action for one
  agent at a time while holding the other agents fixed;
- the projected concave target, unrestricted target, policy-value estimator,
  entropy-temperature update, and decentralized local policy update described
  in the paper.

One complete coordinate sweep is used per target optimization step, as in the
paper's discrete algorithm.  Hypernetwork-generated positive ICNN matrices are
divided by their fan-in and initialized with fan-in-scaled Xavier weights.  The
positive parameterization and represented concave function class are unchanged;
this MATE-specific reparameterization prevents the large centralized state from
making the four-layer product overflow at initialization.
For the HRL action space, exhaustive candidates are evaluated in chunks of 32.
This is memory scheduling only: every one of the 256 actions is still evaluated
for every coordinate, and tests require chunked and unchunked search to return
identical actions and values.

## FOCUS integration

FOCUS first predicts the future target occupancy from the current centralized
training state.  Counterfactual coverage gains define normalized camera
responsibilities `rho`; signal confidence interpolates between a uniform
backward multiplier and `N * rho`:

```text
w_i = 1 - confidence * eta + confidence * eta * N * stop_gradient(rho_i)
x_focus = stop_gradient(x) + w * (x - stop_gradient(x))
```

`x_focus` is exactly equal to `x` in the forward pass.  Consequently each
backbone keeps its original scalar learning objective and target values; only
per-agent backward credit is changed.  The learned occupancy belief has a
separate supervised loss and optimizer.  Integration points are:

- FACMAC: straight-through actor actions before the centralized action critic;
- ResQ: chosen main, residual, and centralized local utilities;
- RiskQ: chosen current quantiles before distributional mixing;
- QFree: transformed chosen advantages on the TD path (IGM constraints remain
  the original constraints);
- ConcaveQ: per-agent soft local actor values before the policy-value operator.

FOCUS is disabled in every baseline config.  It supplies no extra input to the
execution policy and does not alter action selection.  Future state is used as
the belief-training label only and is detached from the backbone objective.

### Learned-belief safeguard modes

An earlier 100,000-step FACMAC+FOCUS diagnostic showed that the learned mean
improved over time but remained worse than a persistence forecast (reuse the
current target position) at every horizon.  The baseline algorithms are
unchanged.  The five new value-factorization +FOCUS configurations, the
MAPPO/TarMAC/SMPE2 policy-gradient +FOCUS configurations, and the HMVFE/HiTMAC
CTCE +FOCUS configurations expose one mutually exclusive option:

- `belief_safeguard_mode="zero"` (default): zero/persistence initialization
  only; the learned residual is used fully after it starts training.
- `belief_safeguard_mode="zero_ema"`: zero initialization plus the causal EMA
  performance gate.
- `belief_safeguard_mode="zero_ema_blend"`: zero initialization, EMA gate,
  and progressive learner-update schedule.

The modes compose the following safeguards:

1. The displacement output head is initialized to zero, so the initial mean is
   exactly `current_position` for every target and horizon.  The Gaussian head
   remains trainable and receives a non-zero supervised gradient on its first
   update.
2. In `zero_ema_blend`, the position supplied to FOCUS is
   `current_position + schedule * gate * learned_displacement`.  The schedule
   stays at zero for 20 learner updates and ramps linearly to one over the next
   80.  The NLL is always computed from the unblended learned mean, so a zero
   FOCUS blend does not stop belief learning.
3. In both EMA modes, the performance gate is zero until an EMA of the learned point error beats
   an EMA of persistence error.  It ramps from zero to one as relative error
   improvement grows from 0% to 20%.  The current batch updates these EMAs only
   **after** its responsibility target has been constructed, so its future
   labels can affect only later learner updates.  EMA values and counters are
   included in checkpoints.

The option is selectable from Python config or with
`--belief-safeguard-mode {zero,zero_ema,zero_ema_blend}` on the value/policy
+FOCUS launchers and sequential suite runner.  The standalone HMVFE/HiTMAC
entrypoints use the namespaced spelling `--focus-belief-safeguard-mode`.
The learner reports raw learned-mean error, effective blended-mean error,
schedule, performance gate, effective blend, and pre/post-update EMA values.
This mechanism uses centralized future state only as a supervised training
label and adds no execution-time input.

### Policy-gradient and CTCE safeguard lifecycle

The safeguard is not limited to the new value-factorization learners:

- MAPPO, TarMAC, SMPE2, and their WM2 model implementations pass the
  persistence initialization into the shared continuous occupancy model.
  Trajectory postprocessing reads a gate synchronized from the learner but
  never updates it.  The supervised learner loss updates the EMA after the
  batch's responsibility weights have already been materialized.  In PPO,
  “learner update” means one optimizer/minibatch loss call.  The estimator's
  registered tensor buffers carry the EMA and update counter through this
  repository's RLlib weight synchronization and checkpoints (with a
  backward-compatible default when loading older checkpoints). MAPPO+FOCUS now uses a non-zero occupancy
  loss coefficient (`0.01`); otherwise its displacement residual and EMA gate
  could never improve beyond persistence.
- HMVFE reuses the continuous occupancy predictor.  Per-timestep CTCE actor
  credits are read-only with respect to the gate; the full rollout sequence
  updates the EMA exactly once.  Its checkpoint contains both estimator
  parameters and explicit runtime gate state.
- HiTMAC predicts binary future visibility rather than target position.  Its
  correct persistence baseline is therefore “the current visibility bit stays
  unchanged.”  The belief network predicts a residual logit around that
  baseline, with a zero-initialized output head.  The raw residual logits are
  always supervised; only the logits consumed by camera responsibility are
  gated/blended.  EMA state is updated once after a completed rollout and is
  saved in checkpoints.

The default remains `zero` everywhere listed above.  In that mode no EMA is
updated and no progressive schedule is applied.  `zero_ema` and
`zero_ema_blend` opt into those mechanisms explicitly.

### Categorical position-belief backend

FOCUS now also supports `belief_distribution="categorical_grid"` in the five
new value-factorization policies, QMix/DuelMIX/QPLEX compatibility paths,
MAPPO/TarMAC/SMPE2 (including their WM2 models), and HMVFE. The default remains
`gaussian` for checkpoint/config compatibility. HiTMAC is not routed through
this backend because its belief target is binary visibility rather than a
two-dimensional target position.

For each horizon and target the model predicts residual logits on an odd 2-D
displacement grid. A fixed symmetric Gaussian-shaped prior over grid cells is
centered at zero displacement. With the `zero` safeguard, the output head is
zero initialized, so the initial expected displacement is zero and the
converted expected position equals the persistence forecast. For a future
displacement `d`, the normalized soft label is

`y_k = softmax_k(-||g_k - d||^2 / (2 sigma_label^2))`.

The supervised loss is spatial soft-label categorical NLL plus
`belief_position_loss_coeff` times Smooth-L1 of the expected-position error,
normalized by the configured maximum displacement. Position MAE, top-1 cell
MAE, categorical NLL, expected-position auxiliary loss, and predicted spread
are logged separately. Expected positions are used for MAE diagnostics and the
causal EMA comparison only.

FOCUS credit does **not** collapse the categorical distribution to its mean.
It computes visibility and unique camera gain for every grid cell, weights the
gain by exact cell probability and horizon weight, and sums the result in
chunks. This preserves multimodal occupancy information. EMA/blending, when
enabled, interpolates full categorical probabilities with the persistence
prior before this integration. The supervised raw distribution is unchanged
by that interpolation. Future state remains a learner-only label; the credit
distribution is detached and no additional execution-time information is
introduced.

Relevant options are `belief_grid_size` (odd), `belief_grid_max_delta`,
`belief_grid_label_std`, `belief_persistence_prior_std`,
`belief_position_loss_coeff`, and `belief_grid_chunk_size`. They are exposed
by the value-factorization suite and MAPPO/TarMAC/SMPE2 launchers; HMVFE uses
the corresponding `focus_belief_*` names. A finite grid necessarily clips
probability support to its configured displacement square; this is an explicit
approximation/deviation from an unbounded continuous density.

## Validation

The dependency-free synthetic suite is:

```bash
conda run -n mate python -m scripts.smoke_value_factorization --focus
```

It covers mixer shapes and signs, transformation behavior, distortion weights,
quantile loss, ICNN concavity, coordinate ascent, forward-preserving FOCUS
gradients, target update cadence, all ten policy learner steps, finite gradients,
and checkpoint round trips.  It passed for all ten configurations.

The real-environment suite is run sequentially on this host:

```bash
WANDB_DISABLED=true conda run -n mate \
  python -m scripts.smoke_value_factorization_mate \
  --checkpoint-dir /tmp/mate-value-factorization-hrl-smoke --restore-check
```

It collected 228 real MATE-HRL environment steps for each configuration,
executed one learner update, verified finite scalar learner statistics, saved
all ten checkpoints, and restored them.  ConcaveQ and ConcaveQ+FOCUS were run
in separate processes after enabling exact candidate chunking; peak resident
memory fell from about 4.2 GB to below 0.8 GB.  These runs are
interface/gradient checks; the single random-policy episode's coverage is not
performance evidence.

Additional architecture-specific regression checks passed on 2026-10-11:

```bash
conda run -n mate python -u tests/test_hmvfe_focus.py
conda run -n mate python tests/test_hitmac_focus_safeguards.py
conda run -n mate python -u tests/test_policy_focus_safeguards.py
```

They verify HMVFE belief gradients and exactly-one EMA commit per rollout,
HiTMAC's exact visibility-persistence initialization and residual gradient,
policy rollout/learner causal separation, zero-mode EMA disablement, and
runtime-state checkpoint round trips. Full real-MATE categorical runs for the
newly wired policy/CTCE paths have not yet been executed; these synthetic
checks are not performance evidence.

Categorical belief checks also passed on 2026-10-11:

```bash
conda run -n mate python tests/test_categorical_focus_belief.py
conda run -n mate python scripts/smoke_value_factorization_mate.py \
  FACMAC_FOCUS --belief-distribution categorical_grid \
  --belief-grid-size 5 --restore-check \
  --checkpoint-dir /tmp/mate-categorical-focus-smoke
```

The dependency-free tests verify exact persistence expected position,
normalized probabilities, finite NLL/spread, non-zero output-head gradient,
HMVFE factory wiring, checkpoint round-trip, and a constructed pair of
multimodal distributions with identical expected position but different exact
FOCUS credit. The real MATE-HRL smoke collected 228 steps, completed one
learner update, saved and restored the trainer checkpoint. All scalar learner
statistics were finite; categorical NLL was 2.77899, the auxiliary position
loss was 0.01298, belief gradient norm was 0.01969, and the initial learned and
persistence position errors were both 58.8589. This is mechanism/stability
evidence only, not coverage performance evidence. The same real-MATE
save/restore smoke also passed with the intended 21x21 grid.

With `zero_ema_blend`, a focused FACMAC+FOCUS MATE-HRL
smoke collected 228 steps, completed one learner update, saved a checkpoint,
and restored it.  At initialization, learned error, effective error, and
persistence error were all exactly 58.8589 world units; effective blend was
zero, the belief gradient norm was finite and non-zero (5.5571), and the EMA
counter changed from zero to one only after responsibility construction.  The
synthetic suite again passed all ten policies and their checkpoint round trips.

A subsequent `zero_ema_blend` FACMAC+FOCUS short run completed 21,456 HRL steps and 34 learner
updates with zero process errors and finite learner statistics.  Its belief
handoff evolved as follows (errors are weighted point errors):

| Steps | Updates | Raw error | Persistence | Schedule | Performance gate | Effective blend |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 5,204 | 1 | 107.571 | 107.571 | 0.0000 | 0.0000 | 0.0000 |
| 10,678 | 12 | 103.670 | 103.638 | 0.0000 | 0.0065 | 0.0000 |
| 15,967 | 23 | 106.210 | 106.958 | 0.0250 | 0.0228 | 0.0006 |
| 21,456 | 34 | 105.764 | 106.790 | 0.1625 | 0.0308 | 0.0050 |

Thus the learned mean began exactly at persistence, retained non-zero belief
gradients, and became slightly better than persistence without causing an
abrupt switch in the position used for credit.  Final rolling training
coverage was 0.38601; this four-report run had no scheduled deterministic
evaluation and is mechanism/stability evidence, not a performance claim.

After changing the default to `zero`, synthetic smoke again passed 10/10
policies.  A 228-step real MATE-HRL smoke verified initial raw error equals
persistence error (58.8589), both schedule and performance multipliers resolve
to one, the effective blend is one, the belief gradient remains finite and
non-zero (5.5571), and checkpoint restoration succeeds.

The following table is from the superseded **direct-control** validation and
must not be compared with HRL-QPLEX.  A corrected 10,000-step FACMAC run
completed after the replay warm-up
fix.  Its first learner report occurred at 5,173 sampled steps, confirming that
the 5,000-step warm-up is enforced; the run ended at 10,628 steps with finite
learner statistics and a checkpoint.  A subsequent sequential suite completed
the other nine configurations.  Every process returned zero, every trial ended
in Ray's `TERMINATED` state, and every scalar learner statistic in every report
was finite.  The final rolling training metrics were:

| Configuration | Steps | Learner updates | Coverage rate |
| --- | ---: | ---: | ---: |
| FACMAC | 10,628 | 12 | 0.25098 |
| FACMAC + FOCUS | 9,438 | 9 | 0.27742 |
| ResQ | 9,121 | 9 | 0.29210 |
| ResQ + FOCUS | 9,230 | 9 | 0.30296 |
| RiskQ | 8,451 | 7 | 0.26957 |
| RiskQ + FOCUS | 9,210 | 9 | 0.27422 |
| QFree | 9,183 | 9 | 0.26516 |
| QFree + FOCUS | 9,067 | 9 | 0.25839 |
| ConcaveQ | 9,424 | 9 | 0.29156 |
| ConcaveQ + FOCUS | 9,310 | 9 | 0.30753 |

These direct-control values are rolling training coverage, not matched deterministic
evaluation.  The requested evaluation interval was five iterations, while the
8,000-step trials ended after four; RLlib's displayed `.nan` evaluation fields
are therefore “not evaluated yet” placeholders, not learner NaNs.  The short
runs show positive relative FOCUS differences for FACMAC (+10.53%), ResQ
(+3.72%), RiskQ (+1.72%), and ConcaveQ (+5.48%), and a negative difference for
QFree (-2.56%).  They are useful stability evidence only: update counts and
sampled episode windows differ, and one short seed cannot establish an
improvement.

## Reproduction status

- Implementation/unit checks: passed.
- Real MATE smoke, gradients, numerical finiteness, checkpoint save/restore:
  passed for 10/10 configurations.
- Direct-control short training: passed for 10/10 configurations but superseded
  because its action interface did not match HRL-QPLEX.
- Matched HRL short training: passed for 10/10 configurations with zero process
  failures and finite final learner statistics.  Final rolling training
  coverage (about 10.6--10.9k steps) was FACMAC 0.36638 / +FOCUS 0.38205,
  ResQ 0.37225 / +FOCUS 0.36680, RiskQ 0.37684 / +FOCUS 0.36298, QFree
  0.36906 / +FOCUS 0.36679, and ConcaveQ 0.36783 / +FOCUS 0.37205.  These
  two-report runs contain no scheduled deterministic evaluation and are
  stability checks, not performance conclusions.
- The initial direct-control 100,000-step W&B suite was stopped and superseded.
  The matched HRL suite was paused after FACMAC at 79,997 sampled steps; its
  deterministic evaluation coverage was 0.50773.  A matched standalone
  FACMAC+FOCUS run then completed 102,087 steps.  Its evaluation coverage was
  0.54773 at 26,565 steps, 0.44322 at 53,294, and 0.45645 at 80,430.  Thus the
  unguarded learned belief produced a promising early peak but did not retain
  it.  The conservative belief safeguards above were implemented after this
  run and do not retroactively validate its performance.
- A later matched FACMAC+FOCUS run with the default `zero` safeguard completed
  101,758 HRL steps and 199 learner updates. Final rolling training coverage
  was 0.52025 and final deterministic evaluation coverage was 0.52285; the
  best scheduled evaluation was 0.54773 at 26,637 steps. At the final learner
  report, the Gaussian belief weighted position error was 107.584 versus
  persistence 105.505, so zero initialization alone did not make the learned
  Gaussian predictor outperform persistence. This run predates the categorical
  backend and motivates its controlled comparison; it is not evidence for
  categorical performance.
- The requested `>= 3%` FOCUS coverage improvement: **not established yet**.

At least multiple seeds and uncertainty estimates are needed to call a
coverage-rate difference consistent; a single 100,000-step seed is reported as
evidence, not as a successful reproduction claim.

## Controlled MATE-4v8-9 report campaign (2026-10-14--16)

The final reporting campaign uses the homogeneous-camera HRL interface and
only the repository's existing `MATE-4v8-9.yaml` environment.  Earlier
generated T100/T70/T40 variants were rejected as out of scope and none of
their results are included in the final report.  The value-factorization
entrypoint accepts an explicit environment selector and writes the selected
YAML into both training and evaluation `env_config`.

CTCE +FOCUS execution is now belief-conditioned during both collection and
deterministic evaluation:

- HMVFE infers responsibility from the current global state, converts it to a
  bounded relative-log camera-logit bias, and uses the identical path in its
  periodic evaluator.  Uniform responsibility is an exact no-op.  The next
  state is retained only as the detached supervised belief label and is never
  used for the action or actor-credit weights.
- HiTMAC forms its pair-action bias from current-state belief logits after the
  configured safeguard, uses the same path during deterministic evaluation,
  and reconstructs the belief/safeguard state for evaluation-only checkpoint
  loading.  Predictive confidence is causal; future visibility is used only in
  the auxiliary BCE target.

Dependency-free regression runs passed three HMVFE causal/execution tests and
four HiTMAC safeguard/execution tests.  These tests establish execution-path,
gradient, and leakage properties, not task performance.

The first eval-first summarizer still inherited an older-RLlib behavior:
RLlib forward-filled the last evaluation result into every following training
row, so a `last-k` row average counted one evaluation several times.  The
reader now uses `evaluation_interval` from `params.json` to select actual
evaluation iterations (with consecutive-value collapse only as a fallback).
The initial one-seed, 100k-step hard-environment runs were used only as pilots
to prioritize scarce hardware.  FACMAC remained a 100k pilot in the final
campaign; ConcaveQ was stopped after 149,732 steps because evaluation coverage
decreased while loss magnitude increased.  Neither is treated as a completed
250k backbone/+FOCUS comparison.

The active report protocol is sequential because a measured QFree/HMVFE
parallel attempt left less than 0.5 GiB available memory and caused heavy swap.
A second measured RiskQ/HMVFE attempt initially had more headroom, but available
memory still fell from 2.7 GiB to 0.38 GiB and swap grew rapidly; the HMVFE run
was stopped at 6,000 steps and marked `interrupted`.
Every run called complete must have queue status `complete` and at least
250,000 observed environment steps.  The report command is:

```bash
conda run -n mate python scripts/build_focus_visibility_report.py \
  outputs/report_250k_20261014 outputs/report_250k_20261014_ctce \
  --output-dir outputs/focus_visibility_report_250k --min-steps 250000
```

It produces `eligible_runs.csv`, the wide `coverage_table.csv`, paired
`focus_gain_by_visibility.csv`, and `focus_gain_by_visibility.png`.  Coverage
is deterministic evaluation coverage; `meets_3pp` means an absolute gain of at
least 0.03, not a three-percent relative change.

The completed seed-0 250k pairs on `MATE-4v8-9.yaml` are ResQ, RiskQ, QFree,
HMVFE, and HiTMAC.  Mean scheduled-evaluation gains of +FOCUS over backbone
are respectively +2.46, +2.73, -2.08, -24.21, and +0.03 percentage points.
FACMAC is only a 100k pilot and ConcaveQ has no matched +FOCUS run, so neither
is used as a completed-pair conclusion.  The milestone table, exact observed
steps, run statuses, and limitations are recorded in
`FOCUS_EXPERIMENT_REPORT.md`.  These are single-seed engineering results and
must not be presented as statistically reliable reproduction evidence.
