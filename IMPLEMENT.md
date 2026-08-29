# IMPLEMENT: Add FOCUS to HMVFE

## 0. Objective

Implement **FOCUS-HMVFE** in `Thanh124pav/mate` without changing HMVFE's native task, policy architecture, reward, critic, GAE, entropy objective, assignment semantics, or environment executor.

The current HMVFE implementation under `hmvfe_mate_d/` is an **A2C + GAE coordinator**, not PPO/MAPPO. Its policy produces a Bernoulli camera-target assignment matrix and currently sums all camera-target log-probabilities into one joint scalar.

The integration must:

1. preserve the existing per-camera axis of the Bernoulli log-probabilities;
2. obtain a per-camera FOCUS responsibility vector `rho`;
3. convert `rho` to baseline-preserving actor weights;
4. weight each camera's contribution before the existing global sum;
5. leave all non-actor terms unchanged.

Read `THEORY.md` before implementing.

---

## 1. Repository target

Primary code path:

```text
hmvfe_mate_d/
    models.py
    trainer.py
    config.py
    ...
```

The repository also contains existing FOCUS work/branches. **Reuse the existing FOCUS responsibility implementation whenever possible.** Do not create a second incompatible implementation of future occupancy prediction, camera footprints, counterfactual gain, or responsibility normalization inside `hmvfe_mate_d/`.

Recommended architecture:

```text
existing FOCUS responsibility engine
              |
              v
      rho [B, N_cam]
      confidence [B] or [B, 1]
              |
              v
HMVFE-specific responsibility adapter
              |
              v
      weights [B, N_cam]
              |
              v
per-camera HMVFE log_probs [B, N_cam]
              |
              v
weighted actor joint log_prob [B]
```

If the existing FOCUS engine currently lives only in a feature branch, port/refactor the minimum reusable engine into an importable common module rather than copying logic into the HMVFE trainer.

---

## 2. Non-goals / DO NOT CHANGE

For the first implementation, do **not**:

- convert HMVFE to PPO;
- add PPO ratios or clipping;
- replace GAE advantage with FOCUS responsibility;
- modify the environmental reward;
- modify return computation;
- modify GAE;
- FOCUS-weight the critic/value loss;
- FOCUS-weight entropy regularization;
- change Bernoulli action semantics;
- change the assignment executor;
- modify HMVFE feature-fusion architecture just to insert FOCUS;
- let actor loss backpropagate into the FOCUS estimator;
- multiply the entire actor loss by FOCUS confidence;
- average over cameras if the old implementation used a sum;
- introduce action-sensitive counterfactual FOCUS advantage in this MVP.

The scientific intervention must remain isolated to **camera-wise actor gradient allocation**.

---

## 3. Tensor contract

Codex should verify the exact batch conventions in the current code and adapt dimensions accordingly. The conceptual shapes are:

```text
logits / probs:            [..., N_cam, N_target]
sampled assignment:        [..., N_cam, N_target]
pair_log_prob:             [..., N_cam, N_target]
camera_log_prob:           [..., N_cam]
joint_log_prob:            [...]

rho:                       [..., N_cam]
focus_confidence:          [...] or [..., 1]
active_camera_mask:        [..., N_cam]  # optional
focus_weight:              [..., N_cam]

advantage:                 [...]         # team-level HMVFE GAE advantage
value:                     [...]
```

Important:

```python
camera_log_prob = pair_log_prob.sum(dim=-1)
joint_log_prob = camera_log_prob.sum(dim=-1)
```

`camera_log_prob.sum(dim=-1)` must reproduce the old joint log-probability numerically.

Do not assume a leading batch dimension exists during rollout; support both single-decision and batched training forms using the code's existing conventions.

---

## 4. Step A — expose per-camera policy log-probabilities in `models.py`

Locate the HMVFE coordinator policy method that creates the Bernoulli distribution, samples/receives the assignment matrix, and currently computes something equivalent to:

```python
log_prob = dist.log_prob(action).sum(...)
```

Refactor without changing distribution semantics.

Required decomposition:

```python
pair_log_prob = dist.log_prob(action)          # [..., N_cam, N_target]
camera_log_prob = pair_log_prob.sum(dim=-1)  # [..., N_cam]
joint_log_prob = camera_log_prob.sum(dim=-1)  # [...]
```

The new `joint_log_prob` must be exactly equal to the old value when run on the same logits and assignment.

### Backward-compatible API

Prefer returning an auxiliary info structure rather than breaking all callers. For example:

```python
output = {
    "action": action,
    "log_prob": joint_log_prob,
    "camera_log_prob": camera_log_prob,
    "entropy": entropy,
    ...
}
```

or add an optional flag such as:

```python
return_per_camera_log_prob=False
```

Use whichever is most consistent with the existing codebase.

Do not create a second Bernoulli distribution or resample actions just to obtain the decomposition.

### Entropy

Preserve the old entropy calculation/reduction exactly. Per-camera entropy may optionally be exposed for diagnostics, but the training objective must continue using the old entropy term in the MVP.

---

## 5. Step B — create/reuse an HMVFE FOCUS adapter

Do not embed normalization and masking logic ad hoc in the actor loss.

Recommended module if no equivalent common abstraction already exists:

```text
hmvfe_mate_d/focus_adapter.py
```

or preferably a common FOCUS module outside the HMVFE-specific package if the repository already has multiple FOCUS backbones.

Suggested interface:

```python
class HMVFEFocusAdapter:
    def __init__(
        self,
        eta: float,
        use_confidence: bool = True,
        eps: float = 1e-8,
    ):
        ...

    @torch.no_grad()
    def compute_weights(
        self,
        rho: torch.Tensor,
        confidence: torch.Tensor | None = None,
        active_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        ...
```

The exact class name is flexible; the semantics are not.

### All cameras active

Normalize defensively:

```python
rho = rho.clamp_min(0.0)
rho = rho / rho.sum(dim=-1, keepdim=True).clamp_min(eps)
```

Let:

```python
eta_t = eta
if use_confidence and confidence is not None:
    eta_t = eta * confidence.clamp(0.0, 1.0)
```

Broadcast `eta_t` over the camera dimension and compute:

```python
n_cam = rho.shape[-1]
weights = 1.0 + eta_t * (n_cam * rho - 1.0)
```

### Active-camera mask

If HMVFE exposes inactive cameras:

```python
mask = active_mask.to(rho.dtype)
masked_rho = rho.clamp_min(0.0) * mask
rho = masked_rho / masked_rho.sum(-1, keepdim=True).clamp_min(eps)
n_active = mask.sum(-1, keepdim=True).clamp_min(1.0)
weights = 1.0 + eta_t * (n_active * rho - 1.0)
weights = weights * mask
```

If an edge case has no active camera, handle it explicitly according to HMVFE's existing action semantics rather than allowing NaNs.

### Detachment

Return detached/no-grad weights.

Even if the FOCUS engine itself uses trainable modules, the HMVFE actor loss must not update those modules through `weights`.

---

## 6. Step C — reuse the existing FOCUS Responsibility Engine

Find the existing FOCUS code in the repository/FOCUS branches and isolate the backbone-independent part that maps joint environment information to:

```text
rho_t:        [N_cam]
confidence_t: scalar or [1]     # if available
```

The engine should retain the existing conceptual pipeline:

```text
joint state/context
    -> future target occupancy / belief
    -> future camera footprints
    -> counterfactual non-redundant coverage gains
    -> normalize gains
    -> rho
    -> optional confidence
```

### Do not duplicate FOCUS semantics

There must be one canonical implementation of:

- future occupancy prediction;
- footprint/visibility calculation;
- counterfactual unique coverage gain;
- responsibility normalization;
- confidence, if already implemented.

HMVFE should have only an adapter between the canonical `rho` output and its actor loss.

### Camera ordering

Verify that FOCUS camera index `i` corresponds exactly to row `i` of HMVFE's assignment matrix.

Add an assertion or explicit mapping if camera IDs can be reordered.

This is a high-risk silent bug: shapes can match while responsibilities are assigned to the wrong cameras.

---

## 7. Step D — align FOCUS with HMVFE decision ticks in `trainer.py`

Locate the rollout path where the HMVFE high-level coordinator:

1. builds its state/input;
2. samples a camera-target assignment;
3. stores log-prob/value/action information;
4. eventually computes returns/GAE and performs the actor-critic update.

At the exact same high-level decision tick, compute or obtain FOCUS responsibility.

Store together per decision:

```python
transition = {
    ... existing HMVFE fields ...
    "camera_log_prob": camera_log_prob,
    "focus_rho": rho,
    "focus_confidence": confidence,   # optional
    "camera_active_mask": mask,       # optional
}
```

If the trainer stores tensors in parallel Python lists rather than transition dictionaries, preserve that style but keep indices strictly aligned.

### Critical temporal rule

If one HMVFE assignment is held for multiple environment steps, do **not** generate an unrelated new `rho` every low-level step and then attach it to the same high-level log-probability.

`rho_t`, action `Y_t`, log-probability, and the advantage later computed for that high-level action must refer to the same decision.

---

## 8. Step E — modify only the actor term

Identify the existing A2C actor loss in `trainer.py`. It should conceptually be equivalent to:

```python
policy_loss = -(advantage.detach() * joint_log_prob).mean()
```

or an algebraically equivalent reduction.

### Vanilla path

Keep an explicit vanilla path when FOCUS is disabled:

```python
if not cfg.focus_enabled:
    actor_log_prob = joint_log_prob
```

This makes regression testing easier and minimizes risk.

### FOCUS path

For enabled FOCUS:

```python
weights = focus_adapter.compute_weights(
    rho=focus_rho,
    confidence=focus_confidence,
    active_mask=active_mask,
)

weighted_joint_log_prob = (
    camera_log_prob * weights.detach()
).sum(dim=-1)

policy_loss = -(
    advantage.detach() * weighted_joint_log_prob
).mean()
```

Adapt the final `.mean()` or other reductions to match the existing trainer exactly. The key transformation is only:

```text
old: camera_log_prob.sum(camera)
new: (camera_log_prob * weight).sum(camera)
```

### DO NOT divide by `N_cam`

Do not write:

```python
(camera_log_prob * weights).mean(dim=-1)
```

if the old code uses a sum.

Uniform `rho`, `eta=0`, or zero confidence must exactly reproduce the old `joint_log_prob`:

```python
assert_close(weighted_joint_log_prob, joint_log_prob)
```

under those conditions.

---

## 9. Step F — leave critic and entropy unchanged

The overall existing objective likely combines terms similar to:

```python
loss = policy_loss + value_coef * value_loss - entropy_coef * entropy
```

Keep:

```python
value_loss      # unchanged
entropy         # unchanged
value_coef      # unchanged
entropy_coef    # unchanged
```

Only substitute the FOCUS-weighted log-probability inside `policy_loss`.

Do not apply `weights` to value loss or entropy in the MVP.

---

## 10. Step G — configuration

Add configuration fields in `hmvfe_mate_d/config.py` using the project's existing config style.

Minimum fields:

```python
focus_enabled = False
focus_eta = 1.0
focus_use_confidence = True
focus_eps = 1e-8
```

Recommended experiment sweep:

```text
focus_eta in {0.25, 0.5, 1.0}
```

The default **must remain `focus_enabled=False`** so existing HMVFE commands still run as the original baseline.

If the repository already has canonical FOCUS configuration fields, reuse them rather than defining duplicate predictor-specific parameters under HMVFE.

Only HMVFE-adapter-specific parameters should live with HMVFE.

---

## 11. Step H — logging

Add diagnostics without changing optimization behavior.

Recommended metrics:

```text
focus/rho_entropy
focus/rho_l1_uniform
focus/rho_min
focus/rho_max
focus/effective_num_cameras
focus/confidence_mean
focus/weight_min
focus/weight_max
focus/weight_mean
focus/weighted_log_prob_mean
focus/vanilla_log_prob_mean
focus/actor_weight_delta
```

Optional:

```text
focus/rho_camera_0
focus/rho_camera_1
...
```

only when the number of cameras is small and logging cost is acceptable.

### Runtime invariants

During debugging, periodically verify:

```python
assert torch.isfinite(rho).all()
assert torch.isfinite(weights).all()
```

For all-active cameras:

```python
assert_close(weights.mean(dim=-1), torch.ones_like(...), atol=...)
```

For active masks, check mean weight only over active cameras.

Do not leave expensive assertions enabled in production training if they materially slow execution; retain them in tests/debug mode.

---

## 12. Required unit tests

Create tests in the repository's existing test structure. If HMVFE currently has no test directory, add a small focused one without reorganizing unrelated code.

### Test 1 — camera log-prob decomposition

Given fixed logits and a fixed assignment:

```python
old_joint = old_equivalent_log_prob(...)
new_joint = camera_log_prob.sum(dim=-1)
```

Require numerical equality.

### Test 2 — FOCUS disabled parity

With `focus_enabled=False`, the policy loss and total loss must match the baseline implementation for a fixed synthetic batch.

### Test 3 — eta zero parity

With arbitrary non-uniform `rho` and:

```python
focus_eta = 0
```

require:

```python
weights == 1
weighted_joint_log_prob == joint_log_prob
```

### Test 4 — uniform responsibility parity

For:

```python
rho = torch.full(..., 1.0 / N_cam)
```

require exact/near-exact baseline recovery.

### Test 5 — zero-confidence parity

For arbitrary `rho` and `confidence=0`, require baseline recovery.

### Test 6 — weight mean invariant

For normalized `rho` and all cameras active:

```python
weights.mean(dim=-1) == 1
```

within numerical tolerance.

### Test 7 — active-mask invariant

If masks are supported:

- inactive camera contribution is zero;
- responsibility is normalized over active cameras;
- active-camera mean weight is one.

### Test 8 — gradient isolation

Construct a synthetic actor loss and call backward.

Require:

- HMVFE policy parameters receive finite gradients;
- `rho`/FOCUS parameters do not receive gradients from the actor loss.

### Test 9 — non-uniform weighting changes relative camera gradient

Use a simple synthetic example with at least two camera rows and non-uniform `rho`.

Verify the camera with larger weight contributes proportionally more strongly to the policy gradient, holding log-prob structure fixed.

### Test 10 — fixed-seed single-update regression

For `focus_enabled=False`, a fixed seed and fixed batch should produce the same parameter update as the pre-FOCUS HMVFE code within floating-point tolerance.

This is one of the most important regression tests.

---

## 13. Smoke training

Before running full experiments:

### Smoke A — baseline

Run the existing HMVFE command with FOCUS disabled.

Verify:

- training starts;
- losses match expected scale;
- no new dependency is required for the baseline path if avoidable;
- evaluation works unchanged.

### Smoke B — uniform FOCUS

Force:

```python
rho_i = 1 / N_cam
```

with FOCUS enabled.

Training curves/losses should match baseline under fixed seeds up to ordinary nondeterminism.

### Smoke C — synthetic non-uniform FOCUS

Temporarily inject a deterministic responsibility vector, e.g. for four cameras:

```text
[0.55, 0.25, 0.15, 0.05]
```

Verify:

- weights have mean 1;
- actor loss changes;
- critic loss does not change from FOCUS logic;
- no NaNs;
- camera log-prob decomposition remains correct.

### Smoke D — real FOCUS engine

Enable the actual FOCUS responsibility estimator and inspect responsibility/weight traces before committing to long training runs.

---

## 14. Experimental switches / ablations

Implement experimental controls cleanly so they do not require code edits between runs.

Recommended responsibility modes:

```text
focus_mode = real
focus_mode = uniform
focus_mode = shuffled
focus_mode = random       # optional
```

Semantics:

### `real`

Use canonical FOCUS output.

### `uniform`

Use:

```python
rho_i = 1 / N_cam
```

This must recover HMVFE and acts as an implementation sanity check.

### `shuffled`

Compute real FOCUS `rho`, then randomly permute the camera dimension per decision or episode according to an explicitly documented protocol.

This preserves responsibility concentration/statistics while destroying camera identity. It is a key semantic control.

### `random`

Optional. Generate random normalized responsibilities, preferably with concentration statistics documented or matched to real FOCUS if used in the paper.

Do not overload `focus_enabled` with these semantics; keep enable/disable and ablation mode separate.

---

## 15. Suggested implementation order for Codex

Implement in this exact order so failures are localizable.

### Commit 1 — expose per-camera HMVFE log-probability

- Refactor `models.py`.
- Preserve old joint log-prob exactly.
- Add decomposition test.
- No FOCUS dependency yet.

Expected behavior: **zero training change**.

### Commit 2 — add responsibility-to-weight adapter

- Implement normalization, confidence interpolation, masks, detachment.
- Add pure unit tests for all invariants.
- No trainer integration yet.

### Commit 3 — add optional rollout storage for `rho`

- Wire canonical FOCUS engine to HMVFE decision ticks.
- Store responsibility/confidence in aligned trajectory data.
- Add shape/order checks.
- Do not modify actor loss yet if practical.

### Commit 4 — add FOCUS-weighted actor path

- Replace only the camera sum in actor log-prob when enabled.
- Preserve critic and entropy exactly.
- Add fixed-batch loss and gradient tests.

### Commit 5 — configs and logging

- Add `focus_enabled`, `focus_eta`, confidence switch, responsibility mode.
- Add diagnostics.

### Commit 6 — regression and smoke tests

- fixed-seed baseline parity;
- uniform-responsibility parity;
- real FOCUS smoke run.

Do not combine all changes into one opaque patch.

---

## 16. Pseudocode reference

### Rollout / decision

```python
# Existing HMVFE policy forward
policy_out = coordinator(...)
action = policy_out["action"]
joint_log_prob = policy_out["log_prob"]
camera_log_prob = policy_out["camera_log_prob"]
value = policy_out["value"]  # adapt to actual API

# FOCUS only if enabled
if cfg.focus_enabled:
    with torch.no_grad():
        focus_out = focus_engine.compute(...joint_environment_context...)
        rho = focus_out.rho
        confidence = getattr(focus_out, "confidence", None)
else:
    rho = None
    confidence = None

# Store all fields at the same high-level decision index
rollout.add(
    action=action,
    log_prob=joint_log_prob,
    camera_log_prob=camera_log_prob,
    value=value,
    focus_rho=rho,
    focus_confidence=confidence,
    ...
)
```

### Update

```python
advantages = compute_existing_gae(...)

if cfg.focus_enabled:
    weights = focus_adapter.compute_weights(
        rho=batch.focus_rho,
        confidence=batch.focus_confidence,
        active_mask=getattr(batch, "camera_active_mask", None),
    )

    actor_log_prob = (
        batch.camera_log_prob * weights.detach()
    ).sum(dim=-1)
else:
    actor_log_prob = batch.log_prob

policy_loss = existing_actor_reduction(
    advantages.detach(),
    actor_log_prob,
)

value_loss = existing_value_loss(...)       # unchanged
entropy = existing_entropy_term(...)        # unchanged

total_loss = existing_loss_combination(
    policy_loss,
    value_loss,
    entropy,
)
```

`existing_actor_reduction` is conceptual notation. Do not add a new abstraction if the trainer is simpler without one; preserve its current reduction semantics.

---

## 17. Baseline-parity checklist

Before considering implementation complete, verify every item.

When `focus_enabled=False`:

- [ ] same HMVFE action distribution;
- [ ] same sampled action under fixed RNG state;
- [ ] same joint log-probability;
- [ ] same entropy;
- [ ] same value prediction;
- [ ] same GAE;
- [ ] same policy loss;
- [ ] same value loss;
- [ ] same total loss;
- [ ] same gradients within tolerance;
- [ ] same optimizer update within tolerance;
- [ ] same evaluation behavior.

When `focus_enabled=True` and `rho=uniform`:

- [ ] all weights are 1;
- [ ] weighted joint log-prob equals vanilla joint log-prob;
- [ ] actor loss equals vanilla actor loss;
- [ ] critic/entropy are unchanged.

When `confidence=0`:

- [ ] all active-camera weights are 1;
- [ ] update falls back to vanilla HMVFE.

---

## 18. Common implementation mistakes

### Mistake 1 — treating HMVFE as MAPPO

Do not add PPO clipping/ratios. The current HMVFE coordinator uses A2C + GAE.

### Mistake 2 — using `rho` as the advantage

Wrong:

```python
policy_loss = -(rho * log_prob)
```

FOCUS responsibility and team GAE advantage have different semantics.

Correct conceptual form:

```python
policy_loss = -(advantage * sum_i(weight_i * camera_log_prob_i))
```

### Mistake 3 — weighting after all camera log-probs were summed

Once the camera dimension is lost, per-camera responsibility cannot be applied correctly.

Recover/preserve the camera axis in `models.py` first.

### Mistake 4 — applying one weight to the entire joint loss

FOCUS is useful precisely because weights differ by camera.

### Mistake 5 — dividing by number of cameras

This breaks exact recovery of the current HMVFE `.sum()` behavior.

### Mistake 6 — confidence suppresses baseline learning

Wrong:

```python
policy_loss *= confidence
```

Correct:

```python
eta_t = eta * confidence
weights = 1 + eta_t * (N * rho - 1)
```

### Mistake 7 — actor gradients update FOCUS

Use detached/no-grad responsibility weights for this MVP.

### Mistake 8 — mismatch of camera ordering

Explicitly verify that FOCUS index `i` equals HMVFE assignment row `i`.

### Mistake 9 — temporal off-by-one

The `rho`, action/log-prob, and GAE advantage must correspond to the same high-level coordinator decision.

### Mistake 10 — silently changing entropy

Keep entropy regularization exactly as the baseline until it is explicitly studied as an ablation.

---

## 19. Definition of Done

The implementation is complete only when all of the following hold:

### Code

- [ ] HMVFE exposes per-camera log-probabilities.
- [ ] Old joint log-prob is reproduced by their sum.
- [ ] Canonical FOCUS engine is reused rather than duplicated.
- [ ] HMVFE adapter maps `rho` to detached baseline-preserving weights.
- [ ] FOCUS data are aligned with HMVFE high-level decisions.
- [ ] Actor loss weights camera log-probabilities before summation.
- [ ] Critic, GAE, entropy, reward, and action semantics are unchanged.
- [ ] Baseline remains default behavior.

### Tests

- [ ] decomposition test passes;
- [ ] disabled parity passes;
- [ ] `eta=0` parity passes;
- [ ] uniform `rho` parity passes;
- [ ] zero-confidence parity passes;
- [ ] mean-weight invariant passes;
- [ ] mask test passes if masks are relevant;
- [ ] actor-gradient isolation test passes;
- [ ] fixed-seed single-update baseline regression passes;
- [ ] smoke training has no NaNs.

### Experiment readiness

- [ ] real/uniform/shuffled responsibility modes are selectable by config;
- [ ] `eta` is configurable;
- [ ] confidence can be enabled/disabled;
- [ ] responsibility and weight diagnostics are logged;
- [ ] baseline and FOCUS evaluation use identical evaluation semantics.

---

## 20. Final instruction to Codex

Implement the smallest possible change that realizes this equation:

\[
\boxed{
\mathcal L_\pi^{\mathrm{FOCUS-HMVFE}}
=
-\mathbb E_t\left[
\operatorname{sg}(A_t)
\sum_i
\operatorname{sg}(w_{i,t})
\sum_j
\log\pi_\theta(Y_{ij,t}\mid s_t)
\right]
}
\]

with

\[
\boxed{
w_{i,t}=1+\eta c_t(N_c\rho_{i,t}-1)}
\]

for the all-camera-active case, using the active-mask version when necessary.

The implementation should demonstrate that FOCUS is an **optimizer-side responsibility adapter for HMVFE**, not a redesign of HMVFE itself.
