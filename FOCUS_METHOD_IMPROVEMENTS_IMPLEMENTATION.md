# FOCUS Method Improvements — Code Implementation Plan

## 1. Scope

This document describes how to implement the proposed FOCUS improvements on top of the current `QPLEX_FOCUS` codebase.

The current implementation is centered around the existing QPLEX_FOCUS learner and mixer, especially:

- `ray/rllib/agents/qplex_focus/qplex_policy.py`
- `ray/rllib/agents/qplex_focus/mixers.py`
- the existing QPLEX/QPLEX_V2 mixer implementation under `ray/rllib/agents/`

Current behavior to preserve:

1. QPLEX computes the ordinary TD objective.
2. The occupancy model predicts future target positions.
3. Sobol / Monte-Carlo samples estimate unique future coverage.
4. Predictive gains \(g_i\) are normalized into \(\rho_i\).
5. `credit_prior(states)` exposes the QPLEX-side state-conditioned allocation prior.
6. Cross-entropy/KL aligns detached \(\rho\) with that prior.
7. Total loss is approximately

\[
L
=
L_{\mathrm{TD}}
+
\alpha L_{\mathrm{credit}}
+
\beta L_{\mathrm{belief}}.
\]

The implementation plan is incremental. Keep the current implementation frozen as a baseline and add new variants separately.

---

# 2. Recommended branch/module structure

Suggested algorithm variants:

```text
ray/rllib/agents/
├── qplex_focus/                 # frozen current baseline
├── qplex_focus_dyn/             # + dynamic/reachable footprint
├── qplex_focus_dyn_conf/        # + improved reliability
├── qplex_focus_dual/            # + action/policy guidance
└── focus_common/                # reusable FOCUS modules
```

Suggested reusable modules:

```text
focus_common/
├── occupancy.py
├── reachable_visibility.py
├── responsibility.py
├── uncertainty.py
├── teachers.py
├── policy_guidance.py
├── predictive_dynamics.py
└── utils.py
```

The key software principle is:

> Keep future-occupancy responsibility estimation independent from the MARL optimization adapter.

This makes it possible to reuse the same FOCUS engine with QPLEX, QMIX, DuelMIX, MAPPO, or IPPO.

---

# 3. Refactor current FOCUS before changing the algorithm

The first change should be a **no-behavior-change refactor**.

## 3.1 Extract the occupancy predictor

Move future-occupancy logic out of `qplex_policy.py`.

Suggested interface:

```python
class FutureOccupancyModel(nn.Module):
    def forward(self, states):
        """
        states:
            [B, T, state_dim]

        returns
        -------
        mu:
            [B, T, H, N_target, 2]
        sigma:
            [B, T, H, N_target, 2]
        """
        return mu, sigma

    def loss(self, mu, sigma, future_positions, mask):
        ...
```

Keep exactly the same Gaussian parameterization as the current baseline first.

---

## 3.2 Extract responsibility computation

```python
class PredictiveResponsibility(nn.Module):
    def forward(
        self,
        occupancy_samples,
        visibility,
        horizon_weights,
        valid_mask,
    ):
        """
        returns
        -------
        gains:
            [B, T, N_camera]
        rho:
            [B, T, N_camera]
        total_gain:
            [B, T]
        """
        ...
```

This module should not know whether the downstream MARL backbone is QPLEX or PPO.

---

## 3.3 Wrap the QPLEX allocation adapter

The current mixer already exposes:

```python
mixer.credit_prior(states)
```

Keep it as the reference adapter.

```python
class QPLEXCreditAdapter:
    def __call__(self, mixer, states, **kwargs):
        return mixer.credit_prior(states)
```

Do not silently switch from the state-conditioned prior to the complete effective \(\lambda_i\).

---

# 4. Improvement 1 — Horizon-dependent reachable visibility

## 4.1 Goal

Replace the current conceptual operation

```python
visible = visibility(
    current_camera_state,
    current_action,
    future_samples,
)
```

with

```python
visible_h = reachable_visibility(
    current_camera_state,
    current_action,
    horizon=h,
    future_samples=future_samples_h,
)
```

Expected tensor:

```text
visibility:
[B, T, H, N_camera, N_target, N_sample]
```

If memory becomes large, calculate one horizon or one sample chunk at a time.

---

## 4.2 First implementation: geometric reachable footprint

This should be the first experiment because it fixes the main approximation without introducing a learned world model.

Suppose the camera has a maximum rotation change and zoom/range change per action.

After executing the current action, at horizon \(h\) approximate the reachable orientation interval as

```python
theta_center = theta_after_current_action

theta_min_h = theta_center - h * max_rotation_step
theta_max_h = theta_center + h * max_rotation_step
```

and similarly for viewing angle / range if the action space supports zoom.

Do **not** enumerate all future action sequences initially.

Instead discretize the reachable camera configurations.

Example:

```python
theta_candidates = torch.linspace(
    theta_min_h,
    theta_max_h,
    num_orientation_bins,
)

zoom_candidates = torch.linspace(
    zoom_min_h,
    zoom_max_h,
    num_zoom_bins,
)
```

Then calculate visibility for the candidate configurations.

---

## 4.3 Reachability modes

### Mode A — `union`

A location is considered reachable if any legal future configuration can observe it.

```python
reachable_visibility = candidate_visibility.max(dim=config_dim).values
```

This is optimistic but simple and useful as an MVP.

### Mode B — `policy_weighted`

Estimate a probability over future camera configurations and use

```python
reachable_visibility = (
    config_prob * candidate_visibility
).sum(dim=config_dim)
```

This is a better long-term default because it represents what the current policy is likely to do rather than everything physically possible.

### Mode C — `conservative`

Use a low quantile or lower-confidence estimate.

Useful as an ablation rather than the default.

---

## 4.4 New file

Create:

```text
ray/rllib/agents/focus_common/reachable_visibility.py
```

Suggested class:

```python
class ReachableVisibility:
    def __init__(
        self,
        horizon,
        rotation_step,
        zoom_step,
        mode="union",
        num_orientation_bins=7,
        num_zoom_bins=3,
    ):
        self.horizon = horizon
        self.rotation_step = rotation_step
        self.zoom_step = zoom_step
        self.mode = mode
        ...

    def compute(
        self,
        camera_states,
        current_actions,
        query_points,
    ):
        """
        camera_states:
            [B, T, N_camera, camera_state_dim]

        current_actions:
            [B, T, N_camera, ...]

        query_points:
            [B, T, H, N_target, N_sample, 2]

        returns:
            [B, T, H, N_camera, N_target, N_sample]
        """
        ...
```

---

# 5. Dynamic predictive gain

The current unique-coverage concept remains unchanged:

```python
unique_gain_i
=
visibility_i
*
product(1 - visibility_other_agents)
```

Only the visibility now depends on horizon.

Conceptual implementation:

```python
# vis:
# [B, T, H, C, J, M]

unique = []

for i in range(n_cameras):
    vis_i = vis[:, :, :, i]  # [B,T,H,J,M]

    others = torch.cat(
        [
            vis[:, :, :, :i],
            vis[:, :, :, i + 1:],
        ],
        dim=3,
    )

    no_other = (1.0 - others).prod(dim=3)
    gain_i = vis_i * no_other
    unique.append(gain_i)

unique = torch.stack(unique, dim=3)
# [B,T,H,C,J,M]

gain_by_horizon = (
    unique
    .mean(dim=-1)     # samples
    .sum(dim=-1)      # targets
)
# [B,T,H,C]

gains = (
    gain_by_horizon
    * horizon_weights.view(1, 1, H, 1)
).sum(dim=2)

# [B,T,C]
```

Preserve the same normalization convention as the current FOCUS baseline.

---

# 6. Improvement 2 — Better confidence/reliability

## 6.1 Keep current reliability as a baseline

Do not remove the existing prediction-error based weighting. Rename it clearly in logs, for example:

```text
focus/reliability_retrospective
```

Then add stronger alternatives as ablations.

---

## 6.2 Responsibility ambiguity gate

Even a perfectly predicted scene can be uninformative if every camera has similar responsibility.

Compute responsibility entropy:

```python
rho_entropy = -(
    rho * torch.log(rho + eps)
).sum(dim=-1)
```

Normalize:

```python
max_entropy = math.log(n_cameras)

credit_confidence = (
    1.0 - rho_entropy / max_entropy
).clamp(0.0, 1.0)
```

Combine with prediction reliability:

```python
final_confidence = (
    prediction_confidence
    * credit_confidence
    * valid_mask.float()
)
```

Detach the final gate:

```python
final_confidence = final_confidence.detach()
```

Otherwise the model may learn to reduce the loss by manipulating the gate.

---

## 6.3 Optional ensemble uncertainty

A lightweight ensemble can estimate epistemic uncertainty.

Use a shared encoder with \(K=3\) prediction heads:

```python
features = shared_encoder(states)

mus = torch.stack(
    [head(features)["mu"] for head in heads],
    dim=0,
)
```

Epistemic uncertainty:

```python
epistemic = mus.var(dim=0).mean(
    dim=(-1, -2, -3)
)
```

Aleatoric uncertainty can be estimated from predicted Gaussian variance.

Then:

```python
uncertainty = (
    epistemic_coef * epistemic
    + aleatoric_coef * aleatoric
)

prediction_confidence = torch.exp(
    -kappa * uncertainty
)
```

Do not introduce this before the entropy-gated version has been tested.

---

# 7. Improvement 3 — Teacher/fallback responsibility

Create:

```text
focus_common/teachers.py
```

Interface:

```python
class ResponsibilityTeacher:
    def compute(self, batch):
        """
        returns:
            rho_teacher [B,T,N_camera]
        """
        ...
```

Recommended teacher implementations:

### `realized_unique`

Use realized camera-target visibility from replay.

### `oracle_future`

Use realized future target positions rather than predicted future occupancy.

This should be implemented even if it is never used in the final method, because it gives a useful upper bound on the occupancy predictor.

### `greedy_geometry`

Use a geometric or greedy camera controller if available.

---

## 7.1 Blend

```python
rho_final = (
    confidence.unsqueeze(-1) * rho_pred
    + (1.0 - confidence.unsqueeze(-1)) * rho_teacher
)
```

Then:

```python
rho_final = rho_final.detach()
```

before credit alignment.

Important ablation:

```text
predicted responsibility
vs.
oracle future responsibility
```

This tells you whether the performance bottleneck is prediction accuracy or the credit formulation itself.

---

# 8. Improvement 4 — Action-sensitive predictive advantage

## 8.1 Goal

Compute not only

```text
which camera matters?
```

but also

```text
which current camera action improves future unique coverage?
```

For discrete camera actions, evaluate all candidate actions.

Expected tensor:

```text
action_gain:
[B, T, N_camera, N_action]
```

Pseudocode:

```python
action_gains = []

for action_id in range(n_actions):
    candidate_actions = executed_actions.clone()
    candidate_actions[..., i] = action_id

    gain = compute_predictive_gain(
        states=states,
        actions=candidate_actions,
        ...
    )

    action_gains.append(gain[..., i])

action_gain = torch.stack(
    action_gains,
    dim=-1,
)
```

This can be expensive. Vectorize after the MVP is correct.

---

## 8.2 Executed-action gain

```python
g_exec = action_gain.gather(
    dim=-1,
    index=actions.unsqueeze(-1),
).squeeze(-1)
```

For an explicit stochastic policy:

```python
g_baseline = (
    action_prob.detach()
    * action_gain
).sum(dim=-1)
```

Then

```python
focus_advantage = g_exec - g_baseline
```

Normalize per batch:

```python
focus_advantage = (
    focus_advantage
    - masked_mean(focus_advantage, mask)
) / (
    masked_std(focus_advantage, mask)
    + 1e-6
)
```

Always log the unnormalized and normalized versions.

---

# 9. Policy-level integration

There are two different implementations depending on whether the backbone remains QPLEX or becomes policy-based.

---

## 9.1 QPLEX + auxiliary policy head

QPLEX does not naturally optimize a stochastic actor.

If you want to test direct policy guidance while retaining QPLEX, attach an auxiliary policy head to the agent recurrent representation:

```text
agent hidden representation
       |
       +---- Q-value head
       |
       +---- auxiliary policy head
```

Training loss:

```python
log_probs = F.log_softmax(
    policy_logits,
    dim=-1,
)

chosen_log_prob = log_probs.gather(
    -1,
    actions.unsqueeze(-1),
).squeeze(-1)

policy_focus_loss = -masked_mean(
    focus_advantage.detach()
    * chosen_log_prob,
    valid_focus_mask,
)
```

The original Q head still determines actions at execution.

Treat this as an ablation; it is not the cleanest policy-based FOCUS formulation.

---

## 9.2 MAPPO/IPPO FOCUS adapter

The cleaner extension is to reuse the FOCUS predictive engine with an explicit actor.

Existing PPO objective:

```python
ppo_loss = ...
value_loss = ...
entropy = ...
```

Add:

```python
focus_policy_loss = -masked_mean(
    focus_advantage.detach()
    * new_log_prob,
    valid_focus_mask,
)
```

Total:

```python
loss = (
    ppo_loss
    + value_coef * value_loss
    - entropy_coef * entropy
    + belief_coef * belief_loss
    + focus_policy_coef * focus_policy_loss
)
```

The same module should generate future occupancy, reachable visibility, gains, reliability, and FOCUS advantage for both QPLEX and MAPPO.

---

# 10. Potential-based shaping variant

Implement this as a separate variant.

```python
phi_t = predictive_potential_t.detach()
phi_tp1 = predictive_potential_tp1.detach()

focus_shaping = (
    gamma * phi_tp1
    - phi_t
)
```

Then:

```python
agent_reward = (
    team_reward.unsqueeze(-1)
    + eta * focus_shaping
)
```

Do not combine reward shaping and the policy auxiliary loss in the first experiment. Otherwise it will be impossible to identify which mechanism produced the gain.

Mandatory logs:

```text
focus/shaping_abs_mean
focus/env_reward_abs_mean
focus/shaping_reward_ratio
```

---

# 11. Optional local responsibility distillation

Create:

```text
focus_common/local_responsibility.py
```

Model:

```python
class LocalResponsibilityPredictor(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, agent_hidden):
        return self.head(agent_hidden).squeeze(-1)
```

Centralized training target:

```python
rho_target = rho_central.detach()
```

Do not require global normalization at execution time.

A practical version is to train a local scalar confidence:

```python
local_resp = torch.sigmoid(
    local_resp_logit
)
```

and feed its latent feature into the Q or policy head.

This preserves decentralized execution.

---

# 12. Changes to `qplex_focus/mixers.py`

Current baseline behavior:

```python
credit_prior(states)
```

should remain unchanged.

Add an explicit adapter method:

```python
def focus_allocation(
    self,
    states,
    mode="state_prior",
    **kwargs,
):
    if mode == "state_prior":
        return self.credit_prior(states)

    if mode == "effective_lambda":
        return self.effective_lambda_distribution(
            states=states,
            **kwargs,
        )

    raise ValueError(mode)
```

Do not make `effective_lambda` the default.

This supports an ablation between:

```text
rho -> state-conditioned allocation prior
```

and

```text
rho -> complete effective lambda distribution
```

while preserving the scientifically accurate interpretation of the current implementation.

Recommended logs:

```text
focus/state_prior_entropy
focus/effective_lambda_entropy
focus/rho_prior_kl
focus/rho_lambda_kl
```

---

# 13. Refactor `qplex_focus/qplex_policy.py`

The loss code should be separated into explicit stages.

```python
def build_qplex_focus_loss(
    policy,
    model,
    dist_class,
    train_batch,
):
    # =========================================================
    # 1. Original QPLEX
    # =========================================================
    q_values, agent_hidden = compute_agent_q(...)
    q_tot, mixer_aux = compute_qplex_mix(...)
    td_loss = compute_td_loss(...)

    # =========================================================
    # 2. Occupancy prediction
    # =========================================================
    occupancy = policy.focus_occupancy(
        states
    )

    belief_loss, belief_stats = compute_belief_loss(
        occupancy,
        future_target_positions,
        mask,
    )

    # =========================================================
    # 3. Sample future target positions
    # =========================================================
    samples = sample_future_positions(
        occupancy,
        mode=policy.config[
            "focus_sampling_mode"
        ],
    )

    # =========================================================
    # 4. Visibility
    # =========================================================
    if policy.config[
        "focus_dynamic_footprint"
    ]:
        visibility = (
            policy.reachable_visibility.compute(
                camera_states,
                actions,
                samples,
            )
        )
    else:
        visibility = compute_current_focus_visibility(
            camera_states,
            actions,
            samples,
        )

    # =========================================================
    # 5. Predictive responsibility
    # =========================================================
    gains, rho_pred, total_gain = (
        policy.focus_responsibility(
            visibility,
            horizon_weights,
            mask,
        )
    )

    # =========================================================
    # 6. Reliability
    # =========================================================
    confidence = (
        policy.focus_uncertainty.compute(
            occupancy=occupancy,
            rho=rho_pred,
            belief_stats=belief_stats,
            total_gain=total_gain,
            mask=mask,
        )
    )

    # =========================================================
    # 7. Optional teacher blend
    # =========================================================
    if policy.config["focus_use_teacher"]:
        rho_teacher = (
            policy.focus_teacher.compute(
                train_batch
            )
        )

        rho = (
            confidence.unsqueeze(-1)
            * rho_pred
            + (1.0 - confidence.unsqueeze(-1))
            * rho_teacher
        )
    else:
        rho = rho_pred

    rho = rho.detach()

    # =========================================================
    # 8. Backbone allocation
    # =========================================================
    p_dist = policy.mixer.focus_allocation(
        states,
        mode=policy.config[
            "focus_allocation_mode"
        ],
    )

    credit_loss = weighted_cross_entropy(
        target=rho,
        pred=p_dist,
        weight=confidence * valid_focus_mask,
    )

    # =========================================================
    # 9. Optional action guidance
    # =========================================================
    if policy.config[
        "focus_action_guidance"
    ]:
        action_gain = (
            compute_counterfactual_action_gain(...)
        )
        focus_advantage = (
            compute_focus_advantage(...)
        )
        policy_focus_loss = (
            compute_policy_focus_loss(...)
        )
    else:
        policy_focus_loss = 0.0

    # =========================================================
    # 10. Total
    # =========================================================
    loss = (
        td_loss
        + alpha * credit_loss
        + beta * belief_loss
        + zeta * policy_focus_loss
    )

    return loss
```

---

# 14. Configuration

Add a dedicated FOCUS block.

```python
FOCUS_CONFIG = {
    # ---------------------------------------------------------
    # baseline
    # ---------------------------------------------------------
    "focus_enabled": True,
    "focus_horizon": 4,
    "focus_credit_coef": 0.1,
    "focus_belief_coef": 0.1,

    # ---------------------------------------------------------
    # dynamic footprint
    # ---------------------------------------------------------
    "focus_dynamic_footprint": False,
    "focus_reachable_mode": "union",
    "focus_orientation_bins": 7,
    "focus_zoom_bins": 3,

    # ---------------------------------------------------------
    # uncertainty
    # ---------------------------------------------------------
    "focus_uncertainty_mode": "current",
    "focus_entropy_gate": False,
    "focus_confidence_kappa": 1.0,
    "focus_ensemble_size": 1,

    # ---------------------------------------------------------
    # teacher
    # ---------------------------------------------------------
    "focus_use_teacher": False,
    "focus_teacher": "realized_unique",

    # ---------------------------------------------------------
    # backbone adapter
    # ---------------------------------------------------------
    "focus_allocation_mode": "state_prior",

    # ---------------------------------------------------------
    # policy guidance
    # ---------------------------------------------------------
    "focus_action_guidance": False,
    "focus_policy_coef": 0.0,

    # ---------------------------------------------------------
    # latent dynamics
    # ---------------------------------------------------------
    "focus_predictive_dynamics": False,
    "focus_latent_dim": 128,
}
```

Do not activate all options in the default config.

---

# 15. Logging

Mandatory diagnostics:

```text
focus/belief_loss
focus/credit_loss
focus/total_gain_mean
focus/valid_fraction

focus/rho_entropy
focus/p_dist_entropy
focus/rho_p_kl

focus/prediction_confidence
focus/credit_confidence
focus/final_confidence
```

Per-horizon:

```text
focus/gain_h1
focus/gain_h2
...
focus/gain_hH
```

For dynamic reachability:

```text
focus/reachable_area_h1
focus/reachable_area_h2
...
focus/reachable_area_hH
```

For policy guidance:

```text
focus/action_gain_mean
focus/action_gain_std
focus/action_adv_mean
focus/action_adv_std
focus/policy_focus_loss
focus/chosen_action_focus_rank
```

For teacher blending:

```text
focus/rho_teacher_kl
focus/teacher_weight
```

These logs are necessary to distinguish three failure modes:

```text
predictor is inaccurate
```

vs.

```text
responsibility target is uninformative
```

vs.

```text
responsibility is good but the MARL optimizer cannot exploit it
```

---

# 16. Numerical stability

## 16.1 Unique-coverage product

The term

```python
product(1 - visibility)
```

can become numerically fragile with many cameras.

A robust implementation can use log space:

```python
vis_safe = visibility.clamp(
    min=0.0,
    max=1.0 - eps,
)

log_no_cover = torch.log1p(
    -vis_safe
)

log_no_other = ...
no_other = torch.exp(log_no_other)
```

For current MATE camera counts this may not be critical, but it is a cheap robustness improvement.

---

## 16.2 Responsibility normalization

Use:

```python
total_gain = gains.sum(
    dim=-1,
    keepdim=True,
)

valid = (
    total_gain > gain_eps
)

rho = gains / total_gain.clamp_min(
    gain_eps
)
```

Do not train on a uniform fallback when `valid=False`.

---

## 16.3 Stop gradients

Detach:

```python
rho = rho.detach()
confidence = confidence.detach()
```

before the credit loss unless a future experiment explicitly studies end-to-end differentiation through these quantities.

This prevents the occupancy model or confidence estimator from reducing the auxiliary loss by manipulating its own target/gate.

---

# 17. Unit tests

Implement tests before expensive training.

## Test 1 — Unique coverage

Synthetic geometry:

```text
C1 covers target
C2 does not
```

Expected:

```text
g1 > 0
g2 ≈ 0
```

---

## Test 2 — Redundant coverage

```text
C1 covers target
C2 also covers target
```

Expected:

```text
unique gain of each camera decreases
```

relative to the unique-coverage case.

---

## Test 3 — Dynamic reachability

Construct:

```text
h=1: target region unreachable
h=3: camera can rotate enough to reach it
```

Expected:

```text
v_h1 ≈ 0
v_h3 > 0
```

---

## Test 4 — Baseline equivalence

With:

```python
focus_dynamic_footprint = False
focus_entropy_gate = False
focus_use_teacher = False
focus_action_guidance = False
```

the refactored implementation should match current QPLEX_FOCUS within numerical tolerance.

This is the most important regression test.

---

## Test 5 — Stop-gradient

Backpropagate only `credit_loss`.

Expected:

```text
occupancy predictor gradient == 0
QPLEX allocation gradient != 0
```

---

## Test 6 — Confidence zero

Set confidence to zero.

Expected:

```text
credit gradient == 0
TD gradient unchanged
```

---

## Test 7 — Action-sensitive advantage

Construct a scene where one candidate action creates unique future coverage.

Expected:

```text
A_focus(best action) > A_focus(worse actions)
```

---

# 18. Experiment sequence

Do not start with the most complex version.

## Experiment A — Refactor equivalence

```text
current QPLEX_FOCUS
vs.
refactored QPLEX_FOCUS
```

Goal: ensure no hidden behavior change.

---

## Experiment B — Dynamic footprint

```text
QPLEX
Current FOCUS
FOCUS-Dyn
```

This tests the main theoretical improvement.

---

## Experiment C — Reliability

```text
FOCUS-Dyn
FOCUS-Dyn + current reliability
FOCUS-Dyn + entropy gate
FOCUS-Dyn + ensemble uncertainty
```

---

## Experiment D — Oracle future

```text
FOCUS predicted future
FOCUS-Dyn predicted future
FOCUS-Dyn oracle future
```

This estimates the performance ceiling if occupancy prediction were perfect.

---

## Experiment E — Action-sensitive guidance

```text
FOCUS-Dyn-Conf
FOCUS-Dual
```

Then compare directly with the policy-based method that currently exceeds QPLEX_FOCUS.

---

## Experiment F — Latent predictive dynamics

Run only if the reachable-footprint version demonstrates that future camera dynamics matter but the geometric approximation still saturates.

---

# 19. Minimal implementation recommended first

If the goal is to obtain a stronger result quickly, implement only:

```text
1. Refactor current occupancy/responsibility logic.
2. Add horizon-dependent geometric reachable footprints.
3. Add responsibility-entropy gating.
4. Add oracle-future responsibility for diagnosis.
```

Do **not** immediately add:

```text
RSSM / stochastic latent dynamics
full world model
PPO integration
local responsibility distillation
potential shaping
```

The first experiment should answer:

> Is the current fixed prospective footprint the real bottleneck?

If the answer is yes, the method can improve substantially without a major architecture rewrite.

---

# 20. Second implementation milestone

If FOCUS-Dyn improves QPLEX_FOCUS but still loses to the policy-based competitor:

```text
1. Enumerate counterfactual camera actions.
2. Compute G_i(a_i) for all actions.
3. Construct A_i^FOCUS.
4. Reuse the same FOCUS engine in MAPPO/IPPO.
5. Add the action-sensitive auxiliary policy loss.
```

At this point the architecture becomes:

```text
                    FOCUS predictive engine
                           |
          +----------------+----------------+
          |                                 |
     QPLEX adapter                     MAPPO/IPPO adapter
          |                                 |
   mixer alignment                    policy guidance
```

This is the cleanest way to generalize FOCUS beyond a value-based backbone.

---

# 21. Optional latent predictive dynamics

Only after the above experiments, create:

```text
focus_common/predictive_dynamics.py
```

A minimal deterministic recurrent model is sufficient initially:

```python
class FocusPredictiveDynamics(nn.Module):
    def __init__(
        self,
        state_dim,
        action_dim,
        hidden_dim,
        horizon,
    ):
        super().__init__()

        self.state_encoder = ...
        self.action_encoder = ...
        self.gru = nn.GRUCell(...)
        self.transition = ...

        self.target_head = ...
        self.camera_head = ...
        self.horizon = horizon

    def encode(
        self,
        state,
        joint_action,
        hidden,
    ):
        ...

    def imagine(
        self,
        latent,
    ):
        """
        returns task-sufficient predictions for
        h=1,...,H
        """
        ...
```

Suggested outputs:

```text
target_mu:
[B,T,H,N_target,2]

target_sigma:
[B,T,H,N_target,2]

camera_pose:
[B,T,H,N_camera,pose_dim]

camera_uncertainty:
[B,T,H,N_camera,pose_dim]
```

Do not add stochastic latent variables until there is evidence that deterministic latent dynamics are insufficient.

---

# 22. Final target software architecture

```text
                         Replay batch
                              |
                   +----------+-----------+
                   |                      |
              MARL backbone          FOCUS engine
                   |                      |
             TD / PPO loss        occupancy predictor
                   |                      |
                   |              reachable visibility
                   |                      |
                   |               unique future gain
                   |                      |
                   |                responsibility
                   |                      |
             +-----+----------------------+------+
             |                                   |
       value adapter                         policy adapter
             |                                   |
       credit alignment                 action guidance /
                                         potential shaping
             |                                   |
             +------------------+----------------+
                                |
                           total objective
```

The core design rule is:

> The FOCUS engine estimates predictive responsibility; a separate optimization adapter decides how a particular MARL algorithm uses that signal.

This separation makes the method easier to ablate, easier to defend, and much easier to extend from QPLEX to a policy-based backbone.
