# FOCUS for Policy-Based MARL

## 1. Objective

The goal is to make FOCUS independent of QPLEX-specific mixer variables so that the same predictive responsibility signal can be used by policy-based cooperative MARL algorithms such as MAPPO and IPPO.

The new abstraction is

\[
\boxed{
\text{FOCUS Responsibility Engine}
\rightarrow
\boldsymbol{\rho}_t
\rightarrow
\text{Backbone-Specific Optimization Adapter}
}
\]

The **Responsibility Engine** answers:

> Which agents should receive more or less responsibility for the current joint decision according to predicted non-redundant future coverage?

The **Optimization Adapter** answers:

> How should a particular MARL backbone use that responsibility?

For value-factorized methods:

\[
\boldsymbol{\rho}_t
\rightarrow
\text{mixer-allocation alignment}.
\]

For policy-gradient methods:

\[
\boldsymbol{\rho}_t
\rightarrow
\text{responsibility-weighted actor update}.
\]

The first implementation target should be **MAPPO**. After MAPPO works, IPPO and other policy-gradient algorithms can reuse the same adapter pattern.

---

## 2. Important scope restriction

The first policy-based implementation must **not change the current FOCUS responsibility estimator**.

Keep:

- the current future-occupancy predictor,
- the current prospective footprint approximation,
- the current multi-horizon weighting,
- the current counterfactual unique-coverage computation,
- the current \(g_i \rightarrow \rho_i\) normalization,
- the current reliability/validity logic.

Do not simultaneously add:

- reachable future camera footprints,
- RSSM/world models,
- Oracle/grid responsibility,
- potential-based reward shaping,
- a new action-level counterfactual advantage.

Those are separate experiments.

The first question is only:

\[
\boxed{
\text{Can the existing FOCUS responsibility signal improve MAPPO?}
}
\]

---

# 3. Refactor FOCUS into a reusable Responsibility Engine

Current FOCUS conceptually computes

\[
s_t
\rightarrow
b_{\phi}^{1:H}
\rightarrow
g_{i,t}
\rightarrow
\rho_{i,t}.
\]

This logic should be extracted from the QPLEX learner.

Suggested interface:

```python
class FocusResponsibilityEngine(nn.Module):
    def forward(
        self,
        global_state,
        joint_actions,
        camera_state,
        future_target_positions=None,
        valid_mask=None,
    ):
        # Returns:
        # rho:        [B, T, N_agents]
        # gains:      [B, T, N_agents]
        # total_gain: [B, T]
        # confidence: [B, T]
        # valid:      [B, T]
        ...
```

Internally:

```text
global state
    ↓
future occupancy predictor
    ↓
future occupancy samples
    ↓
prospective camera footprints
    ↓
counterfactual unique coverage
    ↓
predictive gains g_i
    ↓
responsibility rho_i
```

The engine must not know whether the downstream learner is QPLEX or MAPPO.

---

# 4. Why MAPPO should use responsibility-weighted policy updates

QPLEX_FOCUS aligns \(\rho_i\) with an internal mixer allocation.

MAPPO has no natural QPLEX-style allocation variable.

Therefore, do not invent a pseudo-\(p_i\) merely to mimic the QPLEX formulation.

Instead use the native policy-gradient term:

\[
\nabla_\theta
\log\pi_i(a_{i,t}\mid\tau_{i,t})
A_t.
\]

The two signals have different roles:

- \(A_t\): how good or bad the observed outcome is according to the original MARL algorithm;
- \(\rho_{i,t}\): how strongly agent \(i\) should participate in that update.

FOCUS should therefore modulate **update magnitude across agents**, not replace the environmental advantage.

---

# 5. Responsibility-to-gradient mapping

Use the following mean-preserving interpolation:

\[
w_{i,t}
=
(1-\eta)
+
\eta N_C\rho_{i,t},
\qquad
0\le\eta\le1.
\]

This mapping is recommended because it has three useful properties.

### Baseline recovery

If

\[
\eta=0,
\]

then

\[
w_{i,t}=1.
\]

The original MAPPO objective is recovered exactly.

### Uniform-responsibility recovery

If

\[
\rho_{i,t}=\frac{1}{N_C},
\]

then

\[
w_{i,t}=1
\]

for every agent.

Therefore an uninformative uniform FOCUS target does not distort MAPPO.

### Mean policy-loss scale is preserved

Since

\[
\sum_i\rho_{i,t}=1,
\]

we have

\[
\frac{1}{N_C}\sum_i w_{i,t}=1.
\]

Thus FOCUS redistributes actor-update emphasis across agents without automatically increasing the average actor-loss scale.

---

# 6. MAPPO + FOCUS objective

Let

\[
r_{i,t}(\theta)
=
\frac{
\pi_\theta(a_{i,t}\mid\tau_{i,t})
}{
\pi_{\theta_{\mathrm{old}}}(a_{i,t}\mid\tau_{i,t})
}
\]

be the PPO probability ratio.

Let \(A_t\) be the advantage already produced by the original MAPPO implementation. If the implementation already uses agent-specific \(A_{i,t}\), keep those values.

The native PPO clipped term is

\[
\ell_{i,t}^{\mathrm{PPO}}
=
\min
\left(
r_{i,t}A_t,\;
\operatorname{clip}(r_{i,t},1-\epsilon,1+\epsilon)A_t
\right).
\]

FOCUS changes only the per-agent weighting:

\[
\mathcal L_{\mathrm{actor}}^{\mathrm{FOCUS}}
=
-
\mathbb E_{i,t}
\left[
\operatorname{sg}(w_{i,t})
\ell_{i,t}^{\mathrm{PPO}}
\right].
\]

Use stop-gradient on \(w_{i,t}\).

The actor must not be able to manipulate the responsibility estimator to reduce its own loss.

---

# 7. Correct PPO implementation

Correct:

```python
ratio = torch.exp(
    current_logp - old_logp
)

surrogate_1 = (
    ratio * advantages
)

surrogate_2 = (
    torch.clamp(
        ratio,
        1.0 - clip_param,
        1.0 + clip_param,
    )
    * advantages
)

surrogate = torch.minimum(
    surrogate_1,
    surrogate_2,
)

actor_loss = -masked_mean(
    focus_weight.detach()
    * surrogate,
    valid_sequence_mask,
)
```

Do **not** do:

```python
ratio = focus_weight * current_prob / old_prob
```

Do **not** change the PPO clipping interval with \(\rho\).

Do **not** replace \(A_t\) by \(\rho_t\).

The flow must be:

```text
native PPO ratio + advantage
          ↓
native clipped PPO surrogate
          ↓
FOCUS agent-responsibility weight
          ↓
actor loss
```

---

# 8. Critic and PPO machinery remain unchanged

For the MVP:

```text
Actor loss       → FOCUS-weighted
Critic loss      → unchanged
GAE              → unchanged
Value targets    → unchanged
Entropy bonus    → unchanged
PPO clipping     → unchanged
```

The total loss becomes conceptually

\[
\mathcal L
=
\mathcal L_{\mathrm{actor}}^{\mathrm{FOCUS}}
+
c_v\mathcal L_{\mathrm{value}}
-
c_e\mathcal H(\pi)
+
\beta\mathcal L_{\mathrm{belief}}.
\]

There is no QPLEX-style

\[
D_{\mathrm{KL}}(\rho\Vert p^{\mathcal M})
\]

term in MAPPO because MAPPO does not expose a mixer allocation \(p^{\mathcal M}\).

---

# 9. Confidence should interpolate back to vanilla MAPPO

Current FOCUS has a confidence/reliability signal \(c_t\).

Do not multiply the complete MAPPO actor loss by confidence, because that would suppress native policy learning when FOCUS is uncertain.

Instead define

\[
\eta_t
=
\eta c_t
\]

and use

\[
w_{i,t}
=
(1-\eta_t)
+
\eta_t N_C\rho_{i,t}.
\]

Then:

- if \(c_t=1\), FOCUS acts at full configured strength;
- if \(c_t=0\), \(w_{i,t}=1\) and the original MAPPO update is recovered.

Recommended implementation:

```python
eta_t = (
    focus_eta
    * focus_confidence.detach()
)

focus_weight = (
    1.0 - eta_t
    + eta_t
      * n_agents
      * focus_rho.detach()
)
```

For invalid FOCUS transitions, simply use:

```python
effective_weight = 1.0
```

rather than suppressing actor learning.

---

# 10. Shared-policy and separate-policy cases

## Shared camera policy

If all cameras share actor parameters, attach one scalar responsibility weight to each agent-time sample:

```text
(agent i, time t)
    ↓
rho_i,t
    ↓
w_i,t
    ↓
that sample's PPO surrogate
```

The shared actor receives the weighted sum of gradients across camera samples.

## Separate camera policies

If every camera has its own actor:

\[
\theta_1,\ldots,\theta_{N_C},
\]

then

\[
\mathcal L_i
=
-
\mathbb E_t[
w_{i,t}
\ell_{i,t}^{\mathrm{PPO}}
].
\]

No conceptual change is required.

---

# 11. Main implementation challenge: joint data alignment

The difficult part is not modifying the PPO loss.

The difficult part is computing a correct joint responsibility vector

\[
\boldsymbol\rho_t
\]

before PPO data are flattened into independent agent samples.

FOCUS needs joint camera information:

```text
global state
joint camera actions
camera geometry
future target supervision for the belief model
```

while PPO minibatches may contain individual agent rows.

Therefore the preferred pipeline is:

```text
environment rollout
      ↓
complete multi-agent trajectory fragment
      ↓
assemble joint camera transition at each t
      ↓
FOCUS Responsibility Engine
      ↓
rho[t, agent]
      ↓
attach rho_i,t to each agent training row
      ↓
PPO minibatching / SGD
      ↓
responsibility-weighted surrogate
```

Compute \(\rho_t\) **before PPO minibatch flattening**.

---

# 12. New SampleBatch fields

Add:

```python
FOCUS_RHO = "focus_rho"
FOCUS_WEIGHT = "focus_weight"
FOCUS_VALID = "focus_valid"
FOCUS_CONFIDENCE = "focus_confidence"
FOCUS_GAIN = "focus_gain"
```

Each individual agent row stores:

```text
focus_rho:
    rho_i,t

focus_weight:
    w_i,t

focus_valid:
    whether FOCUS can distinguish responsibility

focus_confidence:
    current FOCUS reliability

focus_gain:
    g_i,t
```

The joint \(\rho_t\) distribution should also be logged separately for diagnostics.

---

# 13. Joint trajectory postprocessing

Use the multi-agent trajectory/postprocessing stage to assemble other-agent information.

Conceptual implementation:

```python
def add_focus_to_trajectory(
    policy,
    sample_batch,
    other_agent_batches,
    episode,
):
    joint_batch = assemble_joint_camera_batch(
        sample_batch=sample_batch,
        other_agent_batches=other_agent_batches,
        episode=episode,
    )

    with torch.no_grad():
        focus = policy.focus_engine(
            global_state=joint_batch.global_state,
            joint_actions=joint_batch.actions,
            camera_state=joint_batch.camera_state,
            valid_mask=joint_batch.valid_mask,
        )

    attach_focus_fields(
        sample_batch=sample_batch,
        rho=focus.rho,
        gains=focus.gains,
        confidence=focus.confidence,
        valid=focus.valid,
        current_agent_id=get_agent_id(...),
    )

    return sample_batch
```

The exact RLlib hook can follow the current MAPPO wrapper, but the responsibility should be computed at the joint-trajectory stage.

---

# 14. Synchronization requirements

Never align camera trajectories merely by row index.

Align using:

```text
episode id
environment id
timestep
agent id
```

Add assertions:

```python
assert same_episode
assert same_timestep
assert n_joint_agents == expected_n_agents
```

If the joint transition is incomplete because of padding, truncation, or missing agent rows:

```python
focus_valid = False
```

and the policy should fall back to weight \(1\).

---

# 15. Occupancy-predictor training

The policy-based branch still needs the existing future-occupancy belief loss.

However, if camera policies share parameters, the same global future-target supervision may appear once per camera row.

Avoid multiplying the belief loss by \(N_C\).

The recommended MVP is an **anchor-agent rule**:

```python
focus_anchor = (
    agent_index == 0
)
```

Compute

\[
\mathcal L_{\mathrm{belief}}
\]

only on anchor rows.

Alternative later:

- explicitly deduplicate joint timesteps;
- move the responsibility engine into a separate centralized learner.

Do not add this complexity before the MAPPO MVP works.

---

# 16. Recommended repository layout

Keep existing baselines frozen.

Suggested structure:

```text
ray/rllib/agents/
├── ppo/
│   └── ...                    # existing baseline PPO
│
├── focus_common/
│   ├── responsibility_engine.py
│   ├── occupancy.py
│   ├── geometry.py
│   ├── batch_utils.py
│   └── adapters.py
│
└── ppo_focus/
    ├── __init__.py
    ├── ppo_focus.py
    ├── ppo_focus_torch_policy.py
    └── focus_postprocessing.py
```

Experiment code:

```text
examples/
├── mappo/
│   └── ...                    # frozen baseline
│
└── mappo_focus/
    └── camera/
        ├── agent.py
        ├── config.py
        ├── train.py
        └── __main__.py
```

Script:

```text
scripts/camera.mappo_focus.sh
```

Do not modify the existing MAPPO script in place.

---

# 17. What to extract from QPLEX_FOCUS

Reuse:

```text
future occupancy model
future-position prediction loss
Sobol / Monte-Carlo future sampling
camera visibility geometry
counterfactual unique-coverage computation
predictive gain calculation
rho normalization
validity mask
current reliability logic
```

Do not reuse:

```text
credit_prior(states)
QPLEX lambda/head logic
mixer allocation distribution
QPLEX-specific CE/KL credit alignment
```

Those belong only to the value-based adapter.

---

# 18. Stable Responsibility Engine output

Use one output contract:

```python
@dataclass
class FocusOutput:
    rho: torch.Tensor
    gains: torch.Tensor
    total_gain: torch.Tensor
    confidence: torch.Tensor
    valid: torch.Tensor
    belief_loss: torch.Tensor | None = None
```

Joint shapes:

```text
rho:
[B, T, N_agents]

gains:
[B, T, N_agents]

total_gain:
[B, T]

confidence:
[B, T]

valid:
[B, T]
```

Optimization adapters must not inspect occupancy samples or camera geometry.

---

# 19. MAPPO adapter

```python
class MAPPOFocusAdapter:
    def __init__(self, eta):
        self.eta = eta

    def responsibility_to_weight(
        self,
        rho,
        n_agents,
        confidence=None,
    ):
        if confidence is None:
            eta_t = self.eta
        else:
            eta_t = (
                self.eta
                * confidence.detach()
            )

        weight = (
            1.0 - eta_t
            + eta_t
              * n_agents
              * rho.detach()
        )

        return weight
```

Do not clip weights in the first experiment.

If later training becomes unstable, add optional clipping and then renormalize across agents.

---

# 20. PPO policy-loss pseudocode

```python
ratio = torch.exp(
    current_logp
    - old_logp
)

surrogate_1 = (
    ratio
    * advantages
)

surrogate_2 = (
    torch.clamp(
        ratio,
        1.0 - clip_param,
        1.0 + clip_param,
    )
    * advantages
)

ppo_surrogate = torch.minimum(
    surrogate_1,
    surrogate_2,
)

focus_weight = train_batch[
    FOCUS_WEIGHT
].detach()

focus_valid = train_batch[
    FOCUS_VALID
].bool()

effective_weight = torch.where(
    focus_valid,
    focus_weight,
    torch.ones_like(
        focus_weight
    ),
)

actor_loss = -masked_mean(
    effective_weight
    * ppo_surrogate,
    recurrent_mask,
)
```

Then:

```python
total_loss = (
    actor_loss
    + vf_coeff * value_loss
    - entropy_coeff * entropy
    + belief_coeff * belief_loss
)
```

---

# 21. Configuration

Suggested configuration:

```python
FOCUS_POLICY_CONFIG = {
    "focus_enabled": True,

    # optimization adapter
    "focus_policy_eta": 0.5,
    "focus_use_confidence": True,

    # responsibility engine
    "focus_horizon": 4,
    "focus_sampling_mode": "MC",
    "focus_gain_eps": 1e-6,

    # auxiliary occupancy learning
    "focus_belief_coef": 1.0,

    # debugging
    "focus_log_per_agent": True,
}
```

Two controls must exactly recover MAPPO:

```python
focus_enabled = False
```

and

```python
focus_policy_eta = 0.0
```

---

# 22. First hyperparameter sweep

Only sweep adapter strength:

\[
\eta
\in
\{0,\;0.25,\;0.5,\;0.75,\;1.0\}.
\]

Do not simultaneously retune the occupancy model.

---

# 23. Mandatory logging

Log:

```text
focus/rho_entropy
focus/rho_max
focus/rho_min

focus/weight_mean
focus/weight_std
focus/weight_max
focus/weight_min

focus/confidence_mean
focus/valid_fraction

focus/belief_loss
focus/total_gain_mean

focus/native_actor_loss
focus/weighted_actor_loss
focus/actor_loss_delta
```

If practical, also log per-agent values.

---

# 24. Gradient diagnostics

For early experiments, inspect:

```text
rho_i
focus_weight_i
agent actor-gradient contribution
```

Expected qualitative behavior:

```text
higher responsibility
    ↓
larger actor-update weight
    ↓
stronger contribution to shared actor gradient
```

This should be verified rather than assumed.

---

# 25. Required unit tests

## Test A — Uniform responsibility

Set

\[
\rho_i=\frac1{N_C}.
\]

Expected:

```python
focus_weight == 1.0
```

for every agent.

MAPPO+FOCUS loss must equal MAPPO loss.

## Test B — `eta = 0`

For arbitrary \(\rho\):

```python
eta = 0
```

Expected:

```python
focus_weight == 1.0
```

and baseline actor loss is recovered exactly.

## Test C — Mean-one weights

For normalized \(\rho\):

```python
focus_weight.mean(agent_dim)
```

must be approximately `1.0`.

Example:

\[
\rho=[0.7,0.2,0.1],\quad N_C=3,\quad \eta=1
\]

gives

\[
w=[2.1,0.6,0.3].
\]

The mean is exactly \(1\).

## Test D — Invalid FOCUS signal

Set:

```python
focus_valid = False
```

Expected:

```python
effective_weight = 1.0
```

and native MAPPO learning continues.

## Test E — Stop-gradient

Backpropagate actor loss only.

Expected:

```text
actor parameters:
    gradient exists

rho / responsibility:
    no gradient

occupancy predictor:
    no actor-loss gradient
```

## Test F — PPO clipping unchanged

Given identical:

```text
ratio
advantage
clip_param
```

baseline and FOCUS must choose the same clipped surrogate branch.

Only final sample weighting may differ.

## Test G — Agent permutation

For a shared policy, consistently permuting camera indices and \(\rho\) should leave the total policy loss unchanged up to numerical precision.

---

# 26. Baseline experiments

Minimum comparison:

| Method | Responsibility | Optimization |
|---|---|---|
| MAPPO | None | Native PPO |
| MAPPO + Uniform-FOCUS | Uniform | Weighted PPO |
| MAPPO + Shuffled-FOCUS | Shuffled \(\rho\) | Weighted PPO |
| MAPPO + FOCUS | Predicted \(\rho\) | Weighted PPO |
| QPLEX | None | Native TD |
| QPLEX + FOCUS | Predicted \(\rho\) | Mixer alignment |

`Uniform-FOCUS` tests implementation equivalence.

`Shuffled-FOCUS` tests whether any improvement comes from meaningful responsibility rather than generic reweighting.

---

# 27. Main scientific test

The most important comparison is

\[
\boxed{
\text{MAPPO+FOCUS}
\overset{?}{>}
\text{MAPPO}
}
\]

If yes, then the central claim becomes stronger:

> Future-occupancy responsibility is useful across distinct MARL optimization families.

---

# 28. If MAPPO+FOCUS fails

Diagnose before changing the method.

### Case 1 — responsibility weights are almost uniform

If:

```text
rho entropy high
weight_std ≈ 0
```

FOCUS provides little agent differentiation.

### Case 2 — responsibility is differentiated but performance does not improve

Then the responsibility-weighted PPO surrogate may be the wrong policy adapter.

Only then consider:

- auxiliary actor loss,
- responsibility-conditioned critic,
- responsibility distillation,
- potential-based shaping.

### Case 3 — training becomes unstable

Inspect:

```text
weight_max
policy KL
PPO clip fraction
gradient norm
entropy
```

If extreme weights are the problem, introduce bounded and renormalized weights.

---

# 29. IPPO extension

The same adapter applies to IPPO:

\[
\mathcal L_{\mathrm{IPPO+FOCUS}}
=
-
\mathbb E_{i,t}
[
w_{i,t}
\ell_{i,t}^{\mathrm{IPPO}}
].
\]

The only difference is the native critic/advantage estimator.

Because FOCUS uses centralized information during training to compute \(\rho_t\), describe this as:

> IPPO backbone with centralized FOCUS training supervision.

Execution remains decentralized.

---

# 30. Generic policy-gradient adapter

For REINFORCE/A2C-style methods:

\[
\mathcal L_{\mathrm{PG}}
=
-
\mathbb E[
A_{i,t}
\log\pi_i(a_{i,t}\mid\tau_{i,t})
].
\]

FOCUS gives:

\[
\mathcal L_{\mathrm{PG+FOCUS}}
=
-
\mathbb E[
w_{i,t}
A_{i,t}
\log\pi_i(a_{i,t}\mid\tau_{i,t})
].
\]

Thus the generic policy-based adapter is:

\[
\boxed{
\text{native per-agent policy-update term}
\times
\text{FOCUS responsibility weight}
}
\]

---

# 31. Recommended implementation order

### Milestone 1

```text
1. Freeze current QPLEX_FOCUS.
2. Extract the current responsibility estimator into focus_common/.
3. Create a MAPPO_FOCUS branch/config.
4. Compute joint rho before PPO minibatch flattening.
5. Attach rho_i,t to each agent sample.
6. Use mean-preserving responsibility weights.
7. Weight only the clipped PPO actor surrogate.
8. Keep critic/GAE/entropy/clipping unchanged.
9. Train the existing occupancy predictor with the existing belief loss.
10. Verify eta=0 and uniform-rho equivalence.
```

### Milestone 2

After MAPPO+Predicted-FOCUS is stable, add a second responsibility source:

```text
Responsibility Engine
    ├── PredictedFOCUS
    └── OracleFOCUS

Optimization Adapter
    ├── QPLEX
    └── MAPPO
```

This allows a clean 2×2 experimental design later without changing the MAPPO adapter.

---

# 32. Final target architecture

```text
                    CENTRALIZED TRAINING
                            |
                   joint trajectory
                            |
                 FOCUS Responsibility
                        Engine
                            |
                          rho_t
                            |
          +-----------------+-----------------+
          |                                   |
   Value-based adapter                 Policy-based adapter
          |                                   |
   mixer alignment                responsibility-weighted
                                    policy-gradient update
          |                                   |
     QPLEX / QMIX                        MAPPO / IPPO


                 DECENTRALIZED EXECUTION

           local history / observation
                      |
                  local policy
                      |
                    action

           No FOCUS engine required.
           No future occupancy required.
           No additional communication.
```

---

# 33. Core statement for the revised method

> **FOCUS does not prescribe how a MARL algorithm must represent credit. It estimates relative task responsibility, while a backbone-specific adapter determines how that responsibility modulates learning.**

For QPLEX:

\[
\boxed{
\text{FOCUS responsibility}
\rightarrow
\text{mixer-allocation alignment}
}
\]

For MAPPO:

\[
\boxed{
\text{environmental advantage}
\times
\text{FOCUS responsibility weight}
\rightarrow
\text{policy update}
}
\]

The responsibility estimator is shared. Only the optimization adapter changes.
