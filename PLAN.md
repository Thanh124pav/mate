# PLAN.md — Predictive Occupancy-Guided Credit Assignment for Sparse Multi-Agent Tracking

## 0. One-line thesis

In sparse and occluded multi-agent tracking, realized binary coverage events provide noisy and delayed credit signals. We improve credit assignment by replacing sparse realized credit with a lower-variance predictive credit signal computed as the conditional expectation of each camera's counterfactual marginal coverage over a future target occupancy belief.

---

## 1. Motivation

### 1.1 Problem setting

We consider a multi-camera multi-target tracking environment such as MATE.

There are:

- `N_c` camera agents.
- `N_t` target agents.
- A 2D map/grid `Omega`.
- Each camera has a field of view (FoV), controlled by action `a_i`.
- Targets may be hidden by obstacles, leave FoV, or behave strategically to avoid cameras.
- Camera reward is team-level, so individual camera credit is not directly observed.

For camera `i` and target `j`, define the binary visibility event:

```math
Z_{ij,t}
=
1[\text{camera } i \text{ tracks target } j \text{ at time } t].
```

The team-level coverage event for target `j` is:

```math
C_{j,t}
=
1[\exists i: Z_{ij,t}=1].
```

A typical team reward contains sparse coverage feedback:

```math
R_t
\propto
\sum_j C_{j,t}.
```

When the environment is sparse or occluded:

```math
P(Z_{ij,t}=1) \ll 1.
```

This makes credit assignment difficult because the reward is:

1. sparse: most transitions have no coverage signal;
2. delayed: useful camera moves may only lead to target coverage several steps later;
3. aliased: when targets are invisible, many latent target states collapse into the same local observation;
4. redundant: multiple cameras may observe the same target, so naive credit double-counts overlapping FoV.

---

### 1.2 Why QPLEX-like credit becomes noisy

QPLEX decomposes the joint action value as:

```math
Q_{\text{tot}}
=
V_{\text{tot}}
+
\sum_i \lambda_i A_i,
```

where:

```math
A_i = Q_i - V_i
```

is the local advantage, and:

```math
\lambda_i \geq 0
```

is an importance or credit-like coefficient.

The TD loss is:

```math
\mathcal L_{\text{TD}}
=
(y_{\text{tot}} - Q_{\text{tot}})^2.
```

Ignoring derivative terms through `lambda_i`, the gradient received by the individual advantage branch is approximately:

```math
\frac{\partial \mathcal L_{\text{TD}}}{\partial A_i}
\approx
-2\delta_t \lambda_i,
```

where:

```math
\delta_t = y_{\text{tot}} - Q_{\text{tot}}.
```

Therefore, the learned credit signal depends heavily on sparse TD errors. If target coverage events are rare, then `lambda_i` is learned from weak, noisy, and infrequent evidence.

The central hypothesis of this project is:

```math
\boxed{
\text{Sparse visibility creates high-variance credit, while predictive occupancy belief provides a lower-variance credit target.}
}
```

---

## 2. Analysis

### 2.1 Realized binary credit

Define the realized non-redundant marginal contribution of camera `i` for target `j`:

```math
Y_{ij,t}
=
C_{j,t}(\mathbf a_t)
-
C_{j,t}(\mathbf a_{-i,t}).
```

If coverage is binary:

```math
C_{j,t}
=
1-
\prod_k (1-Z_{kj,t}),
```

then:

```math
Y_{ij,t}
=
Z_{ij,t}
\prod_{k\neq i}(1-Z_{kj,t}).
```

This means camera `i` receives realized marginal credit only when it sees target `j` and no other camera sees the same target.

Let:

```math
\alpha_{ij}
=
P(Y_{ij,t}=1).
```

Then:

```math
Y_{ij,t}
\sim \text{Bernoulli}(\alpha_{ij}).
```

The Monte Carlo estimator of expected credit is:

```math
\widehat \mu_{ij}
=
\frac{1}{n}
\sum_{t=1}^n
Y_{ij,t}.
```

It is unbiased:

```math
E[\widehat \mu_{ij}]
=
\alpha_{ij}.
```

Its variance is:

```math
\operatorname{Var}[\widehat \mu_{ij}]
=
\frac{\alpha_{ij}(1-\alpha_{ij})}{n}.
```

However, the relative standard error is:

```math
\frac{
\sqrt{\operatorname{Var}[\widehat \mu_{ij}]}
}{
E[\widehat \mu_{ij}]
}
=
\sqrt{
\frac{1-\alpha_{ij}}{n\alpha_{ij}}
}.
```

When sparse visibility implies:

```math
\alpha_{ij}\to 0,
```

we get:

```math
\frac{
\sqrt{\operatorname{Var}[\widehat \mu_{ij}]}
}{
E[\widehat \mu_{ij}]
}
\to \infty.
```

Therefore, realized binary credit becomes statistically unreliable in sparse environments.

---

### 2.2 Predictive credit as Rao-Blackwellized credit

Let the local/global history available for prediction be:

```math
h_t.
```

Define predictive credit as the conditional expectation of realized marginal credit:

```math
g_{ij,t}
=
E[Y_{ij,t}\mid h_t,\mathbf a_t].
```

This is the Rao-Blackwellized version of the realized binary credit.

By the law of total variance:

```math
\operatorname{Var}(Y_{ij})
=
E[
\operatorname{Var}(Y_{ij}\mid h_t,\mathbf a_t)
]
+
\operatorname{Var}(
E[Y_{ij}\mid h_t,\mathbf a_t]
).
```

Therefore:

```math
\operatorname{Var}(g_{ij})
=
\operatorname{Var}(
E[Y_{ij}\mid h_t,\mathbf a_t]
)
\leq
\operatorname{Var}(Y_{ij}).
```

So predictive credit `g` has lower or equal variance than realized binary credit `Y`.

Interpretation:

```math
\boxed{
g_i \text{ is the expected counterfactual marginal coverage contribution of camera } i
\text{ under the current future occupancy belief.}
}
```

It is not an environment reward. It is an auxiliary credit target used to guide the learned credit coefficients of a value-decomposition algorithm.

---

### 2.3 Occupancy belief

For each target `j`, predict a future occupancy field:

```math
B_{j,t}^{H}(x)
=
\sum_{h=1}^{H}
\omega_h
P_\phi(X_{j,t+h}=x\mid h_t),
```

where:

- `x` is a grid cell or continuous coordinate;
- `H` is the prediction horizon;
- `omega_h` is a temporal discount/weight;
- `P_phi` may be produced by a learned world model, a particle filter, a known target-policy simulator, or a hybrid model.

If the target behavior is known, for example greedy, then the belief model may be algorithm-aware rather than fully learned.

If target behavior is adaptive or learned, use a behavior-conditioned belief:

```math
B_{j,t}^{H}(x)
=
\sum_z
q_\psi(z\mid h_t)
B_{j,t}^{H}(x\mid z),
```

where `z` is a latent target behavior type.

---

### 2.4 Visibility kernel

For camera `i`, action `a_i`, and position/grid cell `x`, define:

```math
v_i(x,a_i)\in[0,1].
```

This is the probability or soft score that camera `i` observes a target at `x`.

A simple hard version is:

```math
v_i(x,a_i)
=
1[x\in \mathcal F_i(a_i)]1[\text{not occluded}].
```

A soft version may include:

```math
v_i(x,a_i)
=
\text{FoVSoftness}(x,a_i)
\cdot
\text{ObstacleTransmittance}(x).
```

---

### 2.5 Predictive counterfactual marginal credit

The joint predictive coverage of target `j` under all cameras is:

```math
C_j^{\text{pred}}(\mathbf a)
=
\int_\Omega
B_{j,t}^{H}(x)
\left[
1-
\prod_{k=1}^{N_c}
(1-v_k(x,a_k))
\right]
dx.
```

The marginal predictive credit of camera `i` for target `j` is:

```math
\Delta_{ij}^{\text{pred}}
=
C_j^{\text{pred}}(\mathbf a)
-
C_j^{\text{pred}}(\mathbf a_{-i}).
```

Expanding:

```math
\Delta_{ij}^{\text{pred}}
=
\int_\Omega
B_{j,t}^{H}(x)
v_i(x,a_i)
\prod_{k\neq i}
(1-v_k(x,a_k))
dx.
```

Total predictive credit for camera `i` is:

```math
g_i
=
\sum_j
w_j
\Delta_{ij}^{\text{pred}}.
```

Normalize this into a credit distribution:

```math
\rho_i
=
\frac{g_i+\epsilon}
{\sum_k(g_k+\epsilon)}.
```

This `rho_i` is the predictive credit target for QPLEX-like `lambda_i`.

---

## 3. Method

### 3.1 Base architecture

Use a QPLEX/Qatten-style value-decomposition architecture:

```math
Q_{\text{tot}}
=
V_{\text{tot}}
+
\sum_i
\lambda_i A_i.
```

Use softmax credit coefficients:

```math
\lambda_i
=
\frac{\exp z_i}{\sum_k \exp z_k}.
```

The main TD objective is:

```math
\mathcal L_{\text{TD}}
=
(y_{\text{tot}}-Q_{\text{tot}})^2.
```

The predictive credit regularizer is:

```math
\mathcal L_\lambda
=
D_{\mathrm{KL}}
(
\operatorname{sg}(\rho)
\|
\lambda
).
```

In code, this is usually cross-entropy:

```math
\mathcal L_\lambda
=
-\sum_i
\operatorname{sg}(\rho_i)
\log(\lambda_i+\epsilon).
```

The belief/world-model loss is:

```math
\mathcal L_{\text{belief}}
=
-\sum_{j,h}
\log
P_\phi(X_{j,t+h}^{\text{true}}\mid h_t).
```

Full loss:

```math
\mathcal L
=
\mathcal L_{\text{TD}}
+
\alpha
\mathcal L_\lambda
+
\beta
\mathcal L_{\text{belief}}.
```

Recommended first implementation:

```math
\beta > 0
```

only if the belief model is learned.

If the belief is computed from a known target-policy simulator or particle filter, set:

```math
\beta = 0.
```

---

### 3.2 What is exact and what is approximate?

#### Exact or nearly exact

1. Camera geometry/FoV under action `a_i`.
2. Obstacle transmittance or binary occlusion if map geometry is available.
3. Current true target state during centralized training, if provided by env info.
4. Visibility map `V[i, cell]` once camera pose/action and map are known.
5. Predictive credit `g_i` given occupancy grid and visibility map.

#### Approximate

1. Future occupancy belief `B`.
2. Adaptive or learned target behavior.
3. Continuous-space integral over `Omega`.
4. Counterfactual action baselines if regularizing local advantages.
5. Q-values and TD targets.
6. Any credit target derived from a learned belief model.

---

### 3.3 Grid approximation of the credit integral

Discretize the map into grid cells:

```math
\Omega \approx \mathcal G.
```

Let:

```math
B[j, c]
```

be the probability mass of target `j` at cell `c`.

Let:

```math
V[i, c]
```

be the visibility score of camera `i` at cell `c`.

Then:

```math
g_i
\approx
\sum_j
w_j
\sum_{c\in\mathcal G}
B[j,c]
V[i,c]
\prod_{k\neq i}
(1-V[k,c]).
```

This is the main computation to implement.

---

### 3.4 Pseudocode: predictive credit computation

```python
def compute_predictive_credit(
    occ_belief,        # Tensor [B, N_t, G] probability mass over grid
    visibility,        # Tensor [B, N_c, G] soft visibility map for current camera actions
    target_weight,     # Tensor [B, N_t]
    eps=1e-8,
    mask_no_signal=True,
    min_signal=1e-6,
):
    """
    Returns:
        rho: Tensor [B, N_c], normalized predictive credit distribution
        g:   Tensor [B, N_c], unnormalized predictive credit
        valid_mask: Tensor [B], whether enough predictive signal exists
    """

    Bsz, N_t, G = occ_belief.shape
    _, N_c, _ = visibility.shape

    # g[b, i] = sum_j w[b,j] sum_c B[b,j,c] V[b,i,c] prod_{k!=i}(1 - V[b,k,c])
    g = zeros(Bsz, N_c)

    one_minus_v = 1.0 - visibility.clamp(0.0, 1.0)

    for i in range(N_c):
        # product over all cameras except i
        prod_not_seen_by_others = ones(Bsz, G)

        for k in range(N_c):
            if k == i:
                continue
            prod_not_seen_by_others *= one_minus_v[:, k, :]

        # unique coverage of camera i at each cell
        unique_vis_i = visibility[:, i, :] * prod_not_seen_by_others  # [B, G]

        # contribution for each target
        # occ_belief: [B, N_t, G]
        # unique_vis_i: [B, G] -> [B, 1, G]
        contrib_ij = (occ_belief * unique_vis_i[:, None, :]).sum(dim=-1)  # [B, N_t]

        # weighted sum over targets
        g[:, i] = (target_weight * contrib_ij).sum(dim=-1)

    total_g = g.sum(dim=-1, keepdim=True)  # [B, 1]

    valid_mask = (total_g.squeeze(-1) > min_signal)

    rho = (g + eps) / (total_g + eps * N_c)

    return rho, g, valid_mask
```

---

### 3.5 Pseudocode: training step

```python
def train_step(batch, networks, optimizers, cfg):
    """
    batch contains:
        obs/history
        actions
        rewards
        next_obs/history
        dones
        global_state or centralized info if available
        target future positions for belief training, if learned belief
    """

    # 1. Individual agent networks
    q_i, v_i, a_i_adv = networks.agent_q(batch.obs, batch.actions)
    # q_i:       [B, N_c, A]
    # a_i_adv:  [B, N_c] selected action advantages

    # 2. Mixer/QPLEX forward
    q_tot, lambda_credit = networks.mixer(
        agent_advantages=a_i_adv,
        state=batch.state,
        joint_action=batch.actions,
        return_lambda=True,
    )
    # q_tot: [B]
    # lambda_credit: [B, N_c], softmax-normalized if using proposed variant

    # 3. TD target
    with no_grad():
        target_q_tot = compute_qplex_target(batch, networks.target_networks, cfg)
        y_tot = batch.reward + cfg.gamma * (1 - batch.done) * target_q_tot

    td_loss = mse_loss(q_tot, y_tot)

    # 4. Future occupancy belief
    if cfg.belief_mode == "learned":
        occ_belief = networks.belief_model(batch.history, batch.map_info)
        belief_loss = occupancy_nll_loss(
            occ_belief,
            batch.future_target_positions,
        )
    elif cfg.belief_mode == "known_greedy_rollout":
        with no_grad():
            occ_belief = rollout_known_target_policy_to_occupancy(
                history=batch.history,
                map_info=batch.map_info,
                target_policy="greedy",
                horizon=cfg.pred_horizon,
                num_particles=cfg.num_particles,
            )
        belief_loss = 0.0
    elif cfg.belief_mode == "particle_filter":
        with no_grad():
            occ_belief = particle_filter_predictive_occupancy(
                history=batch.history,
                map_info=batch.map_info,
                horizon=cfg.pred_horizon,
                num_particles=cfg.num_particles,
            )
        belief_loss = 0.0
    else:
        raise ValueError("Unknown belief mode")

    # occ_belief: [B, N_t, G]

    # 5. Visibility maps for current joint camera actions
    with no_grad():
        visibility = compute_visibility_grid(
            camera_state=batch.camera_state,
            camera_actions=batch.actions,
            map_info=batch.map_info,
            grid=cfg.grid,
            soft=cfg.soft_visibility,
        )
        # visibility: [B, N_c, G]

    # 6. Target weights
    with no_grad():
        target_weight = compute_target_weights(
            batch.target_state_or_info,
            mode=cfg.target_weight_mode,
        )
        # target_weight: [B, N_t]

    # 7. Predictive credit target rho
    with no_grad():
        rho, g, valid_mask = compute_predictive_credit(
            occ_belief=occ_belief.detach(),
            visibility=visibility,
            target_weight=target_weight,
            eps=cfg.eps,
            min_signal=cfg.min_credit_signal,
        )

    # 8. Credit loss: align lambda with predictive responsibility
    # lambda_credit: [B, N_c]
    credit_loss_per_sample = -(rho * log(lambda_credit + cfg.eps)).sum(dim=-1)

    if cfg.mask_no_credit_signal:
        credit_loss = (credit_loss_per_sample * valid_mask.float()).sum() / (
            valid_mask.float().sum() + cfg.eps
        )
    else:
        credit_loss = credit_loss_per_sample.mean()

    # 9. Total loss
    loss = (
        td_loss
        + cfg.alpha_credit * credit_loss
        + cfg.beta_belief * belief_loss
    )

    optimizers.main.zero_grad()
    loss.backward()
    clip_grad_norm_(networks.parameters(), cfg.grad_clip)
    optimizers.main.step()

    return {
        "loss": loss.item(),
        "td_loss": td_loss.item(),
        "credit_loss": credit_loss.item(),
        "belief_loss": float(belief_loss),
        "mean_credit_signal": g.sum(dim=-1).mean().item(),
        "valid_credit_ratio": valid_mask.float().mean().item(),
        "lambda_entropy": entropy(lambda_credit).mean().item(),
        "rho_entropy": entropy(rho).mean().item(),
    }
```

---

### 3.6 Possible variants

#### Variant A: regularize only lambda

This is the recommended first variant.

```math
\mathcal L_\lambda
=
D_{\mathrm{KL}}(\rho\|\lambda).
```

Pros:

- Stable.
- Does not require matching the scale of `A_i`.
- Natural when `lambda` is softmax-normalized.

Cons:

- Only controls credit distribution, not advantage magnitude.

#### Variant B: regularize local advantage

Define action-level predictive advantage:

```math
\widetilde g_i
=
g_i(a_i,\mathbf a_{-i})
-
\sum_{a_i'}
\pi_i(a_i'\mid \tau_i)
g_i(a_i',\mathbf a_{-i}).
```

Then:

```math
\mathcal L_A
=
\sum_i
(A_i-\eta \operatorname{sg}(\widetilde g_i))^2.
```

Pros:

- Stronger connection to QPLEX advantage decomposition.

Cons:

- Requires enumerating or sampling counterfactual actions.
- Requires careful scaling because `A_i` is return-scale while `g_i` is coverage-scale.

#### Variant C: reward shaping baseline

Use:

```math
r_i^{\text{aux}} = g_i.
```

This is easy but should not be the main contribution because it may be viewed as dense reward shaping.

---

## 4. How to estimate whether `g` has lower variance than `Y`

### 4.1 Empirical variance comparison

Collect a replay/evaluation dataset.

For each transition, compute:

1. realized binary marginal credit `Y_i`;
2. predictive credit `g_i`.

Then estimate:

```math
\widehat{\operatorname{Var}}(Y_i),
\qquad
\widehat{\operatorname{Var}}(g_i).
```

Report variance ratio:

```math
\operatorname{VR}_i
=
\frac{
\widehat{\operatorname{Var}}(g_i)
}{
\widehat{\operatorname{Var}}(Y_i)+\epsilon
}.
```

Expected result:

```math
\operatorname{VR}_i < 1.
```

Also report average variance ratio:

```math
\operatorname{VR}
=
\frac{1}{N_c}
\sum_i
\operatorname{VR}_i.
```

---

### 4.2 Conditional variance decomposition

Group transitions by similar histories, difficulty regimes, visibility bins, or belief states.

Estimate:

```math
\operatorname{Var}(Y_i)
=
E[\operatorname{Var}(Y_i\mid h)]
+
\operatorname{Var}(E[Y_i\mid h]).
```

Since:

```math
g_i \approx E[Y_i\mid h],
```

compare:

```math
\widehat{\operatorname{Var}}(g_i)
```

with:

```math
\widehat{\operatorname{Var}}(Y_i).
```

This supports the Rao-Blackwellization claim.

---

### 4.3 Relative standard error

For sparse Bernoulli credit:

```math
Y_i\sim \text{Bernoulli}(\alpha_i).
```

Estimate:

```math
\widehat \alpha_i = \frac{1}{n}\sum_t Y_{i,t}.
```

The relative standard error of realized credit is approximately:

```math
\operatorname{RSE}(Y_i)
=
\sqrt{
\frac{1-\widehat\alpha_i}
{n\widehat\alpha_i+\epsilon}
}.
```

For predictive credit, use bootstrap over the dataset:

```math
\operatorname{RSE}(g_i)
=
\frac{
\operatorname{Std}_{\text{bootstrap}}(\widehat E[g_i])
}{
|\widehat E[g_i]|+\epsilon
}.
```

Report:

```math
\frac{\operatorname{RSE}(g_i)}{\operatorname{RSE}(Y_i)}.
```

---

### 4.4 SNR of learning signal

For each transition, define approximate credit gradient signal:

```math
G_i^{Y}
=
Y_i \cdot \|\nabla_{\theta_i} A_i\|
```

and predictive-credit signal:

```math
G_i^{g}
=
g_i \cdot \|\nabla_{\theta_i} A_i\|.
```

Estimate:

```math
\operatorname{SNR}(S)
=
\frac{\|\mathbb E[S]\|^2}
{\operatorname{Var}(S)+\epsilon}.
```

Report:

```math
\operatorname{SNR}(G_i^g)
>
\operatorname{SNR}(G_i^Y).
```

This is optional but useful for connecting credit variance to optimization.

---

### 4.5 Correlation with oracle marginal credit

If centralized information is available, compute oracle marginal credit:

```math
\Delta_i^{\text{oracle}}
=
C(\mathbf a)
-
C(\mathbf a_{-i})
```

using true target positions.

Measure:

```math
\operatorname{Corr}(g_i,\Delta_i^{\text{oracle}}),
```

and compare with:

```math
\operatorname{Corr}(\lambda_i,\Delta_i^{\text{oracle}}),
\qquad
\operatorname{Corr}(\lambda_i A_i,\Delta_i^{\text{oracle}}).
```

After training with the proposed method, expected results:

```math
\operatorname{Corr}(\lambda_i,\Delta_i^{\text{oracle}})
\text{ increases.}
```

---

## 5. Experiments

### 5.1 Core research questions

RQ1. Does sparse visibility increase credit noise?

RQ2. Does predictive occupancy credit have lower variance than realized binary credit?

RQ3. Does regularizing QPLEX lambda with predictive credit improve learned credit assignment?

RQ4. Does improved credit assignment translate to better coverage/return in sparse and occluded environments?

RQ5. Does opponent-aware or behavior-conditioned belief help when targets are non-greedy or learned to evade cameras?

---

### 5.2 Environment regimes

Create multiple MATE configurations.

#### Easy regime

- Many targets.
- Few obstacles.
- Wider FoV.
- High visibility rate.

Expected:

```math
p_{\text{vis}} \text{ high},
\quad
\alpha_i \text{ high},
\quad
\text{credit noise low}.
```

#### Sparse regime

- Fewer targets.
- Narrower FoV.
- Larger map or more empty space.

Expected:

```math
p_{\text{vis}} \downarrow,
\quad
\alpha_i \downarrow,
\quad
\text{credit noise} \uparrow.
```

#### Occluded regime

- More obstacles.
- Lower transmittance.
- Targets can hide behind obstacles.

Expected:

```math
I(o_i;X_j)\downarrow,
\quad
D_{\text{invisible}}\uparrow.
```

#### Overlap regime

- Cameras have overlapping FoVs.
- Tests whether the method reduces redundant credit.

Expected:

```math
\text{redundant coverage}\downarrow,
\quad
\text{unique coverage}\uparrow.
```

#### Adaptive target regime

- Targets use different policies:
  - random;
  - greedy;
  - evasive heuristic;
  - learned policy;
  - mixture of policies.

Expected:

```math
\text{opponent-aware belief} > \text{single-policy belief}.
```

---

### 5.3 Baselines

Minimum baselines:

1. GreedyCamera.
2. QMIX.
3. QPLEX.
4. Qatten.
5. QPLEX + softmax lambda.
6. QPLEX + MATE soft coverage auxiliary reward.
7. QPLEX + reward shaping using `g_i`.
8. Proposed: QPLEX + predictive credit lambda regularization.

Optional:

9. MAPPO.
10. MAPPO + predictive credit advantage weighting.
11. Oracle occupancy credit upper bound.

---

### 5.4 Ablations

#### Ablation A: source of belief

Compare:

1. no belief;
2. current visible target only;
3. constant velocity belief;
4. known greedy-policy rollout;
5. learned occupancy world model;
6. hybrid greedy prior + learned residual;
7. oracle future occupancy.

#### Ablation B: role of predictive credit

Compare:

1. TD loss only;
2. TD + softmax lambda;
3. TD + predictive credit lambda regularization;
4. TD + predictive credit reward shaping;
5. TD + predictive credit advantage regularization.

#### Ablation C: horizon

Test:

```math
H\in\{1,3,5,10,20\}.
```

Expected:

- Small `H`: insufficient future information.
- Very large `H`: belief becomes too diffuse.
- Medium `H`: best.

#### Ablation D: grid resolution

Test:

```math
G\in\{16\times16,\;32\times32,\;64\times64\}.
```

Measure runtime, memory, and performance.

#### Ablation E: credit coefficient

Test:

```math
\alpha\in\{0,0.01,0.05,0.1,0.5,1.0\}.
```

Expected:

- Too small: no effect.
- Too large: over-constrains lambda with imperfect belief.

---

### 5.5 Metrics

#### Performance metrics

1. Team return.
2. Coverage rate:

```math
\frac{
\#\text{tracked target timesteps}
}{
N_tT
}.
```

3. Real coverage rate for targets with cargo/bounty.
4. Target transport success rate.
5. Win/loss if environment defines it.

#### Sparsity metrics

1. Visibility probability:

```math
p_{\text{vis}}
=
\frac{1}{TN_cN_t}
\sum_{t,i,j}
Z_{ij,t}.
```

2. Coverage event rate:

```math
r_{\text{cov}}
=
\frac{1}{TN_t}
\sum_{t,j}
1[\exists i:Z_{ij,t}=1].
```

3. Mean invisible duration.
4. Reward-zero ratio:

```math
\zeta_R
=
\frac{1}{T}
\sum_t
1[R_t=0].
```

#### Credit metrics

1. Realized marginal credit probability:

```math
\alpha_i
=
\frac{1}{T}
\sum_t
Y_{i,t}.
```

2. Credit variance ratio:

```math
\operatorname{VR}
=
\frac{
\operatorname{Var}(g_i)
}{
\operatorname{Var}(Y_i)+\epsilon
}.
```

3. Relative standard error:

```math
\operatorname{RSE}(Y_i),
\quad
\operatorname{RSE}(g_i).
```

4. Correlation with oracle marginal credit:

```math
\operatorname{Corr}(\lambda_i,\Delta_i^{\text{oracle}}).
```

5. Credit entropy:

```math
H(\lambda)
=
-\sum_i
\lambda_i\log\lambda_i.
```

6. Redundant overlap:

```math
\operatorname{Redundancy}
=
\sum_j
\left(
\sum_i Z_{ij}
-
1[\exists i:Z_{ij}=1]
\right).
```

#### Belief metrics

1. Occupancy NLL.
2. Calibration error.
3. Top-k mass accuracy.
4. Mass-in-FoV prediction error:

```math
\left|
\int B_j^H(x)v_i(x,a_i)dx
-
1[X_{j,t+h}\in \mathcal F_i(a_i)]
\right|.
```

---

### 5.6 Expected findings

Expected result 1:

```math
p_{\text{vis}}\downarrow
\Rightarrow
\alpha_i\downarrow
\Rightarrow
\operatorname{RSE}(Y_i)\uparrow.
```

Expected result 2:

```math
\operatorname{Var}(g_i) < \operatorname{Var}(Y_i)
```

or at least:

```math
\operatorname{RSE}(g_i) < \operatorname{RSE}(Y_i).
```

Expected result 3:

Proposed method improves:

```math
\operatorname{Corr}(\lambda_i,\Delta_i^{\text{oracle}})
```

over QPLEX and QPLEX+softmax.

Expected result 4:

Performance gains are strongest in:

- sparse visibility;
- high occlusion;
- overlapping FoV;
- adaptive/evasive target policies.

Expected result 5:

In easy environments, proposed method should be comparable to QPLEX, not necessarily much better.

This supports the claim that the method specifically helps under difficult credit-assignment conditions.

---

## 6. Paper positioning

This is primarily a conceptual and algorithmic improvement for credit assignment.

The conceptual contribution is:

```math
\boxed{
\text{From sparse realized credit to predictive Rao-Blackwellized credit.}
}
```

The algorithmic contribution is:

```math
\boxed{
\text{Use future occupancy belief to supervise QPLEX/Qatten credit coefficients.}
}
```

The empirical contribution is:

```math
\boxed{
\text{Show that predictive credit improves learned credit assignment and performance in sparse, occluded, and adaptive multi-agent tracking.}
}
```

This direction should be positioned as:

- not merely reward shaping;
- not merely adding a world model;
- not merely replacing sigmoid by softmax;
- but a credit-assignment method that uses predictive occupancy as a lower-variance counterfactual responsibility signal.

---

## 7. Minimal implementation checklist

1. Implement grid representation of map.
2. Implement visibility grid `V[i, cell]`.
3. Implement occupancy belief provider:
   - start with known greedy rollout or oracle-like supervised occupancy;
   - then learned occupancy model.
4. Implement `compute_predictive_credit`.
5. Modify QPLEX mixer to return `lambda_i`.
6. Change lambda to softmax if not already done.
7. Add credit loss:

```python
credit_loss = -(rho.detach() * torch.log(lambda_credit + eps)).sum(dim=-1)
```

8. Mask samples where total predictive credit is too small.
9. Log:
   - `td_loss`;
   - `credit_loss`;
   - `belief_loss`;
   - `lambda_entropy`;
   - `rho_entropy`;
   - `credit_variance_ratio`;
   - `lambda_oracle_credit_corr`.
10. Run ablations.

---

## 8. Risks and mitigations

### Risk 1: belief model is wrong

Mitigation:

- compare known greedy rollout, learned model, and oracle occupancy;
- report belief NLL/calibration;
- use small `alpha_credit` early;
- anneal credit regularization.

### Risk 2: predictive credit over-constrains lambda

Mitigation:

- use mask when total credit signal is low;
- tune `alpha_credit`;
- regularize only lambda, not advantage magnitude at first.

### Risk 3: method only helps one environment

Mitigation:

- create multiple difficulty regimes within MATE;
- test across target policies;
- show effect correlates with visibility sparsity metrics.

### Risk 4: reviewer says this is dense reward shaping

Mitigation:

- include baseline with MATE soft coverage reward;
- include baseline using `g_i` as reward shaping;
- show lambda-credit alignment improves independently of reward shaping.

### Risk 5: reviewer says this is just a world model

Mitigation:

- include world-model-only reward-imagination baseline;
- show predictive credit regularization is the key component;
- emphasize Rao-Blackwellized counterfactual credit.
