# FOCUS Method Improvements — Theory

## 1. Motivation

The current QPLEX_FOCUS formulation improves credit assignment by replacing sparse realized marginal responsibility with a predictive responsibility target derived from future target occupancy. Its conceptual pipeline is

\[
s_t \rightarrow b_{\phi,j}^{1:H} \rightarrow g_{i,t} \rightarrow \rho_{i,t} \rightarrow p_{i,t}^{\mathrm{QPLEX}}.
\]

Here, \(b_{\phi,j}^{h}\) is the predicted occupancy of target \(j\), \(g_{i,t}\) is expected non-redundant future coverage, \(\rho_{i,t}\) is normalized predictive responsibility, and \(p_{i,t}^{\mathrm{QPLEX}}\) is the QPLEX-side allocation prior regularized by FOCUS.

The method has two important structural limitations.

First, FOCUS predicts **future targets but not future camera controllability**. Target occupancy evolves over \(1{:}H\), while contribution is evaluated using a prospective footprint induced by the current camera decision. Therefore, current FOCUS is a **future occupancy look-ahead with a prospective footprint approximation**, not a full predictive model of joint camera-target dynamics.

Second, FOCUS supervises a **mixer-side allocation variable**. This is natural for QPLEX, but the signal reaches decentralized behavior indirectly through value learning. A strong policy-based method can exploit more direct policy-level supervision.

The improvements below preserve the core idea of FOCUS—predict responsibility before sparse visibility events are realized—while making the predictive model and optimization pathway stronger.

---

## 2. Current FOCUS as the reference formulation

Current FOCUS uses a prospective visibility footprint

\[
\widetilde v_{i,t}(x\mid s_t,a_{i,t})
\]

and computes

\[
g_{i,t}
=
\sum_{h=1}^{H}\omega_h
\sum_{j=1}^{N_T}
\mathbb E_{x\sim b_{\phi,j}^{h}}
\left[
\widetilde v_{i,t}(x)
\prod_{k\neq i}(1-\widetilde v_{k,t}(x))
\right].
\]

The relative responsibility is

\[
\rho_{i,t}
=
\frac{g_{i,t}}{\sum_k g_{k,t}+\epsilon}.
\]

For QPLEX, current FOCUS regularizes the normalized state-conditioned allocation prior exposed by the mixer, not the complete effective \(\lambda_i\):

\[
\mathcal L_{\mathrm{credit}}
=
D_{\mathrm{KL}}
\left(
\operatorname{sg}(\boldsymbol\rho_t)
\Vert
\mathbf p_t^{\mathrm{QPLEX}}
\right).
\]

This formulation should remain frozen as the reference baseline.

---

# Part I — Improve the predictive model

## 3. Improvement 1: latent predictive state

### 3.1 Current limitation

The current prospective footprint asks:

> If camera \(i\) takes action \(a_{i,t}\) now, how useful is the resulting sensing region against target occupancy predicted at \(t+1,\ldots,t+H\)?

This is cheap, but for larger \(H\) the approximation becomes weaker because cameras can rotate and zoom again, observations change, teammates move their FoVs, and future actions depend on future observations.

### 3.2 Task-sufficient latent dynamics

Introduce a predictive latent state

\[
z_{t+h}
\sim
p_{\psi}(z_{t+h}\mid z_t,\mathbf a_{t:t+h-1}),
\]

where \(z_t\) only needs to preserve the information required for FOCUS:

1. future target occupancy,
2. future camera orientation / zoom or reachable sensing region,
3. obstacle-aware visibility,
4. predictive uncertainty.

The goal is **not** to reconstruct the entire environment. A task-sufficient predictive model is preferable to an unnecessarily large world model.

### 3.3 Theoretical benefit

Current FOCUS approximates

\[
\mathbb E[\Delta_{ij,t+h}\mid\mathcal H_t,\mathbf a_t]
\]

while evolving only target location. A latent predictive model instead approximates the future state distribution relevant to responsibility and therefore reduces the strongest approximation in the current method.

---

## 4. Improvement 2: horizon-dependent reachable visibility

A lighter improvement is to replace the fixed prospective footprint

\[
\widetilde v_{i,t}(x)
\]

with a horizon-dependent reachable footprint

\[
\widetilde v_{i,t}^{h}(x).
\]

Then

\[
g_{i,t}^{\mathrm{dyn}}
=
\sum_{h=1}^{H}\omega_h
\sum_j
\mathbb E_{x\sim b_{\phi,j}^{h}}
\left[
\widetilde v_{i,t}^{h}(x)
\prod_{k\neq i}(1-\widetilde v_{k,t}^{h}(x))
\right].
\]

Interpretation:

- \(b_{\phi,j}^{h}\): where target \(j\) may be at horizon \(h\),
- \(\widetilde v_{i,t}^{h}\): where camera \(i\) can plausibly cover at the same horizon.

This aligns the two predictive quantities temporally.

### Natural approximation ladder

**Level A — Current FOCUS**

\[
\widetilde v_{i,t}^{h}=\widetilde v_{i,t}.
\]

**Level B — Reachable geometric footprint**

Construct the set of orientations / zoom states reachable within \(h\) camera actions.

**Level C — Policy-conditioned footprint**

Predict a distribution over future camera configurations under the current policy.

**Level D — Latent predictive dynamics**

Predict a joint latent future state and derive visibility from that state.

These levels provide a clean ablation path.

---

# Part II — Improve the responsibility signal

## 5. Improvement 3: action-sensitive counterfactual responsibility

Current \(\rho_i\) answers **which camera is likely to matter**, but it does not directly answer **whether its selected current action is better than its alternatives**.

Define

\[
G_{i,t}(a_i)
=
\mathbb E[
\text{future unique coverage}
\mid
s_t,a_i,\mathbf a_{-i,t}
].
\]

Then define

\[
A_{i,t}^{\mathrm{FOCUS}}
=
G_{i,t}(a_{i,t})
-
\mathbb E_{\tilde a_i\sim\pi_i}
[G_{i,t}(\tilde a_i)].
\]

The distinction is important:

- \(\rho_i\): which agent should receive responsibility,
- \(A_i^{\mathrm{FOCUS}}\): whether the chosen action improves predictive responsibility relative to alternatives.

This creates a direct route from future occupancy to policy optimization.

---

## 6. Improvement 4: potential-based predictive shaping

A previously proposed route for universal integration is to convert predictive future coverage into a potential.

Let

\[
\Phi_i(s_t)
=
\text{predicted future unique-coverage potential of camera }i.
\]

Define

\[
F_{i,t}
=
\gamma \Phi_i(s_{t+1})-\Phi_i(s_t),
\]

and an augmented learning reward

\[
r_{i,t}^{\mathrm{aug}}
=
r_t+\eta F_{i,t}.
\]

This converts predictive responsibility into a temporally dense policy-learning signal and does not require a QPLEX-style mixer variable.

The interpretation should remain conservative:

> \(F_{i,t}\) is an auxiliary predictive shaping signal, not the true causal reward of camera \(i\).

---

# Part III — Make FOCUS reach the policy directly

## 7. Improvement 5: dual-level FOCUS

The strongest next version should use both value-level and policy-level supervision.

### Value-level path

Retain

\[
\mathcal L_{\mathrm{credit}}
=
D_{\mathrm{KL}}
(
\boldsymbol\rho_t
\Vert
\mathbf p_t^{\mathcal M}
).
\]

### Policy-level path

Use the action-sensitive predictive advantage:

\[
\mathcal L_{\mathrm{policy\text{-}focus}}
=
-\frac{1}{N_C}
\sum_i
\operatorname{sg}(A_{i,t}^{\mathrm{FOCUS}})
\log\pi_i(a_{i,t}\mid\tau_{i,t}).
\]

Conceptually,

\[
\mathcal L
=
\mathcal L_{\mathrm{MARL}}
+
\alpha\mathcal L_{\mathrm{credit}}
+
\beta\mathcal L_{\mathrm{belief}}
+
\zeta\mathcal L_{\mathrm{policy\text{-}focus}}.
\]

For QPLEX, \(\mathcal L_{\mathrm{MARL}}=\mathcal L_{\mathrm{TD}}\). For MAPPO/IPPO-style methods, the mixer-alignment term can be omitted while retaining the same predictive-responsibility engine.

This reframes FOCUS as a **predictive responsibility framework** rather than a module tied only to a value mixer.

---

## 8. Improvement 6: local responsibility distillation

Another previously proposed integration route is to expose predictive responsibility to the agent representation.

A centralized training feature could be

\[
\widetilde o_{i,t}
=
[o_{i,t},\rho_{i,t},\widehat g_{i,t},c_t].
\]

However, \(\rho_i\) depends on centralized information and therefore should not simply be given to the deployed policy.

Instead, train a local estimator

\[
\widehat\rho_{i,t}^{\mathrm{local}}
=
f_\chi(\tau_{i,t})
\]

by distillation from centralized FOCUS. At execution, only the local estimator is retained.

Interpretation:

> Centralized future-occupancy responsibility acts as a teacher for a locally computable anticipatory representation.

---

# Part IV — Improve reliability

## 9. Improvement 7: stronger confidence gating

A more robust FOCUS should distinguish:

1. **aleatoric uncertainty** in target motion,
2. **epistemic uncertainty** of the predictor,
3. **credit ambiguity** when several cameras have nearly identical responsibility.

A simple factorization is

\[
c_t
=
c_t^{\mathrm{pred}}
c_t^{\mathrm{credit}}.
\]

For example,

\[
c_t^{\mathrm{pred}}
=
\exp(-\kappa u_t),
\]

with predictive uncertainty \(u_t\), and

\[
c_t^{\mathrm{credit}}
=
1-
\frac{H(\boldsymbol\rho_t)}{\log N_C}.
\]

The second term is small when responsibility is nearly uniform, meaning that the geometry itself does not strongly distinguish agents.

Then

\[
\mathcal L_{\mathrm{credit}}
=
\mathbb E[
c_t
D_{\mathrm{KL}}(\rho_t\Vert p_t)
].
\]

This prevents overconfident alignment in geometrically ambiguous states.

---

## 10. Improvement 8: teacher/fallback responsibility

A robustness mechanism previously suggested is

\[
\rho_t^{\mathrm{final}}
=
c_t\rho_t^{\mathrm{belief}}
+
(1-c_t)\rho_t^{\mathrm{teacher}}.
\]

Possible training-only teachers:

- realized unique coverage when available,
- a greedy geometric controller,
- an oracle using realized future positions,
- a short rollout teacher.

Interpretation:

- high confidence: trust predictive occupancy,
- low confidence: fall back to a conservative teacher.

The oracle-future version is especially useful as an experimental upper bound even if it is not part of the final deployed method.

---

# Part V — Recommended FOCUS-v2

## 11. Recommended architecture

The recommended next method is:

### Stage 1 — Dynamic responsibility

\[
b_{\phi}^{h}
+
\widetilde v_{i,t}^{h}
\rightarrow
g_{i,t}^{\mathrm{dyn}}
\rightarrow
\rho_{i,t}^{\mathrm{dyn}}.
\]

### Stage 2 — Reliability-aware target

\[
\rho_t^{*}
=
c_t\rho_t^{\mathrm{dyn}}
+
(1-c_t)\rho_t^{\mathrm{teacher}}.
\]

### Stage 3 — Dual-level optimization

Use \(\rho_t^*\) for:

1. mixer allocation alignment,
2. action-sensitive policy guidance.

Overall:

\[
\boxed{
\text{predict future state}
\rightarrow
\text{estimate dynamic responsibility}
\rightarrow
\text{gate by reliability}
\rightarrow
\text{guide value allocation + policy}
}
\]

This directly targets the structural gap exposed when a strong policy-based method outperforms QPLEX_FOCUS.

---

# Part VI — Priority order

## Priority 1 — Horizon-dependent reachable footprint

Most important theoretical correction while preserving QPLEX_FOCUS.

- Risk: low-to-medium
- Code cost: medium

## Priority 2 — Better uncertainty / credit-ambiguity gating

Useful for every later extension.

- Risk: low
- Code cost: low-to-medium

## Priority 3 — Action-sensitive FOCUS advantage

First direct bridge to policy optimization.

- Risk: medium
- Code cost: medium

## Priority 4 — Policy-based FOCUS adapter

Integrate the same predictive engine into MAPPO/IPPO.

- Risk: medium-to-high
- Code cost: medium-to-high

## Priority 5 — Latent predictive dynamics

Use only if the geometric reachable-footprint variant saturates.

- Risk: high
- Code cost: high

---

# Part VII — Ablation matrix

| Variant | Future target | Future camera | Counterfactual | Reliability | Mixer guidance | Policy guidance |
|---|---:|---:|---:|---:|---:|---:|
| QPLEX | No | No | No | No | No | No |
| Current FOCUS | Yes | Fixed footprint | Yes | Basic | Yes | No |
| FOCUS-Dyn | Yes | Reachable footprint | Yes | Basic | Yes | No |
| FOCUS-Dyn-Conf | Yes | Reachable footprint | Yes | Improved | Yes | No |
| FOCUS-Dual | Yes | Reachable footprint | Yes | Improved | Yes | Yes |
| FOCUS-Latent | Yes | Latent dynamics | Yes | Improved | Yes | Optional |

Most important comparisons:

1. Current FOCUS vs. FOCUS-Dyn — does future camera controllability matter?
2. FOCUS-Dyn vs. FOCUS-Dyn-Conf — does reliability prevent harmful alignment?
3. FOCUS-Dyn-Conf vs. FOCUS-Dual — is direct policy guidance needed?
4. Reachable footprint vs. latent dynamics — is the heavier predictive model necessary?

---

# Part VIII — Safe claims

If experiments support them, the improved method can claim:

1. Future occupancy should be evaluated under future camera reachability rather than a purely fixed prospective footprint.
2. Predictive responsibility can supervise more than mixer-side allocation.
3. Reliability gating matters because predictive responsibility is itself uncertain.
4. FOCUS can be formulated as a predictive responsibility framework with value-based and policy-based optimization adapters.

Avoid claiming:

- \(\rho_i\) is ground-truth causal credit,
- the latent model is a full world model unless it truly models the complete joint dynamics,
- policy-based integration has the same theoretical properties as value decomposition,
- potential shaping preserves optimality unless the implemented shaping satisfies the exact potential-based assumptions.

---

# Part IX — Revised research story

> Sparse visibility makes realized agent-distinguishing responsibility rare. Current FOCUS densifies this signal using future target occupancy, but its prospective footprint only approximates future camera controllability. FOCUS-v2 predicts horizon-dependent reachable responsibility, estimates the reliability of that prediction, and uses the resulting signal to guide both centralized value allocation and, optionally, decentralized policy learning.

In one line:

\[
\boxed{
\text{Sparse realized credit}
\rightarrow
\text{future occupancy}
\rightarrow
\text{dynamic reachable responsibility}
\rightarrow
\text{value + policy guidance}
}
\]
