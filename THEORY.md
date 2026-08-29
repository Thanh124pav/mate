# FOCUS for HMVFE: Responsibility-Weighted Policy Optimization

## 1. Scope

This document specifies the theory for integrating FOCUS into the current HMVFE implementation in `Thanh124pav/mate`, specifically the `hmvfe_mate_d/` coordinator.

The current HMVFE optimizer is **not MAPPO/PPO**. Its high-level coordinator is trained with an **A2C-style actor-critic objective using GAE**, and its policy produces a Bernoulli camera-target assignment matrix. Therefore, FOCUS must be adapted to HMVFE's actual policy-gradient decomposition rather than copying a PPO weighting rule.

The goal of the MVP is deliberately narrow:

> Keep HMVFE's policy, reward, critic, GAE, action semantics, and assignment executor unchanged, while using FOCUS to redistribute the actor gradient across cameras according to predicted future responsibility.

FOCUS remains a separate **Responsibility Engine**. HMVFE only consumes its output through an optimizer adapter.

---

## 2. What HMVFE is optimizing

At a high-level decision time `t`, let HMVFE output a binary assignment matrix

\[
Y_t \in \{0,1\}^{N_c \times N_g},
\]

where:

- \(N_c\): number of cameras,
- \(N_g\): number of targets/goals represented by the coordinator,
- \(Y_{ij,t}=1\): camera \(i\) is assigned to target \(j\).

The coordinator models these entries with Bernoulli distributions. Under the factorization used by the implementation,

\[
\pi_\theta(Y_t\mid s_t)
=
\prod_{i=1}^{N_c}\prod_{j=1}^{N_g}
\pi_\theta(Y_{ij,t}\mid s_t).
\]

Hence the joint log-probability is

\[
\log \pi_\theta(Y_t\mid s_t)
=
\sum_{i=1}^{N_c}\sum_{j=1}^{N_g}
\log \pi_\theta(Y_{ij,t}\mid s_t).
\]

The current implementation effectively performs this global sum before computing the actor loss.

Let \(A_t\) denote HMVFE's native GAE advantage. Because HMVFE is cooperative and optimized from the team-level outcome, \(A_t\) is the temporal desirability signal for the high-level joint decision.

Ignoring implementation-specific reduction details, vanilla HMVFE has the actor objective

\[
\mathcal L_\pi^{\mathrm{HMVFE}}
=
-\mathbb E_t\left[
\operatorname{sg}(A_t)
\log \pi_\theta(Y_t\mid s_t)
\right].
\]

Here `sg` denotes stop-gradient.

---

## 3. The hidden camera-wise decomposition

Although HMVFE currently collapses the policy log-probability to one scalar, its Bernoulli structure already contains a natural per-camera decomposition.

Define

\[
\ell_{i,t}
=
\sum_{j=1}^{N_g}
\log \pi_\theta(Y_{ij,t}\mid s_t).
\]

Then

\[
\log \pi_\theta(Y_t\mid s_t)
=
\sum_{i=1}^{N_c}\ell_{i,t}.
\]

Thus the vanilla actor loss can be rewritten exactly as

\[
\mathcal L_\pi^{\mathrm{HMVFE}}
=
-\mathbb E_t\left[
\operatorname{sg}(A_t)
\sum_i \ell_{i,t}
\right].
\]

This decomposition is the insertion point for FOCUS.

No new policy head is required. No per-camera critic is required. We only prevent the implementation from discarding the camera axis too early.

---

## 4. Credit gap in vanilla HMVFE

The GAE advantage \(A_t\) answers:

> Was the joint high-level decision at time \(t\) better or worse than expected?

It does **not** answer:

> Which camera should receive more responsibility for that outcome?

For a cooperative camera network, the same team-level advantage is propagated through every camera's component of the joint policy gradient:

\[
\nabla_\theta \mathcal L_\pi
\propto
-A_t
\sum_i
\nabla_\theta \ell_{i,t}.
\]

This is problematic in sparse and redundant visibility settings. For example:

- one camera may be the only camera capable of covering a future target;
- several cameras may redundantly cover the same target;
- a camera may currently have little meaningful influence on future team coverage;
- the team reward can be positive although responsibility is highly non-uniform across cameras.

HMVFE supplies temporal credit through \(A_t\), but it does not explicitly resolve camera-level responsibility.

FOCUS is designed to supply this missing factor.

---

## 5. FOCUS as a backbone-agnostic Responsibility Engine

FOCUS should remain independent from HMVFE's optimizer.

At each high-level decision time, FOCUS receives the joint environment information required by the existing FOCUS implementation and estimates future non-redundant coverage responsibility.

Conceptually:

\[
s_t
\rightarrow
\text{future occupancy prediction}
\rightarrow
\text{camera future footprint}
\rightarrow
\text{counterfactual unique coverage gain}
\rightarrow
\rho_t.
\]

FOCUS outputs

\[
\boldsymbol\rho_t
=
(\rho_{1,t},\ldots,\rho_{N_c,t}),
\qquad
\rho_{i,t}\ge 0,
\qquad
\sum_i\rho_{i,t}=1.
\]

Interpretation:

> \(\rho_{i,t}\) is the predicted relative responsibility of camera \(i\) for future non-redundant team coverage induced by the current joint decision context.

Optionally, FOCUS also outputs confidence

\[
c_t\in[0,1].
\]

The FOCUS estimator itself should not be reimplemented inside HMVFE. HMVFE should consume its outputs via an adapter.

---

## 6. From responsibility to safe actor weights

Directly multiplying the actor term by \(\rho_i\) would alter the overall loss scale and would not recover vanilla HMVFE when responsibility is uniform.

Instead define

\[
\eta_t=\eta c_t,
\qquad \eta\in[0,1],
\]

and

\[
\boxed{
 w_{i,t}
 =
 (1-\eta_t)+\eta_t N_c\rho_{i,t}
 =
 1+\eta_t(N_c\rho_{i,t}-1)
}.
\]

If confidence is not yet available, use \(c_t=1\) while retaining the interface.

The weights have several useful invariants.

### Uniform responsibility

If

\[
\rho_{i,t}=\frac1{N_c},
\]

then

\[
w_{i,t}=1.
\]

### FOCUS disabled

If \(\eta=0\), then

\[
w_{i,t}=1.
\]

### Zero confidence

If \(c_t=0\), then

\[
w_{i,t}=1.
\]

Therefore an uncertain FOCUS estimate falls back to ordinary HMVFE rather than suppressing learning.

### Mean-preserving redistribution

Because \(\sum_i\rho_i=1\),

\[
\frac1{N_c}\sum_iw_{i,t}=1.
\]

FOCUS therefore redistributes actor-gradient emphasis across cameras without changing the mean per-camera weight.

---

## 7. FOCUS-HMVFE actor objective

The proposed MVP actor objective is

\[
\boxed{
\mathcal L_\pi^{\mathrm{FOCUS-HMVFE}}
=
-\mathbb E_t
\left[
\operatorname{sg}(A_t)
\sum_{i=1}^{N_c}
\operatorname{sg}(w_{i,t})
\ell_{i,t}
\right]
}.
\]

Equivalently,

\[
\mathcal L_\pi^{\mathrm{FOCUS-HMVFE}}
=
-\mathbb E_t
\left[
\operatorname{sg}(A_t)
\sum_i
w_{i,t}
\sum_j
\log \pi_\theta(Y_{ij,t}\mid s_t)
\right].
\]

The policy gradient becomes

\[
\nabla_\theta \mathcal L_\pi
=
-\mathbb E_t\left[
A_t
\sum_i
w_{i,t}
\nabla_\theta \ell_{i,t}
\right].
\]

This gives a clean factorization:

\[
\boxed{
\text{policy-gradient signal}
=
\text{team temporal desirability}
\times
\text{camera responsibility}
}.
\]

- \(A_t\): whether the joint decision was good or bad;
- \(w_{i,t}\): how strongly camera \(i\)'s policy component should be credited for that decision.

---

## 8. Why the weighted sum must NOT be divided by the number of cameras

This is an important implementation constraint.

The old HMVFE joint log-probability is

\[
\ell_t^{\mathrm{old}}
=
\sum_i\ell_{i,t}.
\]

If FOCUS is disabled or responsibilities are uniform, \(w_i=1\). Exact baseline recovery therefore requires

\[
\ell_t^{\mathrm{new}}
=
\sum_iw_i\ell_i
=
\sum_i\ell_i
=
\ell_t^{\mathrm{old}}.
\]

Do **not** replace it by

\[
\frac1{N_c}\sum_iw_i\ell_i.
\]

That would rescale the actor loss and break numerical equivalence with the existing HMVFE baseline.

The mean-one property of \(w_i\) is a responsibility-normalization property, not a reason to average the existing joint log-probability.

---

## 9. Active-camera masks

If every HMVFE decision always contains all cameras, use the simple formulation above.

If some cameras can be inactive, let

\[
m_{i,t}\in\{0,1\},
\qquad
N_t=\sum_i m_{i,t}.
\]

Normalize responsibility only over active cameras:

\[
\bar\rho_{i,t}
=
\frac{m_{i,t}\rho_{i,t}}
{\sum_km_{k,t}\rho_{k,t}+\epsilon}.
\]

For active cameras use

\[
w_{i,t}
=
1+\eta_t(N_t\bar\rho_{i,t}-1).
\]

Inactive camera contributions must be masked out of the weighted actor log-probability.

The invariant is then

\[
\frac1{N_t}\sum_{i:m_i=1}w_{i,t}=1.
\]

---

## 10. Confidence must control interpolation, not learning itself

A tempting implementation is

\[
\mathcal L_\pi'=c_t\mathcal L_\pi.
\]

Do **not** do this.

If FOCUS is uncertain, HMVFE's native policy-gradient signal remains valid. Low FOCUS confidence should only reduce FOCUS's influence:

\[
\eta_t=\eta c_t.
\]

Then

\[
c_t\rightarrow0
\quad\Rightarrow\quad
w_{i,t}\rightarrow1,
\]

and training recovers vanilla HMVFE.

This fallback behavior is an important safety property.

---

## 11. Stop-gradient boundary

For the MVP, FOCUS is an external responsibility estimator.

Use

\[
\operatorname{sg}(\rho_t),
\qquad
\operatorname{sg}(w_t)
\]

inside the HMVFE actor objective.

The HMVFE actor must not optimize the FOCUS responsibility estimator indirectly through the weighted policy loss.

Reasons:

1. it isolates the scientific question: does the FOCUS responsibility signal improve HMVFE optimization?;
2. it prevents degenerate solutions where the responsibility network learns weights that merely reduce actor loss;
3. it preserves the semantics of FOCUS as an independent estimate of environmental responsibility;
4. it makes baseline and ablation interpretation much cleaner.

If the FOCUS predictor has its own supervised/self-supervised belief loss, that loss can continue training it separately.

---

## 12. Temporal alignment

FOCUS weights must correspond to the same high-level decision whose log-probability and return are used by HMVFE.

For every coordinator decision \(t\), store together:

- coordinator state/input;
- sampled assignment \(Y_t\);
- per-camera log-probability \(\ell_{i,t}\);
- value estimate;
- reward/return information needed by GAE;
- FOCUS responsibility \(\rho_{i,t}\);
- optional FOCUS confidence \(c_t\);
- optional active-camera mask.

If an HMVFE assignment persists for several environment steps, FOCUS must be evaluated and stored at the **coordinator decision tick**, not arbitrarily at every low-level environment step.

The resulting advantage \(A_t\) and responsibility \(\rho_t\) must refer to the same high-level action.

A one-step misalignment between \(A_t\), \(Y_t\), and \(\rho_t\) makes the method conceptually incorrect even if tensor shapes still match.

---

## 13. What remains unchanged in the MVP

FOCUS-HMVFE should change only the camera allocation of the actor-gradient contribution.

Keep unchanged:

- HMVFE actor architecture;
- HMVFE feature extraction/fusion modules;
- Bernoulli assignment parameterization;
- assignment sampling;
- environment action executor;
- environment/team reward;
- return computation;
- GAE;
- critic/value target;
- critic/value loss;
- entropy term and its coefficient;
- optimizer unless a purely mechanical parameter-registration change is required;
- HMVFE evaluation/inference behavior.

In particular, the MVP must **not** weight the critic loss by FOCUS.

The critic estimates team outcome. FOCUS estimates camera responsibility. Mixing these roles in the first implementation would make the source of any gain or failure ambiguous.

---

## 14. Entropy regularization

Keep HMVFE's original entropy regularization unchanged in the MVP.

Do not apply \(w_i\) to entropy unless it becomes a separate ablation.

Reason: weighting entropy would mean that FOCUS controls not only responsibility for the policy-gradient signal but also camera-specific exploration pressure. That is a different intervention.

The first experiment should test exactly one hypothesis:

> FOCUS improves HMVFE by allocating the native team actor gradient according to predicted camera responsibility.

---

## 15. Scientific interpretation

FOCUS now has a common abstraction across value-based and policy-based backbones.

### Value-decomposition backbone

FOCUS supervises or aligns an implicit credit carrier in the mixing architecture.

### HMVFE policy-gradient backbone

HMVFE has no mixer credit carrier. Instead, FOCUS modulates each camera's contribution to the joint policy gradient:

\[
\rho_i
\rightarrow
w_i
\rightarrow
w_i\nabla_\theta\ell_i.
\]

Thus the common abstraction is

\[
\boxed{
\text{FOCUS Responsibility Engine}
\rightarrow
\rho_t
\rightarrow
\text{backbone-specific credit adapter}
}.
\]

For HMVFE, the adapter is **responsibility-weighted camera-wise policy gradient**.

If the method works on both QPLEX-style value decomposition and HMVFE's A2C coordinator, this provides evidence that FOCUS is not tied to one mixer architecture.

---

## 16. Baseline recovery properties

The implementation should treat the following as theoretical invariants.

### Proposition 1: disabled FOCUS recovers HMVFE

If \(\eta=0\), then \(w_i=1\), hence

\[
\mathcal L_\pi^{\mathrm{FOCUS-HMVFE}}
=
\mathcal L_\pi^{\mathrm{HMVFE}}.
\]

### Proposition 2: uniform responsibility recovers HMVFE

If \(\rho_i=1/N_c\) for every camera, then \(w_i=1\), hence the actor objective is exactly unchanged.

### Proposition 3: zero confidence recovers HMVFE

If \(c_t=0\), then \(\eta_t=0\), \(w_i=1\), and the actor objective is exactly unchanged at that decision.

### Proposition 4: FOCUS changes relative credit, not mean camera weight

For all-active cameras,

\[
\frac1{N_c}\sum_iw_i=1.
\]

These properties are not merely theoretical conveniences. They should be unit-tested.

---

## 17. Main hypotheses

### H1 — responsibility transfer

FOCUS-HMVFE should outperform vanilla HMVFE when environmental visibility produces non-uniform future camera responsibility.

### H2 — semantic signal matters

Real FOCUS responsibility should outperform shuffled or random responsibility weights with comparable weight statistics.

If shuffled responsibility performs equally well, improvement is likely caused by generic gradient perturbation rather than meaningful responsibility estimation.

### H3 — no harm under uninformative responsibility

When FOCUS is uniform or has zero confidence, performance and updates should match vanilla HMVFE.

### H4 — gains should be stronger under sparse/redundant visibility

The benefit should correlate with scenarios in which camera-level credit ambiguity is stronger: sparse observations, overlapping FoVs, redundant assignments, obstacles, or target configurations with uneven responsibility.

---

## 18. Required ablations

At minimum compare:

1. **HMVFE** — untouched baseline.
2. **HMVFE + FOCUS** — proposed responsibility weights.
3. **HMVFE + uniform responsibility** — implementation sanity control; should recover baseline.
4. **HMVFE + shuffled FOCUS** — preserve weight distribution but destroy camera semantics.

Recommended additional ablations:

5. different \(\eta\): e.g. \(0.25,0.5,1.0\);
6. confidence on/off;
7. random responsibility with matched entropy;
8. FOCUS responsibility entropy vs performance;
9. scenario-wise comparison across visibility sparsity levels.

Use multiple seeds and report both final performance and learning/sample efficiency.

---

## 19. Diagnostics

Useful quantities to log:

\[
H(\rho_t)=-\sum_i\rho_i\log(\rho_i+\epsilon),
\]

\[
\|\rho_t-U\|_1,
\qquad U_i=1/N_c,
\]

plus:

- minimum/maximum responsibility;
- minimum/maximum/mean actor weight;
- confidence;
- effective number of responsible cameras, e.g. \(\exp(H(\rho))\);
- original joint log-prob and weighted joint log-prob;
- actor loss before/after FOCUS weighting as diagnostics only;
- gradient norm;
- realized team coverage;
- optional offline correlation between FOCUS responsibility and realized future unique coverage.

The weight mean should remain approximately 1 over active cameras.

---

## 20. Failure modes to watch

### 20.1 FOCUS collapse

If \(\rho\) collapses repeatedly to one camera without reliable evidence, actor gradients may become overly concentrated.

Mitigations to evaluate later include smaller \(\eta\), confidence interpolation, or bounded weights. Do not add these silently before establishing the base method.

### 20.2 Responsibility is predictive but not action-sensitive

Current \(\rho_i\) identifies which camera is important, but it does not necessarily measure whether that camera's sampled assignment/action was the best alternative.

This is a known limitation of the MVP, not a reason to change the first experiment.

### 20.3 Temporal mismatch

A correct responsibility vector attached to the wrong HMVFE decision is worse than no responsibility signal.

### 20.4 Double normalization

Do not average a log-probability that the old code summed. Exact baseline recovery is more important than adopting a superficially tidy normalization.

### 20.5 Gradient leakage into FOCUS

If the actor loss updates the responsibility predictor, the intended semantics and ablations become invalid.

---

## 21. Future extension: action-sensitive FOCUS advantage

After the responsibility-weighted HMVFE experiment is validated, a stronger policy-specific version can estimate counterfactual action quality.

For camera \(i\), define

\[
G_{i,t}(a_i)
=
\mathbb E[
\text{future unique coverage}
\mid
s_t,a_i,\mathbf a_{-i,t}
].
\]

Then define an action-sensitive FOCUS advantage

\[
A^{\mathrm{FOCUS}}_{i,t}
=
G_{i,t}(a_{i,t})
-
\mathbb E_{\tilde a_i\sim\pi_i}
G_{i,t}(\tilde a_i).
\]

A possible auxiliary policy loss is

\[
\mathcal L_{\mathrm{action\text{-}FOCUS}}
=
-\sum_i
\operatorname{sg}(A^{\mathrm{FOCUS}}_{i,t})
\log\pi_i(a_{i,t}\mid\tau_{i,t}).
\]

This extension answers a different question — whether a camera's selected action is good relative to counterfactual alternatives — and should **not** be included in the initial HMVFE integration.

---

## 22. Final method definition

The MVP is therefore:

\[
\boxed{
\begin{aligned}
&\text{HMVFE produces }Y_t,\;\ell_{i,t},\;V_t,\\
&\text{HMVFE GAE produces }A_t,\\
&\text{FOCUS produces }\rho_{i,t},\;c_t,\\
&\eta_t=\eta c_t,\\
&w_{i,t}=1+\eta_t(N_c\rho_{i,t}-1),\\
&\mathcal L_\pi
=-\mathbb E_t\left[
\operatorname{sg}(A_t)
\sum_i\operatorname{sg}(w_{i,t})\ell_{i,t}
\right].
\end{aligned}
}
\]

Everything else in HMVFE remains unchanged for the MVP.

The core research claim being tested is:

> A predictive, environment-grounded camera responsibility signal can be transferred from value-decomposition MARL to HMVFE's policy-gradient coordinator by redistributing the native team actor gradient across cameras, without modifying the underlying policy architecture or critic.
