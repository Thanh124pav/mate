# PLAN.md — FOCUS Privileged-to-Local Action Guidance

## 0. Goal

The current `QPLEX_FOCUS` uses a strong training-time privileged action prior

\[
B_i^T(s_t,a)=\texttt{focus\_action\_q\_bias}(s_t,a),
\]

and modifies action selection as

\[
Q_i'(a)=Q_i(o_{i,t},a)+\eta B_i^T(s_t,a).
\]

The problem is that decentralized execution does not have the global state \(s_t\). When the privileged bias is removed, the local policy may lose much of the training-time gain.

The new direction is therefore

\[
\boxed{
\text{privileged global-state teacher}
\rightarrow
\text{explicit local action supervision}
\rightarrow
\text{decentralized student}
}
\]

The implementation should first test whether the bottleneck is **teacher-to-student knowledge transfer**, before making the belief model more complex.

Recommended progression:

1. Audit evaluation to guarantee true decentralized execution.
2. Keep the existing `focus_action_q_bias()` as an oracle-like privileged teacher during training.
3. Train a local recurrent student to reproduce the teacher's action preference.
4. Compare hard-label cross entropy and soft-label KL / soft cross entropy.
5. Let the local student bias guide decentralized action selection.
6. Only after direct distillation works, replace the direct bias head by a structured `local history -> belief -> action bias` pipeline.
7. Optionally unify responsibility prediction and action guidance under one recurrent local belief representation.

---

# 1. Core hypothesis

The current implementation implicitly assumes:

> Better teacher-guided trajectories and TD targets will automatically cause the local Q-network to internalize the teacher's preferred actions.

This is not guaranteed.

The new hypothesis is

\[
\boxed{
\text{The main bottleneck is privileged-to-local knowledge transfer, not teacher quality.}
}
\]

We test this by making the teacher preference an explicit supervised target for a local recurrent student.

---

# 2. Target architecture

## 2.1 Privileged teacher

Keep the current teacher unchanged for the MVP:

```python
teacher_bias = focus_action_q_bias(
    q_values,
    global_state,
    focus_config,
    n_agents,
    n_actions,
)
```

Expected sequence shape:

```text
[B, T, N_agents, N_actions]
```

Interpretation:

\[
B_i^T(s_t,a)
=
\text{one-step privileged geometric action utility}.
\]

Do **not** call this true \(Q^*\).

---

## 2.2 Decentralized student

The student must depend only on information available at decentralized execution.

Use local recurrent features

\[
h_{i,t}=f_\theta(o_{i,1:t}),
\]

where `f_theta` is the existing GRU/LSTM/RNN feature extractor.

Add a local action-guidance head

\[
\hat B_i(h_{i,t})\in\mathbb R^{N_{actions}}.
\]

Conceptually:

```text
local observation history
          |
          v
      GRU / LSTM
          |
          v
         h_i
       /     \
      /       \
 Q-head      bias-head
 Q_i(a)      B_hat_i(a)
      \       /
       \     /
   decentralized action
```

At evaluation, `B_hat_i(a)` must be computed without global state.

---

# 3. Phase 0 — Audit decentralized evaluation first

Before adding new losses, make the evaluation path unambiguous.

Potential current ambiguity:

```text
deterministic=None
    ->
explore=None
    ->
config["explore"] may be True
```

## 3.1 Add explicit `decentralized_execution`

In `examples/qplex_focus/camera/agent.py`, add:

```python
decentralized_execution=True
```

When enabled, force:

```python
config["explore"] = False
config["focus"]["action_bias_eta"] = 0.0
```

and in `act()`:

```python
if self.decentralized_execution:
    deterministic = True
```

## 3.2 Safety assertions

During decentralized evaluation, log/assert:

```text
explore == False
action_bias_eta == 0.0
focus_action_q_bias() is not used for action selection
```

## 3.3 Baseline rerun

Rerun at least:

```text
QPLEX
QPLEX_FOCUS current
```

under identical conditions:

```text
same checkpoint selection rule
same environment config
same target agent
same frame skip
same seed set
same number of episodes
```

Recommended:

```text
>= 20 episodes
>= 3 seeds, ideally 5
```

This becomes the trustworthy baseline.

---

# 4. Phase 1 — Convert teacher bias into training labels

The current teacher already gives a score for every discrete action.

## 4.1 Teacher logits

```python
teacher_logits = teacher_bias.detach()
```

Never backpropagate into `focus_action_q_bias()`.

## 4.2 Hard teacher label

```python
teacher_action = teacher_logits.argmax(dim=-1)
```

so

\[
a^T_{i,t}=\arg\max_a B_i^T(s_t,a).
\]

This is the target for the hard cross-entropy ablation.

## 4.3 Soft teacher distribution — preferred

Instead of throwing away the ranking information, form

\[
p_i^T(a\mid s_t)
=
\operatorname{softmax}\left(\frac{B_i^T(s_t,a)}{\tau_T}\right).
\]

Implementation:

```python
teacher_probs = torch.softmax(
    teacher_logits / teacher_temperature,
    dim=-1,
).detach()
```

Initial config:

```python
"teacher_temperature": 1.0,
```

Suggested later sweep:

```text
0.5, 0.75, 1.0, 1.5, 2.0
```

---

# 5. Phase 2 — Add a local action-bias head

## 5.1 MVP: direct bias prediction

Do **not** start by changing the full occupancy model.

Add a small head on local recurrent features:

```python
self.focus_bias_head = nn.Sequential(
    nn.Linear(hidden_dim, hidden_dim),
    nn.ReLU(),
    nn.Linear(hidden_dim, n_actions),
)
```

Input:

```text
[B, T, N, H]
```

Output:

```text
[B, T, N, A]
```

This predicts

\[
\hat B_i(h_{i,t},a).
\]

## 5.2 Reuse the existing recurrent representation

Preferred order:

1. Reuse the existing Q-network recurrent feature if it can be exposed cleanly.
2. Refactor the RNN model to optionally return recurrent features.
3. Avoid creating a second independent LSTM unless necessary.

Desired conceptual interface:

```python
q_values, hidden_features = _unroll_mac(
    self.model,
    obs,
    return_features=True,
)
```

The same local recurrent representation may later serve:

```text
Q prediction
teacher-action distillation
future belief prediction
```

---

# 6. Phase 3 — Explicitly learn the good action

Implement hard CE and soft KL. Do not start with contrastive learning.

## 6.1 Hard cross entropy

Student logits:

```python
student_logits = self.focus_bias_head(hidden_features)
```

Loss:

\[
L_{hard}
=
-\log p_i^S(a^T_{i,t}\mid h_{i,t}).
\]

Implementation:

```python
hard_ce = F.cross_entropy(
    student_logits.reshape(-1, n_actions),
    teacher_action.reshape(-1),
    reduction="none",
)
```

Apply the valid sequence mask afterward.

Purpose:

- simplest classification baseline;
- directly tests the user's "good action as label" idea.

---

## 6.2 Soft cross entropy / KL — preferred main objective

Student distribution:

\[
p_i^S(a\mid h_{i,t})
=
\operatorname{softmax}\left(\frac{\hat B_i(h_{i,t},a)}{\tau_S}\right).
\]

Use

\[
L_{soft}
=
-\sum_a p_i^T(a\mid s_t)\log p_i^S(a\mid h_{i,t}).
\]

Implementation:

```python
student_log_probs = F.log_softmax(
    student_logits / student_temperature,
    dim=-1,
)

soft_ce = -(
    teacher_probs * student_log_probs
).sum(dim=-1)
```

Equivalent KL form:

```python
kl = F.kl_div(
    student_log_probs,
    teacher_probs,
    reduction="none",
).sum(dim=-1)
```

Initial config:

```python
"action_distill_mode": "soft_kl",
"teacher_temperature": 1.0,
"student_temperature": 1.0,
```

Why prefer this over hard CE:

- keeps the full action ranking;
- nearby good actions are not treated the same as obviously bad actions;
- better matches what `focus_action_q_bias()` already computes.

---

# 7. Phase 4 — Teacher confidence

The teacher is only a geometric one-step oracle surrogate, not true \(Q^*\).

Do not trust it equally everywhere.

## 7.1 Entropy confidence

Teacher entropy:

\[
H_T=-\sum_a p_T(a)\log p_T(a).
\]

Normalized entropy:

\[
\bar H_T=\frac{H_T}{\log A}.
\]

Confidence:

\[
c_t=1-\bar H_T.
\]

Implementation:

```python
teacher_entropy = -(
    teacher_probs
    * torch.log(teacher_probs + eps)
).sum(dim=-1)

teacher_confidence = (
    1.0 - teacher_entropy / math.log(n_actions)
).clamp(0.0, 1.0).detach()
```

Weighted action loss:

```python
action_loss = (
    teacher_confidence * distill_per_step * valid_mask
).sum() / (
    (teacher_confidence * valid_mask).sum() + eps
)
```

## 7.2 Optional top-1/top-2 gap confidence

Later ablation:

\[
c_t
=
\sigma\left(
\frac{B_{(1)}-B_{(2)}}{\tau_{gap}}
\right).
\]

Do not implement before entropy confidence is working.

---

# 8. Phase 5 — Total loss

Current QPLEX_FOCUS approximately uses

\[
L=L_{TD}+\beta L_{belief}.
\]

Add explicit action knowledge transfer:

\[
\boxed{
L
=
L_{TD}
+
\beta L_{belief}
+
\lambda_{act}L_{distill}
}
\]

Recommended config:

```python
"action_distill_enabled": True,
"action_distill_coeff": 0.05,
"action_distill_mode": "soft_kl",
"teacher_temperature": 1.0,
"student_temperature": 1.0,
"action_distill_confidence": "entropy",
```

Initial coefficient sweep after smoke test:

```text
0.01, 0.03, 0.05, 0.1
```

Do not jointly tune all FOCUS parameters at this stage.

---

# 9. Phase 6 — Teacher scheduling

Do not keep strong privileged supervision forever by default.

Use

\[
\lambda_{act}(t):\lambda_0\rightarrow\lambda_{final}.
\]

Suggested first schedule:

```text
lambda_0     = 0.05
lambda_final = 0.005
```

Interpretation:

```text
early training:
teacher transfer strong

late training:
RL objective dominates

evaluation:
teacher absent
```

This allows the student to eventually deviate from an imperfect one-step teacher.

---

# 10. Phase 7 — How the teacher should affect behavior

Test these variants separately.

## Variant A — Distillation only

Behavior policy:

\[
a=\epsilon\text{-greedy}(Q).
\]

Teacher is only a supervised target.

Purpose:

```text
pure knowledge-transfer test
```

## Variant B — Teacher collection + distillation

Training behavior:

\[
Q'(a)=Q(a)+\eta B^T(s,a).
\]

Also train

\[
L_{distill}.
\]

This combines:

```text
better trajectories
+
explicit transfer
```

This is the strongest practical MVP candidate.

## Variant C — Student-guided decentralized collection

Use local predicted bias:

\[
Q'(a)=Q(a)+\eta_S\hat B(h,a).
\]

This is decentralized because \(\hat B\) depends only on local history.

Recommended later training handoff:

\[
B^{mix}
=
\alpha_t B^T
+
(1-\alpha_t)\hat B,
\qquad
\alpha_t:1\rightarrow0.
\]

Conceptually:

```text
early:  centralized teacher
middle: teacher + student
late:   decentralized student
```

This explicitly reduces train/eval mismatch.

---

# 11. Phase 8 — Separate privileged data collection from privileged bootstrap

The current method also uses teacher bias during Double-Q bootstrap action selection.

This can blur the causal mechanism.

Add a clean experiment with:

```python
"action_bias_bootstrap": False,
```

Recommended comparison:

```text
A. No teacher
B. Teacher collection only
C. Teacher bootstrap only
D. Teacher collection + bootstrap
E. Distillation only
F. Teacher collection + distillation
G. Teacher collection + bootstrap + distillation
```

Do not assume G is best.

A particularly clean method candidate is:

```text
teacher improves data collection
+
explicit action distillation
+
normal Q-learning bootstrap
```

because the Bellman operator remains standard.

---

# 12. Phase 9 — Metrics for teacher internalization

Add the following metrics.

## 12.1 Teacher-student top-1 agreement

\[
\text{Agreement}_{B}
=
P\left[
\arg\max_a B^T(a)
=
\arg\max_a\hat B(a)
\right].
\]

Log:

```text
focus_teacher_student_top1_agreement
```

## 12.2 Teacher-Q agreement

\[
\text{Agreement}_{Q}
=
P\left[
\arg\max_a B^T(a)
=
\arg\max_a Q(o,a)
\right].
\]

Log:

```text
focus_teacher_q_top1_agreement
```

This directly tests whether standard Q-learning internalizes the teacher.

## 12.3 Teacher-student KL

```text
focus_teacher_student_kl
```

## 12.4 Teacher entropy / confidence

```text
focus_teacher_entropy
focus_teacher_confidence
```

## 12.5 Student bias scale

```text
focus_student_bias_abs_mean
focus_student_bias_std
```

---

# 13. Phase 10 — Evaluation modes

Every checkpoint should be evaluated in three distinct modes.

## Mode 1 — Teacher-on diagnostic

\[
Q+\eta B^T.
\]

Not valid decentralized execution.

Purpose:

```text
privileged teacher ceiling
```

## Mode 2 — Q-only decentralized

\[
Q(o,a).
\]

Purpose:

```text
did ordinary Q internalize the teacher?
```

## Mode 3 — Q + local student bias decentralized

\[
Q(o,a)+\eta_S\hat B(h,a).
\]

Purpose:

```text
can the distilled local surrogate preserve the gain?
```

Mode 3 is the main deployment setting for the new method.

---

# 14. Phase 11 — Structured belief-based action guidance

Only implement this after direct bias distillation gives a positive result.

The direct student tests

\[
h_i\rightarrow\hat B_i(a).
\]

The structured version should instead use

\[
h_i
\rightarrow
\text{future target belief}
\rightarrow
\hat B_i(a).
\]

## 14.1 Local recurrent belief

Desired model:

\[
h_{i,t}=\mathrm{LSTM}(o_{i,1:t}).
\]

Then predict future target occupancy.

Gaussian option:

\[
h_{i,t}
\rightarrow
(\mu_{i,j,t+h},\sigma_{i,j,t+h}).
\]

Grid option later:

\[
h_{i,t}
\rightarrow
P_{i,j,t+h}(x).
\]

Important:

The execution-time belief predictor must use only local history.

## 14.2 Predictive action utility

For each candidate action:

\[
\hat B_i(a)
=
\sum_{h=1}^{H}\gamma_h
\sum_j
\mathbb E_{x\sim p_\phi(x_{j,t+h}\mid h_{i,t})}
[V_i(x;a)].
\]

This is the predictive version of the current reactive global-state action bias.

Start with independent per-camera coverage.

A unique-coverage extension can be added later:

\[
\hat B_i^{unique}(a_i)
=
\sum_h\gamma_h\sum_j
\mathbb E\left[
V_i(x;a_i)
\prod_{k\neq i}(1-V_k(x))
\right].
\]

Do not start with this more complex form.

---

# 15. Phase 12 — Unified FOCUS representation

Final target architecture:

```text
                    local history
                         |
                         v
                      LSTM/GRU
                         |
                         v
                        h_i
              __________/|\__________
             /           |           \
            /            |            \
        Q-head       belief-head    action-head
          |              |              |
       Q_i(a)       future belief    B_hat_i(a)
                         |
                         v
               predictive responsibility
                         |
                         v
                       rho_i
```

Potential total objective:

\[
L
=
L_{TD}
+
\beta_{belief}L_{belief}
+
\lambda_{act}L_{action}
\]

with optional responsibility supervision if retained.

The final method story is:

\[
\boxed{
\text{local history}
\rightarrow
\text{predictive belief}
\rightarrow
\begin{cases}
\text{credit responsibility}\\
\text{decentralized action guidance}
\end{cases}
}
\]

---

# 16. Contrastive learning — optional, not MVP

Do not implement contrastive learning before CE/KL distillation is tested.

Possible formulation:

\[
z_{i,t}=f(o_{i,1:t})
\]

with learnable action embeddings \(e_a\).

Use teacher-best action as the positive and lower-ranked actions as negatives:

\[
L_{NCE}
=
-\log
\frac{
\exp(\operatorname{sim}(z,e_{a^+})/\tau)
}{
\sum_a\exp(\operatorname{sim}(z,e_a)/\tau)
}.
\]

Potential motivation:

- richer representation learning;
- action-geometry structure;
- possible transfer across environments/action grids.

For the current small discrete action space, soft CE/KL is simpler and more directly aligned with the goal.

Use contrastive learning as an ablation only after the main transfer mechanism works.

---

# 17. Code changes

## 17.1 `ray/rllib/agents/focus_utils.py`

Keep:

```python
focus_action_q_bias(...)
```

Add:

```python
def focus_teacher_distribution(
    action_bias,
    temperature=1.0,
    eps=1e-8,
):
    ...
```

Optional:

```python
def focus_teacher_confidence(
    teacher_probs,
    mode="entropy",
):
    ...
```

---

## 17.2 `ray/rllib/agents/qplex_focus/qplex_policy.py`

Add:

```text
- local recurrent feature extraction
- local student action-bias head
- teacher target generation
- hard CE loss
- soft KL / soft CE loss
- confidence weighting
- teacher/student agreement metrics
- optional teacher-to-student schedule
```

Do not change the existing FOCUS responsibility mechanism in the first patch.

---

## 17.3 Q-network / `RNNModel`

If required, expose recurrent features safely.

Possible interface:

```python
def forward_with_features(...):
    ...
```

or cache the feature from the normal recurrent forward pass.

Avoid breaking RLlib's standard model API.

---

## 17.4 `examples/qplex_focus/camera/config.py`

Add backward-compatible defaults:

```python
"action_distill_enabled": False,
"action_distill_mode": "soft_kl",
"action_distill_coeff": 0.05,
"teacher_temperature": 1.0,
"student_temperature": 1.0,
"action_distill_confidence": "entropy",
"student_action_bias_enabled": False,
"student_action_bias_eta": 1.0,
"teacher_student_mix_enabled": False,
"teacher_student_mix_start": 1.0,
"teacher_student_mix_end": 0.0,
```

---

## 17.5 `examples/qplex_focus/camera/agent.py`

Add robust:

```python
decentralized_execution=True
```

and hard-disable centralized action guidance in that mode.

---

# 18. Backward compatibility

With

```python
"action_distill_enabled": False,
"student_action_bias_enabled": False,
```

the implementation should reproduce current `QPLEX_FOCUS` behavior.

Regression checks on a fixed batch:

```text
same TD loss
same rho
same belief loss
same teacher bias
same greedy actions
```

Use `torch.allclose(..., atol=1e-6, rtol=1e-6)` where appropriate.

---

# 19. Unit tests

## Teacher distribution

- probabilities sum to 1;
- no NaN/Inf;
- low temperature produces sharper distribution;
- high temperature produces flatter distribution.

## Distillation

- exact teacher/student match -> KL approximately 0;
- uniform student vs sharp teacher -> positive KL;
- padded timesteps do not contribute;
- unavailable actions are masked correctly.

## Confidence

- uniform teacher -> low confidence;
- sharp teacher -> high confidence.

## Decentralized safety

When `decentralized_execution=True`, assert:

```text
no global state is used for action selection
teacher bias is not added
focus_action_q_bias() is not called by the deployed action path
```

---

# 20. Experiment ladder

## Experiment A — Evaluation audit

```text
A1 teacher ON
A2 teacher OFF, explore=False
A3 teacher OFF, deterministic=True
A4 teacher OFF, eta=0, explore=False, deterministic=True
```

A2-A4 should match closely.

If not, evaluation still has a bug or hidden dependency.

---

## Experiment B — Does explicit good-action learning help?

```text
B0 current QPLEX_FOCUS
B1 hard CE
B2 soft KL
B3 soft KL + entropy confidence
```

Evaluate all with teacher OFF.

Primary metric:

```text
decentralized mean coverage
```

Secondary metrics:

```text
teacher-student agreement
teacher-Q agreement
teacher-student KL
```

---

## Experiment C — Good data vs explicit transfer

```text
C0 no teacher collection, no distillation
C1 teacher collection only
C2 distillation only
C3 teacher collection + distillation
```

This is the most important causal ablation.

---

## Experiment D — Privileged bootstrap

```text
D0 action_bias_bootstrap=True
D1 action_bias_bootstrap=False
```

Both with explicit action distillation.

This tells whether the gain comes from privileged Bellman action selection or from data + transfer.

---

## Experiment E — Local student bias at execution

```text
E0 Q-only
E1 Q + B_hat
```

Both must be fully decentralized.

If E1 >> E0, ordinary Q did not fully internalize the teacher, but the local auxiliary head did.

---

## Experiment F — Teacher-to-student handoff

Train with

\[
B^{mix}=\alpha_tB^T+(1-\alpha_t)\hat B.
\]

Compare:

```text
constant teacher
linear handoff
cosine handoff
student-only late phase
```

Goal:

```text
reduce train/eval mismatch
```

---

## Experiment G — Structured belief action guidance

Only after direct-B works:

```text
G0 direct local B_hat head
G1 local LSTM -> Gaussian future belief -> B_hat
G2 local LSTM -> grid belief -> B_hat
```

Keep the same decentralized evaluation protocol.

---

# 21. Success criteria

The direction is successful if:

1. Decentralized evaluation is reproducible and explicitly teacher-free.
2. Soft-KL distillation reduces teacher-student KL.
3. Teacher-student top-1 agreement rises during training.
4. Decentralized coverage improves over current teacher-off `QPLEX_FOCUS`.
5. `Q + local student bias` retains a meaningful fraction of the teacher-on gain.
6. The structured belief-based version approaches the direct-B student.

A strong pattern would be:

```text
teacher ON:                  ~60%+
teacher OFF, Q only:         ~40%+
teacher OFF, Q + local bias: ~55-60%
```

This would demonstrate explicit privileged-to-local transfer.

---

# 22. Failure interpretation

## Case 1 — Student matches teacher but eval is still poor

```text
teacher KL decreases
agreement increases
eval remains low
```

Interpretation:

The one-step geometric teacher is not sufficiently aligned with long-term return.

Next steps:

```text
advantage-weighted distillation
short-horizon teacher
predictive teacher
```

---

## Case 2 — Student cannot match teacher

```text
teacher KL remains high
```

Interpretation:

The local recurrent representation does not contain enough information to reconstruct the privileged teacher decision.

Next steps:

```text
larger GRU/LSTM
longer temporal context
explicit local target-belief head
agent-specific encoder/adapters
```

---

## Case 3 — Direct-B works but belief->B fails

Interpretation:

The belief model is the bottleneck.

Next:

```text
better recurrent belief architecture
uncertainty calibration
multimodal belief
grid belief
longer horizon
```

---

## Case 4 — Local student bias works but Q-only does not

Interpretation:

Q-learning alone does not reliably internalize privileged action preferences.

The auxiliary local action-guidance head should remain part of the final method.

---

# 23. Recommended implementation order

## Milestone 1 — Evaluation correctness

```text
1. Add decentralized_execution flag.
2. Force explore=False.
3. Force action_bias_eta=0.
4. Force deterministic=True.
5. Rerun the current checkpoint.
```

Do this first.

## Milestone 2 — Direct teacher distillation

```text
1. Expose local recurrent features.
2. Add local action-bias head.
3. Generate teacher scores from global state.
4. Implement hard CE.
5. Implement soft KL.
6. Add entropy confidence.
7. Log teacher-student agreement.
8. Evaluate with teacher fully OFF.
```

This is the main MVP.

## Milestone 3 — Student-guided decentralized action selection

Use

\[
Q+\eta_S\hat B.
\]

with no global state.

## Milestone 4 — Teacher-to-student handoff

Anneal

```text
teacher bias -> student bias
```

during training.

## Milestone 5 — Structured belief-based action guidance

Replace

```text
h -> B_hat
```

with

```text
h -> future belief -> B_hat
```

and reuse the predictive belief for FOCUS responsibility.

## Milestone 6 — Optional contrastive learning

Only after soft CE/KL is established.

---

# 24. Recommended first experiment config

Use the smallest change that directly tests the hypothesis:

```python
"focus": {
    # existing FOCUS
    "enabled": True,
    "belief_mode": "learned",

    # privileged teacher used during collection
    "action_bias_eta": 3.0,
    "action_bias_bootstrap": False,

    # explicit transfer
    "action_distill_enabled": True,
    "action_distill_mode": "soft_kl",
    "action_distill_coeff": 0.05,
    "teacher_temperature": 1.0,
    "student_temperature": 1.0,
    "action_distill_confidence": "entropy",

    # decentralized student guidance
    "student_action_bias_enabled": True,
    "student_action_bias_eta": 1.0,
}
```

Train with:

```text
teacher collection ON
teacher bootstrap OFF
soft-KL distillation ON
```

Evaluate with:

```text
teacher OFF
student local bias ON
```

This directly tests:

\[
\boxed{
\text{Can privileged geometric action knowledge be explicitly transferred into a decentralized recurrent student?}
}
\]

---

# 25. Final target formulation

The long-term FOCUS method should become

\[
\boxed{
\text{Local history}
\rightarrow
\text{predictive belief}
\rightarrow
\begin{cases}
\text{responsibility for credit assignment}\\
\text{action utility for decentralized guidance}
\end{cases}
}
\]

Training may use global state as privileged supervision.

Execution uses only

\[
\boxed{
o_{i,1:t}
\rightarrow
h_{i,t}
\rightarrow
Q_i(a),\hat B_i(a)
\rightarrow
a_i
}
\]

with no global state.

This produces a cleaner CTDE story than retaining `focus_action_q_bias()` only as a training-time heuristic and hoping the local policy internalizes it implicitly.
