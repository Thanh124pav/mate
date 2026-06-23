# Implementation Differences Between `FOCUS_QMC_Methodology_FIXED.md` and Current Code

This note summarizes the main implementation differences between the methodology described in
`FOCUS_QMC_Methodology_FIXED.md` and the current FOCUS code in:

- `ray/rllib/agents/qplex_focus/qplex_policy.py`
- `ray/rllib/agents/qplex_focus/mixers.py`
- `ray/rllib/agents/qmix_focus/qmix_policy.py`
- `ray/rllib/agents/duelmix_focus/duelmix_policy.py`
- `examples/mappo/models.py`

The short version is: the code follows the same high-level FOCUS idea, but it is not a faithful
implementation of the exact QMC-discrete-belief method described in the methodology file.

---

## 1. Shared Core Idea

Both the methodology file and the code use the same motivation:

Sparse and occluded visibility makes realized camera credit noisy. A camera may have useful future
coverage potential, but the immediate binary tracking signal is often zero. This causes low-SNR
credit assignment from team-level TD error.

Both versions therefore build a predictive credit target:

```math
g_i
=
\sum_j
\int
B^H_{j,t}(x)
v_i(x,a_i)
\prod_{k\ne i}(1-v_k(x,a_k))
dx
```

and normalize it into a responsibility target:

```math
\rho_i \approx \frac{g_i}{\sum_k g_k}.
```

This target is then used to regularize the credit mechanism of a value-decomposition method.

---

## 2. Belief Model Output

### Methodology File

The methodology describes a belief model that outputs logits over a fixed set of QMC map points:

```math
\ell_{j,h,m}=s_{\phi,j}^{h}(x_m\mid \mathcal H_t)
```

```math
\pi_{j,h,m}
=
\frac{\exp(\ell_{j,h,m})}
{\sum_r \exp(\ell_{j,h,r})}.
```

So the future occupancy belief is a categorical distribution over QMC support points:

```math
\pi \in [B,T,H,J,M].
```

### Current Code

The code uses a continuous Gaussian belief model instead:

```math
b_\phi(x_{j,t+h}\mid s_t)
=
\mathcal N(
\mu_{\phi,h,j}(s_t),
\operatorname{diag}(\sigma^2_{\phi,h,j}(s_t))
).
```

The model outputs four values per target and horizon:

```math
(\Delta x,\Delta y,\sigma_x,\sigma_y).
```

In code:

```python
out = self.net(state.reshape(-1, self.state_dim)).view(
    B, T, self.horizon, self.n_targets, 4
)
delta = torch.tanh(out[..., :2]) * self.max_delta
std = F.softplus(out[..., 2:]) + self.min_std
mean = current_pos.unsqueeze(2) + delta
```

### Difference

The methodology uses a discrete QMC belief distribution. The code uses a parametric continuous
Gaussian belief.

This is the largest difference between the document and implementation.

---

## 3. Belief Loss

### Methodology File

The methodology uses a Gaussian soft label over QMC points:

```math
q_{j,h,m}
=
\frac{
\exp\left(-\frac{\|x_m-x^*_{j,t+h}\|^2}{2\sigma_b^2}\right)
}{
\sum_r
\exp\left(-\frac{\|x_r-x^*_{j,t+h}\|^2}{2\sigma_b^2}\right)
}.
```

The belief loss is cross entropy:

```math
\mathcal L_{\mathrm{belief}}
=
-
\frac{1}{BTJH}
\sum_{b,t,j,h}
\sum_m
q_{b,t,j,h,m}
\log \pi_{b,t,j,h,m}.
```

### Current Code

The code trains the Gaussian belief with negative log-likelihood:

```math
\mathcal L_{\mathrm{belief}}
=
\sum_{h}
\omega_h
\mathbb E_{t,j}
\left[
\sum_{d\in\{x,y\}}
\frac12
\left(
\frac{x^d_{j,t+h}-\mu^d_{\phi,h,j}}
{\sigma^d_{\phi,h,j}}
\right)^2
+
\log \sigma^d_{\phi,h,j}
+
\frac12\log(2\pi)
\right].
```

In code:

```python
z = (target - pred_mean) / pred_std
nll = 0.5 * (z ** 2) + torch.log(pred_std) + 0.5 * np.log(2.0 * np.pi)
per_step = nll.sum(dim=-1).mean(dim=-1)
```

### Difference

The methodology trains a categorical distribution over QMC points with cross entropy.
The code trains a continuous Gaussian distribution with NLL.

---

## 4. QMC / Monte Carlo Integral

### Methodology File

The methodology samples fixed low-discrepancy QMC points in map space:

```math
u_m\in[0,1]^2,\qquad x_m=T(u_m)\in\Omega.
```

Then it evaluates occupancy mass and visibility at these support points:

```math
\widehat g_i
=
\sum_j
\sum_m
\Pi^H_{j,m}
U_{i,m}.
```

where:

```math
U_{i,m}
=
V_{i,m}
\prod_{k\ne i}(1-V_{k,m}).
```

### Current Code

The default `integral_mode="MC"` uses Sobol points as quasi-random normal samples, not fixed map
points:

```python
uniforms = engine.draw(num_points)
normal = sqrt(2) * erfinv(2 * uniforms - 1)
samples = mean_h.unsqueeze(4) + std_h.unsqueeze(4) * normal_chunk
```

Thus the code samples:

```math
x_m = \mu_{\phi,h,j} + \sigma_{\phi,h,j} z_m,
\qquad z_m\sim \mathcal N(0,I)
```

and estimates:

```math
g_i
\approx
\sum_h \omega_h
\frac{1}{M}
\sum_{m=1}^M
\sum_j
U_i(x_{h,j,m}).
```

The code also supports `integral_mode="grid"` and `integral_mode="sigma"`, but the default config
uses `MC`.

### Difference

The methodology performs QMC integration over fixed map support points. The code performs QMC-style
sampling from the predicted Gaussian belief distribution.

This means the code is still quasi-Monte Carlo in spirit, but not the exact QMC estimator described
in the methodology.

---

## 5. Responsibility Factorization in QPLEX

### Methodology File

The methodology explicitly factorizes the QPLEX credit coefficient:

```math
\lambda_i = p_i m_i
```

where:

```math
p_i = \operatorname{softmax}(z)_i,
\qquad
m_i = 1+\tanh(r_i).
```

The FOCUS loss regularizes only the responsibility factor `p_i`, not the magnitude gate `m_i`:

```math
\mathcal L_{\mathrm{FOCUS}}
=
D_{\mathrm{KL}}(\operatorname{sg}(\rho)\|p).
```

### Current Code

The code keeps the existing QPLEX-style `LamdaWeight` module and obtains:

```math
\lambda_i = \mathrm{LamdaWeight}(s,\mathbf a)_i.
```

It does not explicitly implement:

```math
\lambda_i=p_i m_i.
```

Instead, it adds a `credit_prior(state)` method that returns a softmax-like prior from the
`LamdaWeight` internals:

```python
prior = F.softmax(agent_ext(states) / scale_factor, dim=-1)
prior = (priors * keys).sum(dim=1) / (keys.sum(dim=1) + 1e-10)
```

During QPLEX-FOCUS training, the code regularizes this prior if available:

```python
if hasattr(self.mixer, "credit_prior"):
    p_dist = self.mixer.credit_prior(state)
else:
    p_dist = lambda_dist
```

### Difference

The methodology has an explicit and clean separation:

```math
p_i = responsibility,
\qquad
m_i = magnitude.
```

The code approximates this by extracting a prior from the existing lambda network. This is close in
intent, but not identical to the methodology.

---

## 6. FOCUS Loss Form

### Methodology File

The methodology uses:

```math
\mathcal L_{\mathrm{FOCUS}}
=
D_{\mathrm{KL}}(\operatorname{sg}(\rho)\|p)
=
\sum_i
\operatorname{sg}(\rho_i)
\log
\frac{\operatorname{sg}(\rho_i)}
{p_i+\epsilon}.
```

Since the entropy of `rho` is constant with respect to `p`, this is equivalent for optimization to:

```math
-
\sum_i
\operatorname{sg}(\rho_i)\log(p_i+\epsilon).
```

### Current Code

The code uses the cross-entropy term:

```python
per_step_ce = -(rho * torch.log(p_dist + eps)).sum(dim=-1)
```

and then applies signal-confidence weighting:

```python
focus_loss = weighted_average(per_step_ce, signal_weights)
```

### Difference

The optimization target is effectively the same as the KL objective, except that the code drops the
constant entropy term and adds confidence weighting based on the total credit signal.

The confidence weighting is not part of the methodology file.

---

## 7. Normalization of Predictive Credit

### Methodology File

The methodology normalizes with epsilon added to every agent credit:

```math
\rho_i
=
\frac{g_i+\epsilon}
{\sum_\ell(g_\ell+\epsilon)}.
```

### Current Code

The code uses:

```python
rho = g / (total_g.unsqueeze(-1) + eps)
rho = torch.where(valid.unsqueeze(-1), rho, uniform)
```

where:

```python
valid = (total_g > min_credit_signal) & mask
```

### Difference

The methodology always smooths each agent by adding epsilon to `g_i`. The code instead divides by
the total plus epsilon and falls back to a uniform distribution when the credit signal is too small.

The code's fallback is more practical for numerical stability but is not exactly the same formula.

---

## 8. Target Weight `w_j`

### Methodology File

The predictive credit includes target weights:

```math
g_i
=
\sum_j
w_j
\int
B^H_{j,t}(x)
U_i(x)
dx.
```

### Current Code

The code sums over targets uniformly. There is no explicit configurable `w_j` in the implemented
credit computation.

### Difference

The methodology allows non-uniform target importance. The current implementation assumes all targets
have equal weight.

---

## 9. Action-Selection Masking

### Methodology File

The methodology writes the visibility kernel as:

```math
v_i(x,a_i).
```

This is general and assumes the action influences visibility.

### Current Code

The code has a specific action decoding path for multi-target selection:

```python
if self.n_actions == 2 ** n_targets:
    selection = decode_bits(actions)
    visibility = visibility * selection
```

If the action space does not match `2 ** n_targets`, this action-selection factor is disabled.

### Difference

The methodology is action-general. The code has a concrete implementation for bit-coded target
selection actions.

---

## 10. Support for QMIX, DQN, DuelMIX, and MAPPO

### Methodology File

The methodology is written mainly for a QPLEX-style mixer with responsibility factor `p_i`.

### Current Code

The implementation extends the same FOCUS helper to several algorithms:

- QPLEX-FOCUS: regularizes `credit_prior(state)` or normalized lambda weights.
- DuelMIX-FOCUS: regularizes normalized lambda weights.
- QMIX-FOCUS: regularizes implicit mixer sensitivities from `W_1(s)W_f(s)`.
- DQN-FOCUS: no mixer, so it reweights per-agent TD error by `rho`.
- MAPPO: only adds the belief-model auxiliary loss in `custom_loss`; it does not implement the
  FOCUS credit regularizer.

### Difference

The code is broader than the methodology file, but some extensions are heuristic because algorithms
like DQN and MAPPO do not naturally expose a QPLEX-style responsibility factor.

---

## 11. Why the Code Differs

The current code appears to be an implementation adapted from existing QPLEX/QMIX/DuelMIX code,
rather than a fresh implementation of the exact QMC methodology.

The likely reasons are:

1. **Lower implementation cost.** A Gaussian belief model is easier to add than a full
   logits-over-QMC-points model.
2. **Lower memory cost.** The methodology's belief tensor has shape `[B,T,H,J,M]`; the code stores
   only `[B,T,H,J,4]` for Gaussian parameters.
3. **Compatibility with existing mixers.** The existing QPLEX lambda module was reused instead of
   replacing it with an explicit `lambda_i=p_i m_i` factorization.
4. **Algorithm portability.** The code tries to reuse FOCUS for QMIX, DQN, DuelMIX, and MAPPO, even
   though the methodology is primarily QPLEX-style.
5. **Practical stability.** The code adds validity masks, uniform fallback, and signal-confidence
   weighting that are not described in the methodology.

---

## 12. Practical Consequences

These differences can affect results.

The Gaussian belief model may work well when target future locations are unimodal and smooth, but it
can be weaker when future occupancy is multi-modal. A QMC categorical support model can represent
multi-modal occupancy more naturally.

The current QMC-from-Gaussian estimator samples around the predicted mean. If the predicted mean or
variance is poor early in training, the credit target can be biased. The methodology's fixed map
support avoids this particular sampling dependence, but requires a larger output tensor.

The missing explicit `lambda_i=p_i m_i` factorization means the code does not fully separate
"who should get credit" from "how strongly credit should affect the mixer." This can make the FOCUS
regularizer less aligned with the methodology.

Finally, the current implementation may underperform vanilla QPLEX in environments where the
belief-based proxy is biased or where QPLEX's original lambda mechanism already receives a strong
TD signal.

---

## 13. What Would Be Needed to Match the Methodology Exactly

To make the code match `FOCUS_QMC_Methodology_FIXED.md`, the implementation would need to:

1. Replace the Gaussian belief head with a logits-over-QMC-points head:

   ```math
   f_\phi(s_t)\rightarrow \ell_{j,h,m}.
   ```

2. Replace Gaussian NLL with soft-label cross entropy over QMC points.

3. Use fixed QMC map support points:

   ```math
   x_m=T(u_m)\in\Omega.
   ```

4. Compute:

   ```math
   \Pi^H_{j,m}=\sum_h \omega_h\pi_{j,h,m}.
   ```

5. Estimate:

   ```math
   \hat g_i=\sum_j w_j\sum_m\Pi^H_{j,m}U_{i,m}.
   ```

6. Modify the QPLEX mixer to explicitly output:

   ```math
   p_i=\operatorname{softmax}(z)_i,\qquad
   m_i=1+\tanh(r_i),\qquad
   \lambda_i=p_i m_i.
   ```

7. Regularize `p_i` directly with:

   ```math
   D_{\mathrm{KL}}(\operatorname{sg}(\rho)\|p).
   ```

8. Normalize credit exactly as:

   ```math
   \rho_i=\frac{g_i+\epsilon}{\sum_\ell(g_\ell+\epsilon)}.
   ```

After these changes, the code would be much closer to the methodology file.
