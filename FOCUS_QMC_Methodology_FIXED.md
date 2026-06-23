# Methodology

This section presents **FOCUS** (*Future Occupancy-guided Credit Under Sparse visibility*), a plug-in predictive credit-assignment module for multi-camera tracking. FOCUS is designed for sparse and occluded environments where binary tracking events provide weak and noisy credit signals. The central idea is to replace sparse realized credit with a lower-variance predictive credit signal computed from target future occupancy belief. The resulting credit signal is estimated with Quasi-Monte Carlo and used to regularize the responsibility factor in a QPLEX-style mixer.

---

## 1. Motivation

### 1.1 Sparse visibility and noisy credit assignment

In multi-camera tracking, each camera observes only a limited field of view. A target is informative to a camera only when it falls inside the camera's visible region and is not fully occluded by obstacles. Let

$$
Z_{ij,t}\in\{0,1\}
$$

denote whether camera $i$ observes target $j$ at time $t$. In sparse or obstacle-rich environments, the visibility rate

$$
p_{\mathrm{vis}}
=
\frac{1}{T N_C N_T}
\sum_{t=1}^{T}
\sum_{i=1}^{N_C}
\sum_{j=1}^{N_T}
Z_{ij,t}
$$

can be small. This means that most camera-target pairs produce no direct tracking signal at most timesteps.

For value decomposition methods, the learning signal usually comes from a team-level TD error. The global value is decomposed into agent-wise utilities or advantages, but the environment does not directly tell which camera deserves credit for a successful tracking event. This is especially problematic when tracking events are intermittent. A camera may be moving toward a target that is currently hidden behind an obstacle, but the immediate binary tracking signal is still zero. Conversely, multiple cameras may track the same target, creating redundant coverage and ambiguous credit.

Therefore, the credit assignment problem is not only sparse-reward learning. It is also a **visibility-induced identifiability problem**: the team reward is observed, but the useful marginal contribution of each camera is difficult to infer from realized binary events alone.

### 1.2 From binary visibility to predictive credit

FOCUS addresses this issue by replacing realized binary credit with a predictive, belief-based credit signal.

Let $X_{j,t}\in\Omega$ be the position of target $j$ on the map $\Omega$. Given the history $\mathcal H_t$, a predictive belief model estimates the future occupancy density of target $j$:

$$
b_{\phi,j}^{h}(x)
=
P_{\phi}(X_{j,t+h}=x\mid \mathcal H_t),
\qquad h=1,\dots,H.
$$

We aggregate the multi-step belief into a future occupancy belief:

$$
B_{j,t}^{H}(x)
=
\sum_{h=1}^{H}
\omega_h
b_{\phi,j}^{h}(x),
\qquad
\sum_{h=1}^{H}\omega_h=1.
$$

Here $H$ is the prediction horizon, and $\omega_h$ controls the importance of each future step.

Let

$$
v_i(x,a_i)\in[0,1]
$$

be the visibility kernel of camera $i$ at location $x$ under action $a_i$. This kernel can encode field-of-view geometry, range, angle, and obstacle transmittance. FOCUS defines the predictive marginal contribution of camera $i$ as

$$
g_i
=
\sum_{j=1}^{N_T}
w_j
\int_{\Omega}
B_{j,t}^{H}(x)
v_i(x,a_i)
\prod_{k\neq i}
\left(1-v_k(x,a_k)\right)
dx.
$$

The term

$$
v_i(x,a_i)
\prod_{k\neq i}
(1-v_k(x,a_k))
$$

measures the **unique visibility** of camera $i$: camera $i$ receives credit for covering future target mass that is not already covered by other cameras. Thus, $g_i$ is a predictive counterfactual marginal credit signal.

We normalize the predictive credits into a responsibility distribution:

$$
\rho_i
=
\frac{g_i+\epsilon}
{\sum_{\ell=1}^{N_C}(g_\ell+\epsilon)}.
$$

This distribution tells the value decomposition module which camera should receive relatively more credit according to future target occupancy and camera geometry.

---

## 2. SNR and Sparse Visibility Analysis

### 2.1 Realized binary credit under sparse visibility

Let $Y_i$ denote the realized marginal credit of camera $i$. For example,

$$
Y_i
=
C(\mathbf a)-C(\mathbf a_{-i}),
$$

where $C(\mathbf a)$ is the realized team coverage under joint action $\mathbf a$, and $C(\mathbf a_{-i})$ is the counterfactual coverage when camera $i$'s contribution is removed.

In sparse visibility settings, $Y_i$ is often close to a Bernoulli variable:

$$
Y_i\in\{0,1\},
\qquad
P(Y_i=1)=\alpha_i.
$$

Here $\alpha_i$ is the probability that camera $i$ has a nonzero useful marginal contribution. If visibility is sparse, then $\alpha_i$ is small.

Given $n$ samples, the empirical realized credit is

$$
\widehat{\mu}_i
=
\frac{1}{n}
\sum_{r=1}^{n}Y_i^{(r)}.
$$

Its expectation and variance are

$$
\mathbb E[\widehat{\mu}_i]=\alpha_i,
$$

$$
\operatorname{Var}(\widehat{\mu}_i)
=
\frac{\alpha_i(1-\alpha_i)}{n}.
$$

The relative standard error is

$$
\operatorname{RSE}(Y_i)
=
\frac{
\sqrt{\operatorname{Var}(\widehat{\mu}_i)}
}{
\mathbb E[\widehat{\mu}_i]+\epsilon
}
\approx
\sqrt{
\frac{1-\alpha_i}{n\alpha_i}
}.
$$

As $\alpha_i\to 0$,

$$
\operatorname{RSE}(Y_i)\to\infty.
$$

Therefore, even if the absolute Bernoulli variance may become small when events are extremely rare, the **signal-to-noise ratio** of the credit estimator becomes poor. Sparse visibility makes realized marginal credit statistically unreliable.

### 2.2 SNR view of credit learning

Let the useful credit signal be

$$
\mu_i=\mathbb E[Y_i],
$$

and let the variance of the estimator be $\operatorname{Var}(\widehat{\mu}_i)$. A simple credit signal-to-noise ratio can be written as

$$
\operatorname{SNR}_i
=
\frac{\mu_i^2}
{\operatorname{Var}(\widehat{\mu}_i)+\epsilon}.
$$

For Bernoulli realized credit,

$$
\operatorname{SNR}_i
=
\frac{\alpha_i^2}
{\alpha_i(1-\alpha_i)/n+\epsilon}.
$$

Ignoring $\epsilon$, this becomes

$$
\operatorname{SNR}_i
=
\frac{n\alpha_i}{1-\alpha_i}.
$$

When $\alpha_i$ is small,

$$
\operatorname{SNR}_i\approx n\alpha_i.
$$

Thus, the SNR of realized credit decreases directly with the useful event probability $\alpha_i$. Sparse visibility therefore weakens the statistical reliability of TD-based credit assignment.

### 2.3 Predictive credit as Rao-Blackwellized credit

FOCUS replaces the sparse realized credit $Y_i$ with its conditional expectation under the current history and joint action:

$$
g_i
=
\mathbb E[Y_i\mid \mathcal H_t,\mathbf a_t].
$$

This is a Rao-Blackwellized estimator of credit. By the law of total variance,

$$
\operatorname{Var}(Y_i)
=
\mathbb E[
\operatorname{Var}(Y_i\mid \mathcal H_t,\mathbf a_t)
]
+
\operatorname{Var}
(
\mathbb E[Y_i\mid \mathcal H_t,\mathbf a_t]
).
$$

Since

$$
g_i
=
\mathbb E[Y_i\mid \mathcal H_t,\mathbf a_t],
$$

we obtain

$$
\operatorname{Var}(g_i)
\leq
\operatorname{Var}(Y_i).
$$

Therefore, predictive belief-based credit has lower or equal variance than realized binary credit. This variance reduction is the statistical foundation of FOCUS.

### 2.4 Bias-variance trade-off from approximate belief

The predictive credit is exact only if the belief model is exact. Let $B_j(x)$ be the true future occupancy and $\widehat B_j(x)$ be the predicted occupancy. Define

$$
f_i(x)
=
v_i(x,a_i)
\prod_{k\neq i}(1-v_k(x,a_k)),
\qquad
0\leq f_i(x)\leq 1.
$$

The true credit and predicted credit are

$$
g_i
=
\sum_j w_j
\int_{\Omega}
B_j(x)f_i(x)dx,
$$

$$
\widehat g_i
=
\sum_j w_j
\int_{\Omega}
\widehat B_j(x)f_i(x)dx.
$$

For each target $j$,

$$
\left|
\int_{\Omega}
(\widehat B_j(x)-B_j(x))f_i(x)dx
\right|
\leq
\int_{\Omega}
|\widehat B_j(x)-B_j(x)|dx.
$$

Using total variation distance,

$$
\operatorname{TV}(\widehat B_j,B_j)
=
\frac{1}{2}
\|\widehat B_j-B_j\|_1,
$$

we get

$$
|\widehat g_i-g_i|
\leq
2
\sum_j
w_j
\operatorname{TV}(\widehat B_j,B_j).
$$

Thus, FOCUS trades variance reduction for possible bias from an imperfect belief model. This motivates training the belief model explicitly and evaluating both prediction quality and credit quality.

---

## 3. FOCUS with Quasi-Monte Carlo

### 3.1 Quasi-Monte Carlo approximation of predictive credit

The predictive credit integral

$$
g_i
=
\sum_j w_j
\int_{\Omega}
B_j^H(x)
v_i(x,a_i)
\prod_{k\neq i}(1-v_k(x,a_k))dx
$$

is generally intractable in continuous space. Instead of discretizing the map into a dense grid, FOCUS estimates this integral using Quasi-Monte Carlo (QMC).

Let

$$
\{u_m\}_{m=1}^{M}\subset[0,1]^2
$$

be a low-discrepancy sequence such as a Sobol sequence. A deterministic map transform $T$ converts these points to map coordinates:

$$
x_m=T(u_m)\in\Omega.
$$

For each target $j$, camera $i$, and QMC point $x_m$, define

$$
V_{i,m}=v_i(x_m,a_i).
$$

The unique visibility term is

$$
U_{i,m}
=
V_{i,m}
\prod_{k\neq i}(1-V_{k,m}).
$$

If the belief model provides a continuous density $B_j^H(x)$, the QMC estimator is

$$
\widehat g_i
=
\sum_j w_j
\frac{|\Omega|}{M}
\sum_{m=1}^{M}
B_j^H(x_m)
U_{i,m}.
$$

If the belief model outputs probability mass over QMC points, then

$$
\Pi_{j,m}^H
=
\sum_{h=1}^{H}
\omega_h
\pi_{j,h,m},
\qquad
\sum_{m=1}^{M}
\Pi_{j,m}^{H}=1,
$$

and the estimator becomes

$$
\widehat g_i
=
\sum_j w_j
\sum_{m=1}^{M}
\Pi_{j,m}^{H}
U_{i,m}.
$$

This second form is more convenient in implementation because it avoids dense grid storage and turns the occupancy belief into a probability distribution over QMC support points.

### 3.2 Belief model and belief loss

The belief model predicts future target positions. For each target $j$ and horizon $h$, it outputs logits over QMC points:

$$
\ell_{j,h,m}
=
s_{\phi,j}^{h}(x_m\mid \mathcal H_t).
$$

The corresponding distribution over QMC points is

$$
\pi_{j,h,m}
=
\frac{
\exp(\ell_{j,h,m})
}{
\sum_{r=1}^{M}
\exp(\ell_{j,h,r})
}.
$$

Because the true future target position $x_{j,t+h}^{*}$ will not usually coincide exactly with a QMC point, we use a Gaussian soft target over QMC samples:

$$
q_{j,h,m}
=
\frac{
\exp
\left(
-\frac{\|x_m-x_{j,t+h}^{*}\|^2}{2\sigma_b^2}
\right)
}{
\sum_{r=1}^{M}
\exp
\left(
-\frac{\|x_r-x_{j,t+h}^{*}\|^2}{2\sigma_b^2}
\right)
}.
$$

The belief loss is the cross entropy between the soft target and the predicted belief:

$$
\mathcal L_{\mathrm{belief}}
=
-
\frac{1}{BTJH}
\sum_{b,t,j,h}
\sum_{m=1}^{M}
q_{b,t,j,h,m}
\log
\pi_{b,t,j,h,m}.
$$

This loss trains the belief model to assign high probability to QMC points near the actual future position of each target.

### 3.3 QPLEX-style responsibility factorization

In QPLEX, the total value can be written as

$$
Q_{\mathrm{tot}}
=
V_{\mathrm{tot}}
+
\sum_{i=1}^{N_C}
\lambda_i A_i.
$$

The original QPLEX parameterization only requires

$$
\lambda_i\geq 0
$$

to preserve the individual-global-max consistency. In our implementation, we use a factorized positive credit coefficient:

$$
\lambda_i
=
p_i m_i,
$$

where

$$
p_i
=
\operatorname{softmax}(z)_i
$$

is a normalized responsibility factor, and

$$
m_i
=
1+\tanh(r_i)
$$

is a positive magnitude gate. Therefore,

$$
p_i\in(0,1),
\qquad
\sum_i p_i=1,
\qquad
m_i\in(0,2),
\qquad
\lambda_i\geq 0.
$$

This factorization separates two roles:

$$
p_i:
\text{which camera should receive credit},
$$

$$
m_i:
\text{how strongly that credit should affect the mixer}.
$$

Since FOCUS produces a normalized predictive credit distribution $\rho_i$, it regularizes the responsibility factor $p_i$, not the magnitude gate $m_i$.

### 3.4 FOCUS responsibility regularization

From the QMC credit estimator, we obtain

$$
\widehat g_i.
$$

We normalize it into the FOCUS target responsibility:

$$
\rho_i
=
\frac{\widehat g_i+\epsilon}
{\sum_{\ell=1}^{N_C}(\widehat g_\ell+\epsilon)}.
$$

Then the FOCUS loss is

$$
\mathcal L_{\mathrm{FOCUS}}
=
D_{\mathrm{KL}}
\left(
\operatorname{sg}(\rho)
\|p
\right),
$$

where $\operatorname{sg}(\cdot)$ denotes stop-gradient. Explicitly,

$$
\mathcal L_{\mathrm{FOCUS}}
=
\sum_i
\operatorname{sg}(\rho_i)
\log
\frac{
\operatorname{sg}(\rho_i)
}{
p_i+\epsilon
}.
$$

The stop-gradient is important: the predictive credit target should supervise the value decomposition responsibility, while the belief model itself should be trained by the belief loss.

### 3.5 Overall objective

The final training objective is

$$
\mathcal L
=
\mathcal L_{\mathrm{TD}}
+
\alpha
\mathcal L_{\mathrm{FOCUS}}
+
\beta
\mathcal L_{\mathrm{belief}},
$$

where

$$
\mathcal L_{\mathrm{TD}}
=
(y_{\mathrm{tot}}-Q_{\mathrm{tot}})^2.
$$

The coefficient $\alpha$ controls the strength of predictive responsibility regularization, and $\beta$ controls the strength of future occupancy prediction.

### 3.6 Algorithm

**Input:** replay batch, QPLEX networks, belief model $f_\phi$, QMC points $\{x_m\}_{m=1}^{M}$, prediction horizon $H$.

**Output:** updated QPLEX and belief model parameters.

1. Sample a batch of sequences from replay buffer.
2. Encode agent histories and compute local utilities $Q_i$ and advantages $A_i$.
3. Compute mixer outputs $V_{\mathrm{tot}}$, $z_i$, and $r_i$.
4. Compute
   $$
   p_i=\operatorname{softmax}(z)_i,
   \qquad
   m_i=1+\tanh(r_i),
   \qquad
   \lambda_i=p_i m_i.
   $$
5. Compute
   $$
   Q_{\mathrm{tot}}
   =
   V_{\mathrm{tot}}
   +
   \sum_i \lambda_i A_i.
   $$
6. Compute TD target $y_{\mathrm{tot}}$ and TD loss:
   $$
   \mathcal L_{\mathrm{TD}}
   =
   (y_{\mathrm{tot}}-Q_{\mathrm{tot}})^2.
   $$
7. Use the belief model to predict future target occupancy logits over QMC points:
   $$
   \ell_{j,h,m}=s_{\phi,j}^{h}(x_m\mid \mathcal H_t).
   $$
8. Convert logits to belief distribution:
   $$
   \pi_{j,h,m}=\operatorname{softmax}_m(\ell_{j,h,m}).
   $$
9. Build Gaussian soft labels $q_{j,h,m}$ from the true future target positions and compute $\mathcal L_{\mathrm{belief}}$.
10. Aggregate future occupancy:
    $$
    \Pi_{j,m}^{H}
    =
    \sum_{h=1}^{H}
    \omega_h
    \pi_{j,h,m}.
    $$
11. Compute visibility values $V_{i,m}=v_i(x_m,a_i)$.
12. Compute unique visibility:
    $$
    U_{i,m}
    =
    V_{i,m}
    \prod_{k\neq i}
    (1-V_{k,m}).
    $$
13. Estimate predictive credits:
    $$
    \widehat g_i
    =
    \sum_j w_j
    \sum_m
    \Pi_{j,m}^{H}
    U_{i,m}.
    $$
14. Normalize:
    $$
    \rho_i
    =
    \frac{\widehat g_i+\epsilon}
    {\sum_\ell(\widehat g_\ell+\epsilon)}.
    $$
15. Compute FOCUS loss:
    $$
    \mathcal L_{\mathrm{FOCUS}}
    =
    D_{\mathrm{KL}}(\operatorname{sg}(\rho)\|p).
    $$
16. Optimize:
    $$
    \mathcal L
    =
    \mathcal L_{\mathrm{TD}}
    +
    \alpha\mathcal L_{\mathrm{FOCUS}}
    +
    \beta\mathcal L_{\mathrm{belief}}.
    $$

### 3.7 Practical implementation notes

The belief tensor should not be stored as

$$
[B,T,C,H,J,M],
$$

because the belief is target-centric rather than camera-centric. A memory-efficient implementation stores

$$
\pi\in[B,T,H,J,M],
$$

then aggregates over the horizon:

$$
\Pi^H\in[B,T,J,M].
$$

The visibility tensor is camera-centric:

$$
V\in[B,T,C,M].
$$

Then the predictive credit is computed as

$$
\widehat g\in[B,T,C].
$$

This avoids materializing unnecessary tensors with both camera and horizon dimensions, substantially reducing memory usage.

---

## Summary

FOCUS converts sparse realized tracking events into predictive belief-based credit. The sparse visibility analysis shows that binary realized credit has poor SNR when useful tracking events are rare. The predictive credit $g_i$ is a Rao-Blackwellized conditional expectation of realized credit and therefore has lower variance. With Quasi-Monte Carlo, FOCUS estimates the predictive credit integral efficiently without dense grid storage. The resulting responsibility distribution regularizes the responsibility factor in a QPLEX-style mixer, improving credit assignment in sparse and occluded multi-camera tracking environments.
