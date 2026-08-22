# FOCUS Belief Model + Sigma Integration Upgrade Plan

**Repository:** `Thanh124pav/mate`
**Target module:** `_FOCUS` / `ray/rllib/agents/qplex_focus/`
**Verified commit:** `dbfaa08eebe72806f74b3d2de5b3eb94788b1efb`
**Main file to modify:** `ray/rllib/agents/qplex_focus/qplex_policy.py`
**Main config:** `examples/qplex_focus/camera/config.py`

---

## 1. Mục tiêu của lần chỉnh sửa

Lần chỉnh sửa này **không thay đổi core idea của FOCUS**. Mục tiêu là cải thiện hai thành phần đang có khả năng trở thành bottleneck:

1. **Belief model hiện tại là MLP theo từng timestep**, chưa khai thác trực tiếp temporal dynamics của target.
2. **`integral_mode="sigma"` hiện chỉ dùng 5 support points cố định**, với trọng số đồng đều `1/5`.

Hướng nâng cấp đề xuất:

```text
Current:
state_t
   ↓
MLP
   ↓
(mu, std)
   ↓
5 fixed sigma points
   ↓
FOCUS responsibility

Upgrade:
state_1:t
   ↓
LSTM
   ↓
(mu, std)
   ↓
configurable Gaussian quadrature
   ↓
FOCUS responsibility
```

Quan trọng: **không sửa cả hai cùng lúc ngay từ đầu**.

Thứ tự nên là:

```text
Baseline parity
    ↓
LSTM only
    ↓
Improved sigma only
    ↓
LSTM + improved sigma
```

Mục đích là biết chính xác improvement đến từ đâu.

---

# 2. Snapshot implementation hiện tại

## 2.1. Belief model hiện tại

Trong `qplex_policy.py`, class hiện tại:

```python
class LearnedOccupancyModel(nn.Module):
```

nhận global state tại từng timestep và dùng:

```python
self.net = nn.Sequential(
    nn.Linear(self.state_dim, hidden_dim),
    nn.ReLU(),
    nn.Linear(hidden_dim, hidden_dim),
    nn.ReLU(),
    nn.Linear(hidden_dim, horizon * n_targets * 4),
)
```

Output được reshape thành:

```python
[B, T, horizon, n_targets, 4]
```

với:

```python
delta = torch.tanh(out[..., :2]) * self.max_delta
std = F.softplus(out[..., 2:]) + self.min_std
mean = current_pos.unsqueeze(2) + delta
```

Do đó belief hiện tại là một diagonal Gaussian:

\[
p(x_{j,t+h})
=
\mathcal N(
\mu_{j,t+h},
\operatorname{diag}(\sigma_x^2,\sigma_y^2)
).
\]

Điểm đáng chú ý:

- `mean`: 2 chiều `(x,y)`.
- `std`: 2 chiều `(sigma_x,sigma_y)`.
- Chưa có covariance chéo `cov_xy`.
- Mean được parameterize dưới dạng displacement từ current target position.
- Belief loss là Gaussian negative log-likelihood.
- Loss hiện tại đã multi-horizon.

Không cần đổi NLL khi chỉ thay MLP bằng LSTM.

---

## 2.2. Sigma mode hiện tại

Hàm hiện tại:

```python
def _sigma_points(self, mean, std):
    x = torch.stack(
        [std[..., 0], torch.zeros_like(std[..., 0])],
        dim=-1
    )
    y = torch.stack(
        [torch.zeros_like(std[..., 1]), std[..., 1]],
        dim=-1
    )
    return torch.stack(
        [mean, mean + x, mean - x, mean + y, mean - y],
        dim=4
    )
```

Tức mỗi Gaussian tạo đúng 5 điểm:

```text
                  mu + sigma_y
                       ●
                       |
                       |
mu - sigma_x     ●─────●─────●     mu + sigma_x
                       mu
                       |
                       |
                       ●
                  mu - sigma_y
```

Số point:

\[
N_\sigma=5.
\]

Sau đó `_credit_from_sigma_points()` chia tổng contribution cho:

```python
float(total_samples)
```

nên 5 điểm có trọng số:

\[
w_k=\frac15.
\]

Đây không phải một full Unscented Transform. Nó là một **5-point deterministic support approximation**.

---

# 3. Nguyên tắc khi sửa

## 3.1. Backward compatibility là bắt buộc

Sau khi sửa, config cũ phải vẫn chạy.

Config mặc định nên tiếp tục tương đương với:

```python
'belief_arch': 'mlp',
'integral_mode': 'sigma',
'sigma_method': 'legacy5',
```

Nếu chạy `legacy5`, output numerical phải gần như giống code hiện tại.

Không nên đổi behavior mặc định thành Gauss-Hermite ngay.

---

## 3.2. Không thay interface của belief model

Dù dùng MLP hay LSTM, `forward()` vẫn phải return:

```python
mean, std
```

với shape:

```text
mean: [B, T, H, J, 2]
std:  [B, T, H, J, 2]
```

Trong đó:

- `B`: batch.
- `T`: sequence length.
- `H`: prediction horizon.
- `J`: number of targets.

Nhờ đó:

```text
grid
MC
sigma
```

đều dùng lại được.

---

# 4. PHASE 0 — Tạo baseline parity trước khi sửa

Trước khi chỉnh architecture:

1. Checkout đúng commit hiện tại.
2. Chạy một short experiment với seed cố định.
3. Lưu:
   - reward curve,
   - `focus_belief_loss`,
   - `focus_belief_pos_error_h1/h2/h3`,
   - `focus_belief_pred_std_h1/h2/h3`,
   - `focus_valid_ratio`,
   - `focus_mean_signal`,
   - `focus_credit_loss`,
   - wall-clock training time.
4. Với sigma mode, lưu một batch `mean`, `std`, `rho` để dùng parity test.

Ví dụ config baseline:

```python
'focus': {
    ...
    'belief_mode': 'learned',
    'belief_arch': 'mlp',
    'integral_mode': 'sigma',
    'sigma_method': 'legacy5',
}
```

Nếu chưa thêm các key mới, baseline hiện tại chính là reference.

---

# 5. PHASE 1 — Thay belief MLP bằng LSTM, giữ sigma cũ

Đây là thay đổi nên làm trước nếu mục tiêu là kiểm tra temporal modeling.

---

## 5.1. Vì sao LSTM hợp lý hơn MLP

Current MLP thực chất tính:

\[
s_t
\rightarrow
(\mu_{t+1:t+H},\sigma_{t+1:t+H}).
\]

Nó chỉ biết dynamics thông qua feature trong current state.

LSTM sẽ tính:

\[
s_1,\ldots,s_t
\rightarrow
h_t
\rightarrow
(\mu_{t+1:t+H},\sigma_{t+1:t+H}).
\]

Trong target tracking, hidden state có thể encode:

- moving direction,
- approximate velocity,
- acceleration trend,
- recent motion changes,
- uncertainty accumulated from history.

Mục tiêu không phải để LSTM output một deterministic future point.

Nó vẫn phải output **distribution**:

\[
(\mu,\sigma).
\]

---

# 6. Chỉnh `LearnedOccupancyModel`

## 6.1. Thêm architecture option

Sửa constructor thành dạng:

```python
class LearnedOccupancyModel(nn.Module):
    def __init__(
        self,
        state_dim,
        n_agents,
        n_targets,
        horizon=3,
        hidden_dim=256,
        max_delta=400.0,
        min_std=25.0,
        architecture="mlp",
        num_layers=1,
        dropout=0.0,
    ):
```

Lưu:

```python
self.architecture = architecture
self.num_layers = num_layers
```

---

## 6.2. Giữ nguyên MLP cũ

Để baseline parity, **không refactor MLP quá mạnh tay**.

Nên giữ nguyên:

```python
if self.architecture == "mlp":
    self.net = nn.Sequential(
        nn.Linear(self.state_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, horizon * n_targets * 4),
    )
```

Như vậy `belief_arch="mlp"` phải tái tạo gần như chính xác baseline.

---

## 6.3. Thêm LSTM branch

Thêm:

```python
elif self.architecture == "lstm":
    self.rnn = nn.LSTM(
        input_size=self.state_dim,
        hidden_size=hidden_dim,
        num_layers=num_layers,
        batch_first=True,
        dropout=dropout if num_layers > 1 else 0.0,
    )

    self.head = nn.Linear(
        hidden_dim,
        horizon * n_targets * 4,
    )
```

Không cần decoder phức tạp ở bước đầu.

---

# 7. Sửa `forward()`

Current code:

```python
B, T = state.shape[:2]
current_pos = _extract_target_positions(...)

out = self.net(
    state.reshape(-1, self.state_dim)
).view(
    B, T, self.horizon, self.n_targets, 4
)
```

Sửa thành:

```python
def forward(self, state):
    B, T = state.shape[:2]

    current_pos = _extract_target_positions(
        state,
        self.n_agents,
        self.n_targets,
    )

    state_flat = state.reshape(B, T, self.state_dim)

    if self.architecture == "mlp":
        out = self.net(
            state_flat.reshape(-1, self.state_dim)
        )

        out = out.view(
            B,
            T,
            self.horizon,
            self.n_targets,
            4,
        )

    elif self.architecture == "lstm":
        features, _ = self.rnn(state_flat)

        out = self.head(features).view(
            B,
            T,
            self.horizon,
            self.n_targets,
            4,
        )

    else:
        raise ValueError(
            f"Unknown belief architecture: {self.architecture}"
        )

    delta = torch.tanh(out[..., :2]) * self.max_delta
    std = F.softplus(out[..., 2:]) + self.min_std

    mean = current_pos.unsqueeze(2) + delta

    return mean, std
```

---

# 8. Hidden state của LSTM xử lý thế nào?

Không cần tự maintain hidden state giữa `learn_on_batch()` calls ở bước đầu.

Lý do:

`chop_into_sequences()` đã đưa data về sequence:

```text
[B, T, ...]
```

LSTM được gọi trên toàn sequence:

```python
features, _ = self.rnn(state_flat)
```

PyTorch mặc định khởi tạo hidden state bằng zero cho mỗi sequence.

Vì LSTM là causal:

```text
h_t = f(s_t, h_{t-1})
```

nên output tại timestep `t` không nhìn thấy future state.

Current config cũng đang dùng:

```python
'batch_mode': 'complete_episodes'
'max_seq_len': 10000
```

nên context loss tương đối ít.

### Không làm ở MVP

Chưa cần:

- persistent hidden state giữa replay batches,
- truncated BPTT custom,
- bidirectional LSTM,
- Transformer,
- attention,
- recurrent state trong `compute_actions()`.

Belief model chỉ tham gia training loss, không cần thay execution policy.

---

# 9. Config cho LSTM

Trong:

```text
examples/qplex_focus/camera/config.py
```

thêm:

```python
'belief_arch': 'mlp',
'belief_num_layers': 1,
'belief_dropout': 0.0,
```

Sau đó test LSTM:

```python
'belief_arch': 'lstm',
'belief_hidden_dim': 256,
'belief_num_layers': 1,
'belief_dropout': 0.0,
```

### Performance-first recommendation

Bắt đầu:

```python
hidden_dim = 256
num_layers = 1
```

Không nên ngay lập tức dùng 2–3 LSTM layers.

LSTM đã có 4 gates nên capacity tăng nhanh.

Sau khi biết nó có improvement thì mới làm parameter-matched ablation.

---

# 10. Sửa nơi instantiate occupancy model

Current:

```python
self.occupancy_model = LearnedOccupancyModel(
    self.env_global_state_shape,
    self.n_agents,
    int(focus_config.get("n_targets", 8)),
    horizon=int(focus_config.get("horizon", 3)),
    hidden_dim=int(focus_config.get("belief_hidden_dim", 256)),
    max_delta=float(focus_config.get("belief_max_delta", 400.0)),
    min_std=float(focus_config.get("belief_min_std", 25.0)),
).to(self.device)
```

Sửa thành:

```python
self.occupancy_model = LearnedOccupancyModel(
    self.env_global_state_shape,
    self.n_agents,
    int(focus_config.get("n_targets", 8)),
    horizon=int(focus_config.get("horizon", 3)),
    hidden_dim=int(focus_config.get("belief_hidden_dim", 256)),
    max_delta=float(focus_config.get("belief_max_delta", 400.0)),
    min_std=float(focus_config.get("belief_min_std", 25.0)),
    architecture=str(
        focus_config.get("belief_arch", "mlp")
    ).lower(),
    num_layers=int(
        focus_config.get("belief_num_layers", 1)
    ),
    dropout=float(
        focus_config.get("belief_dropout", 0.0)
    ),
).to(self.device)
```

Optimizer không cần sửa vì hiện tại đã có:

```python
if self.occupancy_model:
    self.params += list(self.occupancy_model.parameters())
```

---

# 11. PHASE 1 TESTS — LSTM only

Giữ:

```python
'integral_mode': 'sigma',
'sigma_method': 'legacy5',
```

rồi chỉ thay:

```python
'belief_arch': 'mlp'
```

thành:

```python
'belief_arch': 'lstm'
```

So sánh:

```text
MLP + sigma-5
vs.
LSTM + sigma-5
```

Metric quan trọng nhất ở đây chưa phải final reward.

Trước hết xem:

```text
focus_belief_pos_error_h1
focus_belief_pos_error_h2
focus_belief_pos_error_h3
focus_belief_loss_h1
focus_belief_loss_h2
focus_belief_loss_h3
focus_belief_pred_std_h1
focus_belief_pred_std_h2
focus_belief_pred_std_h3
```

Nếu LSTM không cải thiện belief error thì chưa có lý do để kỳ vọng MARL performance tăng.

---

# 12. PHASE 2 — Nâng cấp sigma integration

Sau khi LSTM branch chạy ổn, quay lại MLP baseline và cải thiện sigma riêng.

Experiment:

```text
MLP + legacy sigma-5
vs.
MLP + improved sigma
```

---

# 13. Không nên chỉ tăng số point bằng nhiều vòng heuristic

Có thể tự tạo:

```text
mu ± 0.5 sigma
mu ± 1.0 sigma
mu ± 1.5 sigma
...
```

nhưng cách này cần tự thiết kế weights.

Đối với belief Gaussian hiện tại, cách sạch hơn là:

# Gauss–Hermite quadrature

Lý do:

- belief đã Gaussian,
- dimensionality chỉ là 2,
- deterministic,
- có quadrature weights,
- số point dễ điều khiển,
- dễ viết trong paper.

---

# 14. Công thức Gauss–Hermite cần dùng

Với standard normal:

\[
Z\sim\mathcal N(0,1)
\]

ta có:

\[
\mathbb E[f(Z)]
\approx
\frac{1}{\sqrt{\pi}}
\sum_{i=1}^{m}
w_i
f(\sqrt{2}x_i),
\]

trong đó:

- \(x_i\): Hermite nodes,
- \(w_i\): Hermite weights,
- \(m\): quadrature order.

Trong 2D independent Gaussian:

\[
(x,y)
\sim
\mathcal N(
\mu,
\operatorname{diag}(\sigma_x^2,\sigma_y^2)
),
\]

dùng Cartesian product:

\[
N_{\sigma}=m^2.
\]

Ví dụ:

| `sigma_order` | số point |
|---:|---:|
| 2 | 4 |
| 3 | 9 |
| 4 | 16 |
| 5 | 25 |
| 7 | 49 |
| 9 | 81 |

Không cần dùng 128 point ngay.

Recommended sweep:

```text
legacy5
GH-3  = 9 points
GH-5  = 25 points
GH-7  = 49 points
```

---

# 15. Config mới cho sigma

Thêm:

```python
'sigma_method': 'legacy5',
'sigma_order': 3,
'sample_chunk_size': 32,
```

Không đổi default:

```python
sigma_method = legacy5
```

để giữ parity.

---

# 16. Refactor `_sigma_points()`

Nên để function return:

```python
points, weights
```

thay vì chỉ points.

Interface:

```python
def _sigma_points(self, mean, std):
    ...
    return points, weights
```

Trong đó:

```text
points:
[B, T, H, J, P, 2]

weights:
[P]
```

---

# 17. Legacy 5-point branch

Code:

```python
def _legacy_sigma_points(self, mean, std):
    x = torch.stack(
        [
            std[..., 0],
            torch.zeros_like(std[..., 0]),
        ],
        dim=-1,
    )

    y = torch.stack(
        [
            torch.zeros_like(std[..., 1]),
            std[..., 1],
        ],
        dim=-1,
    )

    points = torch.stack(
        [
            mean,
            mean + x,
            mean - x,
            mean + y,
            mean - y,
        ],
        dim=4,
    )

    weights = torch.full(
        (5,),
        1.0 / 5.0,
        device=mean.device,
        dtype=mean.dtype,
    )

    return points, weights
```

Function này phải numerically equivalent với implementation hiện tại.

---

# 18. Gauss–Hermite point generator

Có thể dùng NumPy:

```python
nodes_np, weights_np = np.polynomial.hermite.hermgauss(order)
```

Implementation:

```python
def _gauss_hermite_sigma_points(
    self,
    mean,
    std,
    order,
):
    nodes_np, weights_np = np.polynomial.hermite.hermgauss(
        order
    )

    nodes = torch.as_tensor(
        nodes_np,
        device=mean.device,
        dtype=mean.dtype,
    )

    weights_1d = torch.as_tensor(
        weights_np,
        device=mean.device,
        dtype=mean.dtype,
    )

    # Convert Hermite nodes to standard-normal coordinates.
    nodes = np.sqrt(2.0) * nodes

    # Normalize 1D weights for expectation under N(0,1).
    weights_1d = weights_1d / np.sqrt(np.pi)

    yy, xx = torch.meshgrid(
        nodes,
        nodes,
        indexing="ij",
    )

    wy, wx = torch.meshgrid(
        weights_1d,
        weights_1d,
        indexing="ij",
    )

    normals = torch.stack(
        [
            xx.reshape(-1),
            yy.reshape(-1),
        ],
        dim=-1,
    )

    weights = (wx * wy).reshape(-1)
    weights = weights / weights.sum()

    points = (
        mean.unsqueeze(4)
        +
        std.unsqueeze(4)
        * normals.view(1, 1, 1, 1, -1, 2)
    )

    return points, weights
```

---

# 19. Dispatcher `_sigma_points()`

```python
def _sigma_points(self, mean, std):
    method = str(
        self.focus_config.get(
            "sigma_method",
            "legacy5",
        )
    ).lower()

    if method == "legacy5":
        return self._legacy_sigma_points(mean, std)

    if method in {
        "gauss_hermite",
        "gh",
    }:
        order = int(
            self.focus_config.get(
                "sigma_order",
                3,
            )
        )

        if order < 1:
            raise ValueError(
                f"sigma_order must be >= 1, got {order}"
            )

        return self._gauss_hermite_sigma_points(
            mean,
            std,
            order,
        )

    raise ValueError(
        f"Unknown sigma_method: {method}"
    )
```

---

# 20. Quan trọng: `_credit_from_sigma_points()` phải hỗ trợ weights

Đây là phần dễ bỏ sót nhất.

Current implementation:

```python
chunk_sum = ...
g = g + horizon_weights[h] * chunk_sum / float(total_samples)
```

Điều này chỉ đúng khi mọi sample có weight:

\[
1/P.
\]

Gauss–Hermite có **non-uniform weights**.

Do đó không được chỉ thay point generator.

---

# 21. Sửa `_credit_chunk_sum()`

Thêm parameter:

```python
sample_weights=None
```

Signature:

```python
def _credit_chunk_sum(
    self,
    target_samples,
    cam_pos,
    cam_orient,
    cam_range,
    cam_half_angle,
    selection,
    obstacle_pos=None,
    obstacle_radius=None,
    sample_weights=None,
):
```

Sau:

```python
unique_gain = visible * ...
```

thêm:

```python
if sample_weights is not None:
    unique_gain = (
        unique_gain
        * sample_weights.view(
            1, 1, 1, 1, 1, -1
        )
    )
```

Sau đó giữ:

```python
return unique_gain.sum(dim=-1).sum(dim=-1)
```

Weights chỉ áp vào **support-point dimension**, không áp vào target dimension.

---

# 22. Sửa `_credit_from_sigma_points()`

Signature:

```python
def _credit_from_sigma_points(
    self,
    target_samples,
    state,
    actions,
    n_targets,
    sample_weights=None,
):
```

Ngay sau:

```python
total_samples = target_samples.size(4)
```

thêm:

```python
if sample_weights is None:
    sample_weights = torch.full(
        (total_samples,),
        1.0 / float(total_samples),
        device=target_samples.device,
        dtype=target_samples.dtype,
    )
else:
    sample_weights = sample_weights.to(
        device=target_samples.device,
        dtype=target_samples.dtype,
    )
    sample_weights = (
        sample_weights
        /
        (
            sample_weights.sum()
            + self.focus_config.get("eps", 1e-8)
        )
    )
```

Trong chunk loop:

```python
weights_chunk = sample_weights[
    start : start + chunk_size
]
```

Rồi gọi:

```python
chunk_sum = self._credit_chunk_sum(
    samples,
    cam_pos,
    cam_orient,
    cam_range,
    cam_half_angle,
    selection,
    obstacle_pos=obstacle_pos,
    obstacle_radius=obstacle_radius,
    sample_weights=weights_chunk,
).squeeze(3)
```

Cuối cùng sửa:

```python
g = g + horizon_weights[h] * chunk_sum
```

**Bỏ:**

```python
/ float(total_samples)
```

vì normalization đã nằm trong `sample_weights`.

---

# 23. Sửa sigma dispatcher trong `_focus_credit_target()`

Current:

```python
elif integral_mode == "sigma":
    target_samples = self._sigma_points(mean, std).detach()

    g = self._credit_from_sigma_points(
        target_samples,
        state,
        actions,
        n_targets,
    )
```

Sửa:

```python
elif integral_mode == "sigma":
    target_samples, sample_weights = (
        self._sigma_points(mean, std)
    )

    target_samples = target_samples.detach()
    sample_weights = sample_weights.detach()

    g = self._credit_from_sigma_points(
        target_samples,
        state,
        actions,
        n_targets,
        sample_weights=sample_weights,
    )
```

---

# 24. Oracle ablation không được làm hỏng

Current:

```python
target_samples = _extract_target_positions(...)
target_samples = (
    target_samples
    .unsqueeze(2)
    .unsqueeze(4)
)

g = self._credit_from_sigma_points(
    target_samples,
    state,
    actions,
    n_targets,
)
```

Không cần sửa.

Khi `sample_weights=None` và chỉ có 1 point:

```text
P = 1
weight = 1
```

nên oracle behavior giữ nguyên.

---

# 25. Unit tests tối thiểu cho sigma

Trước khi train dài, test các invariant.

## Test 1 — Legacy parity

Random:

```python
mean = torch.randn(...)
std = torch.rand(...) + 1.0
```

Old sigma result và:

```python
sigma_method="legacy5"
```

phải:

```python
torch.allclose(
    old_result,
    new_result,
    atol=1e-6,
    rtol=1e-6,
)
```

---

## Test 2 — Number of points

```text
GH order 3 -> 9 points
GH order 5 -> 25 points
GH order 7 -> 49 points
```

Check:

```python
points.size(4) == order ** 2
```

---

## Test 3 — Weights sum to one

```python
assert torch.allclose(
    weights.sum(),
    torch.tensor(
        1.0,
        device=weights.device,
        dtype=weights.dtype,
    ),
    atol=1e-6,
)
```

---

## Test 4 — Weighted mean

Với bất kỳ:

```python
mean
std
```

weighted sample mean phải gần:

\[
\mu.
\]

Code:

```python
estimated_mean = (
    points
    * weights.view(1, 1, 1, 1, -1, 1)
).sum(dim=4)
```

Check:

```python
torch.allclose(
    estimated_mean,
    mean,
    atol=1e-5,
)
```

---

## Test 5 — Weighted variance

Với GH order >= 2:

\[
\operatorname{Var}(X)
\approx
\sigma^2.
\]

Code:

```python
centered = points - mean.unsqueeze(4)

estimated_var = (
    centered.pow(2)
    * weights.view(1, 1, 1, 1, -1, 1)
).sum(dim=4)
```

Check với:

```python
std.pow(2)
```

---

# 26. PHASE 2 EXPERIMENTS — Sigma only

Giữ:

```python
'belief_arch': 'mlp'
```

và sweep:

```text
legacy5
GH-3
GH-5
GH-7
```

Tương ứng:

```text
5
9
25
49
```

spatial samples.

So sánh thêm:

```text
MC-32
MC-64
MC-128
MC-256
```

và nếu cần:

```text
grid-16: 256 cells
grid-32: 1024 cells
grid-64: 4096 cells
```

---

# 27. Metric nên plot cho integration method

Không chỉ plot final reward.

Ít nhất nên có:

## A. MARL performance

```text
mean episode reward / coverage rate
```

## B. Compute

```text
training time / iteration
```

hoặc:

```text
samples evaluated per target
```

## C. Responsibility stability

Có thể log:

```text
mean |rho_t - rho_{t-1}|
```

nếu muốn kiểm tra quantization noise.

Sigma-5 có thể nhảy mạnh khi FoV boundary đi qua một support point.

GH-25/GH-49 có thể smooth hơn.

## D. Agreement với dense reference

Có thể lấy một held-out batch.

Tính:

```text
rho_ref = grid-128 hoặc MC-8192
```

rồi so:

\[
\|\rho_{\text{approx}}-\rho_{\text{ref}}\|_1
\]

hoặc:

\[
\operatorname{MSE}(
\rho_{\text{approx}},
\rho_{\text{ref}}
).
\]

Đây là ablation rất sạch để chứng minh improved sigma thực sự approximate integral tốt hơn.

---

# 28. PHASE 3 — Kết hợp LSTM + improved sigma

Chỉ sau khi Phase 1 và Phase 2 có signal.

Recommended configs:

## Baseline

```python
'belief_arch': 'mlp',
'integral_mode': 'sigma',
'sigma_method': 'legacy5',
```

## Temporal only

```python
'belief_arch': 'lstm',
'integral_mode': 'sigma',
'sigma_method': 'legacy5',
```

## Integration only

```python
'belief_arch': 'mlp',
'integral_mode': 'sigma',
'sigma_method': 'gauss_hermite',
'sigma_order': 5,
```

## Combined

```python
'belief_arch': 'lstm',
'integral_mode': 'sigma',
'sigma_method': 'gauss_hermite',
'sigma_order': 5,
```

Ablation table:

| Belief | Integral | Expected role |
|---|---|---|
| MLP | sigma-5 | current baseline |
| LSTM | sigma-5 | temporal modeling |
| MLP | GH-25 | better integration |
| LSTM | GH-25 | combined |

Nếu:

```text
LSTM + GH > LSTM + sigma5 > MLP + sigma5
```

thì temporal model là main bottleneck.

Nếu:

```text
LSTM + GH > MLP + GH > MLP + sigma5
```

thì integration accuracy có ảnh hưởng rõ.

Nếu chỉ combined tăng:

```text
LSTM + GH >> others
```

thì hai bottleneck tương tác với nhau.

---

# 29. PHASE 4 — Full covariance: chỉ làm sau khi các phase trên ổn

Current Gaussian là axis-aligned:

\[
\Sigma
=
\begin{bmatrix}
\sigma_x^2 & 0 \\
0 & \sigma_y^2
\end{bmatrix}.
\]

Nhưng target có thể di chuyển chéo.

Ví dụ uncertainty thực tế:

```text
        /
      /   elongated uncertainty
    /
```

Trong khi model chỉ biểu diễn:

```text
   |
---+---
   |
```

Nếu temporal motion là diagonal, covariance chéo có thể hữu ích.

Nhưng **không làm cùng ngày với LSTM + GH**.

Nó thay đổi:

- output dimension,
- NLL,
- sigma point transform,
- numerical stability.

Nên coi là Phase 4 riêng.

---

# 30. Full covariance implementation outline

Nếu sau này cần:

Network output từ:

```text
4 values per target/horizon
```

thành:

```text
5 values:
delta_x
delta_y
raw_sigma_x
raw_sigma_y
raw_rho
```

Define:

```python
std_x = F.softplus(...) + min_std
std_y = F.softplus(...) + min_std

rho = 0.95 * torch.tanh(raw_rho)
```

Covariance:

\[
\Sigma=
\begin{bmatrix}
\sigma_x^2 &
\rho\sigma_x\sigma_y\\
\rho\sigma_x\sigma_y &
\sigma_y^2
\end{bmatrix}.
\]

Gauss-Hermite point transform phải dùng Cholesky:

\[
x_k=
\mu+Lz_k,
\qquad
LL^\top=\Sigma.
\]

Không dùng:

```python
mean + std * normals
```

nữa.

---

# 31. U-Net / grid branch — giữ là một experiment riêng

Không nên ghép U-Net vào patch LSTM + sigma.

U-Net có ý nghĩa nhất khi output là:

```text
64 x 64 occupancy map
```

và dùng:

```python
integral_mode = "grid"
```

Pipeline:

```text
historical spatial maps
        ↓
ConvLSTM / temporal encoder
        ↓
U-Net
        ↓
64x64 occupancy
        ↓
grid visibility integral
```

Đây là một architecture khác về representation.

Nên coi là:

```text
Distributional branch:
LSTM -> Gaussian -> sigma / MC

Spatial branch:
ConvLSTM/U-Net -> grid belief
```

Không nên biến một experiment thành quá nhiều thay đổi.

---

# 32. Một điểm quan trọng: current LSTM proposal vẫn dùng global state

Current occupancy model nhận:

```python
self.env_global_state_shape
```

và được train trong CTDE loss.

Do đó LSTM version cũng sẽ dùng global-state sequence.

Điều này không làm execution centralized hơn vì occupancy model không được dùng để chọn action trong `compute_actions()`.

Execution policy vẫn theo QPLEX decentralized agent observations.

Không cần đưa belief LSTM state vào action policy.

---

# 33. Không đổi `beta_belief` ngay

Current loss:

\[
L
=
L_{\rm TD}
+
\alpha L_{\rm focus}
+
\beta L_{\rm belief}.
\]

Config hiện tại:

```python
'alpha_credit': 0.05,
'beta_belief': 0.01,
```

Khi test architecture:

**giữ nguyên hai hệ số này trước**.

Nếu đổi architecture và loss coefficient cùng lúc, sẽ khó biết improvement đến từ đâu.

Sau khi tìm được architecture tốt hơn mới sweep:

```text
beta_belief:
0.001
0.003
0.01
0.03
```

nếu thực sự cần.

---

# 34. Debugging checklist cho LSTM

Nếu reward tụt ngay:

## Check 1 — NLL có NaN không?

Log:

```python
torch.isfinite(mean).all()
torch.isfinite(std).all()
torch.isfinite(belief_loss)
```

## Check 2 — std có collapse không?

Theo dõi:

```text
focus_belief_pred_std_h1
focus_belief_pred_std_h2
focus_belief_pred_std_h3
```

Nếu std luôn sát:

```python
belief_min_std
```

thì model đang overconfident/collapse.

Nếu std rất lớn, NLL có thể đang dùng variance để che mean error.

## Check 3 — Mean displacement saturation

Current:

```python
delta = tanh(...) * max_delta
```

Nếu prediction cần displacement > `belief_max_delta`, LSTM cũng không cứu được.

Log tỷ lệ:

```text
abs(delta) > 0.95 * max_delta
```

Nếu cao, cần tăng `belief_max_delta`.

## Check 4 — Gradient norm occupancy model

Có thể log riêng:

```python
sum(
    p.grad.norm().item()
    for p in self.occupancy_model.parameters()
    if p.grad is not None
)
```

Nếu gần zero liên tục thì belief model không học.

---

# 35. Debugging checklist cho sigma

## Check 1 — Weight normalization

Luôn:

```text
sum(weights) ≈ 1
```

## Check 2 — Legacy parity

Bắt buộc pass trước train.

## Check 3 — `rho` range

Check:

```text
rho >= 0
sum_agents rho ≈ 1
```

ở valid samples.

## Check 4 — `total_g`

Nếu improved quadrature làm:

```text
focus_mean_signal
```

thay đổi rất lớn so với sigma-5, kiểm tra weighting trước khi kết luận method tốt/xấu.

## Check 5 — Chunking invariance

Chạy cùng sample points với:

```text
sample_chunk_size = 8
sample_chunk_size = 32
sample_chunk_size = 128
```

Result phải gần giống nhau.

Nếu khác thì weighted chunking đang sai.

---

# 36. Metrics nên bổ sung vào `last_focus_stats`

Không bắt buộc nhưng hữu ích:

```python
"focus_sigma_num_points": float(num_points)
```

Có thể thêm:

```python
"focus_belief_arch_lstm": 1.0 or 0.0
```

nhưng không thật sự cần.

Quan trọng hơn là lưu config trong experiment directory.

Có thể log thêm mean quadrature weight entropy:

\[
H(w)=-\sum_k w_k\log w_k
\]

nhưng chỉ cần nếu phân tích sâu.

---

# 37. Recommended experiment schedule

Không cần chạy full training cho tất cả ngay.

## Stage A — Smoke tests

Mỗi config:

```text
1–3 training iterations
```

Mục tiêu:

- không crash,
- không NaN,
- shape đúng,
- GPU memory ổn.

Configs:

```text
MLP + legacy5
LSTM + legacy5
MLP + GH3
LSTM + GH3
```

---

## Stage B — Short screening

Chạy khoảng 10–20% training budget.

Configs:

```text
MLP + legacy5
LSTM + legacy5
MLP + GH3
MLP + GH5
LSTM + GH3
LSTM + GH5
```

Loại các variant rõ ràng kém.

---

## Stage C — Full run

Chỉ chạy full cho:

```text
current baseline
best LSTM-only
best sigma-only
best combined
MC baseline
grid baseline nếu cần
```

Sau đó mới multi-seed.

---

# 38. Recommended initial settings

### LSTM

```python
'belief_arch': 'lstm',
'belief_hidden_dim': 256,
'belief_num_layers': 1,
'belief_dropout': 0.0,
```

### Sigma

Bắt đầu:

```python
'integral_mode': 'sigma',
'sigma_method': 'gauss_hermite',
'sigma_order': 3,
'sample_chunk_size': 32,
```

Nếu ổn:

```python
sigma_order = 5
```

Không nhảy thẳng lên order 9.

---

# 39. Parameter-count fairness

Performance-first phase:

```text
MLP hidden = 256
LSTM hidden = 256
```

là chấp nhận được.

Nhưng trong paper, reviewer có thể nói:

> LSTM thắng vì có nhiều parameters hơn.

Sau khi biết LSTM thực sự tốt, thêm parameter-matched ablation.

Có thể thử:

```text
MLP-256
LSTM-128
LSTM-256
```

và report parameter count.

Không cần optimize chuyện này trước khi biết LSTM có giúp hay không.

---

# 40. Suggested commit structure

Đừng gộp mọi thứ vào một commit.

## Commit 1

```text
refactor(focus): add configurable belief architecture
```

- add `belief_arch`,
- add LSTM branch,
- preserve MLP.

## Commit 2

```text
feat(focus): add weighted sigma quadrature
```

- weights in sigma integration,
- legacy parity.

## Commit 3

```text
feat(focus): add Gauss-Hermite sigma points
```

- `sigma_method`,
- `sigma_order`.

## Commit 4

```text
test(focus): add belief and sigma integration checks
```

## Commit 5

```text
exp(focus): add LSTM and GH experiment configs
```

Nếu có bug, dễ bisect.

---

# 41. Những thứ KHÔNG nên sửa trong cùng đợt

Ngày mai không nên đồng thời:

- đổi LSTM,
- đổi sigma,
- thêm covariance,
- thêm U-Net,
- đổi `alpha_credit`,
- đổi `beta_belief`,
- đổi horizon,
- đổi QPLEX mixer,
- đổi replay setup,
- đổi confidence gate.

Nếu performance thay đổi, sẽ không thể xác định nguyên nhân.

---

# 42. Thứ tự sửa code ngày mai

Checklist thực tế:

### Step 1

Tạo branch:

```bash
git checkout -b focus-belief-sigma-upgrade
```

### Step 2

Thêm `belief_arch="mlp"` nhưng chưa thêm LSTM.

Run baseline.

Output phải giống cũ.

### Step 3

Implement LSTM branch.

Run:

```text
LSTM + legacy sigma-5
```

Check belief diagnostics.

### Step 4

Refactor sigma code để support weights.

**Chỉ dùng `legacy5` trước.**

Run parity test.

Nếu reward hoặc `rho` thay đổi nhiều, fix trước.

### Step 5

Implement Gauss-Hermite.

Run unit tests:

```text
weights
mean
variance
point count
chunk invariance
```

### Step 6

Run:

```text
MLP + GH3
MLP + GH5
```

### Step 7

Nếu sigma cải thiện:

```text
LSTM + GH3
LSTM + GH5
```

### Step 8

Chỉ sau khi có result mới quyết định:

```text
full covariance?
U-Net?
GRU?
```

---

# 43. Decision tree sau experiments

```text
Does LSTM reduce belief error?
│
├── NO
│   ├── Check max_delta saturation
│   ├── Check sequence construction
│   ├── Check target dynamics
│   └── Do not assume larger recurrent model helps
│
└── YES
    │
    └── Does MARL performance improve?
        │
        ├── YES → temporal belief was a real bottleneck
        │
        └── NO
            └── responsibility integration / coupling may be bottleneck
```

Sau đó:

```text
Does GH reduce rho error vs dense reference?
│
├── NO
│   └── Sigma integration is not the bottleneck
│
└── YES
    │
    └── Does MARL reward improve?
        │
        ├── YES → keep GH
        └── NO → downstream credit loss may be insensitive
```

Đây là cách tránh tiếp tục thêm complexity mù quáng.

---

# 44. Kỳ vọng hợp lý

Không nên kỳ vọng:

```text
MLP -> LSTM
```

tự động tạo huge reward gain.

Điều cần tìm là chuỗi evidence:

\[
\text{better temporal prediction}
\rightarrow
\text{better future belief}
\rightarrow
\text{better responsibility estimate}
\rightarrow
\text{better credit alignment}
\rightarrow
\text{better MARL performance}.
\]

Tương tự với sigma:

\[
\text{better quadrature}
\rightarrow
\text{lower integral error}
\rightarrow
\text{more stable responsibility}
\rightarrow
\text{better optimization}.
\]

Nếu một link trong chain không xảy ra thì dừng ở đó, không cần tăng model complexity.

---

# 45. MVP recommendation

Nếu chỉ có thời gian làm **một bản nâng cấp tối thiểu**:

```text
1. Add LSTM belief architecture.
2. Keep diagonal Gaussian output.
3. Add Gauss-Hermite weighted sigma integration.
4. Use sigma_order=3 or 5.
5. Keep every other FOCUS hyperparameter unchanged.
```

Architecture:

```text
Global state sequence
        ↓
      LSTM
        ↓
 multi-horizon head
        ↓
(mu_x, mu_y, sigma_x, sigma_y)
        ↓
Gauss-Hermite sigma points
        ↓
geometry / FoV visibility
        ↓
responsibility rho
        ↓
FOCUS credit alignment
```

Đây là thay đổi tương đối nhỏ, modular, dễ ablate và dễ rollback.

---

# 46. Config mẫu cuối cùng

## Current-compatible baseline

```python
'focus': {
    'enabled': True,
    'use_env_params': True,

    'alpha_credit': 0.05,
    'beta_belief': 0.01,

    'belief_mode': 'learned',
    'belief_arch': 'mlp',
    'belief_hidden_dim': 256,
    'belief_num_layers': 1,
    'belief_dropout': 0.0,

    'horizon': 3,
    'horizon_discount': 0.9,

    'belief_max_delta': 400.0,
    'belief_min_std': 25.0,

    'integral_mode': 'sigma',

    'sigma_method': 'legacy5',
    'sigma_order': 3,

    'sample_chunk_size': 32,

    'use_action_selection': False,
    'min_credit_signal': 1e-6,

    'use_signal_confidence': True,
    'signal_weight_min': 0.1,
    'signal_weight_max': 3.0,

    'eps': 1e-8,
}
```

## LSTM + Gauss-Hermite experiment

```python
'focus': {
    'enabled': True,
    'use_env_params': True,

    'alpha_credit': 0.05,
    'beta_belief': 0.01,

    'belief_mode': 'learned',
    'belief_arch': 'lstm',
    'belief_hidden_dim': 256,
    'belief_num_layers': 1,
    'belief_dropout': 0.0,

    'horizon': 3,
    'horizon_discount': 0.9,

    'belief_max_delta': 400.0,
    'belief_min_std': 25.0,

    'integral_mode': 'sigma',

    'sigma_method': 'gauss_hermite',
    'sigma_order': 5,

    'sample_chunk_size': 32,

    'use_action_selection': False,
    'min_credit_signal': 1e-6,

    'use_signal_confidence': True,
    'signal_weight_min': 0.1,
    'signal_weight_max': 3.0,

    'eps': 1e-8,
}
```

---

# 47. Final priority

Ngày mai ưu tiên theo đúng thứ tự:

```text
[1] Preserve old behavior
        ↓
[2] LSTM belief model
        ↓
[3] Weighted sigma refactor
        ↓
[4] Gauss-Hermite 9/25 points
        ↓
[5] Short ablation
        ↓
[6] Combined LSTM + GH
```

**Chưa làm full covariance và U-Net cho đến khi biết hai thay đổi trên có signal.**

Nếu LSTM + GH vẫn không cải thiện performance, lúc đó nên quay lại kiểm tra:

```text
belief quality
→ rho quality
→ focus credit target
→ mixer alignment
```

thay vì tiếp tục tăng architecture size.

---

# 48. BỔ SUNG — Roadmap đầy đủ cho MC/QMC và Grid

Phần trước tập trung vào LSTM + sigma. Tuy nhiên `_FOCUS` hiện có ba integral modes:

```text
integral_mode = "MC"
integral_mode = "sigma"
integral_mode = "grid"
```

Ba nhánh này nên được coi là **ba approximation families khác nhau**, không phải chỉ là ba giá trị config.

Roadmap đầy đủ:

```text
                    Belief model
                        │
             ┌──────────┴──────────┐
             │                     │
       Gaussian belief        Grid belief
        (mu, std)               heatmap
             │                     │
      ┌──────┼──────┐              │
      │      │      │              │
   sigma    QMC   Gaussian-grid    learned-grid
      │      │      │              │
      └──────┴──────┴───────┬──────┘
                            │
                     geometry integral
                            │
                            ↓
                    responsibility rho
```

Vì vậy nên tách upgrade thành:

1. **Temporal belief upgrade**: MLP → LSTM/GRU.
2. **Sigma integration upgrade**: legacy-5 → weighted Gauss–Hermite.
3. **MC/QMC upgrade**: Sobol hiện tại → better QMC machinery.
4. **Grid integration upgrade**: fixed dense grid → efficient/adaptive grid.
5. **Grid belief upgrade**: Gaussian rasterization → learned heatmap / U-Net.
6. **Full covariance**: Gaussian axis-aligned → rotated Gaussian.

Các bước 1–4 là incremental. Bước 5 là thay đổi representation lớn nhất.

---

# 49. Current MC mode thực chất là Sobol-QMC

Tên config hiện tại là:

```python
'integral_mode': 'MC',
'mc_num_points': 128,
'mc_chunk_size': 32,
'mc_seed': 0,
```

Nhưng implementation hiện tại dùng:

```python
engine = torch.quasirandom.SobolEngine(
    dimension=2,
    scramble=True,
    seed=seed,
)

uniforms = engine.draw(num_points)

normals = (
    np.sqrt(2.0)
    * torch.erfinv(2.0 * uniforms - 1.0)
)
```

Pipeline thực tế:

```text
Sobol points in [0,1]^2
        ↓
inverse Gaussian CDF
        ↓
quasi-random standard-normal samples
        ↓
mu + std * z
```

Tức:

\[
u_m\in[0,1]^2,
\qquad
z_m=\Phi^{-1}(u_m),
\qquad
x_m=\mu+\sigma\odot z_m.
\]

Vì vậy về mặt paper nên gọi chính xác hơn là **Sobol-QMC**.

Không cần đổi config key `"MC"` ngay vì có thể phá scripts cũ.

Có thể giữ:

```python
integral_mode = "MC"
```

và thêm comment:

```python
# "MC" currently uses scrambled Sobol QMC.
```

---

# 50. Current MC/QMC pipeline

Hiện tại:

```python
def _credit_from_mc_points(...):
    num_points = int(
        self.focus_config.get("mc_num_points", 128)
    )

    normals = self._mc_normals(
        num_points,
        ...
    )

    for h in range(mean.size(2)):
        samples = (
            mean_h.unsqueeze(4)
            + std_h.unsqueeze(4)
            * normal_chunk
        )

        chunk_sum = self._credit_chunk_sum(...)

        g = (
            g
            + horizon_weights[h]
            * chunk_sum
            / float(num_points)
        )
```

Tất cả QMC samples có equal weight:

\[
w_m=\frac1M.
\]

Điều này hợp lý vì points được transform để follow Gaussian belief.

---

# 51. MC/QMC UPGRADE A — Power-of-two Sobol budgets

Nên ưu tiên:

\[
M=2^k.
\]

Current:

```python
mc_num_points = 128
```

đã là:

\[
128=2^7.
\]

Khi sweep, dùng:

```text
32
64
128
256
512
1024
```

thay vì các số tùy ý.

Helper:

```python
def _is_power_of_two(n):
    return n > 0 and (n & (n - 1)) == 0
```

Trong `_mc_normals()`:

```python
if not _is_power_of_two(num_points):
    logger.warning(
        "Sobol QMC is usually best used with "
        f"a power-of-two point count, got {num_points}."
    )
```

Optional strict version:

```python
order = int(np.log2(num_points))

if 2 ** order != num_points:
    raise ValueError(...)

uniforms = engine.draw_base2(order)
```

Để backward-compatible, nên warning trước, chưa raise.

---

# 52. MC/QMC UPGRADE B — Cache Sobol normal points

Current `_mc_normals()` tạo lại `SobolEngine` và draw points mỗi call.

Nếu:

```text
num_points
seed
device
dtype
```

không đổi thì normals cũng không đổi.

Trong `QPLEXFocusLoss.__init__`:

```python
self._mc_normal_cache = {}
```

Helper:

```python
def _mc_cache_key(
    self,
    num_points,
    device,
    dtype,
    seed,
):
    return (
        int(num_points),
        str(device),
        str(dtype),
        int(seed),
    )
```

Sửa `_mc_normals()`:

```python
def _mc_normals(
    self,
    num_points,
    device,
    dtype,
    seed=None,
    eps=None,
):
    if eps is None:
        eps = self.focus_config.get("eps", 1e-8)

    if seed is None:
        seed = int(
            self.focus_config.get("mc_seed", 0)
        )

    use_cache = bool(
        self.focus_config.get(
            "mc_cache_points",
            True,
        )
    )

    key = self._mc_cache_key(
        num_points,
        device,
        dtype,
        seed,
    )

    if use_cache and key in self._mc_normal_cache:
        return self._mc_normal_cache[key]

    engine = torch.quasirandom.SobolEngine(
        dimension=2,
        scramble=True,
        seed=seed,
    )

    uniforms = engine.draw(num_points).to(
        device=device,
        dtype=dtype,
    )

    uniforms = uniforms.clamp(
        min=eps,
        max=1.0 - eps,
    )

    normals = (
        np.sqrt(2.0)
        * torch.erfinv(
            2.0 * uniforms - 1.0
        )
    )

    if use_cache:
        self._mc_normal_cache[key] = normals

    return normals
```

Config:

```python
'mc_cache_points': True,
```

Đây là compute-only upgrade. Mathematical result không nên đổi.

---

# 53. MC/QMC UPGRADE C — Antithetic points

Có thể force symmetry:

\[
z\leftrightarrow -z.
\]

Pipeline:

```text
Sobol half-samples
        ↓
Gaussian transform
        ↓
z1,...,zM/2
        ↓
concat(z, -z)
        ↓
M samples
```

Code:

```python
def _mc_normals_antithetic(
    self,
    num_points,
    device,
    dtype,
    seed,
    eps,
):
    if num_points % 2 != 0:
        raise ValueError(
            "Antithetic QMC requires an even number of points."
        )

    half = num_points // 2

    base = self._mc_normals(
        half,
        device,
        dtype,
        seed,
        eps,
    )

    return torch.cat(
        [base, -base],
        dim=0,
    )
```

Config:

```python
'mc_antithetic': False,
```

Dispatcher:

```python
if self.focus_config.get(
    "mc_antithetic",
    False,
):
    normals = self._mc_normals_antithetic(...)
else:
    normals = self._mc_normals(...)
```

Không mặc định bật. FoV visibility là discontinuous integrand nên antithetic cần ablation.

---

# 54. MC/QMC UPGRADE D — Multiple scrambled Sobol replicates

Current:

```python
mc_seed = 0
```

nghĩa là dùng một scrambled Sobol design cố định.

Có thể thêm:

```python
'mc_num_scrambles': 1,
```

và evaluate:

```text
seed
seed + 1
seed + 2
...
```

Mỗi scramble cho:

\[
\hat I_r.
\]

Final estimate:

\[
\hat I
=
\frac1R
\sum_{r=1}^{R}
\hat I_r.
\]

Diagnostic uncertainty:

\[
s_I
=
\operatorname{Std}
(\hat I_1,\ldots,\hat I_R).
\]

Recommended usage đầu tiên:

```text
R = 4 or 8
```

trên held-out batches, không phải full training.

Nếu QMC-128 có rất ít variation giữa scrambles, tăng points có thể không còn giá trị.

---

# 55. MC/QMC UPGRADE E — Convergence benchmark

Tạo offline utility:

```python
benchmark_qmc_convergence(...)
```

Test:

```text
M = 16
32
64
128
256
512
```

Reference:

```text
QMC-8192
```

hoặc dense grid reference.

Error:

\[
E_M
=
\|
\rho_M-\rho_{\rm ref}
\|_1.
\]

Nếu:

```text
QMC64 ≈ QMC128 ≈ QMC256
```

thì current integration đã gần converged.

Lúc đó performance bottleneck không nằm ở sample count.

---

# 56. MC/QMC UPGRADE F — Adaptive sample budget

Advanced, không phải MVP đầu tiên.

Nếu Gaussian belief hoàn toàn inside/outside FoV thì integral dễ.

Nếu FoV boundary cắt Gaussian thì khó.

Có thể:

```text
M_low  = 32
M_high = 128
```

Bước 1: estimate với 32 points.

Bước 2: nếu visibility fraction gần 0 hoặc 1 thì stop.

Nếu:

\[
\tau
<
\hat p_{\rm vis}
<
1-\tau
\]

thì refine.

Ví dụ:

```python
tau = 0.1
```

Nhược điểm: current code vectorize cùng sample count cho toàn:

```text
B × T × H × J
```

nên adaptive count theo từng target/horizon sẽ phức tạp.

Để phase sau.

---

# 57. MC/QMC UPGRADE G — Boundary-aware importance sampling

Advanced research extension.

Current samples:

\[
x\sim p(x).
\]

Nhưng numerical difficulty tập trung gần:

- angular FoV boundary,
- range boundary,
- obstacle boundary.

Có thể proposal:

\[
q(x)
=
\lambda p(x)
+
(1-\lambda)
q_{\rm boundary}(x).
\]

Importance weight:

\[
w(x)
=
\frac{p(x)}{q(x)}.
\]

Integral:

\[
I
=
\mathbb E_q
\left[
w(x)f(x)
\right].
\]

Không implement trong MVP vì:

- multiple cameras,
- multiple boundaries,
- dễ variance cao,
- dễ bug weights.

Chỉ làm nếu benchmark chứng minh integration error là bottleneck.

---

# 58. Recommended MC/QMC configs

Current-compatible:

```python
'integral_mode': 'MC',
'mc_num_points': 128,
'mc_chunk_size': 32,
'mc_seed': 0,
'mc_cache_points': True,
'mc_antithetic': False,
'mc_num_scrambles': 1,
```

Budget sweep:

```text
32
64
128
256
```

Ablation:

```text
QMC128 normal
QMC128 antithetic
```

Không làm adaptive QMC trước khi biết fixed-budget convergence.

---

# 59. MC/QMC + LSTM

LSTM hoàn toàn tương thích:

```text
state history
     ↓
    LSTM
     ↓
(mu, std)
     ↓
Sobol-QMC
     ↓
visibility
     ↓
rho
```

Ablation:

```text
MLP  + QMC128
LSTM + QMC128
```

Nếu LSTM tăng performance ở sigma, QMC và grid cùng lúc thì evidence rất mạnh rằng **belief model** là bottleneck.

---

# 60. GRID — Current implementation thực sự làm gì?

Current grid mode **không có learned grid heatmap**.

Nó vẫn dùng Gaussian belief:

```text
MLP
 ↓
(mu, std)
 ↓
evaluate Gaussian density on fixed grid
 ↓
normalize grid mass
 ↓
visibility integral
```

`_gaussian_grid_logits(mean, std, grid)` tính Gaussian log-density theo cell centers.

Do đó current grid là:

> **Dense numerical integration of the same Gaussian belief.**

Nó chưa phải U-Net belief prediction.

---

# 61. Current grid sample count

Current:

```python
grid_size = 64
```

tạo:

\[
64\times64=4096
\]

cells.

General:

\[
N_{\rm grid}=G^2.
\]

| `grid_size` | cells |
|---:|---:|
| 8 | 64 |
| 16 | 256 |
| 24 | 576 |
| 32 | 1024 |
| 48 | 2304 |
| 64 | 4096 |
| 96 | 9216 |
| 128 | 16384 |

Current ranges:

```python
grid_x_range = (-1000.0, 1000.0)
grid_y_range = (-1000.0, 1000.0)
```

Range này phù hợp với scenario MATE hiện tại, nhưng khi đổi environment config phải audit lại.

---

# 62. GRID UPGRADE A — Resolution sweep trước architecture

Trước U-Net, sweep current Gaussian-grid:

```text
G = 8
16
32
64
```

Tức:

```text
64
256
1024
4096
```

cells.

Compare với:

```text
sigma5
GH9
GH25
QMC32
QMC64
QMC128
```

Plot:

```text
rho approximation error
vs.
number of visibility evaluations
```

Nếu grid32 ≈ grid64 thì grid64 đang lãng phí compute.

---

# 63. GRID UPGRADE B — Separate x/y resolution

Current square:

```python
grid_size
```

Có thể hỗ trợ:

```python
grid_size_x = 64
grid_size_y = 64
```

Fallback:

```python
grid_size_x = int(
    self.focus_config.get(
        "grid_size_x",
        self.focus_config.get("grid_size", 64),
    )
)

grid_size_y = int(
    self.focus_config.get(
        "grid_size_y",
        self.focus_config.get("grid_size", 64),
    )
)
```

Không bắt buộc cho MATE square domain nhưng general hơn.

---

# 64. GRID UPGRADE C — Audit finite-domain renormalization

Current grid normalize Gaussian chỉ trên finite grid.

Approximation tương đương gần:

\[
p_{\rm grid}(x)
\propto
p(x)
\mathbf 1[
x\in\Omega_{\rm grid}
].
\]

Nếu environment thật sự bounded đúng bằng grid range thì hợp lý.

Nếu grid range nhỏ hơn physical domain thì sẽ distort belief.

Nên thêm diagnostic:

\[
m_{\rm outside}
=
1-
P(
x\in\Omega_{\rm grid}
).
\]

Không cần dùng trong training.

Chỉ cần log khi debug:

```text
focus_grid_outside_mass
```

Nếu outside mass thường > 1–5%, audit range.

---

# 65. GRID UPGRADE D — Cache fixed grid coordinates

`_grid_points()` tạo lại:

```python
torch.linspace(...)
torch.meshgrid(...)
```

mỗi call.

Grid cố định theo:

```text
size
range
device
dtype
```

nên cache được.

Trong `__init__`:

```python
self._grid_cache = {}
```

Cache key:

```python
(
    grid_size_x,
    grid_size_y,
    float(x_min),
    float(x_max),
    float(y_min),
    float(y_max),
    str(device),
    str(dtype),
)
```

Đây là compute-only optimization.

---

# 66. GRID UPGRADE E — Local belief-centered grid

Thay global grid bằng:

\[
x\in[
\mu_x-k\sigma_x,
\mu_x+k\sigma_x
]
\]

\[
y\in[
\mu_y-k\sigma_y,
\mu_y+k\sigma_y
].
\]

Ví dụ:

```python
grid_sigma_extent = 3.0
```

cover khoảng ±3σ.

Ưu:

- resolution tập trung quanh belief,
- ít cell ở vùng mass ≈ 0.

Nhược:

- mỗi `B,T,H,J` có grid riêng,
- vectorization khó hơn,
- bản chất gần deterministic quadrature.

Không ưu tiên hơn Gauss–Hermite ở MVP.

---

# 67. GRID UPGRADE F — Coarse-to-fine grid

Ý tưởng:

```text
coarse 16×16
     ↓
identify important cells
     ↓
refine only high-mass / boundary cells
```

Quan trọng:

Nếu cell sizes khác nhau thì mass phải tính:

\[
p(x_c)\Delta A_c.
\]

Không được softmax trực tiếp mọi cell center như uniform grid.

Đây là hướng compute-efficient nhưng advanced hơn.

---

# 68. GRID UPGRADE G — LSTM + Gaussian grid

Đây là upgrade rất dễ và nên test trước U-Net.

```text
state history
     ↓
    LSTM
     ↓
(mu, std)
     ↓
current Gaussian grid integral
     ↓
rho
```

Ablation:

```text
MLP  + grid32
LSTM + grid32
```

Nếu LSTM consistently thắng ở grid, sigma và QMC thì temporal belief là bottleneck.

---

# 69. GRID UPGRADE H — Learned grid belief

Đây là bước đổi representation:

Current:

```text
state
 ↓
MLP/LSTM
 ↓
Gaussian (mu,std)
 ↓
rasterize Gaussian
```

Upgrade:

```text
state/history
 ↓
spatial neural network
 ↓
arbitrary occupancy heatmap
```

Output:

\[
P_{t,h,j}
\in
\mathbb R^{G\times G},
\]

với:

\[
P[u,v]\ge0,
\qquad
\sum_{u,v}P[u,v]=1.
\]

Ưu điểm:

Gaussian chỉ biểu diễn unimodal elliptical uncertainty.

Learned grid có thể biểu diễn:

- asymmetric belief,
- multimodal belief,
- non-elliptical shape,
- obstacle-conditioned structure.

---

# 70. U-Net không phải drop-in replacement cho current grid

Current `_credit_from_grid()` nhận:

```python
mean, std
```

và tự gọi:

```python
_gaussian_grid_logits(...)
```

Nếu U-Net output heatmap, phải refactor để integration nhận trực tiếp occupancy mass.

Nên tách:

```python
_credit_from_gaussian_grid(...)
```

và:

```python
_credit_from_grid_mass(...)
```

hoặc generic:

```python
_credit_from_grid(
    occupancy_mass,
    grid,
    state,
    actions,
    n_targets,
)
```

---

# 71. Belief representation config

Thêm:

```python
'belief_representation': 'gaussian',
```

Values:

```text
gaussian
grid
```

Sau đó:

Gaussian branch:

```text
belief_arch = mlp
belief_arch = lstm
belief_arch = gru
```

Grid branch:

```text
belief_arch = conv_decoder
belief_arch = unet
belief_arch = convlstm_unet
```

Ví dụ:

```python
'belief_representation': 'gaussian',
'belief_arch': 'lstm',
```

hoặc:

```python
'belief_representation': 'grid',
'belief_arch': 'unet',
```

---

# 72. Learned-grid MVP trước U-Net — LSTM + convolutional decoder

Nếu muốn test dense heatmap nhưng chưa xây true U-Net:

```text
state sequence
     ↓
    LSTM
     ↓
 temporal latent
     ↓
small conv decoder
     ↓
G×G heatmap
```

Ví dụ:

```text
h_t
 ↓
Linear
 ↓
C × 8 × 8
 ↓
ConvTranspose
 ↓
C/2 × 16 × 16
 ↓
ConvTranspose
 ↓
H*J × 32 × 32
```

Đây **không phải U-Net**.

Nhưng nó trả lời một câu rất quan trọng:

> Bỏ Gaussian assumption và dự đoán dense distribution có giúp không?

Nếu không, chưa cần U-Net.

---

# 73. True U-Net cần spatial input

Nếu chỉ có:

```text
h_t ∈ R^d
```

thì không có spatial encoder feature maps để skip-connect.

Do đó:

```text
LSTM vector → upsampling decoder
```

không nên gọi là U-Net.

True U-Net input nên là:

\[
X_t
\in
\mathbb R^{C\times G\times G}.
\]

---

# 74. Rasterize state thành spatial channels

Có thể dùng:

Target channels:

```text
current target locations
```

Camera channels:

```text
camera positions
camera FoV masks
```

Obstacle channel:

```text
obstacle occupancy
```

Motion/history:

```text
target map t-3
target map t-2
target map t-1
target map t
```

Không cần rasterize mọi scalar feature ngay.

---

# 75. Temporal U-Net option 1 — Stack history as channels

MVP:

```text
Map(t-L+1)
Map(t-L+2)
...
Map(t)
       ↓
concatenate channels
       ↓
2D U-Net
       ↓
future heatmaps
```

Ví dụ:

```python
grid_history_len = 4
```

Ưu:

- dễ implement,
- không cần ConvLSTM,
- temporal information vẫn có.

Nhược:

- fixed history length,
- temporal dynamics implicit.

Đây nên là U-Net baseline đầu tiên.

---

# 76. Temporal U-Net option 2 — ConvLSTM + U-Net

Advanced:

```text
spatial map sequence
       ↓
    ConvLSTM
       ↓
temporal-spatial hidden map
       ↓
      U-Net
       ↓
future heatmaps
```

Hidden state:

\[
H_t
\in
\mathbb R^{C\times G\times G}.
\]

Conceptually:

\[
\boxed{
\text{temporal dynamics}
+
\text{spatial structure}
}
\]

Nhưng không nên là first implementation.

---

# 77. Learned-grid output shape

Ví dụ:

```text
H = 3
J = 5
G = 32
```

Output channels:

\[
H\times J=15.
\]

Raw:

```text
[B*T, 15, 32, 32]
```

reshape:

```text
[B,T,3,5,32,32]
```

Spatial softmax:

```python
logits = logits.view(
    B,
    T,
    H,
    J,
    G * G,
)

probs = torch.softmax(
    logits,
    dim=-1,
)

probs = probs.view(
    B,
    T,
    H,
    J,
    G,
    G,
)
```

---

# 78. Learned-grid loss option A — One-hot CE

Future target position:

\[
x^*_{j,t+h}
\]

map vào:

```text
(u*,v*)
```

Loss:

\[
L_{\rm grid}
=
-\log P[u^*,v^*].
\]

Đây là implementation đơn giản nhất.

---

# 79. Learned-grid loss option B — Gaussian-smoothed target

Recommended hơn one-hot.

Target distribution:

\[
Y[u,v]
\propto
\exp
\left(
-\frac{
\|x_{uv}-x^*\|^2
}{
2\sigma_{\rm label}^2
}
\right).
\]

Loss:

\[
L
=
-\sum_{u,v}
Y[u,v]\log P[u,v].
\]

Config:

```python
'grid_label_std': 40.0,
```

Dùng physical coordinate units.

Ưu:

- smoother gradients,
- neighboring cells không bị coi hoàn toàn sai,
- hợp continuous target position.

---

# 80. Refactor grid integration cho learned heatmap

Tạo:

```python
def _credit_from_grid_mass(
    self,
    occ,
    grid,
    state,
    actions,
    n_targets,
):
```

Shape:

```text
occ:
[B,T,H,J,P]

grid:
[P,2]
```

Invariant:

\[
\sum_p
occ_{b,t,h,j,p}
=
1.
\]

Flow:

```text
occupancy mass
    ×
grid visibility
    ×
unique coverage
    ↓
sum over grid
    ↓
horizon weighted sum
```

Generic backend này dùng được cho:

```text
Gaussian-grid
U-Net grid
conv-decoder grid
```

---

# 81. MEMORY WARNING — U-Net 64×64

Current:

```text
train_batch_size = 1024 env steps
H = 3
J ≈ 5
```

Nếu output 64×64:

\[
1024
\times
3
\times
5
\times
4096
=
62,914,560
\]

logits.

FP32 ≈ 252 MB chỉ riêng output.

Chưa tính:

- gradients,
- intermediate U-Net feature maps,
- QPLEX,
- optimizer states.

Do đó không bắt đầu U-Net ở 64×64.

Recommended:

```python
'grid_belief_size': 32,
'unet_base_channels': 16,
```

sau đó mới tăng nếu GPU memory cho phép.

---

# 82. U-Net MVP architecture

Small U-Net:

```text
Input
 C × 32 × 32
     │
 Conv 16
     │
 Downsample
     ↓
 32 × 16 × 16
     │
 Downsample
     ↓
 64 × 8 × 8
     │
 Upsample + skip
     ↓
 32 × 16 × 16
     │
 Upsample + skip
     ↓
 16 × 32 × 32
     │
 1×1 Conv
     ↓
(H × J) × 32 × 32
```

Không cần heavy ResNet U-Net ở MVP.

---

# 83. U-Net input MVP

Để tránh channel explosion:

```text
target maps from recent history
current camera FoV aggregate
current camera position aggregate
optional obstacle map
```

Có thể bắt đầu với:

```text
history target maps + current camera geometry
```

Không cần rasterize tất cả global-state scalar features.

---

# 84. Grid branch ablation ladder

Không nhảy:

```text
MLP Gaussian grid
→
ConvLSTM U-Net
```

Dùng:

### G0

```text
MLP → Gaussian → grid
```

current baseline.

### G1

```text
LSTM → Gaussian → grid
```

test temporal dynamics.

### G2

```text
LSTM → learned conv-decoder heatmap
```

test bỏ Gaussian assumption.

### G3

```text
history maps → U-Net → heatmap
```

test spatial encoder/skip connections.

### G4

```text
ConvLSTM → U-Net → heatmap
```

advanced.

Nếu G2 không hơn G1, chưa chắc U-Net đáng làm.

---

# 85. Compare three integral families công bằng

## Experiment family A — Same Gaussian belief

Giữ same `(mu,std)`:

```text
sigma legacy5
Gauss-Hermite
QMC
Gaussian-grid
```

Đây là **integration-method comparison**.

## Experiment family B — Same temporal architecture

Ví dụ LSTM Gaussian:

```text
LSTM + sigma
LSTM + QMC
LSTM + Gaussian-grid
```

Đây kiểm tra quadrature bottleneck.

## Experiment family C — Different representation

```text
LSTM Gaussian
vs.
learned-grid/U-Net
```

Đây là **belief-model comparison**, không còn chỉ là integral comparison.

---

# 86. Proposed complete config additions

```python
'focus': {
    'enabled': True,
    'belief_mode': 'learned',

    # Belief representation / architecture
    'belief_representation': 'gaussian',
    'belief_arch': 'mlp',
    'belief_hidden_dim': 256,
    'belief_num_layers': 1,
    'belief_dropout': 0.0,

    # Horizon
    'horizon': 3,
    'horizon_discount': 0.9,

    # Gaussian belief
    'belief_max_delta': 400.0,
    'belief_min_std': 25.0,

    # Integral
    'integral_mode': 'MC',

    # Sigma
    'sigma_method': 'legacy5',
    'sigma_order': 3,

    # MC / Sobol-QMC
    'mc_num_points': 128,
    'mc_chunk_size': 32,
    'mc_seed': 0,
    'mc_cache_points': True,
    'mc_antithetic': False,
    'mc_num_scrambles': 1,

    # Grid
    'grid_size': 64,
    'grid_chunk_size': 128,
    'grid_x_range': (-1000.0, 1000.0),
    'grid_y_range': (-1000.0, 1000.0),
    'grid_cache_points': True,

    # Learned-grid experimental branch
    'grid_belief_size': 32,
    'grid_history_len': 4,
    'grid_label_std': 40.0,
    'unet_base_channels': 16,

    # Existing credit/confidence
    'use_action_selection': False,
    'min_credit_signal': 1e-6,
    'use_signal_confidence': True,
    'signal_weight_min': 0.1,
    'signal_weight_max': 3.0,
    'eps': 1e-8,
}
```

Không cần implement mọi key trong một commit.

Đây là target schema.

---

# 87. Revised implementation phases

## PHASE A — Belief architecture

Implement:

```text
MLP
LSTM
```

Giữ Gaussian output.

Run:

```text
MLP  + sigma5
LSTM + sigma5

MLP  + QMC128
LSTM + QMC128

MLP  + grid32
LSTM + grid32
```

Mục tiêu:

> Temporal modeling có cải thiện xuyên integral modes không?

---

## PHASE B — Sigma

Implement:

```text
legacy5
GH3
GH5
GH7
```

Giữ MLP trước.

Mục tiêu:

> 5-point rule có quá coarse không?

---

## PHASE C — QMC

Implement an toàn:

```text
power-of-two warning
cache
optional antithetic
```

Run:

```text
QMC32
QMC64
QMC128
QMC256
```

Mục tiêu:

> QMC converged ở bao nhiêu points?

Không làm adaptive QMC trước khi biết answer.

---

## PHASE D — Grid

Run:

```text
grid16
grid32
grid64
```

Add:

```text
grid cache
domain audit
```

Mục tiêu:

> Dense grid có đáng compute không?

---

## PHASE E — Learned grid

Chỉ nếu Gaussian assumption bị nghi là bottleneck.

Implement trước:

```text
LSTM + conv decoder → grid heatmap
```

Sau đó:

```text
history-map U-Net
```

Cuối cùng mới:

```text
ConvLSTM + U-Net
```

---

# 88. Không chạy full Cartesian product

Không chạy:

```text
all belief models
×
all sigma orders
×
all QMC sizes
×
all grid sizes
```

Dùng staged screening.

---

# 89. Screening matrix đề xuất

## Stage 1 — Belief bottleneck

```text
MLP  + sigma5
LSTM + sigma5

MLP  + QMC128
LSTM + QMC128

MLP  + grid32
LSTM + grid32
```

6 configs.

Nếu LSTM không cải thiện belief diagnostics ở cả 3, dừng recurrent upgrade.

---

# 90. Stage 2 — Integration bottleneck

Dùng best belief architecture từ Stage 1.

Run:

```text
sigma5
GH9
GH25

QMC32
QMC64
QMC128
QMC256

grid16
grid32
grid64
```

Trước hết đo offline:

```text
rho approximation error
wall-clock
```

---

# 91. Stage 3 — End-to-end candidates

Chọn:

```text
best sigma
best QMC
best grid
```

rồi train end-to-end.

Ví dụ:

```text
LSTM + GH25
LSTM + QMC64
LSTM + grid32
```

Nếu performance tương đương, chọn cái có compute tốt nhất.

---

# 92. Stage 4 — Learned grid / U-Net

Chỉ khi có evidence:

```text
Gaussian belief error remains high
```

hoặc future belief rõ ràng multimodal/asymmetric.

Compare:

```text
best Gaussian method
vs.
LSTM conv-decoder grid
vs.
history U-Net grid
```

Nếu learned grid thắng, mới cân nhắc ConvLSTM-U-Net.

---

# 93. Held-out integration benchmark nên tạo ngay

Tạo utility benchmark.

Reference:

```text
QMC-8192
```

hoặc:

```text
grid-128
```

Candidates:

```text
sigma5
GH9
GH25
GH49
QMC32
QMC64
QMC128
grid16
grid32
grid64
```

Measure:

```text
rho L1 error
rho MSE
total_g error
runtime
point evaluations
```

Output table:

| Method | Points | rho L1 | runtime |
|---|---:|---:|---:|
| sigma5 | 5 | ... | ... |
| GH9 | 9 | ... | ... |
| GH25 | 25 | ... | ... |
| QMC32 | 32 | ... | ... |
| QMC64 | 64 | ... | ... |
| grid16 | 256 | ... | ... |
| grid32 | 1024 | ... | ... |

Nếu GH25 hoặc QMC64 đã gần reference, không cần grid64 chỉ vì chi tiết hơn.

---

# 94. Ideal accuracy-compute plot

```text
rho accuracy
↑
│            grid64
│        QMC128
│      QMC64
│   GH25
│ GH9
│sigma5
└──────────────────→ compute / point evaluations
```

Pareto frontier là thứ đáng quan tâm.

---

# 95. Belief benchmark phải tách khỏi integration benchmark

## Belief quality

Compare:

```text
MLP Gaussian
LSTM Gaussian
learned-grid
U-Net grid
```

Metrics:

```text
future position error
NLL
calibration
```

## Integration quality

Giữ belief cố định:

```text
sigma
GH
QMC
grid
```

Metrics:

```text
rho error
runtime
point evaluations
```

Đừng trộn hai câu hỏi.

---

# 96. Calibration metric nên thêm

Current có:

```text
position_error
pred_std
NLL
```

Thêm standardized residual:

\[
z
=
\frac{x^*-\mu}{\sigma}.
\]

Nếu Gaussian calibrated:

\[
E[z^2]
\]

nên ở gần 1 theo từng coordinate.

Log:

```text
focus_belief_z2_h1
focus_belief_z2_h2
focus_belief_z2_h3
```

Interpretation:

```text
z2 >> 1 → overconfident
z2 << 1 → std quá lớn / underconfident
```

Điều này cực quan trọng vì sigma, QMC và Gaussian-grid đều dùng `std`.

Nếu `std` sai, tăng integral points không sửa root cause.

---

# 97. Critical diagnostic chain

Debug theo:

```text
1. Future mean prediction accurate?
        ↓
2. Uncertainty calibrated?
        ↓
3. Numerical integration accurate?
        ↓
4. rho informative/non-uniform?
        ↓
5. Credit prior/mixer match rho?
        ↓
6. MARL reward improve?
```

Nếu fail 1:

```text
upgrade belief architecture
```

Nếu fail 2:

```text
upgrade uncertainty model / covariance
```

Nếu fail 3:

```text
upgrade sigma/QMC/grid
```

Nếu fail 4–5:

```text
problem lies in responsibility definition/coupling
```

Không tiếp tục tăng network size mù quáng.

---

# 98. Priority sửa code ngày mai — phiên bản đầy đủ

## Priority 1

Backward-compatible:

```text
belief_arch = mlp / lstm
```

Run parity.

## Priority 2

Sigma weighted refactor:

```text
legacy5
GH3
GH5
```

Unit tests.

## Priority 3

QMC cleanup:

```text
cache
power-of-two warning
optional antithetic
```

Không adaptive QMC.

## Priority 4

Grid cleanup:

```text
cache
grid16/32/64 sweep
domain audit
```

Không U-Net ngay.

## Priority 5

Offline integration benchmark:

```text
sigma
GH
QMC
grid
```

against high-resolution reference.

## Priority 6

Nếu còn thời gian:

```text
LSTM → conv decoder → 32×32 heatmap
```

Chưa cần true U-Net.

---

# 99. Sau screening, chọn hướng theo evidence

Nếu LSTM thắng rõ:

```text
keep LSTM
```

Nếu GH25 ≈ QMC128 nhưng rẻ hơn:

```text
prefer GH25
```

Nếu QMC64 tốt hơn GH nhưng vẫn rẻ:

```text
prefer QMC64
```

Nếu grid32 tốt hơn hẳn Gaussian quadrature:

```text
spatial boundary resolution matters
```

Nếu learned-grid >> Gaussian-grid:

```text
Gaussian belief assumption is bottleneck
```

Lúc đó U-Net có scientific motivation rõ.

---

# 100. Final architecture roadmap

```text
LEVEL 0 — CURRENT
MLP
 ↓
diagonal Gaussian
 ↓
sigma5 / Sobol-QMC / fixed grid

LEVEL 1 — TEMPORAL
LSTM
 ↓
diagonal Gaussian
 ↓
same three integrators

LEVEL 2 — BETTER NUMERICAL INTEGRATION
LSTM
 ↓
diagonal Gaussian
 ↓
GH / optimized QMC / optimized grid

LEVEL 3 — BETTER UNCERTAINTY GEOMETRY
LSTM
 ↓
full-covariance Gaussian
 ↓
GH / QMC

LEVEL 4 — NON-GAUSSIAN SPATIAL BELIEF
history maps / ConvLSTM
 ↓
conv decoder / U-Net
 ↓
learned grid heatmap
 ↓
grid geometry integral
```

Không nhảy Level 0 → Level 4 nếu chưa biết bottleneck.

---

# 101. Upgrade priority table

| Upgrade | Effort | Risk | Scientific value | Priority |
|---|---|---|---|---|
| LSTM Gaussian belief | Low–Medium | Low | High | **Very high** |
| Sigma GH9/GH25 | Low | Low | High | **Very high** |
| QMC cache | Very low | Very low | Compute only | High |
| QMC point sweep | Very low | Low | High diagnostic | **Very high** |
| QMC antithetic | Low | Low | Medium | Medium |
| Multi-scramble diagnostic | Medium | Low | High diagnostic | Medium |
| Grid resolution sweep | Very low | Low | High diagnostic | **Very high** |
| Grid cache | Very low | Very low | Compute only | High |
| Local/adaptive grid | Medium–High | Medium | Medium–High | Later |
| LSTM + conv grid decoder | Medium | Medium | High | After screening |
| U-Net grid belief | High | Medium–High | High if Gaussian fails | Later |
| ConvLSTM + U-Net | High | High | High but expensive | Much later |
| Full covariance Gaussian | Medium | Medium | High if directional | Later |
| Boundary importance sampling | High | High | Potentially high | Research extension |

---

# 102. Recommended minimum experimental story

Một story gọn nhưng mạnh:

```text
Current FOCUS
    ↓
Temporal belief upgrade:
MLP → LSTM
    ↓
Integration comparison:
legacy sigma / Gauss-Hermite / Sobol-QMC / grid
    ↓
Select best accuracy–compute tradeoff
```

Chỉ thêm U-Net nếu experiments cho thấy Gaussian representation không đủ.

Reviewer-facing framing:

> Chúng tôi tách responsibility estimation thành hai bottleneck độc lập: temporal belief quality và numerical integration quality; sau đó cải thiện từng bottleneck bằng controlled ablations.

Không framing thành:

> Chúng tôi thử nhiều neural architectures.
