# FOCUS và CTDE trong `mate`

## Phạm vi đã triển khai

Repo hiện có các learner FOCUS cho `QPLEX`, `QMIX` và `DuelMIX`. Phần dùng chung hiện được chuẩn hóa qua `FocusResponsibilityEngine` và `ValueDecompositionFocusAdapter`. `QMIX_FOCUS`/`VDN_FOCUS` dùng responsibility để scale per-agent monotonic mixing weights; `DuelMIX_FOCUS` chỉ thay allocation của advantage branch, còn unrestricted value branch được giữ nguyên.

Khi `rho` là uniform, scale của QMIX/VDN là `rho * N = 1`, vì vậy baseline được khôi phục. Responsibility invalid cũng fallback về uniform. Các learner vẫn dùng global state trong centralized training và chỉ dùng centralized action bias trong nhánh exploration được cấu hình.

## Policy-based methods

`MAPPO_FOCUS` đã có hai nhánh tách biệt. Centralized critic và occupancy/FOCUS supervision có thể nhận true normalized global state. Actor chỉ nhận local observation/history và belief dự đoán từ local features. `GlobalStateBelief` trong `ray/rllib/agents/focus_common/belief_state.py` là module dùng chung cho local-to-global regression.

`SMPE2_FOCUS` dùng variational decoder vốn đã được huấn luyện để tái tạo global state. Action-bias path mặc định hiện lấy state dự đoán từ decoder (`action_bias_state_source="belief"`). True state chỉ được dùng khi bật rõ `allow_privileged_action_bias=True`, do đó không còn accidental privileged action path trong decentralized execution.

`HMVFE` là trainer custom ngoài RLlib nên đã được nối trực tiếp ở `hmvfe_mate_d/`. `MATECoordinatorEnv.global_state()` cung cấp normalized privileged state riêng biệt với public discretized observation. Khi bật belief:

```text
public observation -> HMVFE trunk -> predicted global-state belief -> actor
true global state  -> centralized critic + belief target (training only)
```

Actor không nhận true global state trong `act()`/evaluation. Centralized critic có thể dùng state thật trong training; evaluation tự động fallback về local/belief path vì không truyền `critic_state`.

## Chạy HMVFE với belief CTDE

```bash
PYTHONPATH=. python3 -m hmvfe_mate_d \
  --belief-enabled \
  --total-env-steps 10000000 \
  --run-name hmvfe-belief \
  --wandb-mode disabled
```

Các cờ chính là `--belief-hidden-dim`, `--belief-loss-coeff`, `--belief-loss`, và `--no-critic-global-state`. Mặc định `belief_enabled=False` để giữ baseline cũ; `--belief-enabled` bật local belief actor, auxiliary state loss và centralized critic.

Checkpoint belief-enabled phải được evaluate với `--belief-enabled` để khởi tạo đúng head trước khi load:

```bash
PYTHONPATH=. python3 -m hmvfe_mate_d \
  --eval-only \
  --belief-enabled \
  --load hmvfe_mate_d/runs/hmvfe-belief/latest.pt
```

## Kiểm thử

Các kiểm thử mới kiểm tra ba invariant quan trọng: actor HMVFE không phụ thuộc privileged critic state, belief head nhận gradient từ auxiliary loss, và QMIX/VDN responsibility adapter bảo toàn uniform-responsibility baseline. Smoke run đã được xác nhận với HMVFE belief-enabled và evaluation-only trên checkpoint.

Do repo vendored một bản Ray cũ không có native `_raylet` trong sandbox hiện tại, các test phụ thuộc đầy đủ Ray cần chạy trong environment cài đúng Ray build của project. Syntax checker và các test HMVFE/adapter không cần native Ray build.
