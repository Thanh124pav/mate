# FOCUS Improvements

## Summary

This update turns the strongest FOCUS action-prior idea into CTDE-safe training support for value-based methods and MAPPO. Centralized state is allowed during training, but decentralized execution must not depend on centralized state.

The main validated result is QPLEX_FOCUS on `MATE-4v8-9.yaml` with replay buffer capacity 200: decentralized evaluation reached `61.503%` mean coverage at `49,697` timesteps.

## CTDE Rule Used

- Centralized training may use the global MATE state for critic targets, counterfactual responsibility, belief losses, and action-prior auxiliary losses.
- Decentralized execution must select actions from the per-agent/local policy only.
- For Q-learning policies, the FOCUS action prior is applied only when `explore=True`; evaluation uses `explore=False`, so the prior is skipped.
- For MAPPO, the FOCUS action prior is now a training-only auxiliary loss. The actor logits are not modified in `forward_rnn`, so inference remains local-observation based.

## Implemented Changes

### Shared FOCUS Geometry Prior

Added `focus_action_q_bias(...)` in `ray/rllib/agents/focus_utils.py`. It builds a camera-action geometry prior from centralized normalized MATE state, current camera FOV, and target positions. The helper supports `[B, N, A]` and `[B, T, N, A]` Q/logit-like tensors.

### Q-learning FOCUS

Applied the shared prior to:

- `ray/rllib/agents/qplex_focus/qplex_policy.py`
- `ray/rllib/agents/qmix_focus/qmix_policy.py`
- `ray/rllib/agents/duelmix_focus/duelmix_policy.py`

Training behavior:

- Optional prior is added to online exploratory Q-values.
- Optional prior is added to double-Q bootstrap argmax targets.
- Default config keeps `action_bias_eta=0.0`, so existing behavior is unchanged unless enabled.

Execution behavior:

- `compute_actions(..., explore=False)` skips centralized action bias.
- This keeps evaluation decentralized.

### QPLEX_FOCUS Config

Enabled the prior in `examples/qplex_focus/camera/config.py` with `action_bias_eta=3.0` plus geometry/action-grid settings.

### MAPPO_FOCUS Belief State And Optional FOCUS Prior

Added a local-observation belief state head in `examples/mappo/models.py`:

- `MAPPOModel.forward_rnn()` predicts `global_state_hat` from local recurrent actor features only.
- `MAPPOModel.custom_loss()` supervises that prediction against training-only ground-truth global state via `belief_state_coeff`.
- This belief head can run by itself with FOCUS action prior disabled, making it a clean belief-only ablation.
- FOCUS action prior is now optional via `action_prior_enabled`. When enabled, it uses `global_state_hat`, not ground-truth global state, for both forward-time logit bias and auxiliary distillation.
- The old MAPPO_FOCUS global-state dynamics belief loss is skipped when `beta_belief=0.0`, so the belief-only mode measures the new local-observation state reconstruction head cleanly.
- The prior loss remains advantage-weighted: positive-centered PPO advantages receive stronger distillation, while low/negative-advantage samples are down-weighted.
- Added stats to PPO learner output in `ray/rllib/agents/ppo/ppo_torch_policy.py` for `focus/belief_state_loss`, `focus/belief_state_mae`, and prior metrics.
- Enabled knobs in `examples/mappo_focus/camera/config.py`:
  - `beta_belief=0.0`
  - `belief_state_enabled=True`
  - `belief_state_coeff=0.05`
  - `action_prior_enabled=True`
  - `action_prior_state_source=belief`
  - `action_prior_apply_to_logits=True`
  - `action_prior_coeff=0.1`
  - `action_prior_bias_eta=3.0`
  - `action_prior_temperature=0.75`
  - `action_prior_use_positive_advantage=True`
  - `action_prior_center_advantage=True`
  - `action_prior_advantage_clip=5.0`
  - `action_prior_weight_floor=0.05`

Two MAPPO_FOCUS modes are available from the same entrypoint:

- Belief-only: `--focus-prior off` trains local `global_state_hat` reconstruction without FOCUS action prior.
- Belief+FOCUS: `--focus-prior on` trains and executes the FOCUS prior from predicted `global_state_hat`.

### SMPE2_FOCUS Decentralized Evaluation Agent

Added `examples/smpe2_focus/camera/agent.py` and exports in:

- `examples/smpe2_focus/__init__.py`
- `examples/smpe2_focus/camera/__init__.py`

`SMPE2FocusCameraAgent` loads old SMPE2_FOCUS checkpoints but defaults to `decentralized_execution=True`, which forces `action_bias_eta=0.0` at inference.

## Checkpoints Still Available

Policy-based checkpoints found and re-evaluated:

- SMPE2_FOCUS eta=3.0: `/tmp/smpe2_focus_bias30_real_direct_4v8_9_12k/checkpoints/checkpoint_000003/checkpoint-3`
- SMPE2_FOCUS eta=2.5: `/tmp/smpe2_focus_bias25_real_direct_4v8_9_24k/checkpoints/checkpoint_000010/checkpoint-10`
- TarMAC_FOCUS: `/tmp/tarmac_focus_real_direct_4v8_9_300k/checkpoints/checkpoint_000025/checkpoint-25`

Q-learning validated checkpoint:

- QPLEX_FOCUS CTDE: `/tmp/qplex_focus_ctde_ab30_4v8_9_120k_driver_eval/checkpoints/checkpoint_000011/checkpoint-11`

MAPPO_FOCUS CTDE-prior checkpoints:

- Smoke: `/tmp/mappo_focus_ctde_prior_4v8_9_7200/checkpoints/checkpoint_000003/checkpoint-3`
- Advantage-weighted prior 24k best: `/tmp/mappo_focus_advprior_4v8_9_24k/checkpoints/checkpoint_000005/checkpoint-5`
- Belief-only smoke: no checkpoint kept; direct `PPOTrainer` smoke emitted belief metrics.
- Belief+FOCUS predicted-state smoke: no checkpoint kept; direct `PPOTrainer` smoke emitted belief and prior metrics.

## Results

### QPLEX_FOCUS CTDE Run

Config: `MATE-4v8-9.yaml`, buffer capacity `200`, requested cap `120k` timesteps, stopped early after hitting target.

- Timesteps: `49,697`
- Train `real_coverage_rate_mean`: `0.579692`
- Decentralized eval `real_coverage_rate_mean`: `0.615034`
- Decentralized eval reward mean: `3021.5607`
- History: `/tmp/qplex_focus_ctde_ab30_4v8_9_120k_driver_eval/history.jsonl`
- Summary: `/tmp/qplex_focus_ctde_ab30_4v8_9_120k_driver_eval/summary.json`

### Policy-based Checkpoint Re-evaluation

Re-evaluated with `python -m mate.evaluate --no-render --episodes 3 --seed 0 --config MATE-4v8-9.yaml`. For SMPE2_FOCUS, execution forced `decentralized_execution=true`, disabling action bias at inference.

| Algorithm checkpoint | Training metric previously observed | Decentralized eval Mean Coverage Rate | Notes |
| --- | ---: | ---: | --- |
| SMPE2_FOCUS eta=3.0 checkpoint 3 | `real_coverage_rate_mean=0.600866` | `26.366%` | High training score depended on centralized action bias. |
| SMPE2_FOCUS eta=2.5 checkpoint 10 | `real_coverage_rate_mean=0.585156` | `29.967%` | Also drops under local-only execution. |
| TarMAC_FOCUS checkpoint 25 | `real_coverage_rate_mean=0.325308` | `38.892%` | Decentralized agent path works, but still below target. |

Logs are under `/tmp/policy_based_decentralized_eval/`.

### MAPPO_FOCUS CTDE-prior Runs

Smoke config: `MATE-4v8-9.yaml`, reward coefficient `real_coverage_rate`, 7200 timesteps.

- Iteration 1 train `real_coverage_rate_mean`: `0.239325`
- Iteration 2 train `real_coverage_rate_mean`: `0.259405`
- Iteration 3 train `real_coverage_rate_mean`: `0.272896`
- Decentralized `mate.evaluate` 3-episode Mean Coverage Rate: `28.634%`
- History: `/tmp/mappo_focus_ctde_prior_4v8_9_7200/history.jsonl`

Advantage-weighted prior config: `MATE-4v8-9.yaml`, 24k timesteps, best checkpoint selected by train `real_coverage_rate_mean`.

- Best train timestep: `12,000`
- Best train `real_coverage_rate_mean`: `0.305086`
- Best train episode reward mean: `1465.9870`
- Final train timestep: `24,000`
- Final train `real_coverage_rate_mean`: `0.274902`
- Decentralized `mate.evaluate` 3-episode Mean Coverage Rate: `27.655%`
- Best checkpoint: `/tmp/mappo_focus_advprior_4v8_9_24k/checkpoints/checkpoint_000005/checkpoint-5`
- History: `/tmp/mappo_focus_advprior_4v8_9_24k/history.jsonl`
- Summary: `/tmp/mappo_focus_advprior_4v8_9_24k/summary.json`

The advantage-weighted distillation code path is CTDE-safe and trains without error, but this short MAPPO_FOCUS run did not close the gap to the 60% target. Current evidence says the QPLEX_FOCUS route is the strong result; MAPPO needs more algorithmic work or substantially longer tuning.

Belief-head split smoke checks:

- Belief-only direct `PPOTrainer` smoke: `focus/belief_state_loss=1531.3077`, `focus/belief_state_mae=10.4976`, action-prior stats absent as expected.
- Belief+FOCUS predicted-state direct `PPOTrainer` smoke: `focus/belief_state_loss=1609.5206`, `focus/belief_state_mae=10.7366`, `focus/action_prior_loss=2.6471`, `focus/action_prior_weight_mean=0.3410`.

Belief-head 12k ablation runs, fast local-only `PPOTrainer` setup with old dynamics belief disabled:

| Mode | Best train timestep | Best train `real_coverage_rate_mean` | Decentralized eval Mean Coverage Rate | Notes |
| --- | ---: | ---: | ---: | --- |
| Belief+FOCUS predicted prior | `2,400` | `0.306666` | `21.984%` | Early train spike, then collapsed to `0.246373` by 12k. |
| Belief-only | `12,000` | `0.297551` | `36.859%` | Better decentralized eval than prior MAPPO_FOCUS runs, but still far below 60%. |

Artifacts:

- Belief+FOCUS history: `/tmp/mappo_focus_belief_prior_fast_4v8_9_12k/history.jsonl`
- Belief+FOCUS best checkpoint: `/tmp/mappo_focus_belief_prior_fast_4v8_9_12k/checkpoints/checkpoint_000002/checkpoint-2`
- Belief-only history: `/tmp/mappo_focus_belief_only_fast_4v8_9_12k/history.jsonl`
- Belief-only best checkpoint: `/tmp/mappo_focus_belief_only_fast_4v8_9_12k/checkpoints/checkpoint_000010/checkpoint-10`

## Verification

Executed checks:

- `python3 -m py_compile` on modified MAPPO, SMPE2_FOCUS agent, PPO, and Q-learning policy/config files.
- MAPPO_FOCUS local smoke train with `focus/action_prior_loss` present in learner stats.
- QPLEX_FOCUS smoke train/eval and full CTDE run until decentralized eval exceeded 60%.
- `git diff --check`.

## How To Run

QPLEX_FOCUS CTDE training:

```bash
cd /home/pavt1024/vsn/mate
source ~/miniconda3/etc/profile.d/conda.sh
conda activate mate
python -m examples.qplex_focus.camera.train \
  --project mate-camera \
  --config MATE-4v8-9.yaml \
  --timesteps-total 120000 \
  --buffer-capacity 200 \
  --seed 0
```

MAPPO_FOCUS belief-only training:

```bash
cd /home/pavt1024/vsn/mate
source ~/miniconda3/etc/profile.d/conda.sh
conda activate mate
python -m examples.mappo_focus.camera.train \
  --project mate-camera \
  --timesteps-total 120000 \
  --focus-prior off \
  --seed 0
```

MAPPO_FOCUS belief+FOCUS-prior training:

```bash
cd /home/pavt1024/vsn/mate
source ~/miniconda3/etc/profile.d/conda.sh
conda activate mate
python -m examples.mappo_focus.camera.train \
  --project mate-camera \
  --timesteps-total 120000 \
  --focus-prior on \
  --seed 0
```

SMPE2_FOCUS decentralized checkpoint evaluation:

```bash
cd /home/pavt1024/vsn/mate
source ~/miniconda3/etc/profile.d/conda.sh
conda activate mate
python -m mate.evaluate --no-render --episodes 3 --seed 0 \
  --config MATE-4v8-9.yaml \
  --camera-agent examples.smpe2_focus:SMPE2FocusCameraAgent \
  --camera-kwargs '{"checkpoint_path":"/tmp/smpe2_focus_bias30_real_direct_4v8_9_12k/checkpoints/checkpoint_000003/checkpoint-3","decentralized_execution":true}'
```
