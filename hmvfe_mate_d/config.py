"""Configuration and environment factory for HMVFE on MATE.

The factory builds the *same* wrapper stack the HRL camera baselines use
(``examples/hrl/*/camera/config.py``) -- ``MultiCamera`` ->
``RepeatedRewardIndividualDone`` -> ``AuxiliaryCameraRewards`` -- then hands it to
:class:`MATECoordinatorEnv`, which folds in the (vendored) ``HierarchicalCamera``
frame-skip + geometric executor. This keeps the task (env config, opponent,
reward, frame-skip, executor, horizon) identical across algorithms; only the
learner differs.

HMVFE replaces HiT-MAC's attention orchestrator with an FM + Mixture-of-Experts
embedding (Son et al., JNCA 2026). Fairness invariants (kept from
``hit_mac/``): MATE-4v8-9, dense reward, ``GreedyTargetAgent(seed=0)``,
``coverage_rate`` reward (mean), ``frame_skip=5`` + horizon 500 (= 100 macro
decisions/episode), the FROZEN geometric ``track()`` executor (HMVFE's learned FM
worker is intentionally dropped -- training it would be unfair vs the other HRL
algos), a ``MultiBinary(N_cam * N_tgt)`` selection action, and a ~10M env-step
budget.

Variant B (MATE-extended input): on top of HMVFE's native fields (camera id,
target id, distance bin, angle bin, visibility), the coordinator also consumes a
**cargo** field and an **occlusion** field, so it sees the same MATE-specific
signal (obstacle occlusion of the camera->target ray + target cargo) that the
HiT-MAC MATE integration feeds its coordinator -- the fairest information-parity
comparison and the best on-MATE performance. The FM+MoE absorbs the extra
categorical fields natively. All fields are built only from camera observations +
visibility masks (partial observability); this IS the algorithm.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import mate

from hmvfe_mate_d.envs import MATECoordinatorEnv


__all__ = ['HMVFEConfig', 'make_env', 'make_base_env', 'make_output_dir']


@dataclass
class HMVFEConfig:
    # --- environment (kept identical to the HRL camera benchmark) -------------
    env_id: str = 'MultiAgentTracking-v0'
    env_config: str = 'MATE-4v8-9.yaml'
    reward_type: str = 'dense'
    opponent: str = 'greedy'
    frame_skip: int = 5
    horizon: int = 500
    coverage_coefficient: float = 1.0  # AuxiliaryCameraRewards coefficient on coverage_rate
    seed: int = 0

    # --- observation discretisation (HMVFE embedding fields) ------------------
    # Continuous distance/bearing/occlusion are binned into categorical fields that
    # index a shared look-up embedding table. Bin counts derived from MATE-4v8-9
    # geometry (TERRAIN_SIZE 1000, camera max_sight_range 1500, 9 obstacles,
    # FoV up to 180 deg). Variant B adds the occlusion + cargo fields (7 total).
    num_distance_bins: int = 16
    num_angle_bins: int = 16
    num_occlusion_bins: int = 8       # obstacle occlusion of the camera->target ray

    # --- HMVFE coordinator network (paper-optimal: Table 3 + Table 9 ablation) -
    embedding_dim: int = 10           # look-up embedding size d
    num_experts: int = 4             # total MoE experts     (Table 9 best)
    top_k: int = 2                   # active experts / pair (Table 9 best)
    gating_hidden: int = 128         # MoE gating network hidden units (Table 3)
    mlp_hidden: int = 128            # expert MLP hidden units          (Table 3)
    mlp_layers: int = 2              # expert MLP hidden layers          (Table 3)

    # --- Tier B: state-value critic reduction ---------------------------------
    # 'max' = paper default (parameter-free max over interaction scores z, coupled
    # to the actor); 'mean' = param-free mean; 'learned' = a separate value head on
    # the shared trunk (mean-pooled reweighted embeddings), DECOUPLED from the
    # actor's per-pair scores -> lower-variance advantage baseline. Paper reports
    # 'max' best on its DSN env; this variant tests that choice on MATE.
    critic_reduction: str = 'learned'   # max | mean | learned
    value_head_hidden: int = 128        # hidden units of the learned value head

    # --- CTDE belief state ----------------------------------------------------
    # The actor sees only the discretized local observation and its predicted
    # belief.  The true normalized state is reserved for the centralized critic
    # and the auxiliary supervision target.
    belief_enabled: bool = False
    belief_hidden_dim: int = 128
    belief_loss_coeff: float = 0.05
    belief_loss: str = 'smooth_l1'      # smooth_l1 | mse
    critic_use_global_state: bool = True

    # --- A2C / GAE training ---------------------------------------------------

    total_env_steps: int = 10_000_000
    num_envs: int = 8                 # HMVFE's critic is a PARAMETER-FREE max over interaction
                                      # scores (Sec. 4.1.3), so -- unlike HiT-MAC's Shapley critic,
                                      # whose ~1e3 gradient forced num_envs=1 -- there is no
                                      # grad-domination. HMVFE is a normal-gradient learner, so a
                                      # larger, decorrelated batch improves the gradient direction
                                      # (the paper itself used 10-40 async A3C workers). 8 sync envs.
    rollout_length: int = 20          # coordinator steps per update (paper update frequency = 20)
    gamma: float = 0.99               # kept at the MATE-baseline value for cross-algorithm
                                      # comparability (the paper used 0.9 on its own env).
    gae_lambda: float = 0.95          # <1 trades a little bias for much lower variance
    learning_rate: float = 5e-4       # paper Table 3
    anneal_lr: bool = True            # linearly decay LR to 0 over training (tames the late drop)
    entropy_coef: float = 0.01        # paper Table 3
    value_coef: float = 0.5
    max_grad_norm: float = 50.0       # loose safety clip: with the parameter-free max critic the
                                      # gradient is normal-scale, so this rarely binds (we WANT the
                                      # true gradient direction -- opposite of HiT-MAC, where the
                                      # clip was load-bearing).
    normalize_advantage: bool = True
    device: str = 'cpu'

    # --- logging / checkpointing / evaluation ---------------------------------
    log_interval: int = 10            # in updates
    save_interval: int = 50           # in updates
    eval_interval: int = 50           # in updates
    eval_episodes: int = 5
    output_dir: str = 'hmvfe_mate_d/runs'
    run_name: Optional[str] = None

    # --- Weights & Biases -----------------------------------------------------
    wandb_project: Optional[str] = None
    wandb_group: Optional[str] = None
    wandb_name: Optional[str] = None
    wandb_mode: str = 'online'        # online | offline | disabled
    wandb_tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def make_target_agent(name: str):
    name = str(name).lower()
    if name == 'greedy':
        return mate.GreedyTargetAgent(seed=0)
    if name == 'heuristic':
        return mate.HeuristicTargetAgent(seed=0)
    if name == 'random':
        return mate.RandomTargetAgent(seed=0)
    raise ValueError(f'Unknown opponent {name!r} (expected greedy|heuristic|random).')


def make_base_env(config: HMVFEConfig):
    """Build the camera-team stack up to AuxiliaryCameraRewards."""

    env = mate.make(config.env_id, config=config.env_config, reward_type=config.reward_type)
    env = mate.MultiCamera(env, target_agent=make_target_agent(config.opponent))
    env = mate.RepeatedRewardIndividualDone(env)
    env = mate.AuxiliaryCameraRewards(
        env,
        coefficients={'coverage_rate': float(config.coverage_coefficient)},
        reduction='mean',
    )
    env.seed(config.seed)
    return env


def make_env(config: HMVFEConfig) -> MATECoordinatorEnv:
    env = make_base_env(config)
    return MATECoordinatorEnv(
        env,
        frame_skip=config.frame_skip,
        horizon=config.horizon,
        num_distance_bins=config.num_distance_bins,
        num_angle_bins=config.num_angle_bins,
        num_occlusion_bins=config.num_occlusion_bins,
    )


def make_output_dir(config: HMVFEConfig) -> Path:
    name = config.run_name or 'hmvfe'
    path = (Path(config.output_dir).expanduser() / name).resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path
