"""HiTMAC — Phase 1: Config train QPLEX executors với greedy/heuristic coordinator.

Theo paper HiTMAC (NeurIPS 2020, Sec 3.3):
  - Phase 1: executor train độc lập với pseudo-goal heuristic (greedy assignment)
  - Phase 2: coordinator train với executor đã fix (dùng code hrl/mappo)

Wrapper chain (Phase 1):
    mate.DiscreteCamera(levels=5)   → action: Discrete(25)
    mate.MultiCamera                → single-team
    mate.RelativeCoordinates
    mate.RescaledObservation
    mate.RepeatedRewardIndividualDone
    mate.AuxiliaryCameraRewards
    HiTMACWrapper(coordinator=None) → greedy assignment + augments obs với task bits
    RLlibMultiAgentAPI
    RLlibMultiAgentCentralizedTraining
    env.with_agent_groups()         → QPLEX grouped-agent API

coordinator_checkpoint (optional): nếu set → dùng trained MAPPO coordinator (sau Phase 2)
                                   nếu None → dùng greedy heuristic (Phase 1 default)
"""

import copy

from gym import spaces
from ray import tune
from ray.rllib.agents.qplex_v2 import qplex
from ray.rllib.models import MODEL_DEFAULTS

import mate
from examples.hitmac.wrappers import HiTMACWrapper
from examples.utils import (
    CustomMetricCallback,
    RLlibMultiAgentAPI,
    RLlibMultiAgentCentralizedTraining,
)


def target_agent_factory():
    return mate.agents.GreedyTargetAgent(seed=0)


def _detect_coordinator_class(checkpoint_path):
    """Auto-detect coordinator agent class từ params.pkl của checkpoint."""
    from examples.utils import load_checkpoint

    _, _, params = load_checkpoint(checkpoint_path)
    if params is not None and 'mixer' in params:
        from examples.hrl.qplex_v2.camera.agent import HRLQPLEXV2CameraAgent
        return HRLQPLEXV2CameraAgent

    from examples.hrl.mappo.camera.agent import HRLMAPPOCameraAgent
    return HRLMAPPOCameraAgent


def load_coordinator_agents(checkpoint_path, num_cameras, coordinator_config=None):
    """Tải coordinator từ checkpoint, tạo 1 agent per camera.

    Auto-detect agent class từ checkpoint (MAPPO hoặc QPLEX_V2).
    Mỗi camera có instance riêng để duy trì hidden state độc lập.
    Nếu checkpoint_path là None, trả về None → dùng greedy fallback.
    """
    if checkpoint_path is None:
        return None

    coordinator_class = _detect_coordinator_class(checkpoint_path)

    agents = []
    for c in range(num_cameras):
        kwargs = dict(checkpoint_path=checkpoint_path, seed=c)
        if coordinator_config is not None:
            kwargs['config'] = copy.deepcopy(coordinator_config)
        agents.append(coordinator_class(**kwargs))

    return agents


def make_env(env_config):
    env_config = env_config or {}
    env_id = env_config.get('env_id', 'MultiAgentTracking-v0')
    discrete_levels = env_config.get('discrete_levels', 5)

    # Tạo base environment
    base_env = mate.make(
        env_id,
        config=env_config.get('config'),
        **env_config.get('config_overrides', {}),
    )

    if str(env_config.get('enhanced_observation', None)).lower() != 'none':
        base_env = mate.EnhancedObservation(
            base_env, team=env_config['enhanced_observation']
        )

    # DiscreteCamera PHẢI đặt trước MultiCamera
    base_env = mate.DiscreteCamera(base_env, levels=discrete_levels)

    # MultiCamera wrapper
    target_agent = env_config.get('opponent_agent_factory', target_agent_factory)()
    env = mate.MultiCamera(base_env, target_agent=target_agent)

    # Wrappers chuẩn
    env = mate.RelativeCoordinates(env)
    env = mate.RescaledObservation(env)
    env = mate.RepeatedRewardIndividualDone(env)

    if 'reward_coefficients' in env_config:
        env = mate.AuxiliaryCameraRewards(
            env,
            coefficients=env_config['reward_coefficients'],
            reduction=env_config.get('reward_reduction', 'none'),
        )

    # Load MAPPO coordinator (None → greedy fallback)
    coordinator_agents = load_coordinator_agents(
        checkpoint_path=env_config.get('coordinator_checkpoint'),
        num_cameras=env.num_cameras,
        coordinator_config=env_config.get('coordinator_config'),
    )

    # HiTMAC wrapper: augments obs + quản lý coordinator
    env = HiTMACWrapper(
        env,
        coordinator_agents=coordinator_agents,
        coord_period=env_config.get('coord_period', 5),
        frame_skip=env_config.get('frame_skip', 5),
    )

    # RLlib wrappers
    env = RLlibMultiAgentAPI(env)
    env = RLlibMultiAgentCentralizedTraining(env)

    # QPLEX cần grouped-agent API
    action_space = spaces.Tuple((env.action_space,) * len(env.agent_ids))
    observation_space = spaces.Tuple((env.observation_space,) * len(env.agent_ids))
    setattr(observation_space, 'original_space', copy.deepcopy(observation_space))

    env = env.with_agent_groups(
        groups={'camera': env.agent_ids},
        obs_space=observation_space,
        act_space=action_space,
    )
    return env


tune.register_env('mate-hitmac.qplex.camera', make_env)

config = {
    **qplex.DEFAULT_CONFIG,
    'framework': 'torch',
    'seed': 0,
    # === Environment ==============================================================================
    'env': 'mate-hitmac.qplex.camera',
    'env_config': {
        'env_id': 'MultiAgentTracking-v0',
        'config': 'MATE-4v8-9.yaml',
        'config_overrides': {'reward_type': 'dense'},
        'reward_coefficients': {'coverage_rate': 1.0},
        'reward_reduction': 'mean',          # shared team reward
        'discrete_levels': 5,                # DiscreteCamera: 5×5 = 25 actions
        'coord_period': 5,                   # coordinator cập nhật mỗi 5 env steps
        'frame_skip': 5,                     # QPLEX action lặp 5 bước
        'coordinator_checkpoint': None,      # set tại runtime (--coordinator-checkpoint)
        'coordinator_config': None,          # None → dùng default HRL-MAPPO config
        'enhanced_observation': 'none',
        'opponent_agent_factory': target_agent_factory,
    },
    'disable_env_checking': True,
    'horizon': 500,
    'callbacks': CustomMetricCallback,
    # === Model ====================================================================================
    'normalize_actions': True,
    'model': {
        **MODEL_DEFAULTS,
        'fcnet_hiddens': [512, 256],  # không dùng trực tiếp (LSTM override)
        'fcnet_activation': 'tanh',
        'lstm_cell_size': 256,
        'max_seq_len': 10000,        # complete_episodes mode
    },
    # === QPLEX Mixer ==============================================================================
    'mixer': 'qplex_v2',
    'mixing_embed_dim': 128,
    # === Policy ===================================================================================
    'gamma': 0.99,
    # === Exploration ==============================================================================
    'explore': True,
    'exploration_config': {
        'type': 'EpsilonGreedy',
        'initial_epsilon': 1.0,
        'final_epsilon': 0.02,
        'epsilon_timesteps': 50000,  # trained environment steps
    },
    # === Replay Buffer & Optimization =============================================================
    'batch_mode': 'complete_episodes',
    'rollout_fragment_length': 0,   # gửi episodes ngay vào replay buffer
    'buffer_size': 2000,            # số episodes (sẽ được chia theo num_workers)
    'timesteps_per_iteration': 5120,
    'learning_starts': 5000,
    'train_batch_size': 1024,
    'target_network_update_freq': 500,
    'metrics_num_episodes_for_smoothing': 25,
    'grad_norm_clipping': 1000.0,
    'lr': 1e-4,
}
