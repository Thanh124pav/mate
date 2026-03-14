"""HiTMAC Phase 2B — Config train Role-Based Coordinator (single-agent PPO).

1 camera (coordinator_idx=0) có enhanced observation và học assign targets
cho TẤT CẢ cameras. Các cameras khác dùng Phase 1 QPLEX executor (fixed).

Wrapper chain (Phase 2B):
    mate.DiscreteCamera(levels=5)       → action: Discrete(25) [cho executor]
    mate.EnhancedObservation(team='camera')  → coordinator thấy toàn bộ env
    mate.MultiCamera
    mate.RelativeCoordinates
    mate.RescaledObservation
    mate.RepeatedRewardIndividualDone
    mate.AuxiliaryCameraRewards
    HiTMACRoleWrapper(executor_checkpoint, coordinator_idx=0)
        # obs_space:    coordinator's obs (D_enhanced dims)
        # action_space: MultiDiscrete((2,) * num_cameras * num_targets)
        # On step: runs QPLEX for ALL cameras nội bộ
    → Standard gym interface (single-agent PPO)
"""

import copy

from ray import tune
from ray.rllib.models import MODEL_DEFAULTS

import mate
from examples.hitmac.wrappers import HiTMACRoleWrapper
from examples.mappo.models import MAPPOModel
from examples.utils import CustomMetricCallback


def target_agent_factory():
    return mate.agents.GreedyTargetAgent(seed=0)


def make_env(env_config):
    env_config = env_config or {}
    env_id = env_config.get('env_id', 'MultiAgentTracking-v0')
    discrete_levels = env_config.get('discrete_levels', 5)
    executor_checkpoint = env_config.get('executor_checkpoint')
    coordinator_idx = env_config.get('coordinator_idx', 0)

    base_env = mate.make(
        env_id,
        config=env_config.get('config'),
        **env_config.get('config_overrides', {}),
    )

    # EnhancedObservation cho coordinator thấy toàn bộ (global view)
    enhanced = str(env_config.get('enhanced_observation', 'camera')).lower()
    if enhanced != 'none':
        base_env = mate.EnhancedObservation(base_env, team=enhanced)

    # DiscreteCamera PHẢI đặt trước MultiCamera
    base_env = mate.DiscreteCamera(base_env, levels=discrete_levels)

    target_agent = env_config.get('opponent_agent_factory', target_agent_factory)()
    env = mate.MultiCamera(base_env, target_agent=target_agent)

    env = mate.RelativeCoordinates(env)
    env = mate.RescaledObservation(env)
    env = mate.RepeatedRewardIndividualDone(env)

    if 'reward_coefficients' in env_config:
        env = mate.AuxiliaryCameraRewards(
            env,
            coefficients=env_config['reward_coefficients'],
            reduction=env_config.get('reward_reduction', 'none'),
        )

    # Phase 2B: single-agent coordinator wrapping all cameras
    env = HiTMACRoleWrapper(
        env,
        executor_checkpoint=executor_checkpoint,
        coordinator_idx=coordinator_idx,
        frame_skip=env_config.get('frame_skip', 5),
    )

    return env


tune.register_env('mate-hitmac.role.camera', make_env)

config = {
    'framework': 'torch',
    'seed': 0,
    # === Environment ==============================================================================
    'env': 'mate-hitmac.role.camera',
    'env_config': {
        'env_id': 'MultiAgentTracking-v0',
        'config': 'MATE-4v8-9.yaml',
        'config_overrides': {'reward_type': 'dense'},
        'reward_coefficients': {'coverage_rate': 1.0},
        'reward_reduction': 'mean',
        'discrete_levels': 5,
        'frame_skip': 5,
        'coordinator_idx': 0,
        'executor_checkpoint': None,         # set tại runtime (--executor-checkpoint)
        'enhanced_observation': 'camera',    # coordinator thấy toàn bộ
        'opponent_agent_factory': target_agent_factory,
    },
    'horizon': 500,
    'callbacks': CustomMetricCallback,
    # === Model ====================================================================================
    'normalize_actions': True,
    'model': {
        'max_seq_len': 25,
        'custom_model': MAPPOModel,
        'custom_model_config': {
            **MODEL_DEFAULTS,
            'actor_hiddens': [512, 256],
            'actor_hidden_activation': 'tanh',
            'critic_hiddens': [512, 256],
            'critic_hidden_activation': 'tanh',
            'lstm_cell_size': 256,
            'max_seq_len': 25,
            'vf_share_layers': False,
        },
    },
    # === Policy ===================================================================================
    'gamma': 0.99,
    'use_critic': True,
    'use_gae': True,
    'clip_param': 0.3,
    # === Exploration ==============================================================================
    'explore': True,
    'exploration_config': {'type': 'StochasticSampling'},
    # === Replay Buffer & Optimization =============================================================
    'batch_mode': 'truncate_episodes',
    'rollout_fragment_length': 25,
    'train_batch_size': 1024,
    'sgd_minibatch_size': 256,
    'num_sgd_iter': 16,
    'metrics_num_episodes_for_smoothing': 25,
    'grad_clip': None,
    'lr': 5e-4,
    'lr_schedule': [
        (0, 5e-4),
        (4e6, 5e-4),
        (4e6, 1e-4),
        (8e6, 1e-4),
        (8e6, 5e-5),
    ],
    'entropy_coeff': 0.05,
    'entropy_coeff_schedule': [
        (0, 0.05),
        (2e6, 0.01),
        (4e6, 0.001),
        (10e6, 0.0),
    ],
    'vf_clip_param': 10000.0,
}
