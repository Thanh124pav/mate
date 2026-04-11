"""QPLEX_WM2 (pure) — DiscreteCamera, no hierarchical target selection.

Each camera directly controls orientation & viewing angle via discretized actions.
QPLEX decomposes joint Q-values with centralized training.
World model v2 predicts future states via RSSM-style latent dynamics.

Run: python -m examples.qplex_wm2.camera.train --env MATE-4v8-9.yaml
"""

import copy

from gym import spaces
from ray import tune
from ray.rllib.agents.qplex_wm2 import DEFAULT_CONFIG
from ray.rllib.models import MODEL_DEFAULTS

import mate
from examples.utils import (
    CustomMetricCallback,
    FrameSkip,
    RLlibMultiAgentAPI,
    RLlibMultiAgentCentralizedTraining,
)
from ray.rllib.agents.qplex_wm2.evasive_target import EvasiveTargetAgent


def target_agent_factory():
    """Training: evasive targets (avoid cameras)."""
    return EvasiveTargetAgent(seed=0, avoidance_strength=0.5, avoidance_range=0.5)


def greedy_target_factory():
    """Evaluation: standard greedy targets."""
    return mate.agents.GreedyTargetAgent(seed=0)


def make_env(env_config):
    env_config = env_config or {}
    env_id = env_config.get('env_id', 'MultiAgentTracking-v0')
    base_env = mate.make(
        env_id, config=env_config.get('config'), **env_config.get('config_overrides', {})
    )
    if str(env_config.get('enhanced_observation', None)).lower() != 'none':
        base_env = mate.EnhancedObservation(base_env, team=env_config['enhanced_observation'])

    # Discretize camera actions (orientation x viewing_angle)
    discrete_levels = env_config.get('discrete_levels', 5)
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

    frame_skip = env_config.get('frame_skip', 1)
    if frame_skip > 1:
        env = FrameSkip(env, frame_skip=frame_skip)

    env = RLlibMultiAgentAPI(env)
    env = RLlibMultiAgentCentralizedTraining(env)

    # Group agents for QPLEX
    action_space = spaces.Tuple((env.action_space,) * len(env.agent_ids))
    observation_space = spaces.Tuple((env.observation_space,) * len(env.agent_ids))
    setattr(observation_space, 'original_space', copy.deepcopy(observation_space))

    env = env.with_agent_groups(
        groups={'camera': env.agent_ids}, obs_space=observation_space, act_space=action_space
    )
    return env


tune.register_env('mate-qplex_wm2.camera', make_env)

# Register trainer
from ray.rllib.agents.qplex_wm2.qplex import QPlexWM2Trainer
tune.register_trainable('QPLEX_WM2', QPlexWM2Trainer)


config = {
    **DEFAULT_CONFIG,
    'framework': 'torch',
    'seed': 0,
    # === Environment ==============================================================================
    'env': 'mate-qplex_wm2.camera',
    'env_config': {
        'env_id': 'MultiAgentTracking-v0',
        'config': 'MATE-4v8-9.yaml',
        'config_overrides': {'reward_type': 'dense'},
        'reward_coefficients': {'coverage_rate': 1.0},
        'reward_reduction': 'mean',
        'discrete_levels': 5,
        'frame_skip': 5,
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
        'fcnet_hiddens': [512, 256],
        'fcnet_activation': 'tanh',
        'lstm_cell_size': 256,
        'max_seq_len': 10000,
    },
    'mixer': 'qplex_wm2',
    'mixing_embed_dim': 128,
    # === World Model v2 (RSSM-style latent dynamics) =============================================
    'world_model_v2': {
        'stoch_dim': 32,
        'deter_dim': 128,
        'hidden_dim': 128,
        'action_embed_dim': 16,
        'embed_dim': 128,
        'imagination_horizon': 5,
        'kl_coeff': 1.0,
        'free_nats': 1.0,
        'wm_loss_weight': 0.5,
        'reward_bonus_coeff': 0.1,
        'reward_bonus_scale': 0.5,
        'use_imagination_targets': False,
    },
    # === Evaluation: greedy targets (no avoidance) =================================================
    'evaluation_interval': 10,
    'evaluation_duration': 5,
    'evaluation_config': {
        'explore': False,
        'env_config': {
            'opponent_agent_factory': greedy_target_factory,
        },
    },
    # === Policy ===================================================================================
    'gamma': 0.99,
    # === Exploration ==============================================================================
    'explore': True,
    'exploration_config': {
        'type': 'EpsilonGreedy',
        'initial_epsilon': 1.0,
        'final_epsilon': 0.02,
        'epsilon_timesteps': 50000,
    },
    # === Replay Buffer & Optimization =============================================================
    'batch_mode': 'complete_episodes',
    'rollout_fragment_length': 0,
    'buffer_size': 2000,
    'timesteps_per_iteration': 5120,
    'learning_starts': 5000,
    'train_batch_size': 1024,
    'target_network_update_freq': 500,
    'metrics_num_episodes_for_smoothing': 25,
    'grad_norm_clipping': 10.0,
    'lr': 1e-4,
}
