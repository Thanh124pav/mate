import copy

from gym import spaces
from ray import tune
from ray.rllib.agents.qplex_wm4 import DEFAULT_CONFIG, EvasiveTargetAgent
from ray.rllib.models import MODEL_DEFAULTS

import mate
from examples.hrl.wrappers import DiscreteMultiSelection, HierarchicalCamera
from examples.utils import CustomMetricCallback, RLlibMultiAgentAPI, RLlibMultiAgentCentralizedTraining


def target_agent_factory():
    return EvasiveTargetAgent(seed=0, avoidance_strength=0.5, avoidance_range=0.5)


def greedy_target_factory():
    return mate.agents.GreedyTargetAgent(seed=0)


def make_env(env_config):
    env_config = env_config or {}
    env_id = env_config.get('env_id', 'MultiAgentTracking-v0')
    base_env = mate.make(
        env_id, config=env_config.get('config'), **env_config.get('config_overrides', {})
    )
    if str(env_config.get('enhanced_observation', None)).lower() != 'none':
        base_env = mate.EnhancedObservation(base_env, team=env_config['enhanced_observation'])

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

    multi_selection = env_config.get('multi_selection', False)
    env = HierarchicalCamera(
        env, multi_selection=multi_selection, frame_skip=env_config.get('frame_skip', 1)
    )
    if multi_selection:
        env = DiscreteMultiSelection(env)

    env = RLlibMultiAgentAPI(env)
    env = RLlibMultiAgentCentralizedTraining(env)
    action_space = spaces.Tuple((env.action_space,) * len(env.agent_ids))
    observation_space = spaces.Tuple((env.observation_space,) * len(env.agent_ids))
    setattr(observation_space, 'original_space', copy.deepcopy(observation_space))

    env = env.with_agent_groups(
        groups={'camera': env.agent_ids}, obs_space=observation_space, act_space=action_space
    )
    return env


tune.register_env('mate-hrl.qplex_wm4.camera', make_env)

from ray.rllib.agents.qplex_wm4.qplex import QPlexWM4Trainer

tune.register_trainable('QPLEX_WM4', QPlexWM4Trainer)

config = {
    **DEFAULT_CONFIG,
    'framework': 'torch',
    'seed': 0,
    'env': 'mate-hrl.qplex_wm4.camera',
    'env_config': {
        'env_id': 'MultiAgentTracking-v0',
        'config': 'MATE-4v8-9.yaml',
        'config_overrides': {'reward_type': 'dense'},
        'reward_coefficients': {'coverage_rate': 1.0},
        'reward_reduction': 'mean',
        'multi_selection': True,
        'frame_skip': 5,
        'enhanced_observation': 'none',
        'opponent_agent_factory': greedy_target_factory, # target_agent_factory,
    },
    'disable_env_checking': True,
    'horizon': 500,
    'callbacks': CustomMetricCallback,
    'normalize_actions': True,
    'model': {
        **MODEL_DEFAULTS,
        'fcnet_hiddens': [512, 256],
        'fcnet_activation': 'tanh',
        'lstm_cell_size': 256,
        'max_seq_len': 10000,
    },
    'mixer': 'qplex_wm4',
    'mixing_embed_dim': 128,
    'world_model_v4': {
        'stoch_dim': 32,
        'deter_dim': 128,
        'hidden_dim': 128,
        'action_embed_dim': 16,
        'embed_dim': 128,
        'imagination_horizon': 5,
        'kl_coeff': 0.0,
        'free_nats': 1.0,
        'wm_loss_weight': 0.5,
        'reward_bonus_coeff': 0.1,
        'reward_bonus_scale': 0.5,
        'use_imagination_targets': True,
        'imagination_loss_weight': 0.1,
        'ema_decay': 0.995,
    },
    'evaluation_interval': 10,
    'evaluation_duration': 5,
    'evaluation_config': {
        'explore': False,
        'env_config': {
            'opponent_agent_factory': greedy_target_factory,
        },
    },
    'gamma': 0.99,
    'explore': True,
    'exploration_config': {
        'type': 'EpsilonGreedy',
        'initial_epsilon': 1.0,
        'final_epsilon': 0.02,
        'epsilon_timesteps': 50000,
    },
    'batch_mode': 'complete_episodes',
    'rollout_fragment_length': 0,
    'buffer_size': 2000,
    'timesteps_per_iteration': 5120,
    'learning_starts': 5000,
    'train_batch_size': 1024,
    'target_network_update_freq': 500,
    'metrics_num_episodes_for_smoothing': 25,
    'grad_norm_clipping': 1000.0,
    'lr': 5e-4,
}
