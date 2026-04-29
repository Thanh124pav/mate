import copy

from gym import spaces
from ray import tune
from ray.rllib.agents.duelmix import DEFAULT_CONFIG
from ray.rllib.models import MODEL_DEFAULTS

import mate
from examples.utils import CustomMetricCallback, FrameSkip, RLlibMultiAgentAPI, RLlibMultiAgentCentralizedTraining


def target_agent_factory():
    return mate.agents.GreedyTargetAgent(seed=0)


def make_env(env_config):
    env_config = env_config or {}
    env_id = env_config.get('env_id', 'MultiAgentTracking-v0')
    base_env = mate.make(
        env_id, config=env_config.get('config'), **env_config.get('config_overrides', {})
    )
    if str(env_config.get('enhanced_observation', None)).lower() != 'none':
        base_env = mate.EnhancedObservation(base_env, team=env_config['enhanced_observation'])

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
    action_space = spaces.Tuple((env.action_space,) * len(env.agent_ids))
    observation_space = spaces.Tuple((env.observation_space,) * len(env.agent_ids))
    setattr(observation_space, 'original_space', copy.deepcopy(observation_space))

    env = env.with_agent_groups(
        groups={'camera': env.agent_ids}, obs_space=observation_space, act_space=action_space
    )
    return env


tune.register_env('mate-duelmix.camera', make_env)

from ray.rllib.agents.duelmix.duelmix import DuelMixTrainer

tune.register_trainable('DUELMIX', DuelMixTrainer)

config = {
    **DEFAULT_CONFIG,
    'framework': 'torch',
    'seed': 0,
    'env': 'mate-duelmix.camera',
    'env_config': {
        'env_id': 'MultiAgentTracking-v0',
        'config': 'MATE-4v5-0.yaml',
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
    'normalize_actions': True,
    'model': {
        **MODEL_DEFAULTS,
        'fcnet_hiddens': [512, 256],
        'fcnet_activation': 'tanh',
        'lstm_cell_size': 256,
        'max_seq_len': 10000,
    },
    'mixer': 'duelmix',
    'mixing_embed_dim': 128,
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
    'buffer_size': 500,
    'timesteps_per_iteration': 5120,
    'learning_starts': 5000,
    'train_batch_size': 1024,
    'target_network_update_freq': 50,
    'metrics_num_episodes_for_smoothing': 25,
    'grad_norm_clipping': 1000.0,
    'lr': 1e-4,
}
