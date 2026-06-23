from ray import tune
from ray.rllib.models import MODEL_DEFAULTS
from ray.rllib.policy.policy import PolicySpec

import mate
from examples.hrl.wrappers import HierarchicalCamera
from examples.mappo.models import MAPPOModel
from examples.utils import (
    SHARED_POLICY_ID,
    CustomMetricCallback,
    RLlibMultiAgentAPI,
    RLlibMultiAgentCentralizedTraining,
    shared_policy_mapping_fn,
)


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

    env = HierarchicalCamera(
        env,
        multi_selection=env_config.get('multi_selection', False),
        frame_skip=env_config.get('frame_skip', 1),
    )

    env = RLlibMultiAgentAPI(env)
    env = RLlibMultiAgentCentralizedTraining(env)
    return env


tune.register_env('mate-hrl.mappo.camera', make_env)

config = {
    'framework': 'torch',
    'seed': 0,
    # === Environment ==============================================================================
    'env': 'mate-hrl.mappo.camera',
    'env_config': {
        'env_id': 'MultiAgentTracking-v0',
        'config': 'MATE-4v8-9.yaml',
        'config_overrides': {'reward_type': 'dense'},
        'reward_coefficients': {'coverage_rate': 1.0},  # override env's raw reward
        'reward_reduction': 'mean',  # shared reward
        'multi_selection': True,
        'frame_skip': 5,
        'enhanced_observation': 'none',
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
            'focus': {
                'enabled': True,
                'use_env_params': True,
                'beta_belief': 0.01,
                'horizon': 3,
                'horizon_discount': 0.9,
                'belief_hidden_dim': 256,
                'belief_max_delta': 400.0,
                'belief_min_std': 25.0,
                'integral_mode': 'MC',
                'mc_num_points': 128,
                'mc_chunk_size': 32,
                'mc_seed': 0,
                'grid_size': 64,
                'grid_chunk_size': 128,
                'grid_x_range': (-1000.0, 1000.0),
                'grid_y_range': (-1000.0, 1000.0),
                'use_signal_confidence': True,
                'signal_weight_min': 0.1,
                'signal_weight_max': 3.0,
                'eps': 1e-8,
            },
        },
    },
    # === Policy ===================================================================================
    'gamma': 0.99,
    'use_critic': True,
    'use_gae': True,
    'clip_param': 0.3,
    'multiagent': {
        'policies': {
            SHARED_POLICY_ID: PolicySpec(observation_space=None, action_space=None, config=None)
        },
        'policy_mapping_fn': shared_policy_mapping_fn,
    },
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

config['model']['custom_model_config']['env_config'] = config['env_config']
