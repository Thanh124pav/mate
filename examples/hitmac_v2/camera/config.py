"""HiTMAC v2 — Phase 1: Train PPO executors with greedy coordinator.

Matches the original paper (NeurIPS 2020) by using on-policy actor-critic
(PPO with MAPPO shared policy) instead of QPLEX.

Wrapper chain:
    mate.MultiCamera
    → mate.RelativeCoordinates
    → mate.RescaledObservation
    → mate.RepeatedRewardIndividualDone
    → mate.AuxiliaryCameraRewards
    → HiTMACv2Wrapper(coordinator=greedy)
    → RLlibMultiAgentAPI
    → RLlibMultiAgentCentralizedTraining
"""

from ray import tune
from ray.rllib.models import MODEL_DEFAULTS
from ray.rllib.policy.policy import PolicySpec

import mate
from examples.hitmac_v2.wrappers import HiTMACv2Wrapper
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
        env_id,
        config=env_config.get('config'),
        **env_config.get('config_overrides', {}),
    )

    if str(env_config.get('enhanced_observation', None)).lower() != 'none':
        base_env = mate.EnhancedObservation(
            base_env, team=env_config['enhanced_observation']
        )

    # NO DiscreteCamera — PPO handles continuous actions natively
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

    # HiTMAC v2 wrapper: augments obs with task bits + coordinator
    env = HiTMACv2Wrapper(
        env,
        coordinator_agents=None,  # greedy fallback for Phase 1
        coord_period=env_config.get('coord_period', 5),
        frame_skip=env_config.get('frame_skip', 5),
    )

    env = RLlibMultiAgentAPI(env)
    env = RLlibMultiAgentCentralizedTraining(env)
    return env


tune.register_env('mate-hitmac-v2.ppo.camera', make_env)

config = {
    'framework': 'torch',
    'seed': 0,
    # === Environment ==============================================================================
    'env': 'mate-hitmac-v2.ppo.camera',
    'env_config': {
        'env_id': 'MultiAgentTracking-v0',
        'config': 'MATE-4v8-9.yaml',
        'config_overrides': {'reward_type': 'dense'},
        'reward_coefficients': {'coverage_rate': 1.0},
        'reward_reduction': 'mean',
        'coord_period': 5,
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
        },
    },
    # === Policy ===================================================================================
    'gamma': 0.99,
    'use_critic': True,
    'use_gae': True,
    'clip_param': 0.3,
    'multiagent': {
        'policies': {
            SHARED_POLICY_ID: PolicySpec(
                observation_space=None, action_space=None, config=None
            )
        },
        'policy_mapping_fn': shared_policy_mapping_fn,
    },
    # === Exploration ==============================================================================
    'explore': True,
    'exploration_config': {'type': 'StochasticSampling'},
    # === Optimization =============================================================================
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
