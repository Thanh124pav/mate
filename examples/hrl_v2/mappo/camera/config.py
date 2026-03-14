import copy

from gym import spaces
from ray import tune
from ray.rllib.agents.ppo import ppo
from ray.rllib.models import MODEL_DEFAULTS

import mate
from examples.hrl_v2.wrappers import HierarchicalCameraV2
from examples.hrl_v2.high_level import GreedyDistanceAssigner
from examples.utils import CustomMetricCallback, RLlibMultiAgentAPI


def target_agent_factory():
    """Factory for target agents."""
    return mate.agents.GreedyTargetAgent(seed=0)


def make_env(env_config):
    """
    Create HRL V2 environment for MAPPO training.
    
    Low-level policy learns camera control to track assigned targets.
    High-level assignment is done by greedy algorithm.
    """
    env_config = env_config or {}
    env_id = env_config.get('env_id', 'MultiAgentTracking-v0')
    
    # Create base environment
    base_env = mate.make(
        env_id, config=env_config.get('config'), **env_config.get('config_overrides', {})
    )
    
    if str(env_config.get('enhanced_observation', None)).lower() != 'none':
        base_env = mate.EnhancedObservation(base_env, team=env_config['enhanced_observation'])
    
    # Add target agent
    target_agent = env_config.get('opponent_agent_factory', target_agent_factory())()
    env = mate.MultiCamera(base_env, target_agent=target_agent)
    
    # Standard wrappers
    env = mate.RelativeCoordinates(env)
    env = mate.RescaledObservation(env)
    env = mate.RepeatedRewardIndividualDone(env)
    
    # Auxiliary rewards
    if 'reward_coefficients' in env_config:
        env = mate.AuxiliaryCameraRewards(
            env,
            coefficients=env_config['reward_coefficients'],
            reduction=env_config.get('reward_reduction', 'none'),
        )
    
    # HRL V2 wrapper: High-level assignment (greedy) + Low-level control (learned)
    assigner_kwargs = env_config.get('assigner_kwargs', {'max_assignments_per_camera': 1})
    env = HierarchicalCameraV2(
        env,
        assigner_class=GreedyDistanceAssigner,
        assigner_kwargs=assigner_kwargs,
        frame_skip=env_config.get('frame_skip', 1),
        include_assignment_in_obs=env_config.get('include_assignment_in_obs', True),
    )
    
    # RLlib wrapper (no grouping for MAPPO - each camera is independent agent)
    env = RLlibMultiAgentAPI(env)
    
    return env


# Register environment
tune.register_env('mate-hrl_v2.mappo.camera', make_env)


# MAPPO configuration for low-level camera control
config = {
    **ppo.DEFAULT_CONFIG,
    'framework': 'torch',
    'seed': 0,
    
    # === Environment ==============================================================================
    'env': 'mate-hrl_v2.mappo.camera',
    'env_config': {
        'env_id': 'MultiAgentTracking-v0',
        'config': 'MATE-4v5-0.yaml',
        'config_overrides': {'reward_type': 'dense'},
        'reward_coefficients': {'coverage_rate': 1.0},
        'reward_reduction': 'mean',  # Shared reward
        'assigner_kwargs': {'max_assignments_per_camera': 1},
        'frame_skip': 5,
        'include_assignment_in_obs': True,
        'enhanced_observation': 'none',
        'opponent_agent_factory': target_agent_factory,
    },
    'disable_env_checking': True,
    'horizon': 500,
    'callbacks': CustomMetricCallback,
    
    # === Multi-agent ==============================================================================
    'multiagent': {
        'policies': {
            'camera_policy': (None, None, None, {}),
        },
        'policy_mapping_fn': lambda agent_id, **kwargs: 'camera_policy',
        'policies_to_train': ['camera_policy'],
    },
    
    # === Model ====================================================================================
    'model': {
        **MODEL_DEFAULTS,
        'fcnet_hiddens': [256, 256],
        'fcnet_activation': 'tanh',
        'vf_share_layers': False,  # Separate value function network
        'use_lstm': True,
        'lstm_cell_size': 256,
        'max_seq_len': 20,
    },
    
    # === PPO specific =============================================================================
    'use_critic': True,
    'use_gae': True,
    'lambda': 0.95,
    'gamma': 0.99,
    'kl_coeff': 0.2,
    'clip_param': 0.2,
    'vf_clip_param': 10.0,
    'entropy_coeff': 0.01,
    'vf_loss_coeff': 0.5,
    
    # === Optimization =============================================================================
    'lr': 5e-5,
    'lr_schedule': None,
    'sgd_minibatch_size': 128,
    'num_sgd_iter': 10,
    'shuffle_sequences': True,
    
    # === Rollout ==================================================================================
    'batch_mode': 'truncate_episodes',
    'rollout_fragment_length': 200,
    'train_batch_size': 4000,
    
    # === Exploration ==============================================================================
    'explore': True,
    'exploration_config': {
        'type': 'StochasticSampling',
    },
    
    # === Resources ================================================================================
    'num_workers': 0,  # Will be set in train.py
    'num_envs_per_worker': 8,
    'num_cpus_per_worker': 1,
    'num_gpus_per_worker': 0,
    
    # === Advanced =================================================================================
    'grad_clip': 0.5,
    'normalize_actions': True,
    'clip_actions': True,
    'timesteps_per_iteration': 4000,
    'metrics_num_episodes_for_smoothing': 25,
}
