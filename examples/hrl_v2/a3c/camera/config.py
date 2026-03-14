import copy

from gym import spaces
from ray import tune
from ray.rllib.agents.a3c import a3c
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
    Create HRL V2 environment for A3C training.
    
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
    
    # RLlib wrapper (each camera is independent agent for A3C)
    env = RLlibMultiAgentAPI(env)
    
    return env


# Register environment
tune.register_env('mate-hrl_v2.a3c.camera', make_env)


# A3C configuration for low-level camera control
config = {
    **a3c.DEFAULT_CONFIG,
    'framework': 'torch',
    'seed': 0,
    
    # === Environment ==============================================================================
    'env': 'mate-hrl_v2.a3c.camera',
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
        'fcnet_activation': 'relu',
        'vf_share_layers': True,  # Share layers between policy and value function
        'use_lstm': True,
        'lstm_cell_size': 256,
        'max_seq_len': 20,
    },
    
    # === A3C specific =============================================================================
    'use_critic': True,
    'use_gae': True,
    'lambda': 0.95,  # Reduced from 1.0 to prevent numerical instability
    'gamma': 0.99,
    'vf_loss_coeff': 0.5,
    'entropy_coeff': 0.05,  # Increased from 0.01 for better exploration
    
    # === Optimization =============================================================================
    'lr': 5e-5,  # Reduced from 1e-4 to prevent NaN
    'lr_schedule': None,
    
    # === Rollout ==================================================================================
    'batch_mode': 'truncate_episodes',
    'rollout_fragment_length': 20,
    'sample_async': True,  # Asynchronous sampling (key feature of A3C)
    
    # === Exploration ==============================================================================
    'explore': True,
    'exploration_config': {
        'type': 'StochasticSampling',
    },
    
    # === Resources ================================================================================
    # A3C uses asynchronous workers - each worker independently updates global model
    'num_workers': 14,  # Will be set in train.py
    'num_envs_per_worker': 1,  # A3C typically uses 1 env per worker
    'num_cpus_per_worker': 1,
    'num_gpus_per_worker': 0,
    
    # === Advanced =================================================================================
    'grad_clip': None,
    'timesteps_per_iteration': 4000, 
    'normalize_actions': True,
    'clip_actions': True,
    'clip_rewards': 10.0,  # Clip rewards to prevent extreme values
    'metrics_num_episodes_for_smoothing': 25,
}
