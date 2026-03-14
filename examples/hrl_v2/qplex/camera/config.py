import copy

from gym import spaces
from ray import tune
from ray.rllib.agents.qplex import qplex
from ray.rllib.models import MODEL_DEFAULTS

import mate
from examples.hrl_v2.wrappers import HierarchicalCameraV2
from examples.hrl_v2.high_level import GreedyDistanceAssigner
from examples.utils import CustomMetricCallback, RLlibMultiAgentAPI, RLlibMultiAgentCentralizedTraining


def target_agent_factory():
    """Factory for target agents."""
    return mate.agents.GreedyTargetAgent(seed=0)


def make_env(env_config):
    """
    Create HRL V2 environment for QPLEX training.
    
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
    
    # RLlib wrappers
    env = RLlibMultiAgentAPI(env)
    env = RLlibMultiAgentCentralizedTraining(env)
    
    # For QPLEX: need to group agents
    action_space = spaces.Tuple((env.action_space,) * len(env.agent_ids))
    observation_space = spaces.Tuple((env.observation_space,) * len(env.agent_ids))
    setattr(observation_space, 'original_space', copy.deepcopy(observation_space))
    
    env = env.with_agent_groups(
        groups={'camera': env.agent_ids},
        obs_space=observation_space,
        act_space=action_space
    )
    
    return env


# Register environment
tune.register_env('mate-hrl_v2.qplex.camera', make_env)


# QPLEX configuration for low-level camera control
config = {
    **qplex.DEFAULT_CONFIG,
    'framework': 'torch',
    'seed': 0,
    
    # === Environment ==============================================================================
    'env': 'mate-hrl_v2.qplex.camera',
    'env_config': {
        'env_id': 'MultiAgentTracking-v0',
        'config': 'MATE-4v5-0.yaml',
        'config_overrides': {'reward_type': 'dense'},
        'reward_coefficients': {'coverage_rate': 1.0},
        'reward_reduction': 'mean',  # Shared reward
        'assigner_kwargs': {'max_assignments_per_camera': 1},  # Each camera tracks 1 target
        'frame_skip': 5,
        'include_assignment_in_obs': True,
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
    'mixer': 'qplex',
    'mixing_embed_dim': 128,
    
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
    'grad_norm_clipping': 1000.0,
    'lr': 1e-4,
}
