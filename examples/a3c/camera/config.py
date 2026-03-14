from ray import tune
from ray.rllib.models import MODEL_DEFAULTS
from ray.rllib.policy.policy import PolicySpec

import mate
from examples.utils import (
    SHARED_POLICY_ID,
    CustomMetricCallback,
    FrameSkip,
    RLlibMultiAgentAPI,
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

    discrete_levels = env_config.get('discrete_levels', None)
    if discrete_levels is not None:
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
    return env


tune.register_env('mate-a3c.camera', make_env)

config = {
    'framework': 'torch',
    'seed': 0,
    # === Environment ==============================================================================
    'env': 'mate-a3c.camera',
    'env_config': {
        'env_id': 'MultiAgentTracking-v0',
        'config': 'MATE-4v8-9.yaml',
        'config_overrides': {'reward_type': 'dense'},
        'reward_coefficients': {'coverage_rate': 1.0},  # override env's raw reward
        'reward_reduction': 'mean',  # shared reward
        'discrete_levels': 5,
        'frame_skip': 5,
        'enhanced_observation': 'none',
        'opponent_agent_factory': target_agent_factory,
    },
    'horizon': 500,
    'callbacks': CustomMetricCallback,
    # === Model ====================================================================================
    'normalize_actions': True,
    'model': {
        **MODEL_DEFAULTS,
        'fcnet_hiddens': [256, 128],  # Reduced from [512, 256] for stability
        'fcnet_activation': 'relu',  # Changed from tanh - more stable
        'use_lstm': True,
        'lstm_cell_size': 128,  # Reduced from 256
        'max_seq_len': 20,  # Reduced from 25
        'vf_share_layers': False,
    },
    # === Policy ===================================================================================
    'gamma': 0.99,
    'use_critic': True,
    'use_gae': True,
    'lambda': 0.95,
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
    'rollout_fragment_length': 20,
    'train_batch_size': 160,  # 8 workers * 20 rollout length
    'sample_async': False,  # Changed to False for stability
    'metrics_num_episodes_for_smoothing': 25,
    'grad_clip': 5.0,  # CRITICAL: Must clip gradients for A3C!
    'lr': 3e-5,  # Much lower LR for stability
    'lr_schedule': None,  # Keep constant LR initially
    'entropy_coeff': 0.01,
    'entropy_coeff_schedule': None,  # Keep constant initially
    'vf_loss_coeff': 0.5,
    # Observation normalization
    'observation_filter': 'MeanStdFilter',
    'normalize_actions': True,
    'clip_rewards': 10.0,  # Clip rewards to prevent explosion
}
