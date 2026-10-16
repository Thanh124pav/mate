"""Editable defaults for the FOCUSS-HMVFE camera coordinator example."""

from hmvfe_mate_d.config import HMVFEConfig


config = HMVFEConfig(
    # Environment
    env_id='MultiAgentTracking-v0',
    env_config='MATE-4v8-9.yaml',
    reward_type='dense',
    opponent='greedy',
    frame_skip=5,
    horizon=500,
    coverage_coefficient=1.0,
    seed=0,

    # HMVFE observation discretization
    num_distance_bins=16,
    num_angle_bins=16,
    num_occlusion_bins=8,

    # HMVFE coordinator network
    embedding_dim=10,
    num_experts=4,
    top_k=2,
    gating_hidden=128,
    mlp_hidden=128,
    mlp_layers=2,
    critic_reduction='learned',
    value_head_hidden=128,

    # A2C + GAE training
    total_env_steps=10_000_000,
    num_envs=8,
    rollout_length=20,
    gamma=0.99,
    gae_lambda=0.95,
    learning_rate=5e-4,
    anneal_lr=True,
    entropy_coef=0.01,
    value_coef=0.5,
    max_grad_norm=50.0,
    normalize_advantage=True,
    device='cpu',

    # FOCUSS-HMVFE. Strict mode prevents silent fallback to vanilla HMVFE.
    focus_enabled=True,
    focus_strict=True,
    focus_eta=1.0,
    focus_use_confidence=True,
    # Confidence-aware interpolation recovers vanilla HMVFE when FOCUSS is
    # uncertain and preserves mean actor-loss scale across cameras.
    focus_weight_formula='affine',
    focus_rho_temperature=1.0,
    focus_eps=1e-8,
    focus_mode='real',
    focus_belief_mode='learned',
    focus_beta_belief=0.05,
    focus_belief_lr=None,
    focus_belief_hidden_dim=512,
    focus_belief_arch='mlp',
    focus_belief_num_layers=1,
    focus_belief_dropout=0.0,
    focus_belief_distribution='gaussian',
    focus_belief_max_delta=400.0,
    focus_belief_min_std=25.0,
    focus_belief_grid_size=21,
    focus_belief_grid_max_delta=400.0,
    focus_belief_grid_label_std=None,
    focus_belief_persistence_prior_std=None,
    focus_belief_position_loss_coeff=0.1,
    focus_belief_grid_chunk_size=128,
    focus_horizon=9,
    focus_horizon_discount=0.9,
    focus_integral_mode='MC',
    focus_mc_num_points=128,
    focus_mc_chunk_size=32,
    focus_mc_seed=0,
    focus_sample_chunk_size=32,
    focus_grid_size=64,
    focus_grid_chunk_size=128,
    focus_min_credit_signal=1e-6,
    focus_obstacle_transmittance=0.0,
    focus_credit_level='both',
    focus_credit_mechanism='route',
    focus_action_credit_enabled=True,
    focus_action_alpha=1.0,
    focus_action_temperature=1.0,
    focus_action_q_std_floor=0.01,
    focus_action_grad_floor=0.10,
    focus_action_grad_ema=0.9,
    focus_action_grad_mode='match',

    # Logging, checkpoints, evaluation
    log_interval=10,
    save_interval=50,
    eval_interval=50,
    eval_episodes=5,
    output_dir='examples/hmvfe_focuss/camera/runs',
    run_name='hmvfe-focuss-real',

    # W&B. Set wandb_mode='disabled' for local smoke tests.
    wandb_project='mate-camera',
    wandb_group='hmvfe_focuss.camera',
    wandb_name=None,
    wandb_mode='online',
    wandb_tags=['hmvfe_focuss', 'focuss'],
)
