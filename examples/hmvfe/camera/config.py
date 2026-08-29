"""Editable defaults for the HMVFE camera coordinator example."""

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

    # HMVFE baseline must keep FOCUS off.
    focus_enabled=False,
    focus_strict=False,

    # Logging, checkpoints, evaluation
    log_interval=10,
    save_interval=50,
    eval_interval=50,
    eval_episodes=5,
    output_dir='examples/hmvfe/camera/runs',
    run_name='hmvfe',

    # W&B. Set wandb_mode='disabled' for local smoke tests.
    wandb_project='mate-camera',
    wandb_group='hmvfe.camera',
    wandb_name=None,
    wandb_mode='online',
    wandb_tags=['hmvfe'],
)
