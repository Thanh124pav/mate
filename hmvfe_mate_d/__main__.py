"""Command-line entry point: ``python -m hmvfe_mate_d``.

Examples
--------
Smoke test (tiny, no wandb)::

    python -m hmvfe_mate_d --total-env-steps 5000 --rollout-length 8 \
        --eval-interval 5 --eval-episodes 2 --log-interval 1 --wandb-mode disabled

Full benchmark run with W&B tracing::

    python -m hmvfe_mate_d \
        --env-config MATE-4v8-9.yaml --seed 0 --total-env-steps 10000000 \
        --wandb-project mate-hmvfe --wandb-group benchmark-4v8-9 \
        --run-name hmvfe-d-seed0 --eval-interval 50 --eval-episodes 5
"""

from __future__ import annotations

import argparse
import dataclasses

import torch

from hmvfe_mate_d.config import HMVFEConfig, make_env
from hmvfe_mate_d.evaluate import evaluate
from hmvfe_mate_d.focus_adapter import build_hmvfe_focus_engine
from hmvfe_mate_d.models import HMVFECoordinator
from hmvfe_mate_d.trainer import train


def build_config(args: argparse.Namespace) -> HMVFEConfig:
    overrides = {
        field.name: getattr(args, field.name)
        for field in dataclasses.fields(HMVFEConfig)
        if getattr(args, field.name, None) is not None
    }
    return HMVFEConfig(**overrides)


def parse_args(prog: str = 'python -m hmvfe_mate_d') -> argparse.Namespace:
    p = argparse.ArgumentParser(prog=prog, description=__doc__)

    # environment
    p.add_argument('--env-config', dest='env_config', type=str, help='MATE config YAML name.')
    p.add_argument('--reward-type', dest='reward_type', type=str, choices=['dense', 'sparse'])
    p.add_argument('--opponent', type=str, choices=['greedy', 'heuristic', 'random'])
    p.add_argument('--frame-skip', dest='frame_skip', type=int)
    p.add_argument('--horizon', type=int, help='Low-level env-step horizon before truncation.')
    p.add_argument('--coverage-coefficient', dest='coverage_coefficient', type=float,
                   help='Auxiliary coverage_rate reward coefficient.')
    p.add_argument('--seed', type=int)

    # HMVFE coordinator (FM + MoE embedding)
    p.add_argument('--embedding-dim', dest='embedding_dim', type=int)
    p.add_argument('--num-experts', dest='num_experts', type=int,
                   help='Total MoE experts (paper best: 4).')
    p.add_argument('--top-k', dest='top_k', type=int,
                   help='Active experts per pair (paper best: 2).')
    p.add_argument('--gating-hidden', dest='gating_hidden', type=int)
    p.add_argument('--mlp-hidden', dest='mlp_hidden', type=int)
    p.add_argument('--mlp-layers', dest='mlp_layers', type=int)
    p.add_argument('--critic-reduction', dest='critic_reduction',
                   choices=['max', 'mean', 'learned'],
                   help="Tier B: state-value critic. max=paper, mean=param-free, learned=head.")
    p.add_argument('--value-head-hidden', dest='value_head_hidden', type=int,
                   help='Hidden units of the learned value head (critic_reduction=learned).')
    p.add_argument('--num-distance-bins', dest='num_distance_bins', type=int)
    p.add_argument('--num-angle-bins', dest='num_angle_bins', type=int)
    p.add_argument('--num-occlusion-bins', dest='num_occlusion_bins', type=int,
                   help='Bins for obstacle occlusion of the camera->target ray (variant B).')

    # training
    p.add_argument('--total-env-steps', dest='total_env_steps', type=int)
    p.add_argument('--timesteps-total', dest='total_env_steps', type=int,
                   help='Alias for --total-env-steps, matching Ray-based examples.')
    p.add_argument('--num-envs', dest='num_envs', type=int,
                   help='Synchronous parallel envs (bigger, decorrelated A2C batch).')
    p.add_argument('--num-envs-per-worker', dest='num_envs', type=int,
                   help='Alias for --num-envs in the ray-free HMVFE trainer.')
    p.add_argument('--rollout-length', dest='rollout_length', type=int)
    p.add_argument('--gamma', type=float)
    p.add_argument('--gae-lambda', dest='gae_lambda', type=float)
    p.add_argument('--lr', dest='learning_rate', type=float)
    p.add_argument('--no-anneal-lr', dest='anneal_lr', action='store_false', default=None,
                   help='Disable linear LR annealing (default: anneal to 0).')
    p.add_argument('--entropy-coef', dest='entropy_coef', type=float)
    p.add_argument('--value-coef', dest='value_coef', type=float)
    p.add_argument('--max-grad-norm', dest='max_grad_norm', type=float)
    p.add_argument('--device', type=str, help='cpu | cuda | mps')

    # FOCUS
    p.add_argument('--focus-enabled', dest='focus_enabled', action='store_true', default=None,
                   help='Enable FOCUS camera-wise actor weighting.')
    p.add_argument('--focus-strict', dest='focus_strict', action='store_true', default=None,
                   help='Fail instead of silently falling back to vanilla HMVFE when FOCUS is enabled.')
    p.add_argument('--focus-eta', dest='focus_eta', type=float)
    p.add_argument('--focus-weight-formula', dest='focus_weight_formula', choices=['rho', 'affine'],
                   help='FOCUS actor-weight formula: rho = N*rho_i, affine = 1+eta*c*(N*rho_i-1).')
    p.add_argument('--focus-rho-temperature', dest='focus_rho_temperature', type=float,
                   help='Temperature applied to rho before actor weighting; >1 softens toward uniform.')
    p.add_argument('--focus-use-confidence', dest='focus_use_confidence', action='store_true', default=None)
    p.add_argument('--no-focus-use-confidence', dest='focus_use_confidence', action='store_false')
    p.add_argument('--focus-mode', dest='focus_mode', choices=['real', 'uniform', 'shuffled', 'random'])
    p.add_argument('--focus-belief-mode', dest='focus_belief_mode',
                   choices=['oracle_next_ablation', 'learned'])
    p.add_argument('--focus-beta-belief', dest='focus_beta_belief', type=float,
                   help='Coefficient for the learned FOCUS belief NLL loss.')
    p.add_argument('--focus-belief-lr', dest='focus_belief_lr', type=float,
                   help='Optional separate LR for the learned FOCUS belief model.')
    p.add_argument('--focus-belief-hidden-dim', dest='focus_belief_hidden_dim', type=int)
    p.add_argument('--focus-belief-arch', dest='focus_belief_arch', choices=['mlp', 'lstm'])
    p.add_argument('--focus-belief-num-layers', dest='focus_belief_num_layers', type=int)
    p.add_argument('--focus-belief-dropout', dest='focus_belief_dropout', type=float)
    p.add_argument('--focus-belief-distribution', dest='focus_belief_distribution',
                   choices=['gaussian', 'categorical_grid'])
    p.add_argument('--focus-belief-max-delta', dest='focus_belief_max_delta', type=float)
    p.add_argument('--focus-belief-min-std', dest='focus_belief_min_std', type=float)
    p.add_argument('--focus-belief-grid-size', dest='focus_belief_grid_size', type=int)
    p.add_argument('--focus-belief-grid-max-delta', dest='focus_belief_grid_max_delta', type=float)
    p.add_argument('--focus-belief-grid-label-std', dest='focus_belief_grid_label_std', type=float)
    p.add_argument('--focus-belief-persistence-prior-std', dest='focus_belief_persistence_prior_std', type=float)
    p.add_argument('--focus-belief-position-loss-coeff', dest='focus_belief_position_loss_coeff', type=float)
    p.add_argument('--focus-belief-grid-chunk-size', dest='focus_belief_grid_chunk_size', type=int)
    p.add_argument(
        '--focus-belief-safeguard-mode',
        dest='focus_belief_safeguard_mode',
        choices=['zero', 'zero_ema', 'zero_ema_blend'],
        help='Belief safeguard; default is zero-initialized persistence residual.',
    )
    p.add_argument('--focus-belief-error-ema-decay', dest='focus_belief_error_ema_decay', type=float)
    p.add_argument('--focus-belief-gate-margin', dest='focus_belief_gate_margin', type=float)
    p.add_argument('--focus-belief-gate-full-improvement', dest='focus_belief_gate_full_improvement', type=float)
    p.add_argument('--focus-belief-blend-warmup-updates', dest='focus_belief_blend_warmup_updates', type=int)
    p.add_argument('--focus-belief-blend-ramp-updates', dest='focus_belief_blend_ramp_updates', type=int)
    p.add_argument('--focus-horizon', dest='focus_horizon', type=int)
    p.add_argument('--focus-horizon-discount', dest='focus_horizon_discount', type=float)
    p.add_argument('--focus-integral-mode', dest='focus_integral_mode', choices=['MC', 'sigma', 'grid'])
    p.add_argument('--focus-mc-num-points', dest='focus_mc_num_points', type=int)
    p.add_argument('--focus-mc-chunk-size', dest='focus_mc_chunk_size', type=int)
    p.add_argument('--focus-mc-seed', dest='focus_mc_seed', type=int)
    p.add_argument('--focus-sample-chunk-size', dest='focus_sample_chunk_size', type=int)
    p.add_argument('--focus-grid-size', dest='focus_grid_size', type=int)
    p.add_argument('--focus-grid-chunk-size', dest='focus_grid_chunk_size', type=int)
    p.add_argument('--focus-min-credit-signal', dest='focus_min_credit_signal', type=float)
    p.add_argument('--focus-obstacle-transmittance', dest='focus_obstacle_transmittance', type=float)
    p.add_argument('--focus-execution-eta', dest='focus_execution_eta', type=float,
                   help='CTCE camera-logit bias strength from current-state belief responsibility.')

    # logging / eval / io
    p.add_argument('--log-interval', dest='log_interval', type=int)
    p.add_argument('--save-interval', dest='save_interval', type=int)
    p.add_argument('--eval-interval', dest='eval_interval', type=int)
    p.add_argument('--eval-episodes', dest='eval_episodes', type=int)
    p.add_argument('--output-dir', dest='output_dir', type=str)
    p.add_argument('--local-dir', dest='output_dir', type=str,
                   help='Alias for --output-dir, matching Ray-based examples.')
    p.add_argument('--run-name', dest='run_name', type=str)

    # wandb
    p.add_argument('--project', dest='wandb_project', type=str,
                   help='Alias for --wandb-project, matching the examples namespace.')
    p.add_argument('--group', dest='wandb_group', type=str,
                   help='Alias for --wandb-group, matching the examples namespace.')
    p.add_argument('--wandb-project', dest='wandb_project', type=str)
    p.add_argument('--wandb-group', dest='wandb_group', type=str)
    p.add_argument('--wandb-name', dest='wandb_name', type=str)
    p.add_argument('--wandb-mode', dest='wandb_mode', type=str,
                   choices=['online', 'offline', 'disabled'])

    # modes
    p.add_argument('--eval-only', action='store_true',
                   help='Skip training; evaluate a checkpoint (requires --load).')
    p.add_argument('--load', type=str, help='Checkpoint .pt to load for --eval-only.')

    return p.parse_args()


def main() -> None:
    args = parse_args()
    config = build_config(args)

    if args.eval_only:
        assert args.load, '--eval-only requires --load <checkpoint.pt>.'
        state = torch.load(args.load, map_location='cpu')
        if state.get('config'):
            field_names = {field.name for field in dataclasses.fields(HMVFEConfig)}
            saved = {
                key: value for key, value in state['config'].items()
                if key in field_names
            }
            explicit = {
                key: getattr(args, key) for key in field_names
                if getattr(args, key, None) is not None
            }
            config = HMVFEConfig(**{**saved, **explicit})
        env = make_env(config)
        env.seed(config.seed + 10_000)
        model = HMVFECoordinator(
            env.num_cameras,
            env.num_targets,
            env.table_size,
            num_fields=env.num_fields,
            embedding_dim=config.embedding_dim,
            num_experts=config.num_experts,
            top_k=config.top_k,
            gating_hidden=config.gating_hidden,
            mlp_hidden=config.mlp_hidden,
            mlp_layers=config.mlp_layers,
            critic_reduction=config.critic_reduction,
            value_head_hidden=config.value_head_hidden,
        ).to(torch.device(config.device))
        model.load_state_dict(state['model'])
        focus_engine = build_hmvfe_focus_engine(
            config, env, device=torch.device(config.device)
        )
        if focus_engine is not None and 'focus_engine' in state:
            focus_engine.estimator.load_state_dict(state['focus_engine'])
            focus_engine.set_runtime_state(state.get('focus_runtime_state'))
        metrics = evaluate(
            model,
            config,
            num_episodes=config.eval_episodes,
            env=env,
            focus_engine=focus_engine,
        )
        env.close()
        for key, value in metrics.items():
            print(f'{key}: {value:.4f}')
        return

    train(config)


if __name__ == '__main__':
    main()
