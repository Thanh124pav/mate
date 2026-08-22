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
    p.add_argument('--num-envs', dest='num_envs', type=int,
                   help='Synchronous parallel envs (bigger, decorrelated A2C batch).')
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

    # logging / eval / io
    p.add_argument('--log-interval', dest='log_interval', type=int)
    p.add_argument('--save-interval', dest='save_interval', type=int)
    p.add_argument('--eval-interval', dest='eval_interval', type=int)
    p.add_argument('--eval-episodes', dest='eval_episodes', type=int)
    p.add_argument('--output-dir', dest='output_dir', type=str)
    p.add_argument('--run-name', dest='run_name', type=str)

    # wandb
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
        )
        state = torch.load(args.load, map_location='cpu')
        model.load_state_dict(state['model'])
        metrics = evaluate(model, config, num_episodes=config.eval_episodes, env=env)
        env.close()
        for key, value in metrics.items():
            print(f'{key}: {value:.4f}')
        return

    train(config)


if __name__ == '__main__':
    main()
