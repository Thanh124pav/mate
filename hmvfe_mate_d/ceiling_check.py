"""Executor-ceiling diagnostic: does the coordinator's SELECTION policy actually
move coverage, or is the (fair, frozen) geometric executor the ceiling?

Evaluates several fixed selection strategies on the SAME episode layouts (paired):
  * random      : Bernoulli(0.5) select bits
  * all-select  : select every target for every camera (track everything visible)
  * none-select : select nothing
  * trained     : a trained HMVFECoordinator checkpoint (--ckpt)

If random / all-select / trained land at ~the same coverage, the selection policy
has little leverage -> the plateau is the executor, and policy tuning is futile.

    python hmvfe_mate_d/ceiling_check.py --episodes 15 \
        --ckpt hmvfe_mate_d/runs/hmvfe-seed0/best.pt
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

import hmvfe_mate_d  # applies the gym compat shim
from hmvfe_mate_d.config import HMVFEConfig, make_env
from hmvfe_mate_d.models import HMVFECoordinator


def run_episode(env, pick, seed):
    """Run one episode with a `pick(obs)->[N_cam,N_tgt] int array` selector; return mean coverage."""
    env.seed(seed)
    obs = env.reset()
    cov, done = [], False
    while not done:
        obs, _, done, info = env.step(pick(obs))
        if 'coverage_rate' in info:
            cov.append(info['coverage_rate'])
    return float(np.mean(cov)) if cov else 0.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--episodes', type=int, default=15)
    p.add_argument('--seed', type=int, default=10_000)
    p.add_argument('--ckpt', type=str, default='hmvfe_mate_d/runs/hmvfe-seed0/best.pt')
    args = p.parse_args()

    cfg = HMVFEConfig()
    env = make_env(cfg)
    nc, nt = env.num_cameras, env.num_targets
    rng = np.random.RandomState(0)

    strategies = {
        'random':      lambda obs: rng.randint(0, 2, size=(nc, nt)),
        'all-select':  lambda obs: np.ones((nc, nt), dtype=np.int64),
        'none-select': lambda obs: np.zeros((nc, nt), dtype=np.int64),
    }

    # trained policy (optional)
    try:
        state = torch.load(args.ckpt, map_location='cpu')
        model = HMVFECoordinator(
            env.num_cameras,
            env.num_targets,
            env.table_size,
            num_fields=env.num_fields,
            embedding_dim=cfg.embedding_dim,
            num_experts=cfg.num_experts,
            top_k=cfg.top_k,
            gating_hidden=cfg.gating_hidden,
            mlp_hidden=cfg.mlp_hidden,
            mlp_layers=cfg.mlp_layers,
            critic_reduction=cfg.critic_reduction,
            value_head_hidden=cfg.value_head_hidden,
        )
        model.load_state_dict(state['model'])
        model.eval()

        @torch.no_grad()
        def trained_pick(obs):
            a, *_ = model.act(torch.as_tensor(obs, dtype=torch.float32), deterministic=True)
            return a.cpu().numpy()

        strategies['trained'] = trained_pick
        print(f'[ceiling] loaded trained policy from {args.ckpt} '
              f'(env_steps={state.get("environment_steps", "?")})')
    except FileNotFoundError:
        print(f'[ceiling] no checkpoint at {args.ckpt} -- skipping trained policy')

    seeds = [args.seed + k for k in range(args.episodes)]
    print(f'[ceiling] {args.episodes} paired episodes on MATE-{cfg.env_config} '
          f'({nc} cameras x {nt} targets)\n')
    results = {}
    for name, pick in strategies.items():
        covs = [run_episode(env, pick, s) for s in seeds]
        results[name] = (float(np.mean(covs)), float(np.std(covs)))
        print(f'  {name:12s} coverage = {results[name][0]:.4f} +/- {results[name][1]:.4f}')

    env.close()
    print('\n[ceiling] verdict:')
    base = results.get('random', (0, 0))[0]
    if 'trained' in results:
        gain = results['trained'][0] - base
        print(f'  trained - random = {gain:+.4f}  '
              f'({"executor ceiling: selection barely matters" if abs(gain) < 0.02 else "selection DOES matter -> policy tuning can help"})')


if __name__ == '__main__':
    main()
