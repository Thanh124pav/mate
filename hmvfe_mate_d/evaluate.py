"""Evaluation harness for the HiT-MAC coordinator on MATE.

Runs deterministic episodes against the (fixed) opponent and reports the
benchmark metrics used by the MATE camera track: coverage rate first, plus the
target-side transport metrics for context.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import torch

from hmvfe_mate_d.config import HMVFEConfig, make_env
from hmvfe_mate_d.models import HMVFECoordinator


__all__ = ['evaluate']


@torch.no_grad()
def evaluate(
    model: HMVFECoordinator,
    config: HMVFEConfig,
    num_episodes: int = 5,
    env=None,
    deterministic: bool = True,
) -> Dict[str, float]:
    """Evaluate ``model`` for ``num_episodes`` and return mean metrics."""

    owns_env = env is None
    if owns_env:
        env = make_env(config)
        env.seed(config.seed + 10_000)

    device = next(model.parameters()).device
    was_training = model.training
    model.eval()

    coverage, real_coverage, transport, delivered, returns, lengths = [], [], [], [], [], []
    for _ in range(num_episodes):
        obs = env.reset()
        done = False
        ep_cov, ep_real, ep_trans, ep_ret, ep_len = [], [], [], 0.0, 0
        last_info = {}
        while not done:
            state = torch.as_tensor(obs, dtype=torch.float32, device=device)
            action, _, _, _ = model.act(state, deterministic=deterministic)
            obs, reward, done, info = env.step(action.cpu().numpy())
            ep_ret += reward
            ep_len += 1
            if 'coverage_rate' in info:
                ep_cov.append(info['coverage_rate'])
            if 'real_coverage_rate' in info:
                ep_real.append(info['real_coverage_rate'])
            if 'mean_transport_rate' in info:
                ep_trans.append(info['mean_transport_rate'])
            last_info = info

        coverage.append(float(np.mean(ep_cov)) if ep_cov else 0.0)
        real_coverage.append(float(np.mean(ep_real)) if ep_real else 0.0)
        transport.append(float(np.mean(ep_trans)) if ep_trans else 0.0)
        delivered.append(float(last_info.get('num_delivered_cargoes', 0.0)))
        returns.append(ep_ret)
        lengths.append(ep_len)

    if owns_env:
        env.close()
    if was_training:
        model.train()

    return {
        'eval/mean_coverage_rate': float(np.mean(coverage)),
        'eval/mean_real_coverage_rate': float(np.mean(real_coverage)),
        'eval/mean_transport_rate': float(np.mean(transport)),
        'eval/mean_num_delivered_cargoes': float(np.mean(delivered)),
        'eval/mean_episode_return': float(np.mean(returns)),
        'eval/mean_episode_length': float(np.mean(lengths)),
    }
