"""n-step actor-critic (A2C + GAE) trainer for the HiT-MAC coordinator.

Single-process re-implementation of HiT-MAC's ``player_util.optimize`` loop. The
original used asynchronous A3C workers for decorrelated data; here we keep one
process but step ``config.num_envs`` MATE coordinator envs *synchronously*
(:class:`hmvfe_mate_d.vector_env.SyncVectorCoordinatorEnv`) and stack their
transitions, so each update sees a larger, decorrelated batch -> lower-variance
gradients. The coordinator optimizes the shared scalar coverage reward; credit
among cameras is handled inside the Shapley critic. Learning rate is linearly
annealed to 0 over training to stabilise the tail (toggle with ``anneal_lr``).

Tracing follows the MATE paper's camera-side curve style: ``train/mean_coverage_rate``
and rolling ``train/episode_coverage_rate`` against ``environment_steps``, with
periodic ``eval/*`` metrics.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim

from hmvfe_mate_d.config import HMVFEConfig, make_env, make_output_dir
from hmvfe_mate_d.evaluate import evaluate
from hmvfe_mate_d.models import HMVFECoordinator
from hmvfe_mate_d.vector_env import SyncVectorCoordinatorEnv


__all__ = ['train']


def _set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def _init_wandb(config: HMVFEConfig, output_dir: Path):
    if not config.wandb_project:
        return None
    try:
        import wandb
    except ImportError:
        print('[hmvfe_mate_d] wandb not installed; continuing without tracing.')
        return None

    run = wandb.init(
        project=config.wandb_project,
        group=config.wandb_group,
        name=config.wandb_name or config.run_name,
        tags=config.wandb_tags or None,
        mode=config.wandb_mode,
        dir=str(output_dir),
        config=config.to_dict(),
    )
    # x-axis = environment steps for all metrics
    wandb.define_metric('environment_steps')
    wandb.define_metric('train/*', step_metric='environment_steps')
    wandb.define_metric('eval/*', step_metric='environment_steps')
    return run


def _compute_gae(rewards, values, dones, bootstrap_value, gamma, gae_lambda):
    """Standard GAE on a scalar reward/value stream. Returns (advantages, returns)."""

    horizon = len(rewards)
    advantages = [0.0] * horizon
    next_value = bootstrap_value
    gae = 0.0
    for t in reversed(range(horizon)):
        non_terminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * non_terminal - values[t]
        gae = delta + gamma * gae_lambda * non_terminal * gae
        advantages[t] = gae
        next_value = values[t]
    returns = [advantages[t] + values[t] for t in range(horizon)]
    return advantages, returns


def train(config: HMVFEConfig) -> Path:  # pylint: disable=too-many-locals,too-many-statements
    _set_seed(config.seed)
    device = torch.device(config.device)

    output_dir = make_output_dir(config)
    with open(output_dir / 'config.json', 'w', encoding='utf-8') as f:
        json.dump(config.to_dict(), f, indent=2)

    num_envs = max(1, int(config.num_envs))
    venv = SyncVectorCoordinatorEnv(config, num_envs)
    feature_dim = venv.feature_dim

    model = HMVFECoordinator(
        venv.num_cameras,
        venv.num_targets,
        venv.table_size,
        num_fields=venv.num_fields,
        embedding_dim=config.embedding_dim,
        num_experts=config.num_experts,
        top_k=config.top_k,
        gating_hidden=config.gating_hidden,
        mlp_hidden=config.mlp_hidden,
        mlp_layers=config.mlp_layers,
        critic_reduction=config.critic_reduction,
        value_head_hidden=config.value_head_hidden,
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    run = _init_wandb(config, output_dir)

    steps_per_update = config.rollout_length * config.frame_skip * num_envs
    total_updates = max(1, config.total_env_steps // steps_per_update)

    obs = venv.reset()                       # [num_envs, N_cam, N_tgt, F]
    env_steps = 0
    best_eval = -np.inf

    # per-env episode accumulators + rolling completed-episode stats
    ep_return = [0.0] * num_envs
    ep_coverage = [[] for _ in range(num_envs)]
    completed_returns: list = []
    completed_coverage: list = []

    print(
        f'[hmvfe_mate_d] device={device} feature_dim={feature_dim} '
        f'envs={num_envs} updates={total_updates} steps/update={steps_per_update} '
        f'(rollout={config.rollout_length} x frame_skip={config.frame_skip} x envs={num_envs}) '
        f'anneal_lr={config.anneal_lr}'
    )

    for update in range(1, total_updates + 1):
        # --- linear LR annealing ---------------------------------------------
        if config.anneal_lr:
            lr_now = config.learning_rate * (1.0 - (update - 1) / total_updates)
            for group in optimizer.param_groups:
                group['lr'] = lr_now
        else:
            lr_now = config.learning_rate

        rollout = config.rollout_length
        # buffers indexed [t][env]
        log_probs_te = [[None] * num_envs for _ in range(rollout)]
        values_te = [[None] * num_envs for _ in range(rollout)]
        entropies_te = [[None] * num_envs for _ in range(rollout)]
        rewards_te = np.zeros((rollout, num_envs))
        dones_te = np.zeros((rollout, num_envs))
        values_f_te = np.zeros((rollout, num_envs))
        rollout_coverage = []

        for t in range(rollout):
            actions = []
            for i in range(num_envs):
                state = torch.as_tensor(obs[i], dtype=torch.float32, device=device)
                action, log_prob, entropy, value = model.act(state)
                actions.append(action.detach().cpu().numpy())
                log_probs_te[t][i] = log_prob
                entropies_te[t][i] = entropy
                values_te[t][i] = value.squeeze()
                values_f_te[t, i] = float(value.detach().squeeze().cpu())

            next_obs, rewards, dones, infos = venv.step(actions)
            env_steps += config.frame_skip * num_envs

            for i in range(num_envs):
                rewards_te[t, i] = rewards[i]
                dones_te[t, i] = 1.0 if dones[i] else 0.0
                ep_return[i] += float(rewards[i])
                if 'coverage_rate' in infos[i]:
                    ep_coverage[i].append(infos[i]['coverage_rate'])
                    rollout_coverage.append(infos[i]['coverage_rate'])
                if dones[i]:
                    completed_returns.append(ep_return[i])
                    completed_coverage.append(
                        float(np.mean(ep_coverage[i])) if ep_coverage[i] else 0.0
                    )
                    ep_return[i] = 0.0
                    ep_coverage[i] = []

            obs = next_obs

        # --- per-env GAE, then flatten (env-major: i outer, t inner) ----------
        advantages_all, returns_all = [], []
        for i in range(num_envs):
            if dones_te[-1, i] >= 1.0:
                bootstrap_value = 0.0
            else:
                with torch.no_grad():
                    state = torch.as_tensor(obs[i], dtype=torch.float32, device=device)
                    bootstrap_value = float(model.value_only(state).squeeze().cpu())
            adv_i, ret_i = _compute_gae(
                rewards_te[:, i].tolist(), values_f_te[:, i].tolist(),
                dones_te[:, i].tolist(), bootstrap_value, config.gamma, config.gae_lambda,
            )
            advantages_all.extend(adv_i)
            returns_all.extend(ret_i)

        advantages_t = torch.as_tensor(advantages_all, dtype=torch.float32, device=device)
        returns_t = torch.as_tensor(returns_all, dtype=torch.float32, device=device)
        if config.normalize_advantage and advantages_t.numel() > 1:
            advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

        log_probs_stacked = torch.stack(
            [log_probs_te[t][i] for i in range(num_envs) for t in range(rollout)]
        )
        values_stacked = torch.stack(
            [values_te[t][i] for i in range(num_envs) for t in range(rollout)]
        )
        entropy_mean = torch.stack(
            [entropies_te[t][i] for i in range(num_envs) for t in range(rollout)]
        ).mean()

        policy_loss = -(log_probs_stacked * advantages_t).mean()
        value_loss = 0.5 * (returns_t - values_stacked).pow(2).mean()
        loss = policy_loss + config.value_coef * value_loss - config.entropy_coef * entropy_mean

        optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
        optimizer.step()

        # --- logging ----------------------------------------------------------
        if update % config.log_interval == 0:
            metrics = {
                'environment_steps': env_steps,
                'train/mean_coverage_rate': float(np.mean(rollout_coverage))
                if rollout_coverage
                else 0.0,
                'train/policy_loss': float(policy_loss.detach().cpu()),
                'train/value_loss': float(value_loss.detach().cpu()),
                'train/entropy': float(entropy_mean.detach().cpu()),
                'train/grad_norm': float(grad_norm),
                'train/mean_reward': float(rewards_te.mean()),
                'train/lr': float(lr_now),
            }
            if completed_returns:
                metrics['train/episode_return'] = float(np.mean(completed_returns[-25:]))
                metrics['train/episode_coverage_rate'] = float(np.mean(completed_coverage[-25:]))
            print(
                f'[hmvfe_mate_d] update {update}/{total_updates} '
                f'steps={env_steps} '
                f"cov={metrics['train/mean_coverage_rate']:.3f} "
                f"pi={metrics['train/policy_loss']:.3f} "
                f"vf={metrics['train/value_loss']:.3f} "
                f"H={metrics['train/entropy']:.3f} lr={lr_now:.2e}"
            )
            if run is not None:
                run.log(metrics, step=env_steps)

        # --- evaluation -------------------------------------------------------
        if update % config.eval_interval == 0:
            eval_metrics = evaluate(model, config, num_episodes=config.eval_episodes)
            eval_metrics['environment_steps'] = env_steps
            print(
                f'[hmvfe_mate_d]   eval cov={eval_metrics["eval/mean_coverage_rate"]:.3f} '
                f'transport={eval_metrics["eval/mean_transport_rate"]:.3f} '
                f'delivered={eval_metrics["eval/mean_num_delivered_cargoes"]:.2f}'
            )
            if run is not None:
                run.log(eval_metrics, step=env_steps)
            if eval_metrics['eval/mean_coverage_rate'] > best_eval:
                best_eval = eval_metrics['eval/mean_coverage_rate']
                _save(model, optimizer, config, env_steps, output_dir / 'best.pt')

        # --- checkpoint -------------------------------------------------------
        if update % config.save_interval == 0:
            _save(model, optimizer, config, env_steps, output_dir / 'latest.pt')
            _save(
                model, optimizer, config, env_steps, output_dir / f'checkpoint-{update:06d}.pt'
            )

    _save(model, optimizer, config, env_steps, output_dir / 'latest.pt')
    venv.close()
    if run is not None:
        run.finish()
    print(f'[hmvfe_mate_d] done. checkpoints in {output_dir}')
    return output_dir


def _save(model, optimizer, config: HMVFEConfig, env_steps: int, path: Path) -> None:
    torch.save(
        {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'config': config.to_dict(),
            'environment_steps': env_steps,
        },
        path,
    )
