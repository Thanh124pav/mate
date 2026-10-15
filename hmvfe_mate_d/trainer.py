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
from hmvfe_mate_d.focus_adapter import (
    HMVFEFocusAdapter,
    build_hmvfe_focus_engine,
    focus_diagnostics,
    policy_loss_from_log_prob,
    synthetic_responsibility,
    weighted_actor_log_prob,
    execution_camera_logit_bias,
)
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
    wandb.define_metric('focus/*', step_metric='environment_steps')
    return run


def _append_metrics(path: Path, split: str, metrics: dict) -> None:
    """Persist metrics even when W&B is disabled (one JSON object per line)."""

    row = {'split': split, **metrics}
    with open(path, 'a', encoding='utf-8') as handle:
        handle.write(json.dumps(row, sort_keys=True) + '\n')


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
    metrics_path = output_dir / 'metrics.jsonl'
    metrics_path.write_text('', encoding='utf-8')

    num_envs = max(1, int(config.num_envs))
    venv = SyncVectorCoordinatorEnv(config, num_envs)
    feature_dim = venv.feature_dim
    focus_mode = str(config.focus_mode).lower()
    focus_adapter = HMVFEFocusAdapter(
        eta=config.focus_eta,
        use_confidence=config.focus_use_confidence,
        eps=config.focus_eps,
        weight_formula=config.focus_weight_formula,
        rho_temperature=config.focus_rho_temperature,
    )
    if config.focus_enabled and config.focus_strict:
        if focus_mode == 'uniform':
            raise ValueError('focus_strict=True disallows focus_mode=uniform because it is vanilla HMVFE.')
        if focus_adapter.weight_formula == 'affine' and float(config.focus_eta) == 0.0:
            raise ValueError('focus_strict=True disallows focus_eta=0 with affine weights because it is vanilla HMVFE.')

    focus_engine = build_hmvfe_focus_engine(config, venv, device=device)

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
    optimizer_groups = [
        {'params': model.parameters(), 'lr': config.learning_rate, 'initial_lr': config.learning_rate}
    ]
    if focus_engine is not None and focus_engine.has_learned_belief:
        belief_lr = config.focus_belief_lr or config.learning_rate
        optimizer_groups.append(
            {'params': list(focus_engine.parameters()), 'lr': belief_lr, 'initial_lr': belief_lr}
        )
    optimizer = optim.Adam(optimizer_groups)
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
        f'anneal_lr={config.anneal_lr} focus={config.focus_enabled}:{focus_mode}'
    )

    for update in range(1, total_updates + 1):
        if focus_engine is not None:
            # Credits in this rollout use only gate statistics from completed
            # learner updates.  The current rollout updates the EMA once below.
            focus_engine.set_learner_update(update - 1)
        # --- linear LR annealing ---------------------------------------------
        if config.anneal_lr:
            lr_factor = 1.0 - (update - 1) / total_updates
            for group in optimizer.param_groups:
                group['lr'] = group['initial_lr'] * lr_factor
        lr_now = optimizer.param_groups[0]['lr']

        rollout = config.rollout_length
        # buffers indexed [t][env]
        log_probs_te = [[None] * num_envs for _ in range(rollout)]
        camera_log_probs_te = [[None] * num_envs for _ in range(rollout)]
        focus_rhos_te = [[None] * num_envs for _ in range(rollout)]
        focus_confidence_te = [[None] * num_envs for _ in range(rollout)]
        focus_valid_te = [[None] * num_envs for _ in range(rollout)]
        focus_states_t = [None] * rollout
        focus_next_states_t = [None] * rollout
        values_te = [[None] * num_envs for _ in range(rollout)]
        entropies_te = [[None] * num_envs for _ in range(rollout)]
        rewards_te = np.zeros((rollout, num_envs))
        dones_te = np.zeros((rollout, num_envs))
        values_f_te = np.zeros((rollout, num_envs))
        rollout_coverage = []

        for t in range(rollout):
            actions = []
            focus_state = None
            execution_bias = None
            inferred_focus = None
            if config.focus_enabled and focus_mode in ('real', 'shuffled'):
                focus_state = venv.global_state()
                if focus_engine is not None:
                    inferred_focus = focus_engine.infer(
                        torch.as_tensor(
                            focus_state, dtype=torch.float32, device=device
                        )
                    )
                    execution_rho = inferred_focus.rho.squeeze(1)
                    if focus_mode == 'shuffled':
                        execution_rho = synthetic_responsibility(
                            'shuffled',
                            (num_envs, venv.num_cameras),
                            device,
                            torch.float32,
                            config.focus_eps,
                            base_rho=execution_rho,
                        )
                    if float(config.focus_execution_eta) != 0.0:
                        execution_bias = execution_camera_logit_bias(
                            execution_rho,
                            config.focus_execution_eta,
                            config.focus_eps,
                        )
            for i in range(num_envs):
                state = torch.as_tensor(obs[i], dtype=torch.float32, device=device)
                action, log_prob, entropy, value, camera_log_prob = model.act(
                    state,
                    return_per_camera_log_prob=True,
                    camera_logit_bias=(execution_bias[i] if execution_bias is not None else None),
                )
                actions.append(action.detach().cpu().numpy())
                log_probs_te[t][i] = log_prob
                camera_log_probs_te[t][i] = camera_log_prob
                entropies_te[t][i] = entropy
                values_te[t][i] = value.squeeze()
                values_f_te[t, i] = float(value.detach().squeeze().cpu())

            next_obs, rewards, dones, infos = venv.step(actions)
            env_steps += config.frame_skip * num_envs

            if config.focus_enabled:
                confidence_t = torch.ones(num_envs, dtype=torch.float32, device=device)
                if focus_mode in ('real', 'shuffled'):
                    assert focus_engine is not None
                    assert inferred_focus is not None
                    focus_next_state = venv.last_next_global_state
                    focus_states_t[t] = focus_state
                    focus_next_states_t[t] = focus_next_state
                    # Actor credit is the causal current-state inference made
                    # before the action.  ``focus_next_state`` is retained only
                    # for the auxiliary supervised belief loss below.
                    rho_t = inferred_focus.rho.squeeze(1).to(
                        device=device, dtype=torch.float32
                    )
                    confidence_t = inferred_focus.confidence.squeeze(1).to(
                        device=device, dtype=torch.float32
                    )
                    valid_t = inferred_focus.valid.squeeze(1).to(
                        device=device, dtype=torch.bool
                    )
                    if focus_mode == 'shuffled':
                        rho_t = synthetic_responsibility(
                            'shuffled',
                            (num_envs, venv.num_cameras),
                            device,
                            torch.float32,
                            config.focus_eps,
                            base_rho=rho_t,
                        )
                else:
                    rho_t = synthetic_responsibility(
                        focus_mode,
                        (num_envs, venv.num_cameras),
                        device,
                        torch.float32,
                        config.focus_eps,
                    )
                    valid_t = torch.ones(num_envs, dtype=torch.bool, device=device)
                if not torch.isfinite(rho_t).all() or not torch.isfinite(confidence_t).all():
                    raise RuntimeError('FOCUS produced non-finite responsibility or confidence')
                for i in range(num_envs):
                    focus_rhos_te[t][i] = rho_t[i]
                    focus_confidence_te[t][i] = confidence_t[i]
                    focus_valid_te[t][i] = valid_t[i]

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
        camera_log_probs_stacked = torch.stack(
            [camera_log_probs_te[t][i] for i in range(num_envs) for t in range(rollout)]
        )
        values_stacked = torch.stack(
            [values_te[t][i] for i in range(num_envs) for t in range(rollout)]
        )
        entropy_mean = torch.stack(
            [entropies_te[t][i] for i in range(num_envs) for t in range(rollout)]
        ).mean()

        actor_log_probs = log_probs_stacked
        actor_loss_mask = None
        focus_metrics = {}
        belief_metrics = {}
        belief_loss = torch.zeros((), dtype=torch.float32, device=device)
        if config.focus_enabled:
            focus_rho_stacked = torch.stack(
                [focus_rhos_te[t][i] for i in range(num_envs) for t in range(rollout)]
            )
            focus_confidence_stacked = torch.stack(
                [focus_confidence_te[t][i] for i in range(num_envs) for t in range(rollout)]
            )
            focus_valid_stacked = torch.stack(
                [focus_valid_te[t][i] for i in range(num_envs) for t in range(rollout)]
            )
            confidence_arg = focus_confidence_stacked if config.focus_use_confidence else None
            actor_log_probs, focus_weights = weighted_actor_log_prob(
                log_probs_stacked,
                camera_log_probs_stacked,
                focus_adapter,
                rho=focus_rho_stacked,
                confidence=confidence_arg,
            )
            if not torch.isfinite(actor_log_probs).all() or not torch.isfinite(focus_weights).all():
                raise RuntimeError('FOCUS produced non-finite actor log-probability or weights')
            if config.focus_strict:
                actor_loss_mask = focus_valid_stacked.bool()
                if not bool(actor_loss_mask.any().item()):
                    raise RuntimeError('FOCUS strict mode found no valid responsibility decisions in the update batch.')
            focus_metrics = focus_diagnostics(
                focus_rho_stacked,
                focus_weights,
                actor_log_probs,
                log_probs_stacked,
                confidence=confidence_arg,
                eps=config.focus_eps,
            )
            focus_metrics['focus/valid_ratio'] = float(focus_valid_stacked.float().mean().detach().cpu())
            focus_metrics['focus/rho_temperature'] = float(config.focus_rho_temperature)
            if (
                focus_engine is not None
                and focus_engine.has_learned_belief
                and focus_states_t[0] is not None
            ):
                belief_out = focus_engine.belief_loss_from_sequence(
                    torch.as_tensor(np.stack(focus_states_t, axis=1), dtype=torch.float32, device=device),
                    torch.as_tensor(np.stack(focus_next_states_t, axis=1), dtype=torch.float32, device=device),
                )
                if belief_out.belief_loss is not None:
                    belief_loss = belief_out.belief_loss.to(device=device, dtype=torch.float32)
                belief_metrics['focus/belief_loss'] = float(belief_loss.detach().cpu())
                belief_metrics['focus/beta_belief'] = float(config.focus_beta_belief)
                if len(optimizer.param_groups) > 1:
                    belief_metrics['focus/belief_lr'] = float(optimizer.param_groups[1]['lr'])
                for key, value in (belief_out.belief_stats or {}).items():
                    metric_key = 'focus/' + key.replace('focus_belief_', 'belief_')
                    belief_metrics[metric_key] = float(value)

        policy_loss = policy_loss_from_log_prob(actor_log_probs, advantages_t, mask=actor_loss_mask)
        value_loss = 0.5 * (returns_t - values_stacked).pow(2).mean()
        loss = (
            policy_loss
            + config.value_coef * value_loss
            - config.entropy_coef * entropy_mean
            + config.focus_beta_belief * belief_loss
        )

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
            metrics.update(focus_metrics)
            metrics.update(belief_metrics)
            if completed_returns:
                metrics['train/episode_return'] = float(np.mean(completed_returns[-25:]))
                metrics['train/episode_coverage_rate'] = float(np.mean(completed_coverage[-25:]))
            _append_metrics(metrics_path, 'train', metrics)
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
            eval_metrics = evaluate(
                model,
                config,
                num_episodes=config.eval_episodes,
                focus_engine=focus_engine,
            )
            eval_metrics['environment_steps'] = env_steps
            _append_metrics(metrics_path, 'eval', eval_metrics)
            print(
                f'[hmvfe_mate_d]   eval cov={eval_metrics["eval/mean_coverage_rate"]:.3f} '
                f'real_cov={eval_metrics["eval/mean_real_coverage_rate"]:.3f} '
                f'transport={eval_metrics["eval/mean_transport_rate"]:.3f} '
                f'delivered={eval_metrics["eval/mean_num_delivered_cargoes"]:.2f}'
            )
            if run is not None:
                run.log(eval_metrics, step=env_steps)
            if eval_metrics['eval/mean_coverage_rate'] > best_eval:
                best_eval = eval_metrics['eval/mean_coverage_rate']
                _save(model, optimizer, config, env_steps, output_dir / 'best.pt', focus_engine=focus_engine)

        # --- checkpoint -------------------------------------------------------
        if update % config.save_interval == 0:
            _save(model, optimizer, config, env_steps, output_dir / 'latest.pt', focus_engine=focus_engine)
            _save(
                model,
                optimizer,
                config,
                env_steps,
                output_dir / f'checkpoint-{update:06d}.pt',
                focus_engine=focus_engine,
            )

    _save(model, optimizer, config, env_steps, output_dir / 'latest.pt', focus_engine=focus_engine)
    venv.close()
    if run is not None:
        run.finish()
    print(f'[hmvfe_mate_d] done. checkpoints in {output_dir}')
    return output_dir


def _save(
    model,
    optimizer,
    config: HMVFEConfig,
    env_steps: int,
    path: Path,
    focus_engine=None,
) -> None:
    payload = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'config': config.to_dict(),
        'environment_steps': env_steps,
    }
    if focus_engine is not None:
        payload['focus_engine'] = focus_engine.state_dict()
        payload['focus_runtime_state'] = focus_engine.get_runtime_state()
    torch.save(payload, path)
