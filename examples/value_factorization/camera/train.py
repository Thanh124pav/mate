"""Matched Tune/W&B launcher used by all ten algorithm configurations."""

import argparse
import copy
import os
import sys
import uuid
from math import ceil
from pathlib import Path

import ray
import torch
from ray import tune

from examples.target_agents import configure_greedy_evaluation, configure_target_agent
from examples.utils import SymlinkCheckpointCallback, WandbLoggerCallback


DEBUG = getattr(sys, "gettrace", lambda: None)() is not None
SLURM_CPUS_ON_NODE = int(os.getenv("SLURM_CPUS_ON_NODE", str(os.cpu_count())))
NUM_NODE_CPUS = max(1, min(os.cpu_count(), SLURM_CPUS_ON_NODE))
NUM_NODE_GPUS = torch.cuda.device_count()
NUM_CPUS_FOR_TRAINER = 1
DEFAULT_NUM_WORKERS = 0 if DEBUG else min(8, max(0, NUM_NODE_CPUS - 2))


def resolve_trainable(run):
    """Resolve repo-local algorithms without mutating RLlib's global registry."""

    from ray.rllib.agents.concaveq import ConcaveQFocusTrainer, ConcaveQTrainer
    from ray.rllib.agents.facmac import FACMACFocusTrainer, FACMACTrainer
    from ray.rllib.agents.qfree import QFreeFocusTrainer, QFreeTrainer
    from ray.rllib.agents.resq import ResQFocusTrainer, ResQTrainer
    from ray.rllib.agents.riskq import RiskQFocusTrainer, RiskQTrainer

    return {
        "FACMAC": FACMACTrainer,
        "FACMAC_FOCUSS": FACMACFocusTrainer,
        "RESQ": ResQTrainer,
        "RESQ_FOCUSS": ResQFocusTrainer,
        "RISKQ": RiskQTrainer,
        "RISKQ_FOCUSS": RiskQFocusTrainer,
        "QFREE": QFreeTrainer,
        "QFREE_FOCUSS": QFreeFocusTrainer,
        "CONCAVEQ": ConcaveQTrainer,
        "CONCAVEQ_FOCUSS": ConcaveQFocusTrainer,
    }.get(run, run)


def make_experiment(name, run, config, local_dir):
    return tune.Experiment(
        name=name,
        run=resolve_trainable(run),
        config=copy.deepcopy(config),
        local_dir=Path(local_dir),
        stop={"timesteps_total": 100000},
        checkpoint_score_attr="episode_reward_mean",
        checkpoint_freq=10,
        checkpoint_at_end=True,
        max_failures=0,
    )


def train(
    experiment,
    project=None,
    group=None,
    local_dir=None,
    num_gpus=min(NUM_NODE_GPUS, 0.25),
    num_workers=DEFAULT_NUM_WORKERS,
    num_envs_per_worker=2,
    target_agent=None,
    evaluation_interval=5,
    seed=None,
    timesteps_total=None,
    buffer_capacity=500,
    belief_safeguard_mode=None,
    belief_distribution=None,
    belief_grid_size=None,
    belief_grid_max_delta=None,
    belief_grid_label_std=None,
    belief_position_loss_coeff=None,
    restore=None,
    resume_timesteps_offset=0,
    resume=False,
    env=None,
):
    callbacks = [SymlinkCheckpointCallback()]
    wandb_disabled = os.getenv("WANDB_DISABLED", "").lower() in (
        "1", "true", "yes", "on"
    ) or os.getenv("WANDB_MODE", "").lower() == "disabled"
    if WandbLoggerCallback.is_available() and not wandb_disabled:
        callbacks.append(
            WandbLoggerCallback(
                project=project or ("mate-value-factorization" if not DEBUG else "mate-debug"),
                group=group or experiment.name,
                # Independent Ray sessions can generate identical trial IDs
                # when launched simultaneously with the same seed.  Ray uses
                # that ID as the W&B run ID, which makes concurrent lanes
                # overwrite one another unless we add a process-unique token.
                id=f"{experiment.name.lower()}-{os.getpid()}-{uuid.uuid4().hex[:8]}",
            )
        )
    if not ray.is_initialized():
        # Ray 1.12 does not reliably auto-detect CUDA devices under WSL even
        # when torch.cuda does.  Advertise the devices PyTorch verified so the
        # requested fractional trainer GPU is honored.
        ray.init(
            num_cpus=NUM_NODE_CPUS,
            num_gpus=NUM_NODE_GPUS,
            local_mode=DEBUG,
            include_dashboard=False,
            object_store_memory=int(
                os.getenv("MATE_RAY_OBJECT_STORE_BYTES", str(4 * 1024**3))
            ),
        )
    available_cpus = round(ray.cluster_resources()["CPU"])
    available_gpus = ray.cluster_resources().get("GPU", 0.0)
    num_workers = max(0, min(int(num_workers), available_cpus - NUM_CPUS_FOR_TRAINER))
    experiment.spec["config"].update(
        num_cpus_for_driver=NUM_CPUS_FOR_TRAINER,
        num_gpus=min(float(num_gpus), available_gpus),
        num_gpus_per_worker=0,
        num_workers=num_workers,
        num_envs_per_worker=int(num_envs_per_worker),
        buffer_size=ceil(float(buffer_capacity) / max(num_workers, 1)),
    )
    belief_overrides = {
        "belief_safeguard_mode": belief_safeguard_mode,
        "belief_distribution": belief_distribution,
        "belief_grid_size": belief_grid_size,
        "belief_grid_max_delta": belief_grid_max_delta,
        "belief_grid_label_std": belief_grid_label_std,
        "belief_position_loss_coeff": belief_position_loss_coeff,
    }
    if any(value is not None for value in belief_overrides.values()):
        focus_config = experiment.spec["config"].get("focus", {})
        if not focus_config.get("enabled", False):
            raise ValueError(
                "belief options are valid only for +FOCUS configurations"
            )
        focus_config.update(
            {key: value for key, value in belief_overrides.items() if value is not None}
        )
        experiment.spec["config"]["focus"] = focus_config
    configure_target_agent(experiment.spec["config"], target_agent)
    configure_greedy_evaluation(experiment.spec["config"], evaluation_interval)
    if restore is not None and resume_timesteps_offset:
        # RLlib restores policy/model/optimizer state, but Ray 1.12 resets the
        # Tune sampling counter used by the epsilon schedule.  Shift the
        # remaining schedule so resumed training does not restart at epsilon=1.
        exploration = copy.deepcopy(
            experiment.spec["config"].get("exploration_config", {})
        )
        initial = float(exploration.get("initial_epsilon", 1.0))
        final = float(exploration.get("final_epsilon", 0.05))
        duration = max(1, int(exploration.get("epsilon_timesteps", 1)))
        elapsed = min(max(0, int(resume_timesteps_offset)), duration)
        exploration["initial_epsilon"] = initial + (final - initial) * (
            elapsed / duration
        )
        exploration["epsilon_timesteps"] = max(1, duration - elapsed)
        experiment.spec["config"]["exploration_config"] = exploration
    if seed is not None:
        experiment.spec["config"]["seed"] = (
            tune.grid_search(seed) if isinstance(seed, (list, tuple)) else seed
        )
    if timesteps_total is not None:
        experiment.spec["stop"]["timesteps_total"] = int(timesteps_total)
    if local_dir is not None:
        experiment.spec["local_dir"] = str(local_dir)
    if restore is not None:
        # ``tune.run`` ignores its ``restore=`` keyword when passed an already
        # constructed Experiment in Ray 1.12.  The restore path must live in
        # the Experiment spec so it reaches the generated Trial.
        experiment.spec["restore"] = os.path.abspath(os.path.expanduser(restore))
    if env is not None:
        # Keep the registered RLlib env name unchanged; this selects the MATE
        # YAML consumed by ``make_env``.  Without this override every launcher
        # silently used MATE-4v8-9.yaml regardless of the queue job label.
        experiment.spec["config"]["env_config"]["config"] = str(env)
    return tune.run(
        experiment,
        metric="episode_reward_mean",
        mode="max",
        callbacks=callbacks,
        verbose=3,
        resume=resume,
    )


def main_for(name, run, config, package, default_local_dir):
    parser = argparse.ArgumentParser(prog=f"python -m {package}")
    parser.add_argument("--project", default=None)
    parser.add_argument("--group", default=None)
    parser.add_argument("--local-dir", default=str(default_local_dir))
    parser.add_argument("--num-gpus", type=float, default=min(NUM_NODE_GPUS, 0.25))
    parser.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS)
    parser.add_argument("--num-envs-per-worker", type=int, default=2)
    parser.add_argument("--target-agent", choices=("greedy", "evasive"), default=None)
    parser.add_argument("--evaluation-interval", type=int, default=5)
    parser.add_argument("--timesteps-total", type=float, default=100000)
    parser.add_argument("--seed", type=int, nargs="*", default=None)
    parser.add_argument("--buffer-capacity", type=float, default=500)
    parser.add_argument(
        "--env",
        default=None,
        help="MATE YAML environment configuration (for example MATE-4v8-9-T100.yaml)",
    )
    if config.get("focus", {}).get("enabled", False):
        parser.add_argument(
            "--belief-safeguard-mode",
            choices=("zero", "zero_ema", "zero_ema_blend"),
            default=None,
        )
        parser.add_argument(
            "--belief-distribution",
            choices=("gaussian", "categorical_grid"),
            default=None,
        )
        parser.add_argument("--belief-grid-size", type=int, default=None)
        parser.add_argument("--belief-grid-max-delta", type=float, default=None)
        parser.add_argument("--belief-grid-label-std", type=float, default=None)
        parser.add_argument("--belief-position-loss-coeff", type=float, default=None)
    parser.add_argument("--restore", default=None)
    parser.add_argument("--resume-timesteps-offset", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    experiment = make_experiment(name, run, config, args.local_dir)
    return train(experiment, **vars(args))
