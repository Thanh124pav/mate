#!/usr/bin/env python3

import argparse
import copy
import os
import sys
from pathlib import Path

import torch
from ray import tune

from examples.smpe2.camera.train import train as _train
from examples.smpe2_focus.camera.config import config


DEBUG = getattr(sys, "gettrace", lambda: None)() is not None
HERE = Path(__file__).absolute().parent
REPO_ROOT = HERE.parents[2]
LOCAL_DIR = HERE / "ray_results"
if DEBUG:
    LOCAL_DIR = LOCAL_DIR / "debug"

SLURM_CPUS_ON_NODE = int(os.getenv("SLURM_CPUS_ON_NODE", str(os.cpu_count())))
NUM_NODE_CPUS = max(1, min(os.cpu_count(), SLURM_CPUS_ON_NODE))
NUM_NODE_GPUS = torch.cuda.device_count()
NUM_GPUS_FOR_TRAINER = min(NUM_NODE_GPUS, 0.25)
NUM_CPUS_FOR_TRAINER = 1
PRESERVED_NUM_CPUS = 1
MAX_NUM_CPUS_FOR_WORKER = max(0, NUM_NODE_CPUS - PRESERVED_NUM_CPUS - NUM_CPUS_FOR_TRAINER)
NUM_WORKERS = min(32, MAX_NUM_CPUS_FOR_WORKER) if not DEBUG else 0


def _ensure_local_repo_on_pythonpath():
    repo = str(REPO_ROOT)
    paths = [path for path in os.environ.get("PYTHONPATH", "").split(os.pathsep) if path]
    if repo not in paths:
        os.environ["PYTHONPATH"] = os.pathsep.join([repo, *paths])


experiment = tune.Experiment(
    name="SMPE2_FOCUS",
    run="PPO",
    config=copy.deepcopy(config),
    local_dir=LOCAL_DIR,
    stop={"timesteps_total": 10e6},
    checkpoint_score_attr="episode_reward_mean",
    checkpoint_freq=20,
    checkpoint_at_end=True,
    max_failures=-1,
)


def train(
    experiment=experiment,
    project=None,
    group=None,
    local_dir=None,
    num_gpus=NUM_GPUS_FOR_TRAINER,
    num_workers=NUM_WORKERS,
    num_envs_per_worker=8,
    seed=None,
    timesteps_total=None,
):
    _ensure_local_repo_on_pythonpath()
    group = group or f"smpe2_focus.camera.{experiment.name}"
    return _train(
        experiment,
        project=project,
        group=group,
        local_dir=local_dir,
        num_gpus=num_gpus,
        num_workers=num_workers,
        num_envs_per_worker=num_envs_per_worker,
        seed=seed,
        timesteps_total=timesteps_total,
    )


def main():
    parser = argparse.ArgumentParser(prog=f"python -m {__package__}")
    parser.add_argument("--project", type=str, metavar="PROJECT", default=None)
    parser.add_argument("--group", type=str, metavar="GROUP", default=None)
    parser.add_argument("--local-dir", type=str, metavar="DIR", default=LOCAL_DIR)
    parser.add_argument("--num-gpus", type=float, metavar="GPU", default=NUM_GPUS_FOR_TRAINER)
    parser.add_argument("--num-workers", type=int, metavar="WORKER", default=NUM_WORKERS)
    parser.add_argument("--num-envs-per-worker", type=int, metavar="ENV", default=8)
    parser.add_argument("--timesteps-total", type=float, metavar="STEP", default=10e6)
    parser.add_argument("--seed", type=int, metavar="SEED", nargs="*", default=None)
    return train(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
