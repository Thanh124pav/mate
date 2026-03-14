#!/usr/bin/env python3
"""HiTMAC — Phase 1: Train QPLEX executors với greedy coordinator.

Theo paper HiTMAC (NeurIPS 2020, Section 3.3 "Training Strategy"):

  Phase 1 (ĐÂY):
      Train executor độc lập với heuristic pseudo-goal thay coordinator.
      Mỗi k bước, targets trong tầm nhìn của sensor được tự động chọn làm goal.
      Executor được train bằng RL (paper: A3C, chúng ta: QPLEX).

  Phase 2 (sau khi Phase 1 xong):
      Train coordinator với executor đã được fix (dùng geometric/scripted executor
      để tăng tốc độ — paper gọi là "programmed strategy").
      Dùng code có sẵn: python -m examples.hrl.mappo.camera.train
      (HRL-MAPPO dùng HierarchicalCamera.executor() là scripted executor)

  Inference:
      Load cả 2 checkpoint vào HiTMACQPLEXCameraAgent.

Chạy Phase 1:
    python -m examples.hitmac.camera.train --env MATE-4v8-9.yaml

Chạy Phase 2 (sau khi Phase 1 done):
    python -m examples.hrl.mappo.camera.train --env MATE-4v8-9.yaml
"""

import argparse
import copy
import os
import sys
from math import ceil
from pathlib import Path

import ray
import torch
from ray import tune

from examples.hitmac.camera.config import config
from examples.utils import SymlinkCheckpointCallback, WandbLoggerCallback


DEBUG = getattr(sys, 'gettrace', lambda: None)() is not None

HERE = Path(__file__).absolute().parent
LOCAL_DIR = HERE / 'ray_results'
if DEBUG:
    print(f'DEBUG MODE: {DEBUG}')
    LOCAL_DIR = LOCAL_DIR / 'debug'


# Node resources
SLURM_CPUS_ON_NODE = int(os.getenv('SLURM_CPUS_ON_NODE', str(os.cpu_count())))
NUM_NODE_CPUS = max(1, min(os.cpu_count(), SLURM_CPUS_ON_NODE))
assert NUM_NODE_CPUS >= 2
NUM_NODE_GPUS = torch.cuda.device_count()

# Training resources
PRESERVED_NUM_CPUS = 1       # cho raylet
NUM_CPUS_FOR_TRAINER = 1
NUM_GPUS_FOR_TRAINER = min(NUM_NODE_GPUS, 0.25)

MAX_NUM_CPUS_FOR_WORKER = max(0, NUM_NODE_CPUS - PRESERVED_NUM_CPUS - NUM_CPUS_FOR_TRAINER)
MAX_NUM_WORKERS = min(32, MAX_NUM_CPUS_FOR_WORKER)
NUM_WORKERS = MAX_NUM_WORKERS if not DEBUG else 0


experiment = tune.Experiment(
    name='HiTMAC-QPLEX',
    run='QPLEX_V2',
    config=copy.deepcopy(config),
    local_dir=LOCAL_DIR,
    stop={'timesteps_total': 10e6},
    checkpoint_score_attr='episode_reward_mean',
    checkpoint_freq=20,
    checkpoint_at_end=True,
    max_failures=-1,
)


def train(
    experiment,
    project=None,
    group=None,
    local_dir=None,
    num_gpus=NUM_GPUS_FOR_TRAINER,
    num_workers=NUM_WORKERS,
    num_envs_per_worker=8,
    seed=None,
    timesteps_total=None,
    buffer_capacity=2000,
    restore=None,
    resume=False,
    env=None,
    coordinator_checkpoint=None,
    coord_period=None,
):
    tune_callbacks = [SymlinkCheckpointCallback()]
    if WandbLoggerCallback.is_available():
        project = project or ('mate' if not DEBUG else 'mate-debug')
        group = group or f'hitmac.qplex.camera.{experiment.name}'
        tune_callbacks.append(WandbLoggerCallback(project=project, group=group))

    if not ray.is_initialized():
        ray.init(num_cpus=NUM_NODE_CPUS, 
                 num_gpus=NUM_NODE_GPUS, local_mode=DEBUG)

    num_ray_cpus = round(ray.cluster_resources()['CPU'])
    num_ray_gpus = ray.cluster_resources().get('GPU', 0.0)
    num_gpus = min(num_gpus, num_ray_gpus)
    num_workers = max(0, min(num_workers, num_ray_cpus - NUM_CPUS_FOR_TRAINER))

    experiment.spec['config'].update(
        num_cpus_for_driver=NUM_CPUS_FOR_TRAINER,
        num_gpus=num_gpus,
        num_gpus_per_worker=0,
        num_workers=num_workers,
        num_envs_per_worker=num_envs_per_worker,
    )
    if seed is not None:
        seed = tune.grid_search(seed) if isinstance(seed, (list, tuple)) else seed
        experiment.spec['config'].update(seed=seed)
    if timesteps_total is not None:
        experiment.spec['stop'].update(timesteps_total=timesteps_total)
    if local_dir is not None:
        experiment.spec['local_dir'] = local_dir
    if env is not None:
        experiment.spec['config']['env_config']['config'] = env
    if coordinator_checkpoint is not None:
        experiment.spec['config']['env_config']['coordinator_checkpoint'] = (
            coordinator_checkpoint
        )
    if coord_period is not None:
        experiment.spec['config']['env_config']['coord_period'] = coord_period

    # Chia buffer_size theo số workers
    experiment.spec['config'].update(
        buffer_size=ceil(buffer_capacity / max(num_workers, 1))
    )

    analysis = tune.run(
        experiment,
        metric='episode_reward_mean',
        mode='max',
        callbacks=tune_callbacks,
        restore=restore,
        resume=resume,
        verbose=3,
    )
    return analysis


def main():
    parser = argparse.ArgumentParser(prog=f'python -m {__package__}')
    parser.add_argument('--project', type=str, default=None, help='W&B project name')
    parser.add_argument('--group', type=str, default=None, help='W&B group name')
    parser.add_argument(
        '--local-dir', type=str, default=LOCAL_DIR,
        help='Local directory for the experiment (default: %(default)s)',
    )
    parser.add_argument(
        '--num-gpus', type=float, default=NUM_GPUS_FOR_TRAINER,
        metavar='GPU', help='number of GPUs for trainer (default: %(default)g)',
    )
    parser.add_argument(
        '--num-workers', type=int, default=NUM_WORKERS,
        metavar='WORKER', help='number of rollout workers (default: %(default)d)',
    )
    parser.add_argument(
        '--num-envs-per-worker', type=int, default=8,
        metavar='ENV', help='number of environments per worker (default: %(default)d)',
    )
    parser.add_argument(
        '--timesteps-total', type=float, default=10e6,
        metavar='STEP', help='total env steps (default: %(default).1e)',
    )
    parser.add_argument(
        '--seed', type=int, nargs='*', default=None, metavar='SEED',
        help='global seed(s)',
    )
    parser.add_argument(
        '--buffer-capacity', type=float, default=200,
        metavar='EPISODE', help='replay buffer capacity in episodes (default: %(default).0f)',
    )
    parser.add_argument(
        '--restore', type=str, default=None, metavar='PATH',
        help='path to executor checkpoint to restore from',
    )
    parser.add_argument(
        '--resume', action='store_true',
        help='resume from latest checkpoint in local_dir',
    )
    parser.add_argument(
        '--env', default=None,
        help='MATE .yaml config file (e.g. MATE-4v8-9.yaml)',
    )
    parser.add_argument(
        '--coordinator-checkpoint', type=str, default=None, metavar='PATH',
        help='Path to trained MAPPO coordinator checkpoint (Phase 1 output). '
             'Omit to use greedy fallback during executor training.',
    )
    parser.add_argument(
        '--coord-period', type=int, default=None, metavar='N',
        help='coordinator updates task assignment every N env steps (default: 5)',
    )

    args = parser.parse_args()
    analysis = train(experiment, **vars(args))
    return analysis


if __name__ == '__main__':
    main()
