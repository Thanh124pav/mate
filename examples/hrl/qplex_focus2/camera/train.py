#!/usr/bin/env python3

# Run: python -m examples.hrl.qplex_focus.camera.train

import argparse
import copy
import os
import sys
from math import ceil
from pathlib import Path

import ray
import torch
from ray import tune

from examples.hrl.qplex_focus2.camera.config import config
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
assert NUM_NODE_CPUS >= 1
NUM_NODE_GPUS = torch.cuda.device_count()

# Training resources
PRESERVED_NUM_CPUS = 1  # for raylet
NUM_CPUS_FOR_TRAINER = 1
NUM_GPUS_FOR_TRAINER = min(NUM_NODE_GPUS, 1.0)  # can be overridden by command line arguments

MAX_NUM_CPUS_FOR_WORKER = max(0, NUM_NODE_CPUS - PRESERVED_NUM_CPUS - NUM_CPUS_FOR_TRAINER)
MAX_NUM_WORKERS = min(32, MAX_NUM_CPUS_FOR_WORKER)  # use at most 32 workers
NUM_WORKERS = MAX_NUM_WORKERS if not DEBUG else 0  # can be overridden by command line arguments


experiment = tune.Experiment(
    name='HRL-QPLEX-FOCUS2',
    run='QPLEX_FOCUS2',
    config=copy.deepcopy(config),
    local_dir=LOCAL_DIR,
    stop={'timesteps_total': 10e6},
    checkpoint_score_attr='episode_reward_mean',
    checkpoint_freq=20,
    checkpoint_at_end=True,
    max_failures=3,
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
    env = None,
):
    tune_callbacks = [SymlinkCheckpointCallback()]
    if WandbLoggerCallback.is_available():
        project = project or ('mate' if not DEBUG else 'mate-debug')
        group = group or f'hrl.qplex_focus2.camera.{experiment.name}'
        tune_callbacks.append(WandbLoggerCallback(project=project, group=group))

    if not ray.is_initialized():
        ray.init(num_cpus=NUM_NODE_CPUS, num_gpus = NUM_NODE_GPUS, local_mode=DEBUG)

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
    if restore is not None:
        # tune.run ignores `restore` when passed an Experiment object (Ray 1.x bug):
        # it only applies restore when constructing a new Experiment internally.
        experiment.spec['restore'] = restore

    # Update replay buffer capacity
    experiment.spec['config'].update(buffer_size=ceil(buffer_capacity / max(num_workers, 1)))

    analysis = tune.run(
        experiment,
        metric='episode_reward_mean',
        mode='max',
        callbacks=tune_callbacks,
        verbose=3,
        resume=resume
    )
    with open('examples/hrl/qplex_focus2/config.txt', 'w', encoding='utf-8') as f:
        f.write(str(experiment.spec['config']))
    return analysis


def main():
    parser = argparse.ArgumentParser(prog=f'python -m {__package__}')
    parser.add_argument(
        '--project', type=str, metavar='PROJECT', default=None, help='W&B project name'
    )
    parser.add_argument('--group', type=str, metavar='GROUP', default=None, help='W&B group name')
    parser.add_argument(
        '--local-dir',
        type=str,
        metavar='DIR',
        default=LOCAL_DIR,
        help='Local directory for the experiment (default: %(default)s)',
    )
    parser.add_argument(
        '--num-gpus',
        type=float,
        metavar='GPU',
        default=NUM_GPUS_FOR_TRAINER,
        help='number of GPUs for trainer (default: %(default)g)',
    )
    parser.add_argument(
        '--num-workers',
        type=int,
        metavar='WORKER',
        default=NUM_WORKERS,
        help='number of rollout workers (default: %(default)d)',
    )
    parser.add_argument(
        '--num-envs-per-worker',
        type=int,
        metavar='ENV',
        default=8,
        help='number of environments per rollout worker (default: %(default)d)',
    )
    parser.add_argument(
        '--timesteps-total',
        type=float,
        metavar='STEP',
        default=10e6,
        help='number of environment steps (default: %(default).1e)',
    )
    parser.add_argument(
        '--seed', type=int, metavar='SEED', nargs='*', default=None, help='the global seed(s)'
    )
    parser.add_argument(
        '--buffer-capacity',
        type=float,
        metavar='EPISODE',
        default=200,
        help='capacity for the replay buffer (default: %(default).1e episodes)',
    )
    parser.add_argument(
        '--restore',
        type=str,
        metavar='PATH',
        default=None,
        help='path to checkpoint directory to restore from',
    )
    parser.add_argument(
        '--resume',
        action='store_true',
        help='resume from latest checkpoint in local_dir',
    )
    parser.add_argument(
        '--env',
        default=None,
        help='.yaml environment to run experiments'
    )

    args = parser.parse_args()

    analysis = train(experiment, **vars(args))
    return analysis


if __name__ == '__main__':
    main()
