#!/usr/bin/env python3
"""Run all HRL experiments: 9 algorithms × 2 envs = 18 runs.

Each run logs to wandb with project: mate-claude-<env>, group: <algorithm>

Usage:
    python scripts/run_all_experiments.py
    python scripts/run_all_experiments.py --timesteps 3e5   # shorter runs
    python scripts/run_all_experiments.py --only duelmix     # single algorithm
"""

import argparse
import copy
import os
import sys
import gc
from math import ceil
from pathlib import Path

# Ensure mate project root is in sys.path
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
os.chdir(_PROJECT_ROOT)

# Must be set before ray import
os.environ.setdefault('RAY_OBJECT_STORE_MEMORY', str(2 * 1024 * 1024 * 1024))  # 2GB

import ray
import torch
from ray import tune

from examples.utils import SymlinkCheckpointCallback


ENVS = ['MATE-4v5-0.yaml', 'MATE-4v8-9.yaml']
ENV_SHORT = {'MATE-4v5-0.yaml': '4v5-0', 'MATE-4v8-9.yaml': '4v8-9'}

# Algorithm name -> (import path for config, ray run type)
ALG_REGISTRY = {
    'duelmix':  ('examples.hrl.duelmix.camera.config',  'DUELMIX'),
    'spectra':  ('examples.hrl.spectra.camera.config',   'SPECTRA'),
    'qplex':    ('examples.hrl.qplex.camera.config',     'QPLEX'),
    'qplex_v2': ('examples.hrl.qplex_v2.camera.config',  'QPLEX_V2'),
    'qmix':     ('examples.hrl.qmix.camera.config',      'QMIX'),
    'iql':      ('examples.hrl.iql.camera.config',        'DQN'),
    'mappo':    ('examples.hrl.mappo.camera.config',      'PPO'),
    'ippo':     ('examples.hrl.ippo.camera.config',       'PPO'),
    'tarmac':   ('examples.hrl.tarmac.camera.config',     'PPO'),
}

# Value decomposition methods (off-policy, need buffer size adjustment)
VALUE_DECOMP_ALGS = {'duelmix', 'spectra', 'qplex', 'qplex_v2', 'qmix', 'iql'}
# On-policy methods (need batch size adjustment)
ON_POLICY_ALGS = {'mappo', 'ippo', 'tarmac'}

NUM_CPUS = max(1, os.cpu_count())
NUM_GPUS = torch.cuda.device_count()
PRESERVED_NUM_CPUS = 1  # for raylet
NUM_CPUS_FOR_TRAINER = 1
NUM_WORKERS = max(0, min(32, NUM_CPUS - PRESERVED_NUM_CPUS - NUM_CPUS_FOR_TRAINER))


def run_experiment(alg_name, env_yaml, timesteps_total, local_dir, seed=None):
    """Run a single experiment."""
    import importlib

    env_short = ENV_SHORT[env_yaml]
    wandb_project = f'mate-claude-{env_short}'
    print(f'\n{"="*60}')
    print(f'  Starting: {alg_name} on {env_yaml}')
    print(f'  W&B project: {wandb_project}, group: {alg_name}')
    print(f'{"="*60}\n')

    module_path, run_type = ALG_REGISTRY[alg_name]
    mod = importlib.import_module(module_path)
    config = mod.config

    exp_config = copy.deepcopy(config)
    exp_config['env_config']['config'] = env_yaml
    exp_config['num_workers'] = NUM_WORKERS
    exp_config['num_envs_per_worker'] = 8
    exp_config['num_gpus'] = min(NUM_GPUS, 0.25)
    exp_config['num_gpus_per_worker'] = 0
    exp_config['num_cpus_for_driver'] = NUM_CPUS_FOR_TRAINER

    # Adjust buffer size for value decomposition methods to avoid OOM
    if alg_name in VALUE_DECOMP_ALGS:
        if alg_name == 'iql':
            exp_config.setdefault('replay_buffer_config', {})
            exp_config['replay_buffer_config']['capacity'] = 5000
        else:
            exp_config['buffer_size'] = ceil(200 / max(NUM_WORKERS, 1))

    # Adjust batch size for on-policy methods
    if alg_name in ON_POLICY_ALGS and NUM_WORKERS > 0:
        frag_len = exp_config.get('rollout_fragment_length', 25)
        n_envs = exp_config.get('num_envs_per_worker', 4)
        exp_config['train_batch_size'] = NUM_WORKERS * n_envs * frag_len

    if seed is not None:
        seed = tune.grid_search(seed) if isinstance(seed, (list, tuple)) else seed
        exp_config['seed'] = seed

    exp_dir = Path(local_dir) / alg_name / env_short
    exp_dir.mkdir(parents=True, exist_ok=True)

    exp_name = f'HRL-{alg_name}'
    experiment = tune.Experiment(
        name=exp_name,
        run=run_type,
        config=exp_config,
        local_dir=str(exp_dir),
        stop={'timesteps_total': int(timesteps_total)},
        checkpoint_score_attr='episode_reward_mean',
        checkpoint_freq=20,
        checkpoint_at_end=True,
        max_failures=-1,
    )

    # Callbacks
    tune_callbacks = [SymlinkCheckpointCallback()]
    try:
        from examples.utils import WandbLoggerCallback
        if WandbLoggerCallback.is_available():
            tune_callbacks.append(WandbLoggerCallback(
                project=wandb_project,
                group=alg_name,
            ))
    except Exception:
        pass

    analysis = tune.run(
        experiment,
        metric='episode_reward_mean',
        mode='max',
        callbacks=tune_callbacks,
        verbose=3,
    )
    return analysis


def main():
    all_algs = list(ALG_REGISTRY.keys())

    parser = argparse.ArgumentParser()
    parser.add_argument('--timesteps', type=float, default=3e5,
                        help='Total env steps per experiment (default: 3e5)')
    parser.add_argument('--only', type=str, default=None,
                        choices=all_algs,
                        help='Run only this algorithm')
    parser.add_argument('--env', type=str, default=None,
                        choices=['MATE-4v5-0.yaml', 'MATE-4v8-9.yaml'],
                        help='Run only this env')
    parser.add_argument('--seed', type=int, metavar='SEED', nargs='*', default=None,
                        help='the global seed(s)')
    parser.add_argument('--local-dir', type=str,
                        default=str(Path(__file__).parent.parent / 'experiment_results'),
                        help='Base directory for results')
    args = parser.parse_args()

    algorithms = all_algs
    if args.only:
        algorithms = [args.only]

    envs = ENVS
    if args.env:
        envs = [args.env]

    if not ray.is_initialized():
        ray.init(num_cpus=NUM_CPUS, num_gpus=NUM_GPUS)

    for alg in algorithms:
        for env_yaml in envs:
            try:
                run_experiment(alg, env_yaml, args.timesteps, args.local_dir, seed=args.seed)
            except Exception as e:
                print(f'\nERROR in {alg}/{env_yaml}: {e}')
                import traceback
                traceback.print_exc()
                continue
            finally:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    ray.shutdown()
    print('\nAll experiments complete!')


if __name__ == '__main__':
    main()
