#!/usr/bin/env python3
# Run: python -m examples.qplex_focus.camera.train --project mate-4v5-0

import argparse
import copy
import json
import os
import re
import subprocess
import sys
import time
from math import ceil
from pathlib import Path

import ray
import torch
from ray import tune

from examples.qplex_focus.camera.config import config
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
print(f"NUM_GPUS_FOR_TRAINER: {NUM_GPUS_FOR_TRAINER}")
MAX_NUM_CPUS_FOR_WORKER = max(0, NUM_NODE_CPUS - PRESERVED_NUM_CPUS - NUM_CPUS_FOR_TRAINER)
MAX_NUM_WORKERS = min(32, MAX_NUM_CPUS_FOR_WORKER)  # use at most 32 workers
NUM_WORKERS = MAX_NUM_WORKERS if not DEBUG else 0  # can be overridden by command line arguments


experiment = tune.Experiment(
    name='QPLEX_FOCUS',
    run='QPLEX_FOCUS',
    config=copy.deepcopy(config),
    local_dir=LOCAL_DIR,
    stop={'timesteps_total': 10e6},
    checkpoint_score_attr='episode_reward_mean',
    checkpoint_freq=10,
    checkpoint_at_end=True,
    max_failures=-1,
)


class DecentralizedEvaluationCallback(tune.Callback):
    METRICS = {
        'Step / Cargo': 'step_per_cargo',
        'Target Episode Reward': 'target_episode_reward',
        'Mean Transport Rate': 'mean_transport_rate',
        'Mean Coverage Rate': 'mean_coverage_rate',
        'Normalized Target Episode Reward': 'normalized_target_episode_reward',
        'Belief Local Pos Error': 'belief_local_pos_error',
        'Belief Local Visible Pos Error': 'belief_local_visible_pos_error',
        'Belief Local Invisible Pos Error': 'belief_local_invisible_pos_error',
        'Belief Visible Obs Pos Error': 'belief_visible_obs_pos_error',
        'Belief Local Std World': 'belief_local_std_world',
        'Belief Visible Target Ratio': 'belief_visible_target_ratio',
        'Belief Visible Step Ratio': 'belief_visible_step_ratio',
    }

    def __init__(
        self,
        *,
        episodes,
        seed,
        env_config,
        camera_agent='examples.qplex_focus.camera.agent:QPLEXFocusCameraAgent',
        camera_kwargs=None,
        belief_diagnostics_stride=10,
    ):
        self.episodes = int(episodes)
        self.seed = int(seed)
        self.env_config = env_config
        self.camera_agent = camera_agent
        self.camera_kwargs = dict(camera_kwargs or {})
        self.belief_diagnostics_stride = int(belief_diagnostics_stride)
        self.camera_kwargs.setdefault('decentralized_execution', True)
        self.camera_kwargs.setdefault('decentralized_fallback_mode', 'off')
        self.pending_results = {}
        self.ansi = re.compile(r'\x1b\[[0-9;]*m')

    def on_trial_result(self, iteration, trials, trial, result, **info):
        pending = self.pending_results.pop(trial.trial_id, None)
        if pending:
            result.update({f'decentralized_eval/{key}': value for key, value in pending.items()})

    def on_checkpoint(self, iteration, trials, trial, checkpoint, **info):
        if self.episodes <= 0:
            return

        checkpoint_file = self.checkpoint_file(Path(checkpoint.value), trial)
        if checkpoint_file is None:
            self.write_record(
                trial,
                {
                    'status': 'failed',
                    'error': f'No checkpoint file found under {checkpoint.value}',
                    'checkpoint': str(checkpoint.value),
                },
            )
            return

        training_iteration = None
        if getattr(trial, 'last_result', None):
            training_iteration = trial.last_result.get('training_iteration')

        result = self.run_evaluation(checkpoint_file, trial, training_iteration)
        self.write_record(trial, result)
        self.write_combined_record(trial, result)

        if result.get('status') == 'ok':
            eval_metrics = {
                key: value for key, value in result.items() if isinstance(value, (int, float))
            }
            prefixed = {f'decentralized_eval/{key}': value for key, value in eval_metrics.items()}
            if getattr(trial, 'last_result', None):
                trial.last_result.update(prefixed)
            self.pending_results[trial.trial_id] = eval_metrics

    @staticmethod
    def checkpoint_file(path, trial, timeout_s=60):
        candidates = [path]
        if not path.is_absolute():
            candidates.append(Path.cwd() / path)
            candidates.append(Path(trial.logdir) / path)
            candidates.append(Path(trial.logdir) / path.parent.name / path.name)
        candidates.append(Path(trial.logdir) / 'latest-checkpoint')

        deadline = time.time() + timeout_s
        while time.time() <= deadline:
            for candidate in candidates:
                resolved = DecentralizedEvaluationCallback.find_checkpoint_file(candidate)
                if resolved is not None:
                    return resolved
            resolved = DecentralizedEvaluationCallback.find_checkpoint_file_under_logdir(
                Path(trial.logdir), path
            )
            if resolved is not None:
                return resolved
            time.sleep(1)
        return None

    @staticmethod
    def find_checkpoint_file(path):
        if path.is_symlink():
            link = Path(os.readlink(path))
            if link.is_absolute():
                path = link
            else:
                candidates = [path.parent / link, Path.cwd() / link]
                path = next((candidate for candidate in candidates if candidate.exists()), candidates[0])
        if path.is_file() and path.name.startswith('checkpoint-') and not path.name.endswith(
            '.tune_metadata'
        ):
            return path
        if path.is_file():
            return None
        if not path.is_dir():
            return None
        files = sorted(
            item
            for item in path.iterdir()
            if item.is_file()
            and item.name.startswith('checkpoint-')
            and not item.name.endswith('.tune_metadata')
        )
        return files[0] if files else None

    @staticmethod
    def find_checkpoint_file_under_logdir(logdir, path):
        checkpoint_name = path.name if path.name.startswith('checkpoint-') else None
        if checkpoint_name is None:
            return None
        matches = sorted(
            item
            for item in logdir.rglob(checkpoint_name)
            if item.is_file()
            and item.name.startswith('checkpoint-')
            and not item.name.endswith('.tune_metadata')
        )
        return matches[0] if matches else None

    def run_evaluation(self, checkpoint_file, trial, training_iteration):
        log_dir = Path(trial.logdir) / 'decentralized_eval'
        log_dir.mkdir(parents=True, exist_ok=True)
        tag = f'iter_{training_iteration or "unknown"}'

        camera_kwargs = dict(self.camera_kwargs)
        camera_kwargs['checkpoint_path'] = str(checkpoint_file.resolve())

        command = [
            sys.executable,
            '-m',
            'mate.evaluate',
            '--no-render',
            '--episodes',
            str(self.episodes),
            '--seed',
            str(self.seed),
            '--config',
            self.env_config,
            '--camera-agent',
            self.camera_agent,
            '--camera-kwargs',
            json.dumps(camera_kwargs, sort_keys=True),
            '--belief-diagnostics-stride',
            str(self.belief_diagnostics_stride),
        ]
        env = os.environ.copy()
        repo = str(Path(__file__).resolve().parents[3])
        env['PYTHONPATH'] = repo + os.pathsep + env.get('PYTHONPATH', '')

        completed = subprocess.run(
            command,
            cwd=repo,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )

        stdout_path = log_dir / f'{tag}.stdout.log'
        stderr_path = log_dir / f'{tag}.stderr.log'
        stdout_path.write_text(completed.stdout)
        stderr_path.write_text(completed.stderr)

        metrics = self.parse_metrics(completed.stdout)
        record = {
            'status': 'ok' if completed.returncode == 0 and metrics else 'failed',
            'training_iteration': training_iteration,
            'checkpoint_path': str(checkpoint_file.resolve()),
            'episodes': self.episodes,
            'seed': self.seed,
            'env_config': self.env_config,
            'camera_agent': self.camera_agent,
            'camera_kwargs': camera_kwargs,
            'command': command,
            'decentralized_execution': camera_kwargs.get('decentralized_execution'),
            'decentralized_fallback_mode': camera_kwargs.get('decentralized_fallback_mode'),
            'returncode': completed.returncode,
            'stdout_log': str(stdout_path),
            'stderr_log': str(stderr_path),
        }
        record.update(metrics)
        if record['status'] != 'ok':
            lines = completed.stderr.strip().splitlines()
            record['error'] = lines[-1] if lines else 'mate.evaluate produced no metrics'
        print(f'Decentralized eval after checkpoint {checkpoint_file}: {record}')
        return record

    def parse_metrics(self, stdout):
        metrics = {}
        for line in stdout.splitlines():
            clean = self.ansi.sub('', line).strip()
            if not clean.startswith('|'):
                continue
            cells = [cell.strip() for cell in clean.strip('|').split('|')]
            if len(cells) < 2 or cells[0] not in self.METRICS:
                continue
            metrics[self.METRICS[cells[0]]] = self.parse_value(cells[1])
        return metrics

    @staticmethod
    def parse_value(value):
        value = value.strip()
        if value.endswith('%'):
            return float(value[:-1]) / 100.0
        return float(value.replace('+', ''))

    @staticmethod
    def training_record(trial):
        result = getattr(trial, 'last_result', None) or {}
        keys = (
            'training_iteration',
            'timesteps_total',
            'episodes_total',
            'episode_reward_mean',
            'episode_len_mean',
            'time_total_s',
        )
        record = {key: result.get(key) for key in keys if key in result}
        custom_metrics = result.get('custom_metrics') or {}
        for key, value in custom_metrics.items():
            if isinstance(value, (int, float)):
                record[f'custom_metrics/{key}'] = value
        learner = (((result.get('info') or {}).get('learner') or {}).get('default_policy') or {})
        learner_stats = learner.get('learner_stats') or {}
        always_keep = {'loss', 'td_loss', 'grad_norm', 'q_taken_mean', 'target_mean'}
        for key, value in learner_stats.items():
            if key in always_keep or str(key).startswith('focus_'):
                record[f'learner_stats/{key}'] = value
        return record

    @staticmethod
    def write_record(trial, record):
        path = Path(trial.logdir) / 'decentralized_eval.jsonl'
        with path.open('a') as handle:
            handle.write(json.dumps(record, sort_keys=True) + '\n')

    def write_combined_record(self, trial, eval_record):
        eval_metrics = {
            f'decentralized_eval/{key}': value
            for key, value in eval_record.items()
            if isinstance(value, (int, float))
        }
        combined = {
            'status': eval_record.get('status'),
            'checkpoint_path': eval_record.get('checkpoint_path'),
            'stdout_log': eval_record.get('stdout_log'),
            'stderr_log': eval_record.get('stderr_log'),
            'decentralized_eval/camera_kwargs': eval_record.get('camera_kwargs'),
            'decentralized_eval/command': eval_record.get('command'),
            'decentralized_eval/decentralized_execution': eval_record.get(
                'decentralized_execution'
            ),
            'decentralized_eval/decentralized_fallback_mode': eval_record.get(
                'decentralized_fallback_mode'
            ),
            'decentralized_eval/decentralized_local_belief_guide': eval_record.get(
                'camera_kwargs', {}
            ).get('decentralized_local_belief_guide'),
            'decentralized_eval/local_belief_guide_eta': eval_record.get(
                'camera_kwargs', {}
            ).get('local_belief_guide_eta'),
            'decentralized_eval/student_action_bias_eta': eval_record.get(
                'camera_kwargs', {}
            ).get('student_action_bias_eta'),
            'decentralized_eval/local_belief_action_eta': eval_record.get(
                'camera_kwargs', {}
            ).get('local_belief_action_eta'),
        }
        combined.update(self.training_record(trial))
        combined.update(eval_metrics)
        if 'error' in eval_record:
            combined['error'] = eval_record['error']
        path = Path(trial.logdir) / 'checkpoint_eval_results.jsonl'
        with path.open('a') as handle:
            handle.write(json.dumps(combined, sort_keys=True) + '\n')
        print(f'Checkpoint eval joined metrics: {combined}')


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
    env_config=None,
    checkpoint_frequency=10,
    decentralized_eval_episodes=5,
    decentralized_eval_seed=0,
    restore=None,
    resume=False,
):
    print(f"Resume checkpoint: {resume}")
    tune_callbacks = [SymlinkCheckpointCallback()]
    decentralized_eval_episodes = int(decentralized_eval_episodes)
    if decentralized_eval_episodes > 0:
        eval_env_config = env_config or experiment.spec['config']['env_config']['config']
        focus_config = experiment.spec['config'].get('focus', {})
        student_action_bias_eta = float(focus_config.get('student_action_bias_eta', 0.0))
        local_belief_action_eta = float(
            focus_config.get('local_belief_action_eta', student_action_bias_eta)
        )
        tune_callbacks.append(
            DecentralizedEvaluationCallback(
                episodes=decentralized_eval_episodes,
                seed=decentralized_eval_seed,
                env_config=eval_env_config,
                belief_diagnostics_stride=10,
                camera_kwargs={
                    'decentralized_execution': True,
                    'decentralized_fallback_mode': 'off',
                    'decentralized_local_belief_guide': True,
                    'student_action_bias_eta': student_action_bias_eta,
                    'local_belief_action_eta': local_belief_action_eta,
                },
            )
        )
    if WandbLoggerCallback.is_available():
        project = project or ('mate-camera' if not DEBUG else 'mate-debug')
        group = group or f'qplex_focus.camera.{experiment.name}'
        tune_callbacks.append(WandbLoggerCallback(project=project, group=group))

    if not ray.is_initialized():
        ray.init(num_cpus=NUM_NODE_CPUS, num_gpus=NUM_NODE_GPUS, local_mode=DEBUG)

    num_ray_cpus = round(ray.cluster_resources()['CPU'])
    num_ray_gpus = ray.cluster_resources().get('GPU', 0.0)
    print(f"Num ray gpus: {num_ray_gpus}")
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
        experiment.spec['local_dir'] = str(Path(local_dir).expanduser().resolve())
    if env_config is not None:
        experiment.spec['config']['env_config'].update(config=env_config)
    experiment.spec['checkpoint_freq'] = checkpoint_frequency

    # Update replay buffer capacity
    experiment.spec['config'].update(buffer_size=ceil(buffer_capacity / max(num_workers, 1)))

    analysis = tune.run(
        experiment,
        metric='episode_reward_mean',
        mode='max',
        callbacks=tune_callbacks,
        verbose=1,
        restore=restore,
        resume=resume,
    )
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
        default=4,
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
        '--env-config',
        type=str,
        metavar='YAML',
        default=None,
        help='MATE environment config YAML, e.g. MATE-4v8-9.yaml',
    )
    parser.add_argument(
        '--checkpoint-frequency',
        type=int,
        metavar='N',
        default=10,
        help='save a checkpoint every N training iterations / W&B result logs',
    )
    parser.add_argument(
        '--decentralized-eval-episodes',
        type=int,
        metavar='N',
        default=5,
        help='episodes for decentralized mate.evaluate after each checkpoint; 0 disables it',
    )
    parser.add_argument(
        '--decentralized-eval-seed',
        type=int,
        metavar='SEED',
        default=0,
        help='seed for decentralized checkpoint evaluation',
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

    args = parser.parse_args()
    analysis = train(experiment, **vars(args))
    return analysis


if __name__ == '__main__':
    main()
