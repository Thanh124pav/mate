#!/usr/bin/env python3
import json
import os
from pathlib import Path

os.environ.setdefault('PYTHONPATH', '/home/pavt1024/vsn/mate')
os.environ.setdefault('WANDB_MODE', 'disabled')

import ray
import ray.tune.integration.wandb as wandb

wandb.WandbLoggerCallback.is_available = classmethod(lambda cls: False)

from examples.qplex_focus.camera.train import experiment, train

LOCAL_DIR = '/home/pavt1024/vsn/mate/outputs/qplex_focus_ctde_guarded_v2_4v8_9_200k_5120'

experiment.spec['config']['timesteps_per_iteration'] = 5120
experiment.spec['config']['min_sample_timesteps_per_reporting'] = 5120

ray.init(num_cpus=4, num_gpus=0, include_dashboard=False)
analysis = train(
    experiment,
    local_dir=LOCAL_DIR,
    num_gpus=0,
    num_workers=2,
    num_envs_per_worker=2,
    timesteps_total=200000,
    env_config='MATE-4v8-9.yaml',
    checkpoint_frequency=5,
    decentralized_eval_episodes=5,
    decentralized_eval_seed=0,
    seed=0,
    buffer_capacity=200,
)
trial = analysis.trials[0] if analysis.trials else None
joined = Path(trial.logdir) / 'checkpoint_eval_results.jsonl' if trial else None
rows = [json.loads(line) for line in joined.read_text().splitlines()] if joined and joined.exists() else []
ok = [row for row in rows if row.get('status') == 'ok']
best_coverage = max(ok, key=lambda row: row.get('decentralized_eval/mean_coverage_rate', float('-inf'))) if ok else None
best_suppression = min(ok, key=lambda row: row.get('decentralized_eval/normalized_target_episode_reward', float('inf'))) if ok else None
summary = {
    'local_dir': LOCAL_DIR,
    'trial_status': trial.status if trial else None,
    'trial_logdir': trial.logdir if trial else None,
    'joined_file': str(joined) if joined else None,
    'joined_count': len(rows),
    'best_camera_coverage_eval': best_coverage,
    'best_camera_target_suppression_eval': best_suppression,
}
Path(LOCAL_DIR, 'final_summary.json').write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n')
print('GUARDED_FINAL_SUMMARY', json.dumps(summary, sort_keys=True))
