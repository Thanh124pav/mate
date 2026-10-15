#!/usr/bin/env python3
"""Run one real MATE/Ray iteration and checkpoint for each new trainer."""

import argparse
import copy
import json
import os
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import ray

from ray.rllib.agents.concaveq import ConcaveQFocusTrainer, ConcaveQTrainer
from ray.rllib.agents.facmac import FACMACFocusTrainer, FACMACTrainer
from ray.rllib.agents.qfree import QFreeFocusTrainer, QFreeTrainer
from ray.rllib.agents.resq import ResQFocusTrainer, ResQTrainer
from ray.rllib.agents.riskq import RiskQFocusTrainer, RiskQTrainer
from examples.concaveq.camera.config import config as concaveq_config
from examples.concaveq_focus.camera.config import config as concaveq_focus_config
from examples.facmac.camera.config import config as facmac_config
from examples.facmac_focus.camera.config import config as facmac_focus_config
from examples.qfree.camera.config import config as qfree_config
from examples.qfree_focus.camera.config import config as qfree_focus_config
from examples.resq.camera.config import config as resq_config
from examples.resq_focus.camera.config import config as resq_focus_config
from examples.riskq.camera.config import config as riskq_config
from examples.riskq_focus.camera.config import config as riskq_focus_config


ALGORITHMS = {
    "FACMAC": (FACMACTrainer, facmac_config),
    "FACMAC_FOCUS": (FACMACFocusTrainer, facmac_focus_config),
    "RESQ": (ResQTrainer, resq_config),
    "RESQ_FOCUS": (ResQFocusTrainer, resq_focus_config),
    "RISKQ": (RiskQTrainer, riskq_config),
    "RISKQ_FOCUS": (RiskQFocusTrainer, riskq_focus_config),
    "QFREE": (QFreeTrainer, qfree_config),
    "QFREE_FOCUS": (QFreeFocusTrainer, qfree_focus_config),
    "CONCAVEQ": (ConcaveQTrainer, concaveq_config),
    "CONCAVEQ_FOCUS": (ConcaveQFocusTrainer, concaveq_focus_config),
}


def smoke_config(source, belief_distribution=None, belief_grid_size=5):
    config = copy.deepcopy(source)
    config.update(
        num_workers=0,
        num_envs_per_worker=1,
        num_gpus=0,
        learning_starts=0,
        train_batch_size=32,
        timesteps_per_iteration=100,
        buffer_size=8,
        rollout_fragment_length=0,
        evaluation_interval=None,
        simple_optimizer=True,
    )
    if config.get("focus", {}).get("enabled"):
        config["focus"].update(
            integral_mode="sigma",
            sigma_method="legacy5",
            horizon=1,
            belief_hidden_dim=64,
        )
        if belief_distribution is not None:
            config["focus"].update(
                belief_distribution=belief_distribution,
                belief_grid_size=int(belief_grid_size),
                belief_grid_chunk_size=8,
            )
    return config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("algorithms", nargs="*", default=None)
    parser.add_argument("--checkpoint-dir", default="/tmp/mate-value-factorization-smoke")
    parser.add_argument("--restore-check", action="store_true")
    parser.add_argument(
        "--belief-distribution",
        choices=("gaussian", "categorical_grid"),
        default=None,
    )
    parser.add_argument("--belief-grid-size", type=int, default=5)
    args = parser.parse_args()
    unknown = sorted(set(args.algorithms or ()) - set(ALGORITHMS))
    if unknown:
        parser.error(f"unknown algorithms: {', '.join(unknown)}")
    os.environ.setdefault("WANDB_DISABLED", "true")
    root = Path(args.checkpoint_dir)
    root.mkdir(parents=True, exist_ok=True)
    ray.init(
        num_cpus=2,
        local_mode=True,
        include_dashboard=False,
        ignore_reinit_error=True,
    )
    results = {}
    try:
        for name in (args.algorithms or list(ALGORITHMS)):
            trainer_class, source = ALGORITHMS[name]
            config = smoke_config(
                source,
                belief_distribution=args.belief_distribution,
                belief_grid_size=args.belief_grid_size,
            )
            trainer = trainer_class(config=config)
            try:
                result = trainer.train()
                checkpoint = trainer.save(str(root / name.lower()))
                learner = result.get("info", {}).get("learner", {})
                results[name] = {
                    "training_iteration": result.get("training_iteration"),
                    "timesteps_total": result.get("timesteps_total"),
                    "episode_reward_mean": result.get("episode_reward_mean"),
                    "custom_metrics": result.get("custom_metrics", {}),
                    "learner": learner,
                    "checkpoint": checkpoint,
                }
                print(
                    f"PASS {name}: steps={result.get('timesteps_total')} "
                    f"reward={result.get('episode_reward_mean')} checkpoint={checkpoint}",
                    flush=True,
                )
            finally:
                trainer.stop()
            if args.restore_check:
                restored = trainer_class(config=config)
                try:
                    restored.restore(checkpoint)
                finally:
                    restored.stop()
                print(f"PASS {name}: trainer checkpoint restore", flush=True)
    finally:
        ray.shutdown()
    output = root / "results.json"
    output.write_text(json.dumps(results, indent=2, default=str))
    print(f"Results: {output}")


if __name__ == "__main__":
    main()
