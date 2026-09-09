#!/usr/bin/env python3
"""Run one reproducible QPLEX-WM2 ablation on a selected MATE environment."""

import argparse
import copy
import json
import os
import subprocess
from functools import partial
from datetime import datetime, timezone
from pathlib import Path

from examples.hrl.qplex_wm2.camera.train import experiment as base_experiment
from examples.hrl.qplex_wm2.camera.train import train
from examples.target_agents import EvasiveTargetAgent, greedy_target_agent_factory

VARIANTS = {
    # Main WM2 and pre-WM baselines intentionally excluded: already run for the paper.
    "no_local_z": {"world_model_v2.augment_local_obs": False},
    "no_global_z": {"world_model_v2.augment_global_state": False},
    "no_z": {
        "world_model_v2.augment_local_obs": False,
        "world_model_v2.augment_global_state": False,
    },
    "no_state_decoder": {
        "world_model_v2.state_recon_coeff": 0.0,
        # Reward bonus is defined from decoder error, so disable it with the decoder.
        "world_model_v2.reward_bonus_coeff": 0.0,
    },
    "no_reward_head": {"world_model_v2.reward_pred_coeff": 0.0},
    "no_kl": {"world_model_v2.kl_coeff": 0.0},
    "no_reward_bonus": {"world_model_v2.reward_bonus_coeff": 0.0},
    "lr_5e5": {"lr": 5e-5},
    "lr_3e4": {"lr": 3e-4},
    "wm_weight_01": {"world_model_v2.wm_loss_weight": 0.1},
    "wm_weight_10": {"world_model_v2.wm_loss_weight": 1.0},
    "transport_penalty_025": {"env_config.reward_coefficients": {
        "coverage_rate": 1.0,
        "mean_transport_rate": -0.25,
    }},
    "transport_penalty_010": {"env_config.reward_coefficients": {
        "coverage_rate": 1.0,
        "mean_transport_rate": -0.10,
    }},
    "transport_penalty_050": {"env_config.reward_coefficients": {
        "coverage_rate": 1.0,
        "mean_transport_rate": -0.50,
    }},
    "transport_penalty_100": {"env_config.reward_coefficients": {
        "coverage_rate": 1.0,
        "mean_transport_rate": -1.00,
    }},
    "target_greedy": {},
    "target_evasive_strength_025": {},
    "target_evasive_strength_075": {},
    "target_evasive_range_025": {},
    "target_evasive_range_075": {},
    "target_evasive_noise_025": {},
    "target_evasive_noise_075": {},
}

TARGET_TRAINING = {
    "target_greedy": {
        "kind": "greedy", "seed": 0, "factory": greedy_target_agent_factory,
    },
    "target_evasive_strength_025": {
        "kind": "evasive", "seed": 0, "avoidance_strength": 0.25,
        "avoidance_range": 0.5, "noise_scale": 0.5,
    },
    "target_evasive_strength_075": {
        "kind": "evasive", "seed": 0, "avoidance_strength": 0.75,
        "avoidance_range": 0.5, "noise_scale": 0.5,
    },
    "target_evasive_range_025": {
        "kind": "evasive", "seed": 0, "avoidance_strength": 0.5,
        "avoidance_range": 0.25, "noise_scale": 0.5,
    },
    "target_evasive_range_075": {
        "kind": "evasive", "seed": 0, "avoidance_strength": 0.5,
        "avoidance_range": 0.75, "noise_scale": 0.5,
    },
    "target_evasive_noise_025": {
        "kind": "evasive", "seed": 0, "avoidance_strength": 0.5,
        "avoidance_range": 0.5, "noise_scale": 0.25,
    },
    "target_evasive_noise_075": {
        "kind": "evasive", "seed": 0, "avoidance_strength": 0.5,
        "avoidance_range": 0.5, "noise_scale": 0.75,
    },
}


def target_factory(spec):
    if spec["kind"] == "greedy":
        return spec["factory"]
    return partial(
        EvasiveTargetAgent,
        seed=spec["seed"],
        noise_scale=spec["noise_scale"],
        avoidance_strength=spec["avoidance_strength"],
        avoidance_range=spec["avoidance_range"],
    )


def set_nested(config, dotted_key, value):
    target = config
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        target = target[part]
    target[parts[-1]] = value


def jsonable(value):
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("variant", choices=sorted(VARIANTS))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timesteps-total", type=int, default=500_000)
    parser.add_argument("--buffer-capacity", type=int, default=200)
    parser.add_argument("--num-workers", type=int, default=3)
    parser.add_argument("--num-envs-per-worker", type=int, default=8)
    parser.add_argument("--num-gpus", type=float, default=1.0)
    parser.add_argument("--evaluation-interval", type=int, default=5)
    parser.add_argument("--project", default="mate-wm2-ablations")
    parser.add_argument("--env", default="MATE-4v8-9.yaml")
    parser.add_argument("--output-root", type=Path,
                        default=Path("experiments/wm2_ablations"))
    args = parser.parse_args()

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    env_slug = args.env.removeprefix("MATE-").removesuffix(".yaml").lower()
    run_name = f"qplex_wm2__{env_slug}__{args.variant}__seed{args.seed}__{stamp}"
    run_dir = (args.output_root / run_name).resolve()
    ray_dir = run_dir / "ray_results"
    wandb_dir = run_dir / "wandb"
    ray_dir.mkdir(parents=True, exist_ok=False)
    wandb_dir.mkdir(parents=True, exist_ok=True)
    os.environ["WANDB_DIR"] = str(wandb_dir)
    os.environ.setdefault("WANDB_SILENT", "true")

    experiment = copy.deepcopy(base_experiment)
    experiment.spec["name"] = run_name
    experiment.spec["config"]["env_config"]["config"] = args.env
    # Lossless compression keeps the replay buffer viable on the 6 GiB host.
    experiment.spec["config"]["compress_observations"] = True
    # Imagination is intentionally excluded from this paper's ablation suite.
    experiment.spec["config"]["world_model_v2"]["use_imagination_targets"] = False
    for key, value in VARIANTS[args.variant].items():
        set_nested(experiment.spec["config"], key, value)
    target_spec = TARGET_TRAINING.get(args.variant)
    if target_spec is not None:
        experiment.spec["config"]["env_config"]["opponent_agent_factory"] = (
            target_factory(target_spec)
        )

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "algorithm": "QPLEX_WM2",
        "variant": args.variant,
        "seed": args.seed,
        "environment": args.env,
        "timesteps_total": args.timesteps_total,
        "buffer_capacity_episodes_global_requested": args.buffer_capacity,
        "buffer_size_per_worker_effective": -(-args.buffer_capacity // max(args.num_workers, 1)),
        "num_workers": args.num_workers,
        "num_envs_per_worker": args.num_envs_per_worker,
        "num_gpus": args.num_gpus,
        "evaluation_interval": args.evaluation_interval,
        "compress_observations": True,
        "wandb_project": args.project,
        "wandb_group": f"paper-wm2-{env_slug}-qplex-ablations",
        "overrides": VARIANTS[args.variant],
        "training_target": jsonable(target_spec) if target_spec is not None else {
            "kind": "evasive", "seed": 0, "avoidance_strength": 0.5,
            "avoidance_range": 0.5, "noise_scale": 0.5,
        },
        "evaluation_target": {"kind": "greedy", "seed": 0},
        "git_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], check=False, text=True,
            capture_output=True,
        ).stdout.strip(),
        "resolved_config": jsonable(experiment.spec["config"]),
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    train(
        experiment,
        project=args.project,
        group=manifest["wandb_group"],
        local_dir=str(ray_dir),
        num_gpus=args.num_gpus,
        num_workers=args.num_workers,
        num_envs_per_worker=args.num_envs_per_worker,
        evaluation_interval=args.evaluation_interval,
        seed=args.seed,
        timesteps_total=args.timesteps_total,
        buffer_capacity=args.buffer_capacity,
        env=args.env,
    )


if __name__ == "__main__":
    main()
