#!/usr/bin/env python3
"""Run one reproducible WM2 ablation across supported MARL architectures."""

import argparse
import copy
import importlib
import json
import os
import subprocess
from datetime import datetime, timezone
from functools import partial
from math import ceil
from pathlib import Path

from examples.target_agents import EvasiveTargetAgent, greedy_target_agent_factory

ALGORITHMS = {
    "qplex": {"module": "examples.hrl.qplex_wm2.camera.train", "replay": True},
    "duelmix": {"module": "examples.hrl.duelmix_wm2.camera.train", "replay": True},
    "spectra": {"module": "examples.hrl.spectra_wm2.camera.train", "replay": True},
    "mappo": {"module": "examples.hrl.mappo_wm2.camera.train", "replay": False},
}

VARIANTS = (
    "lr_half", "lr_double", "wm_weight_01", "wm_weight_10",
    "transport_penalty_010", "transport_penalty_025",
    "transport_penalty_050", "transport_penalty_100",
    "target_greedy", "target_evasive_default",
    "target_evasive_strength_025", "target_evasive_strength_075",
    "target_evasive_range_025", "target_evasive_range_075",
    "target_evasive_noise_025", "target_evasive_noise_075",
)

TARGET_TRAINING = {
    "target_greedy": {"kind": "greedy", "seed": 0},
    "target_evasive_default": {
        "kind": "evasive", "seed": 0, "avoidance_strength": 0.5,
        "avoidance_range": 0.5, "noise_scale": 0.5,
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


def set_nested(config, dotted_key, value):
    target = config
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        target = target[part]
    target[parts[-1]] = value


def wm_prefix(algorithm):
    if algorithm == "mappo":
        return "model.custom_model_config.world_model_v2"
    return "world_model_v2"


def variant_overrides(config, algorithm, variant):
    prefix = wm_prefix(algorithm)
    if variant == "lr_half":
        return {"lr": float(config["lr"]) * 0.5}
    if variant == "lr_double":
        return {"lr": float(config["lr"]) * 2.0}
    if variant == "wm_weight_01":
        return {f"{prefix}.wm_loss_weight": 0.1}
    if variant == "wm_weight_10":
        return {f"{prefix}.wm_loss_weight": 1.0}
    penalties = {
        "transport_penalty_010": -0.10,
        "transport_penalty_025": -0.25,
        "transport_penalty_050": -0.50,
        "transport_penalty_100": -1.00,
    }
    if variant in penalties:
        return {"env_config.reward_coefficients": {
            "coverage_rate": 1.0,
            "mean_transport_rate": penalties[variant],
        }}
    return {}


def target_factory(spec):
    if spec["kind"] == "greedy":
        return greedy_target_agent_factory
    return partial(
        EvasiveTargetAgent,
        seed=spec["seed"],
        noise_scale=spec["noise_scale"],
        avoidance_strength=spec["avoidance_strength"],
        avoidance_range=spec["avoidance_range"],
    )


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
    parser.add_argument("variant", choices=VARIANTS)
    parser.add_argument("--algorithm", choices=sorted(ALGORITHMS), required=True)
    parser.add_argument("--env", default="MATE-4v8-9.yaml")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timesteps-total", type=int, default=500_000)
    parser.add_argument("--buffer-capacity", type=int, default=200)
    parser.add_argument("--num-workers", type=int, default=3)
    parser.add_argument("--num-envs-per-worker", type=int, default=8)
    parser.add_argument("--num-gpus", type=float, default=1.0)
    parser.add_argument("--evaluation-interval", type=int, default=5)
    parser.add_argument("--project", default="mate-wm2-ablations")
    parser.add_argument("--output-root", type=Path,
                        default=Path("experiments/wm2_ablations"))
    args = parser.parse_args()

    algorithm_spec = ALGORITHMS[args.algorithm]
    uses_replay = algorithm_spec["replay"]
    env_slug = args.env.removeprefix("MATE-").removesuffix(".yaml").lower()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_name = (
        f"{args.algorithm}_wm2__{env_slug}__{args.variant}__"
        f"seed{args.seed}__{stamp}"
    )
    run_dir = (args.output_root / run_name).resolve()
    ray_dir = run_dir / "ray_results"
    wandb_dir = run_dir / "wandb"
    ray_dir.mkdir(parents=True, exist_ok=False)
    wandb_dir.mkdir(parents=True, exist_ok=True)
    os.environ["WANDB_DIR"] = str(wandb_dir)
    os.environ.setdefault("WANDB_SILENT", "true")

    module = importlib.import_module(algorithm_spec["module"])
    experiment = copy.deepcopy(module.experiment)
    experiment.spec["name"] = run_name
    config = experiment.spec["config"]
    config["env_config"]["config"] = args.env
    config["compress_observations"] = uses_replay

    prefix = wm_prefix(args.algorithm)
    wm_config = config
    for part in prefix.split("."):
        wm_config = wm_config[part]
    if "use_imagination_targets" in wm_config:
        wm_config["use_imagination_targets"] = False

    overrides = variant_overrides(config, args.algorithm, args.variant)
    for key, value in overrides.items():
        set_nested(config, key, value)
    target_spec = TARGET_TRAINING.get(args.variant)
    if target_spec is not None:
        config["env_config"]["opponent_agent_factory"] = target_factory(target_spec)

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "algorithm": f"{args.algorithm.upper()}_WM2",
        "variant": args.variant,
        "seed": args.seed,
        "environment": args.env,
        "timesteps_total": args.timesteps_total,
        "uses_replay_buffer": uses_replay,
        "buffer_capacity_episodes_global_requested": (
            args.buffer_capacity if uses_replay else None
        ),
        "buffer_size_per_worker_effective": (
            ceil(args.buffer_capacity / max(args.num_workers, 1))
            if uses_replay else None
        ),
        "num_workers": args.num_workers,
        "num_envs_per_worker": args.num_envs_per_worker,
        "num_gpus": args.num_gpus,
        "evaluation_interval": args.evaluation_interval,
        "evaluation_episodes_per_checkpoint": 5,
        "compress_observations": uses_replay,
        "use_imagination_targets": False,
        "wandb_project": args.project,
        "wandb_group": f"paper-wm2-{env_slug}-{args.algorithm}-ablations",
        "overrides": overrides,
        "training_target": (
            jsonable(target_spec) if target_spec is not None
            else {"kind": "architecture_default"}
        ),
        "evaluation_target": {"kind": "greedy", "seed": 0},
        "git_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], check=False, text=True,
            capture_output=True,
        ).stdout.strip(),
        "resolved_config": jsonable(config),
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    train_kwargs = dict(
        project=args.project,
        group=manifest["wandb_group"],
        local_dir=str(ray_dir),
        num_gpus=args.num_gpus,
        num_workers=args.num_workers,
        num_envs_per_worker=args.num_envs_per_worker,
        evaluation_interval=args.evaluation_interval,
        seed=args.seed,
        timesteps_total=args.timesteps_total,
    )
    if uses_replay:
        train_kwargs.update(buffer_capacity=args.buffer_capacity, env=args.env)
    module.train(experiment, **train_kwargs)


if __name__ == "__main__":
    main()
