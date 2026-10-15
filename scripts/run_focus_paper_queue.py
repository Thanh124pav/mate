#!/usr/bin/env python3
"""Run reproducible backbone vs. backbone+FOCUS experiment queues.

Jobs are deliberately sequential on the local 4 GB GPU / 6 GB RAM host.  Each
job gets an isolated result directory, a plain-text log, and a resumable JSON
state entry.  ``smoke`` verifies integration, ``pilot`` estimates effect sizes,
and ``paper`` is the minimum three-seed protocol used for tables.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Dict, Iterable, List


REPO_ROOT = Path(__file__).resolve().parent.parent

ENV_LADDER = (
    "MATE-4v8-9-T100.yaml",
    "MATE-4v8-9-T70.yaml",
    "MATE-4v8-9-T40.yaml",
    "MATE-4v8-9.yaml",
)

REGISTRY = {
    "facmac": {
        "family": "ctde-value",
        "kind": "ray-new-value",
        "backbone": "examples.facmac.camera",
        "focus": "examples.facmac_focus.camera",
        "categorical": True,
    },
    "resq": {
        "family": "ctde-value",
        "kind": "ray-new-value",
        "backbone": "examples.resq.camera",
        "focus": "examples.resq_focus.camera",
        "categorical": True,
    },
    "riskq": {
        "family": "ctde-value",
        "kind": "ray-new-value",
        "backbone": "examples.riskq.camera",
        "focus": "examples.riskq_focus.camera",
        "categorical": True,
    },
    "qfree": {
        "family": "ctde-value",
        "kind": "ray-new-value",
        "backbone": "examples.qfree.camera",
        "focus": "examples.qfree_focus.camera",
        "categorical": True,
    },
    "concaveq": {
        "family": "ctde-value",
        "kind": "ray-new-value",
        "backbone": "examples.concaveq.camera",
        "focus": "examples.concaveq_focus.camera",
        "categorical": True,
    },
    "qplex": {
        "family": "ctde-value",
        "kind": "ray-value",
        "backbone": "examples.hrl.qplex.camera.train",
        "focus": "examples.hrl.qplex_focus.camera.train",
        "categorical": True,
    },
    "qmix": {
        "family": "ctde-value",
        "kind": "ray-value",
        "backbone": "examples.hrl.qmix.camera.train",
        "focus": "examples.hrl.qmix_focus.camera.train",
    },
    "duelmix": {
        "family": "ctde-value",
        "kind": "ray-value",
        "backbone": "examples.hrl.duelmix.camera.train",
        "focus": "examples.hrl.duelmix_focus.camera.train",
    },
    "mappo": {
        "family": "ctde-policy",
        "kind": "ray-policy",
        "backbone": "examples.mappo.camera.train",
        "focus": "examples.mappo_focus.camera.train",
        "categorical": True,
    },
    "tarmac": {
        "family": "ctde-policy",
        "kind": "ray-policy",
        "backbone": "examples.tarmac.camera.train",
        "focus": "examples.tarmac_focus.camera.train",
        "categorical": True,
    },
    "smpe2": {
        "family": "ctde-policy",
        "kind": "ray-policy",
        "backbone": "examples.smpe2.camera.train",
        "focus": "examples.smpe2_focus.camera.train",
        "categorical": True,
    },
    "hmvfe": {
        "family": "ctce-policy",
        "kind": "hmvfe",
        "backbone": "examples.hmvfe.camera.train",
        "focus": "examples.hmvfe_focus.camera.train",
        "categorical": True,
    },
    "hitmac": {
        "family": "ctce-policy",
        "kind": "hitmac",
        "backbone": "hitmac_focus",
        "focus": "hitmac_focus",
    },
}

TIER_DEFAULTS = {
    "smoke": {"timesteps": 10_000, "seeds": (0,), "envs": (ENV_LADDER[-1],)},
    "pilot": {"timesteps": 300_000, "seeds": (0,), "envs": ENV_LADDER},
    "paper": {"timesteps": 3_000_000, "seeds": (0, 1, 2), "envs": ENV_LADDER},
}


def slug_env(env: str) -> str:
    name = Path(env).stem
    return name.removeprefix("MATE-").lower().replace("_", "-")


def available_memory_gib() -> float:
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        if line.startswith("MemAvailable:"):
            return float(line.split()[1]) / 1024.0 / 1024.0
    return float("nan")


def load_state(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {"schema_version": 1, "jobs": {}}
    return json.loads(path.read_text(encoding="utf-8"))


def save_state(path: Path, state: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def build_command(job: Dict[str, Any], result_dir: Path, args: argparse.Namespace) -> List[str]:
    spec = REGISTRY[job["algorithm"]]
    module = spec[job["variant"]]
    command = [sys.executable, "-m", module]
    if spec["kind"] in ("hmvfe", "hitmac"):
        step_flag = (
            "--total-env-steps" if spec["kind"] == "hitmac" else "--timesteps-total"
        )
        command.extend(
            [
                "--env-config",
                job["env"],
                step_flag,
                str(job["timesteps"]),
                "--seed",
                str(job["seed"]),
                "--num-envs",
                str(args.num_envs),
                "--device",
                args.hmvfe_device,
                "--output-dir",
                str(result_dir),
                "--run-name",
                job["id"],
                "--wandb-mode",
                args.wandb_mode,
                "--wandb-project",
                args.wandb_project,
                "--wandb-group",
                args.wandb_group,
                "--log-interval",
                "1",
                "--save-interval",
                "20",
                "--eval-interval",
                "10",
                "--eval-episodes",
                str(args.eval_episodes),
            ]
        )
        if spec["kind"] == "hitmac":
            if job["variant"] == "backbone":
                command.append("--no-focus")
            else:
                command.extend(
                    [
                        "--focus-belief-safeguard-mode",
                        args.belief_safeguard_mode,
                    ]
                )
        elif job["variant"] == "focus" and args.belief_distribution:
            command.extend(
                [
                    "--focus-belief-safeguard-mode",
                    args.belief_safeguard_mode,
                    "--focus-belief-distribution",
                    args.belief_distribution,
                    "--focus-belief-grid-size",
                    str(args.belief_grid_size),
                    "--focus-belief-grid-max-delta",
                    str(args.belief_grid_max_delta),
                    "--focus-belief-position-loss-coeff",
                    str(args.belief_position_loss_coeff),
                ]
            )
    else:
        command.extend(
            [
                "--timesteps-total",
                str(job["timesteps"]),
                "--seed",
                str(job["seed"]),
                "--num-workers",
                str(args.num_workers),
                "--num-envs-per-worker",
                str(args.num_envs),
                "--num-gpus",
                str(args.num_gpus),
                "--local-dir",
                str(result_dir),
                "--project",
                args.wandb_project,
                "--group",
                args.wandb_group,
            ]
        )
        # Every Ray launcher accepts ``--env`` as the MATE YAML selector.
        # The newer value-factorization launcher previously omitted it, which
        # mislabeled visibility-ladder runs while executing the default env.
        command.extend(["--env", job["env"]])
        if spec["kind"] in ("ray-value", "ray-new-value"):
            command.extend(["--buffer-capacity", str(args.buffer_capacity)])
        if spec["kind"] == "ray-new-value":
            command.extend(["--evaluation-interval", str(args.evaluation_interval)])
        if (
            job["variant"] == "focus"
            and spec.get("categorical", False)
            and args.belief_distribution
        ):
            command.extend(
                [
                    "--belief-safeguard-mode",
                    args.belief_safeguard_mode,
                    "--belief-distribution",
                    args.belief_distribution,
                    "--belief-grid-size",
                    str(args.belief_grid_size),
                    "--belief-grid-max-delta",
                    str(args.belief_grid_max_delta),
                    "--belief-position-loss-coeff",
                    str(args.belief_position_loss_coeff),
                ]
            )
    return command


def make_jobs(args: argparse.Namespace) -> Iterable[Dict[str, Any]]:
    algorithms = args.algorithms or tuple(REGISTRY)
    variants = ("backbone", "focus") if args.variant == "both" else (args.variant,)
    for family in args.families or ():
        unknown = [name for name in algorithms if REGISTRY[name]["family"] != family]
        if len(unknown) == len(algorithms):
            continue
    for algorithm in algorithms:
        if args.families and REGISTRY[algorithm]["family"] not in args.families:
            continue
        for env in args.envs:
            for seed in args.seeds:
                for variant in variants:
                    job_id = f"{REGISTRY[algorithm]['family']}__{algorithm}__{variant}__{slug_env(env)}__s{seed}"
                    yield {
                        "id": job_id,
                        "family": REGISTRY[algorithm]["family"],
                        "algorithm": algorithm,
                        "variant": variant,
                        "env": env,
                        "seed": seed,
                        "timesteps": args.timesteps,
                    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tier", choices=tuple(TIER_DEFAULTS), default="smoke")
    parser.add_argument("--algorithms", nargs="*", choices=tuple(REGISTRY))
    parser.add_argument(
        "--families",
        nargs="*",
        choices=sorted({spec["family"] for spec in REGISTRY.values()}),
    )
    parser.add_argument("--variant", choices=("both", "backbone", "focus"), default="both")
    parser.add_argument("--envs", nargs="*")
    parser.add_argument("--seeds", nargs="*", type=int)
    parser.add_argument("--timesteps", type=int)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--num-gpus", type=float, default=0.20)
    parser.add_argument("--buffer-capacity", type=int, default=32)
    parser.add_argument("--hmvfe-device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--evaluation-interval", type=int, default=5)
    parser.add_argument("--wandb-mode", choices=("disabled", "offline", "online"), default="disabled")
    parser.add_argument("--wandb-project", default="mate-backbone-focus")
    parser.add_argument("--wandb-group", default="backbone-focus-100k")
    parser.add_argument(
        "--belief-safeguard-mode",
        choices=("zero", "zero_ema", "zero_ema_blend"),
        default="zero",
    )
    parser.add_argument(
        "--belief-distribution",
        choices=("gaussian", "categorical_grid"),
        default="categorical_grid",
    )
    parser.add_argument("--belief-grid-size", type=int, default=21)
    parser.add_argument("--belief-grid-max-delta", type=float, default=400.0)
    parser.add_argument("--belief-position-loss-coeff", type=float, default=0.1)
    parser.add_argument("--min-available-memory-gib", type=float, default=2.0)
    parser.add_argument("--max-hours", type=float, default=10.0)
    parser.add_argument("--rerun-failed", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    defaults = TIER_DEFAULTS[args.tier]
    args.timesteps = args.timesteps or defaults["timesteps"]
    args.seeds = tuple(args.seeds) if args.seeds is not None else defaults["seeds"]
    args.envs = tuple(args.envs) if args.envs is not None else defaults["envs"]
    args.output_root = args.output_root or REPO_ROOT / "experiments" / "focus_jnca_20261004" / args.tier
    return args


def main() -> None:
    args = parse_args()
    jobs = list(make_jobs(args))
    if args.dry_run:
        for job in jobs:
            result_dir = args.output_root / job["id"]
            print(" ".join(build_command(job, result_dir, args)))
        print(f"jobs={len(jobs)}")
        return

    args.output_root.mkdir(parents=True, exist_ok=True)
    state_path = args.output_root / "queue_state.json"
    state = load_state(state_path)
    state.update(
        {
            "tier": args.tier,
            "timesteps": args.timesteps,
            "seeds": list(args.seeds),
            "envs": list(args.envs),
            "resource_policy": {
                "sequential": True,
                "num_workers": args.num_workers,
                "num_envs": args.num_envs,
                "num_gpus": args.num_gpus,
                "buffer_capacity": args.buffer_capacity,
            },
        }
    )
    save_state(state_path, state)

    environment = os.environ.copy()
    existing_pythonpath = environment.get("PYTHONPATH", "")
    environment.update(
        {
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "SLURM_CPUS_ON_NODE": str(max(3, args.num_workers + 2)),
            "RAY_OBJECT_STORE_MEMORY": str(512 * 1024 * 1024),
            "WANDB_MODE": args.wandb_mode,
            "WANDB_DISABLED": "true" if args.wandb_mode == "disabled" else "false",
            "PYTHONUNBUFFERED": "1",
            "PYTHONPATH": os.pathsep.join(
                part
                for part in (
                    str(REPO_ROOT),
                    str(REPO_ROOT / "hmvfe-mate"),
                    existing_pythonpath,
                )
                if part
            ),
        }
    )

    started = time.monotonic()
    deadline = started + args.max_hours * 3600.0
    for job in jobs:
        previous = state["jobs"].get(job["id"], {})
        if previous.get("status") == "complete":
            print(f"SKIP complete {job['id']}")
            continue
        if previous.get("status") == "failed" and not args.rerun_failed:
            print(f"SKIP failed {job['id']} (use --rerun-failed)")
            continue
        if time.monotonic() >= deadline:
            print("STOP max-hours reached")
            break

        memory = available_memory_gib()
        if memory < args.min_available_memory_gib:
            print(
                f"STOP only {memory:.2f} GiB available; threshold is "
                f"{args.min_available_memory_gib:.2f} GiB"
            )
            break

        result_dir = args.output_root / job["id"]
        result_dir.mkdir(parents=True, exist_ok=True)
        command = build_command(job, result_dir, args)
        log_path = result_dir / "train.log"
        now = dt.datetime.now(dt.timezone.utc).isoformat()
        state["jobs"][job["id"]] = {
            **job,
            "status": "running",
            "started_at": now,
            "command": command,
            "log": str(log_path),
            "available_memory_gib_at_start": memory,
        }
        save_state(state_path, state)
        print(f"START {job['id']} mem={memory:.2f}GiB", flush=True)

        try:
            with log_path.open("a", encoding="utf-8") as log:
                log.write(f"\n[{now}] command: {' '.join(command)}\n")
                log.flush()
                completed = subprocess.run(
                    command,
                    cwd=REPO_ROOT,
                    env=environment,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
        except KeyboardInterrupt:
            state["jobs"][job["id"]].update(
                {
                    "status": "interrupted",
                    "ended_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                }
            )
            save_state(state_path, state)
            raise

        ended = dt.datetime.now(dt.timezone.utc).isoformat()
        state["jobs"][job["id"]].update(
            {
                "status": "complete" if completed.returncode == 0 else "failed",
                "returncode": completed.returncode,
                "ended_at": ended,
            }
        )
        save_state(state_path, state)
        print(
            f"END {job['id']} returncode={completed.returncode}",
            flush=True,
        )


if __name__ == "__main__":
    main()
