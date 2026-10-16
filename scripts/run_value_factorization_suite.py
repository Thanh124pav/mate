#!/usr/bin/env python3
"""Run matched MATE value-factorization configurations sequentially.

Each algorithm is launched in a fresh process so Ray releases workers, CUDA
contexts, and object-store memory before the next trial.  Stdout/stderr and a
machine-readable suite summary are kept under ``--local-dir``.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


MODULES = {
    "FACMAC": "examples.facmac.camera",
    "FACMAC_FOCUS": "examples.facmac_focus.camera",
    "FACMAC_FOCUSS": "examples.facmac_focuss.camera",
    "RESQ": "examples.resq.camera",
    "RESQ_FOCUS": "examples.resq_focus.camera",
    "RISKQ": "examples.riskq.camera",
    "RISKQ_FOCUS": "examples.riskq_focus.camera",
    "QFREE": "examples.qfree.camera",
    "QFREE_FOCUS": "examples.qfree_focus.camera",
    "CONCAVEQ": "examples.concaveq.camera",
    "CONCAVEQ_FOCUS": "examples.concaveq_focus.camera",
}


def latest_result(root):
    candidates = sorted(root.rglob("result.json"), key=lambda path: path.stat().st_mtime)
    if not candidates:
        return None, None
    path = candidates[-1]
    lines = [line for line in path.read_text().splitlines() if line.strip()]
    return path, json.loads(lines[-1]) if lines else None


def finite_learner_stats(result):
    import math

    failures = []
    learner = (result or {}).get("info", {}).get("learner", {})
    for policy_id, payload in learner.items():
        stats = payload.get("learner_stats", payload)
        for key, value in stats.items():
            if isinstance(value, (int, float)) and not math.isfinite(float(value)):
                failures.append(f"{policy_id}.{key}={value}")
    return failures


def main():
    parser = argparse.ArgumentParser()
    # ``argparse`` on Python 3.9 validates the empty list produced by
    # ``nargs='*'`` against ``choices`` as if ``[]`` were itself a choice.
    # Validate explicitly so omitting this positional reliably means all ten.
    parser.add_argument("algorithms", nargs="*", metavar="ALGORITHM")
    parser.add_argument("--timesteps-total", type=int, default=100000)
    parser.add_argument("--local-dir", type=Path, required=True)
    parser.add_argument("--project", default="mate-value-factorization")
    parser.add_argument("--group", default=None)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--num-envs-per-worker", type=int, default=2)
    parser.add_argument("--num-gpus", type=float, default=0.2)
    parser.add_argument("--evaluation-interval", type=int, default=5)
    parser.add_argument("--buffer-capacity", type=int, default=500)
    parser.add_argument(
        "--belief-safeguard-mode",
        choices=("zero", "zero_ema", "zero_ema_blend"),
        default=None,
    )
    parser.add_argument(
        "--belief-distribution",
        choices=("gaussian", "categorical_grid"),
        default=None,
    )
    parser.add_argument("--belief-grid-size", type=int, default=None)
    parser.add_argument("--belief-grid-max-delta", type=float, default=None)
    parser.add_argument("--belief-grid-label-std", type=float, default=None)
    parser.add_argument("--belief-position-loss-coeff", type=float, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--disable-wandb", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args()

    unknown = sorted(set(args.algorithms) - set(MODULES))
    if unknown:
        parser.error(
            "unknown algorithm(s): "
            + ", ".join(unknown)
            + "; choose from "
            + ", ".join(MODULES)
        )
    selected = args.algorithms or list(MODULES)
    belief_options = (
        args.belief_safeguard_mode,
        args.belief_distribution,
        args.belief_grid_size,
        args.belief_grid_max_delta,
        args.belief_grid_label_std,
        args.belief_position_loss_coeff,
    )
    if any(value is not None for value in belief_options):
        incompatible = [name for name in selected if not name.endswith("_FOCUS")]
        if incompatible:
            parser.error(
                "belief options require only +FOCUS algorithms; got "
                + ", ".join(incompatible)
            )
    args.local_dir.mkdir(parents=True, exist_ok=True)
    summary = {}
    for name in selected:
        output_dir = args.local_dir / name.lower()
        output_dir.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable,
            "-m",
            MODULES[name],
            "--timesteps-total",
            str(args.timesteps_total),
            "--num-workers",
            str(args.num_workers),
            "--num-envs-per-worker",
            str(args.num_envs_per_worker),
            "--num-gpus",
            str(args.num_gpus),
            "--evaluation-interval",
            str(args.evaluation_interval),
            "--buffer-capacity",
            str(args.buffer_capacity),
            "--seed",
            str(args.seed),
            "--project",
            args.project,
            "--local-dir",
            str(output_dir),
        ]
        if args.group:
            command.extend(["--group", args.group])
        if args.belief_safeguard_mode is not None:
            command.extend(
                ["--belief-safeguard-mode", args.belief_safeguard_mode]
            )
        for flag, value in (
            ("--belief-distribution", args.belief_distribution),
            ("--belief-grid-size", args.belief_grid_size),
            ("--belief-grid-max-delta", args.belief_grid_max_delta),
            ("--belief-grid-label-std", args.belief_grid_label_std),
            ("--belief-position-loss-coeff", args.belief_position_loss_coeff),
        ):
            if value is not None:
                command.extend([flag, str(value)])
        environment = os.environ.copy()
        if args.disable_wandb:
            environment["WANDB_DISABLED"] = "true"
        else:
            environment.pop("WANDB_DISABLED", None)
        log_path = args.local_dir / f"{name.lower()}.log"
        start = time.time()
        print(f"START {name}: {' '.join(command)}", flush=True)
        with log_path.open("w") as stream:
            completed = subprocess.run(
                command,
                cwd=Path(__file__).resolve().parents[1],
                env=environment,
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=False,
            )
        result_path, result = latest_result(output_dir)
        nonfinite = finite_learner_stats(result)
        summary[name] = {
            "returncode": completed.returncode,
            "elapsed_seconds": time.time() - start,
            "log": str(log_path),
            "result": str(result_path) if result_path else None,
            "timesteps_total": (result or {}).get("timesteps_total"),
            "episode_reward_mean": (result or {}).get("episode_reward_mean"),
            "coverage_rate_mean": (result or {}).get("custom_metrics", {}).get(
                "coverage_rate_mean"
            ),
            "evaluation_coverage_rate_mean": (result or {}).get(
                "evaluation", {}
            ).get("custom_metrics", {}).get("coverage_rate_mean"),
            "nonfinite_learner_stats": nonfinite,
        }
        summary_path = args.local_dir / "suite_results.json"
        summary_path.write_text(json.dumps(summary, indent=2))
        status = "PASS" if completed.returncode == 0 and not nonfinite else "FAIL"
        print(
            f"{status} {name}: rc={completed.returncode} "
            f"steps={summary[name]['timesteps_total']} "
            f"coverage={summary[name]['coverage_rate_mean']} "
            f"elapsed={summary[name]['elapsed_seconds']:.1f}s",
            flush=True,
        )
        if status == "FAIL" and not args.continue_on_error:
            return completed.returncode or 2
    print(f"Suite results: {args.local_dir / 'suite_results.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
