#!/usr/bin/env python3
"""Export complete W&B histories for *_focuss runs to a plot-ready CSV."""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Any

import pandas as pd
import wandb


HISTORY_PREFIX = "info/learner/default_policy/learner_stats/"
METRICS = [
    "focus_belief_pos_error_weighted",
    "focus_belief_baseline_pos_error_weighted",
    "focuss_action_gradient_norm",
    "focuss_backbone_gradient_norm",
    "focuss_action_gradient_scale",
    "focuss_credit_loss",
    "focuss_action_loss",
]
HISTORY_KEYS = [f"{HISTORY_PREFIX}{metric}" for metric in METRICS]
SYSTEM_KEYS = ["_step", "_timestamp", "_runtime", "timesteps_total"]


def scalar_or_nan(value: Any) -> Any:
    """Keep scalar metric values and turn absent/non-scalar values into NaN."""
    if value is None:
        return math.nan
    if isinstance(value, (str, int, float, bool)):
        return value
    return math.nan


def run_algorithm(run: Any) -> str:
    """Find the algorithm label while tolerating the config variants in this repo."""
    # These jobs use custom W&B IDs such as
    # ``resq_focuss-1429765-d3fa13a1``; that is the most explicit algorithm label.
    run_id_prefix = (run.id or "").split("-", 1)[0]
    if "_focuss" in run_id_prefix.lower():
        return run_id_prefix

    trainer_match = re.match(r"([A-Za-z0-9]+)FocusTrainer(?:_|$)", run.name or "")
    if trainer_match:
        return f"{trainer_match.group(1).lower()}_focuss"

    config = dict(run.config)
    candidates = (
        "algorithm",
        "algorithm_name",
        "alg_name",
        "algo",
        "name",
        "exp_name",
        "experiment_name",
    )
    for key in candidates:
        value = config.get(key)
        if isinstance(value, str) and value:
            return value
    return run.name or ""


def is_focuss_run(run: Any) -> bool:
    """Match the requested algorithm form, with run name as a safe fallback."""
    algorithm = run_algorithm(run).lower()
    run_id = (run.id or "").lower()
    run_name = (run.name or "").lower()
    return (
        "_focuss" in algorithm
        or "_focuss" in run_id
        or "_focuss" in run_name
        or re.match(r"[a-z0-9]+focustrainer(?:_|$)", run_name) is not None
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default="mate-value-factorization-2m")
    parser.add_argument("--entity", default=None)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("wandb_mate_value_factorization_2m_focuss_metrics.csv"),
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=1e-12,
        help="Denominator stabilizer for RAGC (default: 1e-12).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    api = wandb.Api(timeout=60)
    project_path = f"{args.entity}/{args.project}" if args.entity else args.project
    all_runs = list(api.runs(project_path))
    runs = [run for run in all_runs if is_focuss_run(run)]

    if not runs:
        raise RuntimeError(
            f"No run with an algorithm/run name containing '_focuss' found in "
            f"{project_path!r} ({len(all_runs)} total runs)."
        )

    records: list[dict[str, Any]] = []
    for run in runs:
        algorithm = run_algorithm(run)
        logging_index = 0
        # scan_history streams the complete unsampled history. Do not pass
        # ``keys=...``: W&B then keeps only rows that contain every requested key,
        # which would silently drop partial metric logging events.
        for item in run.scan_history(min_step=0):
            metric_values = {
                metric: scalar_or_nan(item.get(full_key))
                for metric, full_key in zip(METRICS, HISTORY_KEYS)
            }
            # scan_history can surface rows containing only system fields. They
            # are not requested metric logging events, so exclude those rows.
            if all(pd.isna(value) for value in metric_values.values()):
                continue

            record = {
                "run_name": run.name,
                "run_id": run.id,
                "algorithm": algorithm,
                "logging_index": logging_index,
                "_step": item.get("_step"),
                "_timestamp": item.get("_timestamp"),
                "_runtime": item.get("_runtime"),
                "timesteps_total": item.get("timesteps_total"),
                **metric_values,
            }

            error = record["focus_belief_pos_error_weighted"]
            baseline = record["focus_belief_baseline_pos_error_weighted"]
            if pd.notna(error) and pd.notna(baseline) and baseline != 0:
                record["prediction_error_reduction_percent"] = 100.0 * (
                    1.0 - error / baseline
                )
            else:
                record["prediction_error_reduction_percent"] = math.nan

            action_norm = record["focuss_action_gradient_norm"]
            action_scale = record["focuss_action_gradient_scale"]
            backbone_norm = record["focuss_backbone_gradient_norm"]
            if all(pd.notna(v) for v in (action_norm, action_scale, backbone_norm)):
                scaled_action_norm = action_norm * action_scale
                record["relative_action_gradient_contribution_percent"] = (
                    100.0
                    * scaled_action_norm
                    / (scaled_action_norm + backbone_norm + args.epsilon)
                )
            else:
                record["relative_action_gradient_contribution_percent"] = math.nan

            records.append(record)
            logging_index += 1

        print(f"{run.id}\t{run.name}\t{algorithm}\t{logging_index} rows")

    columns = [
        "run_name",
        "run_id",
        "algorithm",
        "logging_index",
        "_step",
        "_timestamp",
        "_runtime",
        "timesteps_total",
        *METRICS,
        "prediction_error_reduction_percent",
        "relative_action_gradient_contribution_percent",
    ]
    frame = pd.DataFrame.from_records(records, columns=columns)
    frame.sort_values(["run_name", "run_id", "logging_index"], inplace=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    print(f"Wrote {len(frame)} rows from {len(runs)} runs to {args.output.resolve()}")


if __name__ == "__main__":
    main()
