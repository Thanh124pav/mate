#!/usr/bin/env python3
"""Summarize a FOCUS queue into run-level and paired comparison tables."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional, Sequence


RAY_METRICS = (
    "eval/mean_coverage_rate",
    "evaluation/custom_metrics/coverage_rate_mean",
    "evaluation/custom_metrics/real_coverage_rate_mean",
    "custom_metrics/real_coverage_rate_mean",
    "sampler_results/custom_metrics/real_coverage_rate_mean",
    "real_coverage_rate_mean",
    "evaluation/custom_metrics/coverage_rate_mean",
    "custom_metrics/coverage_rate_mean",
    "sampler_results/custom_metrics/coverage_rate_mean",
    "coverage_rate_mean",
    "episode_reward_mean",
)

HMVFE_METRICS = (
    "eval/mean_coverage_rate",
    "eval/mean_real_coverage_rate",
    "train/episode_coverage_rate",
    "train/mean_coverage_rate",
)


def finite(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def choose_metric(rows: Sequence[Dict[str, Any]], candidates: Iterable[str]) -> Optional[str]:
    for candidate in candidates:
        if any(finite(row.get(candidate)) is not None for row in rows):
            return candidate
    return None


def score_rows(
    rows: Sequence[Dict[str, Any]],
    candidates: Iterable[str],
    last_k: int,
    collapse_consecutive: bool = False,
):
    metric = choose_metric(rows, candidates)
    if metric is None:
        return None, None, 0
    values = [value for row in rows if (value := finite(row.get(metric))) is not None]
    if collapse_consecutive:
        # Older RLlib versions forward-fill the most recent evaluation result
        # into every subsequent training iteration. Count each actual eval
        # checkpoint once instead of weighting it by the interval length.
        values = [
            value for index, value in enumerate(values)
            if index == 0 or value != values[index - 1]
        ]
    if not values:
        return metric, None, 0
    tail = values[-last_k:]
    return metric, mean(tail), len(values)


def read_ray(result_dir: Path, last_k: int):
    progress_files = sorted(result_dir.rglob("progress.csv"), key=lambda path: path.stat().st_mtime)
    if not progress_files:
        return None, None, 0, None
    path = progress_files[-1]
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    metric = choose_metric(rows, RAY_METRICS)
    is_evaluation = (
        metric is not None
        and (metric.startswith("eval/") or metric.startswith("evaluation/"))
    )
    event_rows = rows
    used_schedule = False
    if is_evaluation:
        params_path = path.with_name("params.json")
        try:
            interval = int(json.loads(params_path.read_text(encoding="utf-8"))["evaluation_interval"])
        except (FileNotFoundError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            interval = 0
        if interval > 0:
            scheduled = [
                row for row in rows
                if int(finite(row.get("training_iteration")) or -1) % interval == 0
            ]
            if any(finite(row.get(metric)) is not None for row in scheduled):
                event_rows = scheduled
                used_schedule = True
    metric, score, points = score_rows(
        event_rows,
        (metric,) if metric is not None else (),
        last_k,
        collapse_consecutive=is_evaluation and not used_schedule,
    )
    return metric, score, points, path


def read_hmvfe(result_dir: Path, last_k: int):
    metric_files = sorted(result_dir.rglob("metrics.jsonl"), key=lambda path: path.stat().st_mtime)
    if not metric_files:
        return None, None, 0, None
    path = metric_files[-1]
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    metric, score, points = score_rows(rows, HMVFE_METRICS, last_k)
    return metric, score, points, path


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("queue_root", type=Path)
    parser.add_argument("--last-k", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    state_path = args.queue_root / "queue_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    run_rows: List[Dict[str, Any]] = []

    for job_id, job in state.get("jobs", {}).items():
        result_dir = args.queue_root / job_id
        reader = (
            read_hmvfe
            if job.get("algorithm") in ("hmvfe", "hitmac")
            else read_ray
        )
        metric, score, points, source = reader(result_dir, args.last_k)
        run_rows.append(
            {
                "job_id": job_id,
                "status": job.get("status", "unknown"),
                "family": job.get("family"),
                "algorithm": job.get("algorithm"),
                "variant": job.get("variant"),
                "env": job.get("env"),
                "seed": job.get("seed"),
                "timesteps": job.get("timesteps"),
                "metric": metric,
                "score_last_k": score,
                "metric_points": points,
                "source": str(source) if source else None,
            }
        )

    index = {
        (row["family"], row["algorithm"], row["env"], row["seed"], row["variant"]): row
        for row in run_rows
    }
    pair_rows: List[Dict[str, Any]] = []
    pair_keys = sorted({key[:4] for key in index})
    for family, algorithm, env, seed in pair_keys:
        backbone = index.get((family, algorithm, env, seed, "backbone"))
        focus = index.get((family, algorithm, env, seed, "focus"))
        base_score = finite(backbone.get("score_last_k")) if backbone else None
        focus_score = finite(focus.get("score_last_k")) if focus else None
        absolute = focus_score - base_score if base_score is not None and focus_score is not None else None
        relative = (
            100.0 * absolute / max(abs(base_score), 1e-8)
            if absolute is not None and base_score is not None
            else None
        )
        pair_rows.append(
            {
                "family": family,
                "algorithm": algorithm,
                "env": env,
                "seed": seed,
                "backbone_score": base_score,
                "focus_score": focus_score,
                "absolute_gain": absolute,
                "absolute_gain_pp": 100.0 * absolute if absolute is not None else None,
                "relative_gain_percent": relative,
                # The reporting target is +3 percentage points of coverage,
                # not a 3% relative change.
                "meets_3pp": absolute is not None and absolute >= 0.03,
                "pair_complete": base_score is not None and focus_score is not None,
            }
        )

    write_csv(args.queue_root / "run_summary.csv", run_rows)
    write_csv(args.queue_root / "paired_summary.csv", pair_rows)
    (args.queue_root / "run_summary.json").write_text(
        json.dumps(run_rows, indent=2) + "\n", encoding="utf-8"
    )
    (args.queue_root / "paired_summary.json").write_text(
        json.dumps(pair_rows, indent=2) + "\n", encoding="utf-8"
    )

    for row in pair_rows:
        print(
            f"{row['family']}/{row['algorithm']} {row['env']} s{row['seed']}: "
            f"base={row['backbone_score']} focus={row['focus_score']} "
            f"gain_pp={row['absolute_gain_pp']}"
        )


if __name__ == "__main__":
    main()
