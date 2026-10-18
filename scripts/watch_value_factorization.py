#!/usr/bin/env python3
"""Periodically summarize value-factorization priority queues."""

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def process_alive(pid):
    if not pid:
        return False
    try:
        os.kill(int(pid), 0)
    except (OSError, ValueError):
        return False
    return True


def load_rows(path):
    if not path:
        return []
    result_path = Path(path)
    if not result_path.is_file():
        return []
    rows = []
    for line in result_path.read_text(encoding="utf-8").splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def resumed_rows(item):
    """Combine pre-checkpoint history with the reset counters of a restored run."""

    current = load_rows(item.get("result_path"))
    restored_from = item.get("restored_from")
    iteration_offset = int(item.get("restore_iteration") or 0)
    timesteps_offset = int(item.get("restore_timesteps") or 0)
    if not restored_from:
        return current
    checkpoint = Path(restored_from)
    previous = load_rows(checkpoint.parent / "result.json")
    previous = [
        row
        for row in previous
        if int(row.get("training_iteration") or 0) <= iteration_offset
    ]
    adjusted = []
    for row in current:
        row = dict(row)
        if not item.get("restore_counters_preserved"):
            if row.get("training_iteration") is not None:
                row["training_iteration"] += iteration_offset
            if row.get("timesteps_total") is not None:
                row["timesteps_total"] += timesteps_offset
        adjusted.append(row)
    return previous + adjusted


def coverage(row, evaluation=False):
    source = row.get("evaluation", {}) if evaluation else row
    metrics = source.get("custom_metrics", {}) if isinstance(source, dict) else {}
    return metrics.get("coverage_rate_mean")


def train_trend(rows):
    values = [coverage(row) for row in rows]
    values = [float(value) for value in values if value is not None]
    if len(values) < 20:
        return {"plateau": "insufficient_data"}
    recent = values[-20:]
    before = sum(recent[:10]) / 10
    after = sum(recent[10:]) / 10
    delta = after - before
    return {
        "plateau": abs(delta) <= 0.01,
        "recent_mean": after,
        "previous_mean": before,
        "delta": delta,
    }


def eval_trend(rows):
    points = []
    for row in rows:
        value = coverage(row, evaluation=True)
        if value is not None:
            evaluation = row.get("evaluation", {})
            points.append(
                {
                    "iteration": row.get("training_iteration"),
                    "timesteps_total": row.get("timesteps_total"),
                    "coverage_rate": value,
                    "real_coverage_rate": evaluation.get("custom_metrics", {}).get(
                        "real_coverage_rate_mean"
                    ),
                    "episode_reward_mean": evaluation.get("episode_reward_mean"),
                }
            )
    if len(points) < 3:
        trend = "insufficient_data"
        plateau = "insufficient_data"
    else:
        recent = points[-3:]
        change = float(recent[-1]["coverage_rate"]) - float(
            recent[0]["coverage_rate"]
        )
        last_change = float(recent[-1]["coverage_rate"]) - float(
            recent[-2]["coverage_rate"]
        )
        spread = max(float(p["coverage_rate"]) for p in recent) - min(
            float(p["coverage_rate"]) for p in recent
        )
        plateau = abs(change) <= 0.01 and spread <= 0.02
        trend = (
            "plateau"
            if plateau
            else ("improving" if last_change > 0.01 else "declining" if last_change < -0.01 else "mixed")
        )
    return {"plateau": plateau, "trend": trend, "points": points[-5:]}


def summarize(status_path):
    status = json.loads(status_path.read_text(encoding="utf-8"))
    jobs = []
    queued = []
    for name, item in status.get("jobs", {}).items():
        state = item.get("state")
        if state == "queued":
            queued.append(name)
        rows = resumed_rows(item)
        latest = rows[-1] if rows else {}
        train_plateau = train_trend(rows)
        if item.get("restored_from"):
            resumed_iterations = int(item.get("training_iteration") or 0) - int(
                item.get("restore_iteration") or 0
            )
            if resumed_iterations < 20:
                train_plateau = {
                    "plateau": "resume_warmup",
                    "resumed_iterations": max(0, resumed_iterations),
                }
        entry = {
            "algorithm": name,
            "state": state,
            "pid": item.get("pid"),
            "process_alive": process_alive(item.get("pid")) if state == "running" else None,
            "training_iteration": item.get(
                "training_iteration", latest.get("training_iteration")
            ),
            "timesteps_total": item.get("timesteps_total", latest.get("timesteps_total")),
            "train_coverage_rate": item.get(
                "coverage_rate_mean", coverage(latest) if latest else None
            ),
            "train_plateau": train_plateau,
            "evaluation": eval_trend(rows),
            "result_path": item.get("result_path"),
            "restored_from": item.get("restored_from"),
        }
        jobs.append(entry)
    return {
        "status_path": str(status_path),
        "supervisor_pid": status.get("supervisor_pid"),
        "supervisor_alive": process_alive(status.get("supervisor_pid")),
        "current_algorithm": status.get("current_algorithm"),
        "jobs": jobs,
        "queued": queued,
    }


def write_report(status_paths, output_dir):
    report = {
        "generated_at": utc_now(),
        "lanes": [summarize(path) for path in status_paths if path.is_file()],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    temporary = output_dir / "latest_report.json.tmp"
    latest = output_dir / "latest_report.json"
    payload = json.dumps(report, indent=2, allow_nan=True)
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(latest)
    with (output_dir / "reports.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(report, allow_nan=True) + "\n")
    print(payload, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--status", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--interval-seconds", type=int, default=30 * 60)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    while True:
        write_report(args.status, args.output_dir)
        if args.once:
            return
        time.sleep(args.interval_seconds)


if __name__ == "__main__":
    main()
