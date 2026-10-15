#!/usr/bin/env python3
"""Build paper-ready coverage tables and sparse-visibility gain curves.

Only completed runs that reached ``--min-steps`` are admitted.  Coverage is
read with the same eval-first policy as ``summarize_focus_paper_queue.py``.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

from summarize_focus_paper_queue import read_hmvfe, read_ray


ENVIRONMENT_ORDER = (
    "MATE-4v8-9-T100.yaml",
    "MATE-4v8-9-T70.yaml",
    "MATE-4v8-9-T40.yaml",
    "MATE-4v8-9.yaml",
)
TRANSMITTANCE = {
    "MATE-4v8-9-T100.yaml": 1.0,
    "MATE-4v8-9-T70.yaml": 0.7,
    "MATE-4v8-9-T40.yaml": 0.4,
    "MATE-4v8-9.yaml": 0.1,
}


def finite_float(value, default=0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def observed_steps(source: Path | None, algorithm: str) -> int:
    if source is None or not source.exists():
        return 0
    if algorithm in ("hmvfe", "hitmac"):
        steps = 0
        for line in source.read_text(encoding="utf-8").splitlines():
            if line:
                steps = max(steps, int(finite_float(json.loads(line).get("environment_steps"))))
        return steps
    with source.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return max(
        (int(finite_float(row.get("timesteps_total"))) for row in rows),
        default=0,
    )


def collect(queue_roots: List[Path], last_k: int, min_steps: int):
    runs: Dict[Tuple[str, str, str, int], dict] = {}
    for root in queue_roots:
        state_path = root / "queue_state.json"
        if not state_path.exists():
            continue
        state = json.loads(state_path.read_text(encoding="utf-8"))
        for job_id, job in state.get("jobs", {}).items():
            algorithm = str(job["algorithm"])
            reader = read_hmvfe if algorithm in ("hmvfe", "hitmac") else read_ray
            metric, score, _points, source = reader(root / job_id, last_k)
            steps = observed_steps(source, algorithm)
            eligible = (
                job.get("status") == "complete"
                and score is not None
                and steps >= min_steps
            )
            key = (algorithm, str(job["variant"]), str(job["env"]), int(job["seed"]))
            runs[key] = {
                "algorithm": algorithm,
                "variant": str(job["variant"]),
                "env": str(job["env"]),
                "seed": int(job["seed"]),
                "status": job.get("status", "unknown"),
                "steps": steps,
                "metric": metric,
                "coverage": score,
                "eligible": eligible,
                "source": str(source) if source else None,
            }
    return runs


def write_csv(path: Path, rows: List[dict], fieldnames: List[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("queue_roots", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/focus_visibility_report"))
    parser.add_argument("--last-k", type=int, default=5)
    parser.add_argument("--min-steps", type=int, default=250_000)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    runs = collect(args.queue_roots, args.last_k, args.min_steps)
    run_rows = sorted(runs.values(), key=lambda row: (
        row["algorithm"], row["variant"], ENVIRONMENT_ORDER.index(row["env"])
        if row["env"] in ENVIRONMENT_ORDER else 999, row["seed"]
    ))
    write_csv(
        args.output_dir / "eligible_runs.csv",
        run_rows,
        ["algorithm", "variant", "env", "seed", "status", "steps", "metric", "coverage", "eligible", "source"],
    )

    algorithms = sorted({row["algorithm"] for row in run_rows})
    wide_rows = []
    gain_rows = []
    for algorithm in algorithms:
        for variant in ("backbone", "focus"):
            row = {"algorithm_variant": algorithm + ("+FOCUS" if variant == "focus" else "")}
            for env in ENVIRONMENT_ORDER:
                values = [
                    item["coverage"] for item in run_rows
                    if item["algorithm"] == algorithm
                    and item["variant"] == variant
                    and item["env"] == env
                    and item["eligible"]
                ]
                row[env] = 100.0 * sum(values) / len(values) if values else None
            wide_rows.append(row)

        for env in ENVIRONMENT_ORDER:
            base = [
                item["coverage"] for item in run_rows
                if item["algorithm"] == algorithm and item["variant"] == "backbone"
                and item["env"] == env and item["eligible"]
            ]
            focus = [
                item["coverage"] for item in run_rows
                if item["algorithm"] == algorithm and item["variant"] == "focus"
                and item["env"] == env and item["eligible"]
            ]
            if base and focus:
                base_mean = sum(base) / len(base)
                focus_mean = sum(focus) / len(focus)
                gain_rows.append({
                    "algorithm": algorithm,
                    "environment": env,
                    "transmittance": TRANSMITTANCE[env],
                    "visibility_sparsity": 1.0 - TRANSMITTANCE[env],
                    "backbone_coverage_percent": 100.0 * base_mean,
                    "focus_coverage_percent": 100.0 * focus_mean,
                    "gain_percentage_points": 100.0 * (focus_mean - base_mean),
                    "focus_wins": focus_mean > base_mean,
                    "meets_3pp": focus_mean - base_mean >= 0.03,
                })

    write_csv(
        args.output_dir / "coverage_table.csv",
        wide_rows,
        ["algorithm_variant", *ENVIRONMENT_ORDER],
    )
    write_csv(
        args.output_dir / "focus_gain_by_visibility.csv",
        gain_rows,
        ["algorithm", "environment", "transmittance", "visibility_sparsity",
         "backbone_coverage_percent", "focus_coverage_percent",
         "gain_percentage_points", "focus_wins", "meets_3pp"],
    )

    try:
        import matplotlib.pyplot as plt

        figure, axis = plt.subplots(figsize=(7.2, 4.4), constrained_layout=True)
        for algorithm in sorted({row["algorithm"] for row in gain_rows}):
            points = sorted(
                (row for row in gain_rows if row["algorithm"] == algorithm),
                key=lambda row: row["visibility_sparsity"],
            )
            axis.plot(
                [row["visibility_sparsity"] for row in points],
                [row["gain_percentage_points"] for row in points],
                marker="o",
                linewidth=2,
                label=algorithm.upper(),
            )
        axis.axhline(0.0, color="black", linewidth=1)
        axis.axhline(3.0, color="tab:green", linewidth=1, linestyle="--", label="+3 pp")
        axis.set_xlabel("Visibility sparsity (1 - obstacle transmittance)")
        axis.set_ylabel("FOCUS improvement (coverage percentage points)")
        axis.grid(alpha=0.25)
        if gain_rows:
            axis.legend(frameon=False)
        figure.savefig(args.output_dir / "focus_gain_by_visibility.png", dpi=200)
        plt.close(figure)
    except ImportError:
        pass

    print(f"eligible_runs={sum(row['eligible'] for row in run_rows)}")
    print(f"paired_cells={len(gain_rows)}")
    print(args.output_dir.resolve())


if __name__ == "__main__":
    main()
