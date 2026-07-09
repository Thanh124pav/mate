#!/bin/sh
"exec" "python3" "$0" "$@"
"exit" "$?"
"""Select and evaluate the best Ray Tune checkpoints for MATE camera agents."""

import argparse
import csv
import json
import math
import os
import re
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Pattern, Tuple


REPO = Path(__file__).resolve().parent.parent
HOME_RAY_RESULTS = Path("/home/pavt1024/ray_results")
DEFAULT_CONFIGS = ["4v5-0", "4v2-9", "4v4-9", "4v8-9"]
DEFAULT_ALGORITHMS = [
    "QPLEX",
    "QMIX",
    "DuelMIX",
    "SPECTRA",
    "QPLEX_V2",
    "QPLEX_WM2",
    "QPLEX_FOCUS",
]

ALGORITHM_SPECS = {
    "qplex": {
        "label": "QPLEX",
        "slug": "qplex",
        "agent": "examples.hrl.qplex:HRLQPLEXCameraAgent",
    },
    "qmix": {
        "label": "QMIX",
        "slug": "qmix",
        "agent": "examples.hrl.qmix:HRLQMIXCameraAgent",
    },
    "duelmix": {
        "label": "DuelMIX",
        "slug": "duelmix",
        "agent": "examples.hrl.duelmix:HRLDuelMixCameraAgent",
    },
    "spectra": {
        "label": "SPECTRA",
        "slug": "spectra",
        "agent": "examples.hrl.spectra:HRLSPECTraCameraAgent",
    },
    "qplex_v2": {
        "label": "QPLEX_V2",
        "slug": "qplex_v2",
        "agent": "examples.hrl.qplex_v2:HRLQPLEXV2CameraAgent",
    },
    "qplex_wm2": {
        "label": "QPLEX_WM2",
        "slug": "qplex_wm2",
        "agent": "examples.hrl.qplex_wm2:HRLQPLEXWM2_CameraAgent",
    },
    "qplex_focus": {
        "label": "QPLEX_FOCUS",
        "slug": "qplex_focus",
        "agent": "examples.hrl.qplex_focus:HRLQPLEXFocusCameraAgent",
    },
}

METRIC_ALIASES = {
    "real_coverage_rate_mean": [
        "real_coverage_rate_mean",
        "custom_metrics/real_coverage_rate_mean",
        "sampler_results/custom_metrics/real_coverage_rate_mean",
    ],
    "real_coverage_rate_min": [
        "real_coverage_rate_min",
        "custom_metrics/real_coverage_rate_min",
        "sampler_results/custom_metrics/real_coverage_rate_min",
    ],
    "real_coverage_rate_max": [
        "real_coverage_rate_max",
        "custom_metrics/real_coverage_rate_max",
        "sampler_results/custom_metrics/real_coverage_rate_max",
    ],
}

EVAL_METRICS = [
    "Step / Cargo",
    "Target Episode Reward",
    "Mean Transport Rate",
    "Mean Coverage Rate",
    "Normalized Target Episode Reward",
]


@dataclass
class ProgressPoint:
    iteration: int
    mean: float
    minimum: float
    maximum: float


@dataclass
class SelectedCheckpoint:
    algorithm: str
    config: str
    run_dir: Path
    checkpoint_dir: Path
    checkpoint_file: Path
    checkpoint_iteration: int
    metric_value: float
    global_best_iteration: int
    global_best_value: float
    progress: List[ProgressPoint]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate best checkpoints selected from Ray Tune progress.csv files.",
    )
    parser.add_argument(
        "--specific-run",
        default="all",
        help="all, an existing path, a run basename, or a Python regex matched against run paths.",
    )
    parser.add_argument(
        "--specific-config",
        nargs="*",
        default=DEFAULT_CONFIGS,
        help="Environment configs to evaluate, e.g. 4v5-0 4v8-9. Default: %(default)s",
    )
    parser.add_argument("--episodes", type=int, default=20, help="Evaluation episodes.")
    parser.add_argument("--seed", type=int, default=0, help="Evaluation seed.")
    parser.add_argument(
        "--output-dir",
        default=str(REPO / "outputs" / "evaluate_best_checkpoints"),
        help="Directory for markdown, plots, logs, and machine-readable outputs.",
    )
    parser.add_argument(
        "--metric",
        default="real_coverage_rate_mean",
        help="Training metric used to select checkpoints. Default: %(default)s",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Select checkpoints and write reports without running mate.evaluate.",
    )
    parser.add_argument(
        "--algorithms",
        nargs="*",
        default=DEFAULT_ALGORITHMS,
        help="Algorithms to evaluate. Default: %(default)s",
    )
    return parser.parse_args()


def normalize_config(value: Any) -> str:
    name = Path(str(value)).name if value is not None else "unknown"
    if name.endswith(".yaml"):
        name = name[:-5]
    if name.startswith("MATE-"):
        name = name[len("MATE-") :]
    return name or "unknown"


def config_file(config: str) -> str:
    config = normalize_config(config)
    return f"MATE-{config}.yaml"


def normalize_algorithm(value: str) -> str:
    value = value.strip().replace("-", "_")
    value = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", value)
    return value.lower().strip("_")


def algorithm_from_run_dir(run_dir: Path) -> Optional[str]:
    text = run_dir.resolve().as_posix()
    match = re.search(r"/examples/hrl/([^/]+)/camera/ray_results/", text)
    if match:
        slug = normalize_algorithm(match.group(1))
        return slug if slug in ALGORITHM_SPECS else None

    name = run_dir.name
    if "mate-hrl." in name:
        slug = name.split("mate-hrl.", 1)[1].split(".camera", 1)[0]
        slug = normalize_algorithm(slug)
        return slug if slug in ALGORITHM_SPECS else None

    return None


def safe_resolve(path: Path) -> Path:
    try:
        return path.resolve()
    except OSError:
        return path.absolute()


def candidate_roots() -> List[Path]:
    roots = []  # type: List[Path]
    examples = REPO / "examples"
    if examples.exists():
        roots.extend(sorted(examples.glob("**/ray_results")))
    roots.append(HOME_RAY_RESULTS)
    return roots


def discover_run_dirs() -> List[Path]:
    seen = set()
    runs = []  # type: List[Path]
    for root in candidate_roots():
        if not root.exists():
            continue
        for progress in root.rglob("progress.csv"):
            run_dir = progress.parent
            resolved = safe_resolve(run_dir)
            if resolved in seen:
                continue
            seen.add(resolved)
            runs.append(run_dir)
    return sorted(runs, key=lambda path: safe_resolve(path).as_posix())


def matches_specific_run(run_dir: Path, spec: str) -> bool:
    if spec == "all":
        return True
    direct = Path(spec).expanduser()
    if not direct.is_absolute():
        direct = REPO / direct
    if direct.exists():
        resolved = safe_resolve(run_dir)
        target = safe_resolve(direct)
        return resolved == target or target in resolved.parents
    if run_dir.name == spec:
        return True
    candidates = [run_dir.name, safe_resolve(run_dir).as_posix()]
    try:
        candidates.append(safe_resolve(run_dir).relative_to(REPO).as_posix())
    except ValueError:
        pass
    try:
        pattern = re.compile(spec)
    except re.error:
        return False
    return any(pattern.search(candidate) for candidate in candidates)


def parse_float(value: Any) -> float:
    if value in (None, "", "nan", "NaN", "None"):
        return math.nan
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return math.nan
    return parsed if math.isfinite(parsed) else math.nan


def choose_column(fieldnames: Optional[List[str]], aliases: List[str]) -> Optional[str]:
    if not fieldnames:
        return None
    for alias in aliases:
        if alias in fieldnames:
            return alias
    return None


def metric_aliases(metric: str) -> List[str]:
    return METRIC_ALIASES.get(metric, [metric])


def read_params(run_dir: Path) -> Dict[str, Any]:
    path = run_dir / "params.json"
    if not path.exists():
        return {}
    try:
        with path.open() as handle:
            params = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return {}
    env_config = params.get("env_config") or {}
    return env_config if isinstance(env_config, dict) else {}


def read_progress(run_dir: Path, metric: str) -> List[ProgressPoint]:
    path = run_dir / "progress.csv"
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        mean_column = choose_column(reader.fieldnames, metric_aliases(metric))
        min_column = choose_column(reader.fieldnames, METRIC_ALIASES["real_coverage_rate_min"])
        max_column = choose_column(reader.fieldnames, METRIC_ALIASES["real_coverage_rate_max"])
        iteration_column = choose_column(reader.fieldnames, ["training_iteration", "iterations_since_restore"])
        if mean_column is None or iteration_column is None:
            return []

        points = []  # type: List[ProgressPoint]
        for index, row in enumerate(reader, start=1):
            try:
                iteration = int(float(row.get(iteration_column) or index))
            except ValueError:
                iteration = index
            mean_value = parse_float(row.get(mean_column))
            if not math.isfinite(mean_value):
                continue
            points.append(
                ProgressPoint(
                    iteration=iteration,
                    mean=mean_value,
                    minimum=parse_float(row.get(min_column)) if min_column else math.nan,
                    maximum=parse_float(row.get(max_column)) if max_column else math.nan,
                )
            )
    return points


def checkpoint_file(checkpoint_dir: Path, iteration: int) -> Optional[Path]:
    preferred = checkpoint_dir / f"checkpoint-{iteration}"
    if preferred.exists() and preferred.is_file():
        return preferred
    files = sorted(path for path in checkpoint_dir.iterdir() if path.is_file() and path.name.startswith("checkpoint-") and not path.name.endswith(".tune_metadata"))
    return files[0] if files else None


def available_checkpoints(run_dir: Path) -> Dict[int, Tuple[Path, Path]]:
    checkpoints = {}  # type: Dict[int, Tuple[Path, Path]]
    for checkpoint_dir in sorted(run_dir.glob("checkpoint_*")):
        if not checkpoint_dir.is_dir():
            continue
        match = re.search(r"checkpoint_(\d+)$", checkpoint_dir.name)
        if not match:
            continue
        iteration = int(match.group(1))
        ckpt_file = checkpoint_file(checkpoint_dir, iteration)
        if ckpt_file is not None:
            checkpoints[iteration] = (checkpoint_dir, ckpt_file)
    return checkpoints


def select_checkpoint_for_run(run_dir: Path, algorithm: str, config: str, metric: str) -> Optional[SelectedCheckpoint]:
    progress = read_progress(run_dir, metric)
    checkpoints = available_checkpoints(run_dir)
    if not progress or not checkpoints:
        return None

    global_best = max(progress, key=lambda point: point.mean)
    progress_by_iteration = {point.iteration: point for point in progress}
    checkpoint_points = [
        progress_by_iteration[iteration]
        for iteration in checkpoints
        if iteration in progress_by_iteration
    ]
    if not checkpoint_points:
        return None

    selected = max(
        checkpoint_points,
        key=lambda point: (point.mean, -abs(point.iteration - global_best.iteration)),
    )
    checkpoint_dir, ckpt_file = checkpoints[selected.iteration]
    return SelectedCheckpoint(
        algorithm=algorithm,
        config=config,
        run_dir=run_dir,
        checkpoint_dir=checkpoint_dir,
        checkpoint_file=ckpt_file,
        checkpoint_iteration=selected.iteration,
        metric_value=selected.mean,
        global_best_iteration=global_best.iteration,
        global_best_value=global_best.mean,
        progress=progress,
    )


def select_best_checkpoints(args: argparse.Namespace) -> Dict[Tuple[str, str], SelectedCheckpoint]:
    wanted_algorithms = {normalize_algorithm(name) for name in args.algorithms}
    wanted_configs = {normalize_config(config) for config in args.specific_config}
    selected = {}  # type: Dict[Tuple[str, str], SelectedCheckpoint]

    for run_dir in discover_run_dirs():
        if not matches_specific_run(run_dir, args.specific_run):
            continue
        algorithm = algorithm_from_run_dir(run_dir)
        if algorithm is None or algorithm not in wanted_algorithms:
            continue
        env_config = read_params(run_dir)
        config = normalize_config(env_config.get("config"))
        if config not in wanted_configs:
            continue
        candidate = select_checkpoint_for_run(run_dir, algorithm, config, args.metric)
        if candidate is None:
            continue
        key = (config, algorithm)
        old = selected.get(key)
        if old is None or candidate.metric_value > old.metric_value:
            selected[key] = candidate
    return selected


def slugify(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return slug.strip("_") or "unknown"


def ensure_output_dirs(output_dir: Path) -> Dict[str, Path]:
    paths = {
        "root": output_dir,
        "plots": output_dir / "plots",
        "logs": output_dir / "logs",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def import_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_training_curve(selection: SelectedCheckpoint, plot_dir: Path) -> Optional[Path]:
    try:
        plt = import_matplotlib()
    except Exception as exc:  # pragma: no cover - optional plotting dependency
        print(f"WARNING: could not import matplotlib for plot: {exc}")
        return None

    points = [point for point in selection.progress if point.iteration <= selection.checkpoint_iteration]
    if not points:
        return None
    xs = [point.iteration for point in points]
    means = [point.mean for point in points]
    mins = [point.minimum for point in points]
    maxs = [point.maximum for point in points]

    fig, ax = plt.subplots(figsize=(10.5, 5.5))
    ax.plot(xs, means, linewidth=2.2, label="real_coverage_rate_mean")
    if any(math.isfinite(value) for value in mins) and any(math.isfinite(value) for value in maxs):
        ax.fill_between(xs, mins, maxs, alpha=0.16, label="min/max")
    ax.axvline(selection.checkpoint_iteration, color="black", linestyle="--", linewidth=1.3, label=f"checkpoint {selection.checkpoint_iteration}")
    ax.set_title(f"{ALGORITHM_SPECS[selection.algorithm]['label']} on {selection.config}", loc="left", fontweight="bold")
    ax.set_xlabel("training iteration")
    ax.set_ylabel("real_coverage_rate")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    path = plot_dir / f"{slugify(selection.config)}__{slugify(selection.algorithm)}.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def parse_eval_table(stdout: str) -> Dict[str, str]:
    metrics = {}  # type: Dict[str, str]
    ansi = re.compile(r"\x1b\[[0-9;]*m")
    for line in stdout.splitlines():
        clean = ansi.sub("", line).strip()
        if not clean.startswith("|") or "|" not in clean[1:]:
            continue
        cells = [cell.strip() for cell in clean.strip("|").split("|")]
        if len(cells) < 2:
            continue
        metric, value = cells[0], cells[1]
        if metric in EVAL_METRICS:
            metrics[metric] = value
    return metrics


def run_evaluation(selection: SelectedCheckpoint, args: argparse.Namespace, log_dir: Path) -> Dict[str, Any]:
    spec = ALGORITHM_SPECS[selection.algorithm]
    command = [
        sys.executable,
        "-m",
        "mate.evaluate",
        "--no-render",
        "--episodes",
        str(args.episodes),
        "--seed",
        str(args.seed),
        "--config",
        config_file(selection.config),
        "--camera-agent",
        spec["agent"],
        "--camera-kwargs",
        json.dumps({"checkpoint_path": safe_resolve(selection.checkpoint_file).as_posix()}),
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO) + os.pathsep + env.get("PYTHONPATH", "")
    name = f"{slugify(selection.config)}__{slugify(selection.algorithm)}"
    stdout_path = log_dir / f"{name}.stdout.log"
    stderr_path = log_dir / f"{name}.stderr.log"
    completed = subprocess.run(
        command,
        cwd=str(REPO),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    stdout_path.write_text(completed.stdout)
    stderr_path.write_text(completed.stderr)
    metrics = parse_eval_table(completed.stdout)
    return {
        "status": "ok" if completed.returncode == 0 and metrics else "failed",
        "returncode": completed.returncode,
        "command": command,
        "metrics": metrics,
        "stdout_log": stdout_path.as_posix(),
        "stderr_log": stderr_path.as_posix(),
        "error": "" if completed.returncode == 0 else completed.stderr.strip().splitlines()[-1:] or ["unknown error"],
    }


def base_record(selection: SelectedCheckpoint, plot_path: Optional[Path]) -> Dict[str, Any]:
    spec = ALGORITHM_SPECS[selection.algorithm]
    return {
        "config": selection.config,
        "algorithm": spec["label"],
        "algorithm_slug": selection.algorithm,
        "run_dir": safe_resolve(selection.run_dir).as_posix(),
        "checkpoint_dir": safe_resolve(selection.checkpoint_dir).as_posix(),
        "checkpoint_file": safe_resolve(selection.checkpoint_file).as_posix(),
        "checkpoint_iteration": selection.checkpoint_iteration,
        "training_metric": selection.metric_value,
        "global_best_iteration": selection.global_best_iteration,
        "global_best_value": selection.global_best_value,
        "plot": plot_path.as_posix() if plot_path is not None else "",
    }


def write_selected_csv(path: Path, records: List[Dict[str, Any]]) -> None:
    fields = [
        "config",
        "algorithm",
        "checkpoint_iteration",
        "training_metric",
        "global_best_iteration",
        "global_best_value",
        "run_dir",
        "checkpoint_file",
        "plot",
        "status",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow({field: record.get(field, "") for field in fields})


def write_jsonl(path: Path, records: List[Dict[str, Any]]) -> None:
    with path.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def markdown_value(record: Dict[str, Any], key: str) -> str:
    metrics = record.get("metrics") or {}
    value = metrics.get(key, "")
    return str(value) if value != "" else "-"


def relative_link(path: str, output_dir: Path) -> str:
    if not path:
        return "-"
    path_obj = Path(path)
    try:
        return path_obj.relative_to(output_dir).as_posix()
    except ValueError:
        return path_obj.as_posix()


def write_markdown(path: Path, records: List[Dict[str, Any]], args: argparse.Namespace) -> None:
    output_dir = path.parent
    by_config = defaultdict(list)  # type: Dict[str, List[Dict[str, Any]]]
    for record in records:
        by_config[record["config"]].append(record)

    lines = [
        "# Best Checkpoint Evaluation",
        "",
        f"- Episodes: `{args.episodes}`",
        f"- Seed: `{args.seed}`",
        f"- Training metric: `{args.metric}`",
        f"- Specific run: `{args.specific_run}`",
        "",
    ]
    for config in sorted(by_config):
        lines.extend([
            f"## {config}",
            "",
            "| Algorithm | Status | Checkpoint | Train coverage | Step / Cargo | Target Episode Reward | Mean Transport Rate | Mean Coverage Rate | Normalized Target Episode Reward | Plot | Run |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
        ])
        for record in sorted(by_config[config], key=lambda item: item["algorithm"]):
            plot = relative_link(record.get("plot", ""), output_dir)
            plot_cell = f"[plot]({plot})" if plot != "-" else "-"
            run_cell = f"`{record['run_dir']}`"
            lines.append(
                "| {algorithm} | {status} | {checkpoint_iteration} | {training_metric:.6f} | {step_cargo} | {target_reward} | {transport} | {coverage} | {normalized} | {plot} | {run} |".format(
                    algorithm=record["algorithm"],
                    status=record.get("status", "pending"),
                    checkpoint_iteration=record["checkpoint_iteration"],
                    training_metric=record["training_metric"],
                    step_cargo=markdown_value(record, "Step / Cargo"),
                    target_reward=markdown_value(record, "Target Episode Reward"),
                    transport=markdown_value(record, "Mean Transport Rate"),
                    coverage=markdown_value(record, "Mean Coverage Rate"),
                    normalized=markdown_value(record, "Normalized Target Episode Reward"),
                    plot=plot_cell,
                    run=run_cell,
                )
            )
        lines.append("")

    failures = [record for record in records if record.get("status") == "failed"]
    if failures:
        lines.extend(["## Failures", ""])
        for record in failures:
            lines.append(f"- `{record['config']}` / `{record['algorithm']}`: `{record.get('error')}`")
        lines.append("")

    path.write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    paths = ensure_output_dirs(output_dir)
    selected = select_best_checkpoints(args)

    records = []  # type: List[Dict[str, Any]]
    for key in sorted(selected):
        selection = selected[key]
        plot_path = plot_training_curve(selection, paths["plots"])
        record = base_record(selection, plot_path)
        if args.dry_run:
            record.update({"status": "dry-run", "metrics": {}, "returncode": None})
        else:
            print(f"Evaluating {record['algorithm']} on {record['config']} at checkpoint {record['checkpoint_iteration']}")
            record.update(run_evaluation(selection, args, paths["logs"]))
        records.append(record)

    write_selected_csv(output_dir / "selected_checkpoints.csv", records)
    write_jsonl(output_dir / "evaluation_results.jsonl", records)
    write_markdown(output_dir / "evaluation_results.md", records, args)

    print(f"Selected checkpoints: {len(records)}")
    print(output_dir / "evaluation_results.md")
    print(output_dir / "evaluation_results.jsonl")
    print(output_dir / "selected_checkpoints.csv")


if __name__ == "__main__":
    main()
