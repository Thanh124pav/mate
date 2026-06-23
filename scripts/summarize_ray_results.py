#!/bin/sh
"exec" "python3" "$0" "$@"
"exit" "$?"
"""Summarize Ray Tune results by environment and algorithm."""

import argparse
import csv
import glob
import json
import math
import re
import shutil
import tarfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Dict, Iterable, List, Optional, Pattern, Set, Tuple

REPO = Path(__file__).resolve().parent.parent
HOME_RAY_RESULTS = Path("/home/pavt1024/ray_results")
DEFAULT_SCAN_ROOTS = [REPO, HOME_RAY_RESULTS]
MIN_ROWS = 20

MEAN_COLUMNS = [
    "real_coverage_rate_mean",
    "custom_metrics/real_coverage_rate_mean",
    "sampler_results/custom_metrics/real_coverage_rate_mean",
]
MIN_COLUMNS = [
    "real_coverage_rate_min",
    "custom_metrics/real_coverage_rate_min",
    "sampler_results/custom_metrics/real_coverage_rate_min",
]
MAX_COLUMNS = [
    "real_coverage_rate_max",
    "custom_metrics/real_coverage_rate_max",
    "sampler_results/custom_metrics/real_coverage_rate_max",
]


@dataclass
class Run:
    path: Path
    algorithm: str
    env: str
    rows: int
    score: Optional[float]
    metric_mean_last: Optional[float]
    metric_min_last: Optional[float]
    metric_max_last: Optional[float]
    series: List[float]
    min_series: List[float]
    max_series: List[float]
    env_config: Dict[str, Any]
    invalid_reason: Optional[str] = None

    @property
    def valid(self) -> bool:
        return self.invalid_reason is None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate Ray Tune progress.csv results by algorithm and env_config.",
    )
    parser.add_argument(
        "--exps",
        nargs="*",
        default=None,
        help=(
            "Experiment roots, glob specs, or Python regex specs. "
            "Default: /home/pavt1024/ray_results and examples/**/ray_results."
        ),
    )
    parser.add_argument(
        "--k",
        type=int,
        required=True,
        help="Number of final finite metric steps used to score each run.",
    )
    parser.add_argument(
        "--name",
        required=True,
        help="Output name. Files are written to outputs/<name>/.",
    )
    parser.add_argument(
        "--zip",
        action="store_true",
        default=None,
        help="Copy selected best experiment folders into outputs/<name>/details and archive them as <name>_details.tar.gz.",
    )
    return parser.parse_args()


def has_glob_chars(spec: str) -> bool:
    return any(ch in spec for ch in "*?[")


def safe_resolve(path: Path) -> Path:
    try:
        return path.resolve()
    except OSError:
        return path.absolute()


def contains_path(parent: Path, child: Path) -> bool:
    try:
        child.relative_to(parent)
        return True
    except ValueError:
        return False


def default_roots() -> List[Path]:
    roots = [HOME_RAY_RESULTS]
    examples = REPO / "examples"
    if examples.exists():
        roots.extend(sorted(examples.glob("**/ray_results")))
    return roots


def run_dirs_under(root: Path) -> List[Path]:
    if not root.exists():
        return []
    if root.is_file():
        return [root.parent] if root.name == "progress.csv" else []
    if (root / "progress.csv").exists():
        return [root]
    return sorted({progress.parent for progress in root.rglob("progress.csv")})


def all_candidate_run_dirs() -> List[Path]:
    candidates = []  # type: List[Path]
    for root in DEFAULT_SCAN_ROOTS:
        candidates.extend(run_dirs_under(root))
    return dedupe_paths(candidates)


def dedupe_paths(paths: Iterable[Path]) -> List[Path]:
    result = []  # type: List[Path]
    seen = set()  # type: Set[Path]
    for path in paths:
        resolved = safe_resolve(path)
        if resolved in seen:
            continue
        seen.add(resolved)
        result.append(path)
    return sorted(result, key=lambda p: safe_resolve(p).as_posix())


def repo_relative(path: Path) -> Optional[str]:
    resolved = safe_resolve(path)
    try:
        return resolved.relative_to(REPO).as_posix()
    except ValueError:
        return None


def regex_matches_run(pattern: Pattern[str], run_dir: Path) -> bool:
    resolved = safe_resolve(run_dir)
    candidates = [resolved.as_posix(), run_dir.as_posix()]
    rel = repo_relative(run_dir)
    if rel is not None:
        candidates.append(rel)
    else:
        try:
            candidates.append(resolved.relative_to(Path("/home/pavt1024")).as_posix())
        except ValueError:
            pass
    return any(pattern.search(candidate) for candidate in candidates)


def expand_glob_spec(spec: str) -> List[Path]:
    specs = [spec]
    path = Path(spec).expanduser()
    if not path.is_absolute():
        specs.append((REPO / path).as_posix())

    matches = []  # type: List[Path]
    for item in specs:
        matches.extend(Path(match) for match in glob.glob(item, recursive=True))
    return dedupe_paths(matches)


def resolve_exp_specs(specs: Optional[List[str]]) -> List[Path]:
    if not specs:
        paths = []  # type: List[Path]
        for root in default_roots():
            paths.extend(run_dirs_under(root))
        return dedupe_paths(paths)

    run_dirs = []  # type: List[Path]
    regex_specs = []  # type: List[str]
    for spec in specs:
        expanded = Path(spec).expanduser()
        direct = expanded if expanded.is_absolute() else REPO / expanded
        if direct.exists():
            run_dirs.extend(run_dirs_under(direct))
            continue

        glob_matches = expand_glob_spec(spec) if has_glob_chars(spec) else []
        if glob_matches:
            for match in glob_matches:
                run_dirs.extend(run_dirs_under(match))
            continue

        regex_specs.append(spec)

    if regex_specs:
        candidates = all_candidate_run_dirs()
        for spec in regex_specs:
            try:
                pattern = re.compile(spec)
            except re.error as exc:
                raise SystemExit(f"Invalid --exps regex {spec!r}: {exc}") from exc
            run_dirs.extend(run for run in candidates if regex_matches_run(pattern, run))

    return dedupe_paths(run_dirs)


def normalize_algorithm(value: str) -> str:
    if value.endswith("Trainer"):
        value = value[: -len("Trainer")]
    value = value.replace("-", "_")
    value = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", value)
    value = value.replace("__", "_")
    return value.strip("_").lower()


def algorithm_from_path(run_dir: Path) -> str:
    name = run_dir.name
    prefix = re.match(r"(.+?)_mate-", name)
    if prefix:
        return normalize_algorithm(prefix.group(1))

    text = safe_resolve(run_dir).as_posix()
    match = re.search(r"/examples/hitmac/coordinator/([^/]+)/camera/ray_results/", text)
    if match:
        return normalize_algorithm(f"hitmac_{match.group(1)}")

    match = re.search(r"/examples/(?:hrl/|hrl_v2/)?([^/]+)/camera/ray_results/", text)
    if match:
        return normalize_algorithm(match.group(1))

    for marker in ("mate-hrl.", "mate-"):
        if marker in name:
            tail = name.split(marker, 1)[1].split(".camera", 1)[0]
            return normalize_algorithm(tail)

    return normalize_algorithm(name.split("_", 1)[0])


def normalize_env_name(raw: Any) -> str:
    if raw is None:
        return "unknown"
    name = Path(str(raw)).name
    if name.endswith(".yaml"):
        name = name[:-5]
    if name.startswith("MATE-"):
        name = name[len("MATE-") :]
    return name or "unknown"


def choose_column(fieldnames: Optional[List[str]], candidates: List[str]) -> Optional[str]:
    if not fieldnames:
        return None
    return next((column for column in candidates if column in fieldnames), None)


def parse_float(value: Any) -> float:
    if value in (None, "", "nan", "NaN", "None"):
        return math.nan
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return math.nan
    return parsed if math.isfinite(parsed) else math.nan


def finite_tail(values: List[float], k: int) -> List[float]:
    finite = [value for value in values if math.isfinite(value)]
    return finite[-k:]


def read_progress(progress_path: Path, k: int) -> Tuple[
    int,
    List[float],
    List[float],
    List[float],
    Optional[float],
    Optional[float],
    Optional[float],
    Optional[str],
]:
    try:
        with progress_path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            mean_column = choose_column(reader.fieldnames, MEAN_COLUMNS)
            min_column = choose_column(reader.fieldnames, MIN_COLUMNS)
            max_column = choose_column(reader.fieldnames, MAX_COLUMNS)
            if mean_column is None:
                rows = sum(1 for _ in reader)
                return rows, [], [], [], None, None, None, "missing_metric"

            rows = 0
            mean_series = []  # type: List[float]
            min_series = []  # type: List[float]
            max_series = []  # type: List[float]
            for row in reader:
                rows += 1
                mean_series.append(parse_float(row.get(mean_column)))
                if min_column is not None:
                    min_series.append(parse_float(row.get(min_column)))
                if max_column is not None:
                    max_series.append(parse_float(row.get(max_column)))
    except (OSError, csv.Error) as exc:
        return 0, [], [], [], None, None, None, f"read_error:{exc}"

    if rows < MIN_ROWS:
        return rows, mean_series, min_series, max_series, None, None, None, "too_short"

    score_tail = finite_tail(mean_series, k)
    if not score_tail:
        return rows, mean_series, min_series, max_series, None, None, None, "no_finite_metric"

    min_tail = finite_tail(min_series, k)
    max_tail = finite_tail(max_series, k)
    return (
        rows,
        mean_series,
        min_series,
        max_series,
        mean(score_tail),
        mean(min_tail) if min_tail else None,
        mean(max_tail) if max_tail else None,
        None,
    )


def read_run(run_dir: Path, k: int) -> Run:
    params_path = run_dir / "params.json"
    progress_path = run_dir / "progress.csv"
    env_config = {}  # type: Dict[str, Any]
    invalid_reason = None  # type: Optional[str]

    if not params_path.exists():
        invalid_reason = "missing_params"
    else:
        try:
            params = json.loads(params_path.read_text())
            loaded_env_config = params.get("env_config") or {}
            if isinstance(loaded_env_config, dict):
                env_config = loaded_env_config
        except (OSError, json.JSONDecodeError) as exc:
            invalid_reason = f"bad_params:{exc}"

    if not progress_path.exists():
        invalid_reason = invalid_reason or "missing_progress"
        rows, series, min_series, max_series, score, metric_min, metric_max = 0, [], [], [], None, None, None
    else:
        rows, series, min_series, max_series, score, metric_min, metric_max, progress_reason = read_progress(progress_path, k)
        invalid_reason = invalid_reason or progress_reason

    return Run(
        path=run_dir,
        algorithm=algorithm_from_path(run_dir),
        env=normalize_env_name(env_config.get("config")),
        rows=rows,
        score=score,
        metric_mean_last=score,
        metric_min_last=metric_min,
        metric_max_last=metric_max,
        series=series,
        min_series=min_series,
        max_series=max_series,
        env_config=env_config,
        invalid_reason=invalid_reason,
    )


def group_valid_runs(runs: List[Run]) -> Dict[Tuple[str, str], List[Run]]:
    grouped = defaultdict(list)  # type: Dict[Tuple[str, str], List[Run]]
    for run in runs:
        if run.valid:
            grouped[(run.env, run.algorithm)].append(run)
    return grouped


def select_best_runs(grouped: Dict[Tuple[str, str], List[Run]]) -> Dict[str, Dict[str, Run]]:
    best = defaultdict(dict)  # type: Dict[str, Dict[str, Run]]
    for (env, algorithm), runs in grouped.items():
        best_run = max(runs, key=lambda run: run.score if run.score is not None else -math.inf)
        best[env][algorithm] = best_run
    return {env: dict(sorted(alg_runs.items())) for env, alg_runs in sorted(best.items())}


def score_stats(runs: List[Run]) -> Tuple[float, float, int]:
    scores = [run.score for run in runs if run.score is not None and math.isfinite(run.score)]
    if not scores:
        return math.nan, math.nan, 0
    return mean(scores), pstdev(scores) if len(scores) > 1 else 0.0, len(scores)


def rolling_mean_std(values: List[float], window: int) -> Tuple[List[float], List[float]]:
    means = []  # type: List[float]
    stds = []  # type: List[float]
    for index in range(len(values)):
        start = max(0, index - window + 1)
        finite = [value for value in values[start : index + 1] if math.isfinite(value)]
        if not finite:
            means.append(math.nan)
            stds.append(math.nan)
            continue
        means.append(mean(finite))
        stds.append(pstdev(finite) if len(finite) > 1 else 0.0)
    return means, stds


def slugify(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return slug.strip("_") or "unknown"


def plot_env(env: str, best_runs: Dict[str, Run], grouped: Dict[Tuple[str, str], List[Run]], output_dir: Path, name: str) -> Path:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "matplotlib is required to write PNG plots. Install project requirements first "
            "(requirements.txt pins matplotlib==3.9.2)."
        ) from exc

    fig, ax = plt.subplots(figsize=(12, 6.5))
    for algorithm, run in best_runs.items():
        xs = list(range(1, len(run.series) + 1))
        mean_score, std_score, count = score_stats(grouped[(env, algorithm)])
        label = f"{algorithm} ({run.score:.3f}, std {std_score:.3f}, n={count})"
        window = max(3, min(15, max(1, len(run.series) // 12)))
        smoothed, rolling_std = rolling_mean_std(run.series, window)
        lower = [
            smooth - std if math.isfinite(smooth) and math.isfinite(std) else math.nan
            for smooth, std in zip(smoothed, rolling_std)
        ]
        upper = [
            smooth + std if math.isfinite(smooth) and math.isfinite(std) else math.nan
            for smooth, std in zip(smoothed, rolling_std)
        ]
        line = ax.plot(xs, smoothed, linewidth=2.2, label=label)[0]
        ax.fill_between(xs, lower, upper, color=line.get_color(), alpha=0.12, linewidth=0)

    ax.set_title(f"{env}: best run per algorithm", fontsize=15, loc="left", pad=14, fontweight="bold")
    ax.set_xlabel("training iteration")
    ax.set_ylabel("real_coverage_rate_mean (rolling mean +/- rolling std)")
    ax.grid(alpha=0.25)
    ax.legend(loc="best", fontsize=8, frameon=True)
    finite_values = [
        value
        for run in best_runs.values()
        for value in run.series
        if math.isfinite(value)
    ]
    if finite_values:
        low = min(0.0, min(finite_values) * 0.95)
        high = min(1.05, max(finite_values) * 1.08) if max(finite_values) <= 1 else max(finite_values) * 1.08
        if high > low:
            ax.set_ylim(low, high)
    fig.tight_layout()
    path = output_dir / f"{name}__{slugify(env)}.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def clean_details_dir(details_dir: Path) -> None:
    resolved = safe_resolve(details_dir)
    output_root = safe_resolve(REPO / "outputs")
    if details_dir.name != "details" or not contains_path(output_root, resolved):
        raise SystemExit(f"Refusing to clean unexpected details directory: {details_dir}")
    if details_dir.exists():
        shutil.rmtree(str(details_dir))
    details_dir.mkdir(parents=True, exist_ok=True)


def copy_selected_details(best: Dict[str, Dict[str, Run]], output_dir: Path, name: str) -> Path:
    details_dir = output_dir / "details"
    clean_details_dir(details_dir)

    copied = 0
    for env, alg_runs in best.items():
        for algorithm, run in alg_runs.items():
            target = details_dir / slugify(env) / slugify(algorithm) / slugify(run.path.name)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(str(run.path), str(target))
            copied += 1

    archive_path = output_dir / f"{name}_details.tar.gz"
    if archive_path.exists():
        archive_path.unlink()
    with tarfile.open(str(archive_path), "w:gz") as archive:
        archive.add(str(details_dir), arcname="details")
    print(f"\nDetails copied: {copied}")
    print(f"Details archive: {archive_path}")
    return archive_path


def record_for_run(env: str, algorithm: str, run: Run, grouped: Dict[Tuple[str, str], List[Run]]) -> Dict[str, Any]:
    group_mean, group_std, group_count = score_stats(grouped[(env, algorithm)])
    return {
        "env_config": env,
        "algorithm": algorithm,
        "score": run.score,
        "metric_mean_last_k": run.metric_mean_last,
        "metric_min_last_k": run.metric_min_last,
        "metric_max_last_k": run.metric_max_last,
        "algorithm_score_mean": group_mean,
        "algorithm_score_std": group_std,
        "algorithm_valid_runs": group_count,
        "rows": run.rows,
        "path": safe_resolve(run.path).as_posix(),
        "env_config_full": run.env_config,
    }


def csv_value(value: float) -> Optional[float]:
    return value if math.isfinite(value) else None


def write_summary_csv(path: Path, records: List[Dict[str, Any]]) -> None:
    fieldnames = [
        "env_config",
        "algorithm",
        "score",
        "metric_mean_last_k",
        "metric_min_last_k",
        "metric_max_last_k",
        "algorithm_score_mean",
        "algorithm_score_std",
        "algorithm_valid_runs",
        "rows",
        "path",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow({key: record.get(key) for key in fieldnames})


def write_lite_progress_csv(path: Path, best: Dict[str, Dict[str, Run]], grouped: Dict[Tuple[str, str], List[Run]]) -> None:
    fieldnames = [
        "env_config",
        "algorithm",
        "step",
        "real_coverage_rate_mean",
        "real_coverage_rate_min",
        "real_coverage_rate_max",
        "selected_score",
        "algorithm_score_mean",
        "algorithm_score_std",
        "algorithm_valid_runs",
        "rows",
        "path",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for env, alg_runs in best.items():
            for algorithm, run in alg_runs.items():
                group_mean, group_std, group_count = score_stats(grouped[(env, algorithm)])
                max_len = max(len(run.series), len(run.min_series), len(run.max_series))
                for index in range(max_len):
                    mean_value = run.series[index] if index < len(run.series) else math.nan
                    min_value = run.min_series[index] if index < len(run.min_series) else math.nan
                    max_value = run.max_series[index] if index < len(run.max_series) else math.nan
                    writer.writerow(
                        {
                            "env_config": env,
                            "algorithm": algorithm,
                            "step": index + 1,
                            "real_coverage_rate_mean": csv_value(mean_value),
                            "real_coverage_rate_min": csv_value(min_value),
                            "real_coverage_rate_max": csv_value(max_value),
                            "selected_score": run.score,
                            "algorithm_score_mean": group_mean,
                            "algorithm_score_std": group_std,
                            "algorithm_valid_runs": group_count,
                            "rows": run.rows,
                            "path": safe_resolve(run.path).as_posix(),
                        }
                    )


def write_jsonl(path: Path, records: List[Dict[str, Any]]) -> None:
    with path.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def progress_points_for_run(run: Run) -> List[Dict[str, Any]]:
    points = []  # type: List[Dict[str, Any]]
    max_len = max(len(run.series), len(run.min_series), len(run.max_series))
    for index in range(max_len):
        mean_value = run.series[index] if index < len(run.series) else math.nan
        min_value = run.min_series[index] if index < len(run.min_series) else math.nan
        max_value = run.max_series[index] if index < len(run.max_series) else math.nan
        points.append(
            {
                "step": index + 1,
                "real_coverage_rate_mean": csv_value(mean_value),
                "real_coverage_rate_min": csv_value(min_value),
                "real_coverage_rate_max": csv_value(max_value),
            }
        )
    return points


def lite_progress_record(env: str, algorithm: str, run: Run, grouped: Dict[Tuple[str, str], List[Run]]) -> Dict[str, Any]:
    record = record_for_run(env, algorithm, run, grouped)
    record["progress"] = progress_points_for_run(run)
    return record


def write_lite_progress_jsonl(path: Path, best: Dict[str, Dict[str, Run]], grouped: Dict[Tuple[str, str], List[Run]]) -> None:
    with path.open("w") as handle:
        for env, alg_runs in best.items():
            for algorithm, run in alg_runs.items():
                handle.write(json.dumps(lite_progress_record(env, algorithm, run, grouped), sort_keys=True) + "\n")


def print_summary(runs: List[Run], best: Dict[str, Dict[str, Run]], grouped: Dict[Tuple[str, str], List[Run]], plot_paths: List[Path]) -> None:
    reasons = Counter(run.invalid_reason for run in runs if not run.valid)
    valid_count = sum(1 for run in runs if run.valid)
    algorithms = sorted({run.algorithm for run in runs if run.valid})

    print("\nRay results summary")
    print("===================")
    print(f"Runs scanned: {len(runs)}")
    print(f"Valid runs:   {valid_count}")
    print(f"Invalid runs: {len(runs) - valid_count}")
    for reason, count in sorted(reasons.items()):
        print(f"  - {reason}: {count}")
    print(f"Env configs:  {len(best)}")
    print(f"Algorithms:   {len(algorithms)}")

    print("\nPer-env best/worst")
    print("------------------")
    for env, alg_runs in best.items():
        selected = [(algorithm, run) for algorithm, run in alg_runs.items() if run.score is not None]
        scores = [run.score for _, run in selected if run.score is not None]
        if not selected:
            continue
        best_alg, best_run = max(selected, key=lambda item: item[1].score or -math.inf)
        worst_alg, worst_run = min(selected, key=lambda item: item[1].score or math.inf)
        print(
            f"{env}: algorithms={len(selected)}, valid_runs={sum(len(grouped[(env, alg)]) for alg, _ in selected)}, "
            f"mean={mean(scores):.4f}, best={best_alg}:{best_run.score:.4f}, "
            f"worst={worst_alg}:{worst_run.score:.4f}"
        )

    print("\nPlots")
    print("-----")
    for path in plot_paths:
        print(path)


def main() -> None:
    args = parse_args()
    if args.k <= 0:
        raise SystemExit("--k must be a positive integer")

    output_dir = REPO / "outputs" / args.name
    output_dir.mkdir(parents=True, exist_ok=True)

    run_dirs = resolve_exp_specs(args.exps)
    runs = [read_run(run_dir, args.k) for run_dir in run_dirs]
    grouped = group_valid_runs(runs)
    best = select_best_runs(grouped)

    records = []  # type: List[Dict[str, Any]]
    for env, alg_runs in best.items():
        for algorithm, run in alg_runs.items():
            records.append(record_for_run(env, algorithm, run, grouped))

    write_lite_progress_csv(output_dir / f"{args.name}.csv", best, grouped)
    write_summary_csv(output_dir / f"{args.name}_summary.csv", records)
    write_lite_progress_jsonl(output_dir / f"{args.name}.jsonl", best, grouped)
    write_jsonl(output_dir / f"{args.name}_summary.jsonl", records)

    archive_path = None
    if args.zip:
        archive_path = copy_selected_details(best, output_dir, args.name)

    plot_paths = []  # type: List[Path]
    for env, alg_runs in best.items():
        plot_paths.append(plot_env(env, alg_runs, grouped, output_dir, args.name))

    print_summary(runs, best, grouped, plot_paths)

    print("\nOutputs")
    print("-------")
    print(output_dir / f"{args.name}.csv")
    print(output_dir / f"{args.name}_summary.csv")
    print(output_dir / f"{args.name}.jsonl")
    print(output_dir / f"{args.name}_summary.jsonl")
    if archive_path is not None:
        print(archive_path)


if __name__ == "__main__":
    main()
