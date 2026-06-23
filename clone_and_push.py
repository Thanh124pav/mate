#!/usr/bin/env python3
"""Replay local Ray Tune results into W&B projects.

The script scans experiment_results/ and every */ray_results/ directory for
trial folders containing params.json + non-empty progress.csv. Each trial folder
is pushed as one W&B run, with the W&B run name equal to the trial folder name.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

os.environ.setdefault("WANDB_SILENT", "true")

import wandb


ROOT = Path(__file__).resolve().parent
ENV_TO_PROJECT = {
    "MATE-4v8-9.yaml": "4v8-9",
    "MATE-4v5-0.yaml": "4v5-0",
}
DEFAULT_STATE_FILE = ROOT / ".wandb_clone_push_state.json"
EXCLUDED_PROGRESS_KEYS = {
    "date",
    "done",
    "experiment_id",
    "hostname",
    "node_ip",
    "pid",
    "trial_id",
}


@dataclass(frozen=True)
class Trial:
    params_path: Path
    progress_path: Path
    run_dir: Path
    run_name: str
    project: str
    env_config: str
    group: str | None
    rel_path: str


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'\"")
        os.environ.setdefault(key, value)


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def progress_has_rows(path: Path) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        try:
            next(reader)
        except StopIteration:
            return False
        return any(row for row in reader)


def iter_params_files(root: Path) -> Iterable[Path]:
    experiment_root = root / "experiment_results"
    if experiment_root.exists():
        yield from experiment_root.rglob("params.json")

    for ray_results in root.rglob("ray_results"):
        if ".git" in ray_results.parts or "wandb" in ray_results.parts:
            continue
        yield from ray_results.rglob("params.json")


def infer_group(run_dir: Path) -> str | None:
    parent_name = run_dir.parent.name
    if parent_name and parent_name != ".":
        return parent_name
    return None


def collect_trials(root: Path) -> list[Trial]:
    trials: list[Trial] = []
    seen: set[Path] = set()

    for params_path in iter_params_files(root):
        params_path = params_path.resolve()
        if params_path in seen:
            continue
        seen.add(params_path)

        run_dir = params_path.parent
        progress_path = run_dir / "progress.csv"
        if not progress_has_rows(progress_path):
            continue

        try:
            params = read_json(params_path)
        except json.JSONDecodeError as exc:
            print(f"skip invalid params.json: {params_path}: {exc}", file=sys.stderr)
            continue

        env_config = params.get("env_config", {}).get("config")
        project = ENV_TO_PROJECT.get(env_config)
        if project is None:
            print(
                f"skip unknown env_config.config={env_config!r}: {params_path}",
                file=sys.stderr,
            )
            continue

        rel_path = str(run_dir.relative_to(root))
        trials.append(
            Trial(
                params_path=params_path,
                progress_path=progress_path,
                run_dir=run_dir,
                run_name=run_dir.name,
                project=project,
                env_config=env_config,
                group=infer_group(run_dir),
                rel_path=rel_path,
            )
        )

    return sorted(trials, key=lambda trial: (trial.project, trial.rel_path))


def load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"completed": {}}
    with path.open("r", encoding="utf-8") as f:
        state = json.load(f)
    state.setdefault("completed", {})
    return state


def save_state(path: Path, state: dict[str, Any]) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, sort_keys=True)
        f.write("\n")
    tmp_path.replace(path)


def stable_run_id(trial: Trial) -> str:
    digest = hashlib.sha1(trial.rel_path.encode("utf-8")).hexdigest()[:12]
    return f"clone-{digest}"


def coerce_value(value: str) -> Any | None:
    if value == "":
        return None
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    try:
        number = float(value)
    except ValueError:
        return None
    if not math.isfinite(number):
        return None
    return int(number) if number.is_integer() else number


def row_to_metrics(row: dict[str, str]) -> tuple[dict[str, Any], int | None]:
    metrics: dict[str, Any] = {}
    step = None

    for key, raw_value in row.items():
        if key in EXCLUDED_PROGRESS_KEYS:
            continue
        value = coerce_value(raw_value)
        if value is None:
            continue
        metrics[key] = value

    if "training_iteration" in metrics and isinstance(metrics["training_iteration"], int):
        step = metrics["training_iteration"]
    elif "timesteps_total" in metrics and isinstance(metrics["timesteps_total"], int):
        step = metrics["timesteps_total"]

    return metrics, step


def count_progress_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8", newline="") as f:
        return sum(1 for _ in csv.DictReader(f))


def ensure_projects(projects: Iterable[str], entity: str) -> None:
    api = wandb.Api()
    for project in sorted(set(projects)):
        try:
            api.project(project, entity=entity)
        except Exception:
            print(f"creating W&B project {entity}/{project}")
            api.create_project(project, entity)


def push_trial(trial: Trial, entity: str, upload_files: bool) -> int:
    params = read_json(trial.params_path)
    tags = ["cloned-local-ray-results", trial.env_config]
    if "experiment_results" in trial.run_dir.parts:
        tags.append("experiment_results")
    if "ray_results" in trial.run_dir.parts:
        tags.append("ray_results")

    run = wandb.init(
        entity=entity,
        project=trial.project,
        id=stable_run_id(trial),
        name=trial.run_name,
        group=trial.group,
        config=params,
        tags=tags,
        notes=f"Cloned from local Ray Tune folder: {trial.rel_path}",
        resume="allow",
        reinit="finish_previous",
    )

    logged = 0
    with trial.progress_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            metrics, step = row_to_metrics(row)
            if not metrics:
                continue
            run.log(metrics, step=step)
            logged += 1

    run.summary["source_path"] = trial.rel_path
    run.summary["source_params_json"] = str(trial.params_path.relative_to(ROOT))
    run.summary["source_progress_csv"] = str(trial.progress_path.relative_to(ROOT))
    run.summary["cloned_rows"] = logged

    if upload_files:
        run.save(str(trial.params_path), policy="now")
        run.save(str(trial.progress_path), policy="now")

    run.finish()
    return logged


def resolve_entity(explicit_entity: str | None) -> str:
    if explicit_entity:
        return explicit_entity
    api = wandb.Api()
    viewer = api.viewer
    entity = getattr(viewer, "entity", None) or getattr(viewer, "username", None)
    if not entity:
        raise RuntimeError("Cannot infer W&B entity. Pass --entity explicitly.")
    return entity


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Clone local Ray Tune progress.csv results into W&B projects."
    )
    parser.add_argument("--entity", default=None, help="W&B entity/team. Default: API viewer entity.")
    parser.add_argument("--state-file", type=Path, default=DEFAULT_STATE_FILE)
    parser.add_argument("--limit", type=int, default=None, help="Push at most N pending runs.")
    parser.add_argument("--include-completed", action="store_true", help="Re-push runs already in state.")
    parser.add_argument("--no-upload-files", action="store_true", help="Do not attach params/progress files.")
    parser.add_argument("--execute", action="store_true", help="Actually push to W&B. Without this, dry-run only.")
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")
    trials = collect_trials(ROOT)
    state = load_state(args.state_file)
    completed = state["completed"]

    pending = [
        trial
        for trial in trials
        if args.include_completed or stable_run_id(trial) not in completed
    ]
    if args.limit is not None:
        pending = pending[: args.limit]

    print(f"found trials with non-empty progress.csv: {len(trials)}")
    print(f"pending trials selected: {len(pending)}")
    project_counts: dict[str, int] = {}
    for trial in pending:
        project_counts[trial.project] = project_counts.get(trial.project, 0) + 1
    for project, count in sorted(project_counts.items()):
        print(f"  {project}: {count}")

    if not args.execute:
        print("\ndry-run only. Re-run with --execute to push.")
        for trial in pending[:10]:
            print(f"  {trial.project}: {trial.run_name} ({count_progress_rows(trial.progress_path)} rows)")
        if len(pending) > 10:
            print(f"  ... {len(pending) - 10} more")
        return 0

    entity = resolve_entity(args.entity)
    print(f"pushing to W&B entity: {entity}")
    ensure_projects((trial.project for trial in pending), entity)

    for index, trial in enumerate(pending, start=1):
        run_id = stable_run_id(trial)
        print(f"[{index}/{len(pending)}] {trial.project}/{trial.run_name}")
        logged_rows = push_trial(
            trial,
            entity=entity,
            upload_files=not args.no_upload_files,
        )
        completed[run_id] = {
            "project": trial.project,
            "run_name": trial.run_name,
            "source": trial.rel_path,
            "rows": logged_rows,
        }
        save_state(args.state_file, state)

    print("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
