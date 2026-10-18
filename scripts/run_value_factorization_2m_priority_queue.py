#!/usr/bin/env python3
"""Run value-factorization experiments one at a time in priority order.

The MATE allocation exposes eight CPUs, while environment sampling is the
bottleneck.  Giving one experiment six rollout workers produces completed
results sooner than sharing those CPUs between all ten experiments.
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


MODULES = {
    "facmac": "examples.facmac.camera",
    "facmac_focuss": "examples.facmac_focuss.camera",
    "resq": "examples.resq.camera",
    "resq_focuss": "examples.resq_focuss.camera",
    "riskq": "examples.riskq.camera",
    "riskq_focuss": "examples.riskq_focuss.camera",
    "qfree": "examples.qfree.camera",
    "qfree_focuss": "examples.qfree_focuss.camera",
    "concaveq": "examples.concaveq.camera",
    "concaveq_focuss": "examples.concaveq_focuss.camera",
}


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=True), encoding="utf-8")
    temporary.replace(path)


def latest_metrics(root, iteration_offset=0, timesteps_offset=0):
    paths = list(root.rglob("result.json"))
    if not paths:
        return {}
    path = max(paths, key=lambda item: item.stat().st_mtime)
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line]
    if not lines:
        return {"result_path": str(path)}
    try:
        result = json.loads(lines[-1])
    except json.JSONDecodeError:
        return {"result_path": str(path)}
    training_iteration = result.get("training_iteration")
    timesteps_total = result.get("timesteps_total")
    return {
        "result_path": str(path),
        "training_iteration": (
            training_iteration + iteration_offset
            if training_iteration is not None
            else None
        ),
        "timesteps_total": (
            timesteps_total + timesteps_offset
            if timesteps_total is not None
            else None
        ),
        "episode_reward_mean": result.get("episode_reward_mean"),
        "coverage_rate_mean": result.get("custom_metrics", {}).get(
            "coverage_rate_mean"
        ),
    }


def restore_progress(checkpoint_path):
    """Return the Tune iteration and env steps captured by a checkpoint."""

    try:
        iteration = int(checkpoint_path.name.rsplit("_", 1)[-1])
    except ValueError:
        iteration = 0
    result_path = checkpoint_path.parent / "result.json"
    if not result_path.is_file():
        return iteration, 0
    timesteps = 0
    for line in result_path.read_text(encoding="utf-8").splitlines():
        try:
            result = json.loads(line)
        except json.JSONDecodeError:
            continue
        if result.get("training_iteration") == iteration:
            timesteps = int(result.get("timesteps_total") or 0)
            break
    return iteration, timesteps


def restore_payload(checkpoint_path):
    """Resolve a Tune checkpoint directory to its trainer payload file."""

    payloads = [
        path
        for path in checkpoint_path.glob("checkpoint-*")
        if not path.name.endswith(".tune_metadata") and path.is_file()
    ]
    if len(payloads) != 1:
        raise ValueError(
            f"expected one checkpoint payload in {checkpoint_path}, found {payloads}"
        )
    return payloads[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--api-key-file", type=Path, required=True)
    parser.add_argument("--timesteps-total", type=int, default=2_000_000)
    parser.add_argument("--report-after-seconds", type=int, default=7 * 60 * 60)
    parser.add_argument("--num-workers", type=int, default=6)
    parser.add_argument("--num-envs-per-worker", type=int, default=2)
    parser.add_argument("--num-gpus", type=float, default=1.0)
    parser.add_argument("--evaluation-interval", type=int, default=10)
    parser.add_argument("--project", default="mate-value-factorization-2m")
    parser.add_argument("--group", default="4v8-9-2m-priority")
    parser.add_argument("--max-retries", type=int, default=1)
    parser.add_argument(
        "--restore",
        action="append",
        default=[],
        metavar="ALGORITHM=CHECKPOINT",
        help="Restore an algorithm from a Tune checkpoint (repeatable).",
    )
    parser.add_argument(
        "--algorithms", nargs="+", choices=tuple(MODULES), default=list(MODULES)
    )
    args = parser.parse_args()
    # Children run from /tmp so imports always come from the controlled
    # PYTHONPATH overlay.  Resolve output paths before spawning them, otherwise
    # a relative --local-dir silently writes trials under /tmp/outputs while
    # the supervisor watches the repository path.
    args.output_dir = args.output_dir.expanduser().resolve()

    restore_paths = {}
    for item in args.restore:
        try:
            name, checkpoint = item.split("=", 1)
        except ValueError as error:
            raise ValueError(
                f"invalid --restore value {item!r}; expected ALGORITHM=CHECKPOINT"
            ) from error
        if name not in MODULES:
            raise ValueError(f"unknown restore algorithm: {name}")
        checkpoint_path = Path(checkpoint).expanduser().resolve()
        if not checkpoint_path.is_dir():
            raise FileNotFoundError(checkpoint_path)
        restore_paths[name] = checkpoint_path

    restore_offsets = {
        name: restore_progress(checkpoint) for name, checkpoint in restore_paths.items()
    }
    restore_payloads = {
        name: restore_payload(checkpoint) for name, checkpoint in restore_paths.items()
    }

    repo = Path(__file__).resolve().parents[1]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    api_key = args.api_key_file.read_text(encoding="utf-8").strip()
    if len(api_key) != 86:
        raise ValueError(f"expected an 86-character W&B token, got {len(api_key)}")

    environment = os.environ.copy()
    environment.update(
        {
            "PYTHONPATH": str(repo / ".runtime_pythonpath"),
            "WANDB_API_KEY": api_key,
            "WANDB_MODE": "online",
            "CUDA_VISIBLE_DEVICES": "0",
            # One CPU for the trainer plus one for each rollout worker.  The
            # allocation only exposes eight CPUs, so advertising an extra idle
            # Ray CPU in each of two concurrent lanes causes oversubscription.
            "SLURM_CPUS_ON_NODE": str(args.num_workers + 1),
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "MATE_RAY_OBJECT_STORE_BYTES": str(4 * 1024**3),
        }
    )
    environment.pop("WANDB_DISABLED", None)
    environment.pop("CUBLAS_WORKSPACE_CONFIG", None)

    started = time.time()
    deadline = started + args.report_after_seconds
    states = {
        name: {"state": "queued", "attempts": 0, "updated_at": utc_now()}
        for name in args.algorithms
    }
    current = None
    stopping = False

    def snapshot(final=False):
        payload = {
            "supervisor_pid": os.getpid(),
            "started_at": datetime.fromtimestamp(started, timezone.utc).isoformat(),
            "updated_at": utc_now(),
            "report_deadline_epoch": deadline,
            "current_algorithm": current[0] if current else None,
            "jobs": states,
        }
        atomic_json(args.output_dir / "status.json", payload)
        if time.time() >= deadline and not (args.output_dir / "seven_hour_summary.json").exists():
            atomic_json(args.output_dir / "seven_hour_summary.json", payload)
        if final:
            atomic_json(args.output_dir / "final_summary.json", payload)

    def stop_current(signum, _frame):
        nonlocal stopping
        stopping = True
        if current and current[1].poll() is None:
            try:
                os.killpg(current[1].pid, signum)
            except ProcessLookupError:
                pass

    signal.signal(signal.SIGTERM, stop_current)
    signal.signal(signal.SIGINT, stop_current)

    for index, name in enumerate(args.algorithms):
        if stopping:
            break
        module = MODULES[name]
        output = args.output_dir / name
        output.mkdir(parents=True, exist_ok=True)
        for attempt in range(args.max_retries + 1):
            if stopping:
                break
            log = (args.output_dir / f"{name}.log").open(
                "a", encoding="utf-8", buffering=1
            )
            iteration_offset, timesteps_offset = restore_offsets.get(name, (0, 0))
            command = [
                sys.executable,
                "-m",
                module,
                "--timesteps-total",
                str(args.timesteps_total),
                "--env",
                "MATE-4v8-9.yaml",
                "--num-workers",
                str(args.num_workers),
                "--num-envs-per-worker",
                str(args.num_envs_per_worker),
                "--num-gpus",
                str(args.num_gpus),
                "--evaluation-interval",
                str(args.evaluation_interval),
                "--buffer-capacity",
                "500",
                "--seed",
                "0",
                "--project",
                args.project,
                "--group",
                args.group,
                "--local-dir",
                str(output),
            ]
            restore_path = restore_paths.get(name)
            if restore_path is not None:
                command.extend(("--restore", str(restore_payloads[name])))
            child_env = environment.copy()
            child_env["RAY_TMPDIR"] = f"/tmp/mate_priority_{index}_{name}_{attempt}"
            log.write(f"START {utc_now()} {' '.join(command)}\n")
            process = subprocess.Popen(
                command,
                cwd="/tmp",
                env=child_env,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            current = (name, process)
            states[name].update(
                state="running",
                attempts=attempt + 1,
                pid=process.pid,
                started_at=utc_now(),
                updated_at=utc_now(),
            )
            if restore_path is not None:
                states[name]["restored_from"] = str(restore_path)
                states[name]["restore_iteration"] = iteration_offset
                states[name]["restore_timesteps"] = timesteps_offset
                states[name]["restore_counters_preserved"] = True
            while process.poll() is None and not stopping:
                states[name].update(latest_metrics(output))
                states[name]["updated_at"] = utc_now()
                snapshot()
                time.sleep(30)
            if stopping and process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                process.wait()
            returncode = process.poll()
            states[name].update(latest_metrics(output))
            states[name].update(returncode=returncode, updated_at=utc_now())
            log.close()
            if returncode == 0:
                states[name]["state"] = "completed"
                break
            states[name]["state"] = "stopped" if stopping else "retrying"
            snapshot()
        if states[name]["state"] == "retrying":
            states[name]["state"] = "failed"
        current = None
        snapshot()

    snapshot(final=True)
    return 0 if all(item["state"] == "completed" for item in states.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
