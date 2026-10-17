#!/usr/bin/env python3
"""Run the ten matched value-factorization experiments concurrently.

The supervisor is intentionally independent of a Ray cluster: every algorithm
gets a fresh local cluster, its own short temporary path, log file, Tune output
directory, and W&B run.  It periodically writes a compact status snapshot and
keeps unfinished 2M-step jobs alive after the seven-hour reporting mark.
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


def latest_result(root):
    candidates = list(root.rglob("result.json"))
    if not candidates:
        return None, None
    path = max(candidates, key=lambda candidate: candidate.stat().st_mtime)
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        return path, None
    try:
        return path, json.loads(lines[-1])
    except json.JSONDecodeError:
        return path, None


def summarize(name, process, output_dir, start_time):
    result_path, result = latest_result(output_dir)
    result = result or {}
    learner = result.get("info", {}).get("learner", {}).get("default_policy", {})
    learner_stats = learner.get("learner_stats", learner)
    return {
        "algorithm": name,
        "pid": process.pid,
        "returncode": process.poll(),
        "running": process.poll() is None,
        "elapsed_seconds": time.time() - start_time,
        "result_path": str(result_path) if result_path else None,
        "training_iteration": result.get("training_iteration"),
        "timesteps_total": result.get("timesteps_total"),
        "episode_reward_mean": result.get("episode_reward_mean"),
        "coverage_rate_mean": result.get("custom_metrics", {}).get("coverage_rate_mean"),
        "evaluation_coverage_rate_mean": result.get("evaluation", {})
        .get("custom_metrics", {})
        .get("coverage_rate_mean"),
        "loss": learner_stats.get("loss"),
        "belief_loss": learner_stats.get("belief_loss"),
        "updated_at": utc_now(),
    }


def atomic_json(path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=True), encoding="utf-8")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--api-key-file", type=Path, required=True)
    parser.add_argument("--timesteps-total", type=int, default=2_000_000)
    parser.add_argument("--report-after-seconds", type=int, default=7 * 60 * 60)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--num-envs-per-worker", type=int, default=2)
    parser.add_argument("--num-gpus", type=float, default=0.1)
    parser.add_argument("--evaluation-interval", type=int, default=10)
    parser.add_argument("--project", default="mate-value-factorization-2m")
    parser.add_argument("--group", default="4v8-9-2m")
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[1]
    runtime_path = repo / ".runtime_pythonpath"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    api_key = args.api_key_file.read_text(encoding="utf-8").strip()
    if len(api_key) != 86:
        raise ValueError(f"expected the new 86-character W&B token, got {len(api_key)}")

    environment = os.environ.copy()
    environment.update(
        {
            "PYTHONPATH": str(runtime_path),
            "WANDB_API_KEY": api_key,
            "WANDB_MODE": "online",
            "CUDA_VISIBLE_DEVICES": "0",
            "SLURM_CPUS_ON_NODE": str(args.num_workers + 2),
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "MATE_RAY_OBJECT_STORE_BYTES": str(4 * 1024**3),
        }
    )
    environment.pop("WANDB_DISABLED", None)
    environment.pop("CUBLAS_WORKSPACE_CONFIG", None)

    processes = {}
    logs = {}
    starts = {}
    for index, (name, module) in enumerate(MODULES.items()):
        output_dir = args.output_dir / name
        output_dir.mkdir(parents=True, exist_ok=True)
        log_path = args.output_dir / f"{name}.log"
        log_stream = log_path.open("a", encoding="utf-8", buffering=1)
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
            str(output_dir),
        ]
        child_env = environment.copy()
        child_env["RAY_TMPDIR"] = f"/tmp/mate_2m_{index}_{name}"
        log_stream.write(f"START {utc_now()} {' '.join(command)}\n")
        process = subprocess.Popen(
            command,
            cwd="/tmp",
            env=child_env,
            stdout=log_stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        processes[name] = process
        logs[name] = log_stream
        starts[name] = time.time()
        time.sleep(2)

    start_time = time.time()
    deadline = start_time + args.report_after_seconds
    snapshot_written = False
    pid_payload = {
        "supervisor_pid": os.getpid(),
        "started_at": utc_now(),
        "report_deadline_epoch": deadline,
        "children": {name: process.pid for name, process in processes.items()},
    }
    atomic_json(args.output_dir / "pids.json", pid_payload)

    stopping = False

    def stop_children(signum, _frame):
        nonlocal stopping
        stopping = True
        for process in processes.values():
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signum)
                except ProcessLookupError:
                    pass

    signal.signal(signal.SIGTERM, stop_children)
    signal.signal(signal.SIGINT, stop_children)

    while True:
        status = {
            name: summarize(name, process, args.output_dir / name, starts[name])
            for name, process in processes.items()
        }
        payload = {
            "started_at": pid_payload["started_at"],
            "updated_at": utc_now(),
            "report_deadline_epoch": deadline,
            "jobs": status,
        }
        atomic_json(args.output_dir / "status.json", payload)
        if not snapshot_written and time.time() >= deadline:
            atomic_json(args.output_dir / "seven_hour_summary.json", payload)
            snapshot_written = True
        if stopping or all(process.poll() is not None for process in processes.values()):
            atomic_json(args.output_dir / "final_summary.json", payload)
            break
        time.sleep(30)

    for stream in logs.values():
        stream.close()
    return 0 if all(process.returncode == 0 for process in processes.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
