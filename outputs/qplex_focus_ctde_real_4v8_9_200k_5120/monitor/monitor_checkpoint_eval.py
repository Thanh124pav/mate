#!/usr/bin/env python3
import json
import pathlib
import time


TRIAL_DIR = pathlib.Path(
    "/home/pavt1024/vsn/mate/outputs/qplex_focus_ctde_real_4v8_9_200k_5120/"
    "QPLEX_FOCUS/QPLEX_FOCUS_mate-qplex_focus.camera_2650f_00000_0_2026-08-24_23-02-04"
)
RESULT_JSON = TRIAL_DIR / "result.json"
JOINED_JSONL = TRIAL_DIR / "checkpoint_eval_results.jsonl"
MONITOR_DIR = pathlib.Path("/home/pavt1024/vsn/mate/outputs/qplex_focus_ctde_real_4v8_9_200k_5120/monitor")
EVENTS_JSONL = MONITOR_DIR / "iteration5_events.jsonl"
SUMMARY_JSON = MONITOR_DIR / "latest_summary.json"
POLL_SECONDS = 30


def read_last_jsonl(path):
    if not path.exists():
        return None
    last = None
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if line:
                last = line
    return json.loads(last) if last else None


def read_joined_records():
    if not JOINED_JSONL.exists():
        return []
    records = []
    with JOINED_JSONL.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def pick_training_metrics(result):
    stats = (
        result.get("info", {})
        .get("learner", {})
        .get("default_policy", {})
        .get("learner_stats", {})
    )
    return {
        "training_iteration": result.get("training_iteration"),
        "timesteps_total": result.get("timesteps_total"),
        "episodes_total": result.get("episodes_total"),
        "episode_reward_mean": result.get("episode_reward_mean"),
        "episode_len_mean": result.get("episode_len_mean"),
        "loss": stats.get("loss"),
        "td_loss": stats.get("td_loss"),
        "focus_action_distill_loss": stats.get("focus_action_distill_loss"),
        "focus_belief_loss": stats.get("focus_belief_loss"),
        "focus_valid_ratio": stats.get("focus_valid_ratio"),
        "focus_confidence_mean": stats.get("focus_confidence_mean"),
        "focus_belief_loss_h1": stats.get("focus_belief_loss_h1"),
        "focus_belief_pos_error_h1": stats.get("focus_belief_pos_error_h1"),
        "focus_belief_pred_std_h1": stats.get("focus_belief_pred_std_h1"),
        "focus_belief_loss_h2": stats.get("focus_belief_loss_h2"),
        "focus_belief_pos_error_h2": stats.get("focus_belief_pos_error_h2"),
        "focus_belief_pred_std_h2": stats.get("focus_belief_pred_std_h2"),
        "focus_belief_loss_h3": stats.get("focus_belief_loss_h3"),
        "focus_belief_pos_error_h3": stats.get("focus_belief_pos_error_h3"),
        "focus_belief_pred_std_h3": stats.get("focus_belief_pred_std_h3"),
        "focus_teacher_student_kl": stats.get("focus_teacher_student_kl"),
        "done": result.get("done"),
    }


def pick_eval_metrics(record):
    return {
        "checkpoint": record.get("checkpoint"),
        "training_iteration": record.get("training_iteration"),
        "timesteps_total": record.get("timesteps_total"),
        "episode_reward_mean": record.get("episode_reward_mean"),
        "decentralized_eval/normalized_target_episode_reward": record.get(
            "decentralized_eval/normalized_target_episode_reward"
        ),
        "decentralized_eval/mean_transport_rate": record.get(
            "decentralized_eval/mean_transport_rate"
        ),
        "decentralized_eval/mean_coverage_rate": record.get(
            "decentralized_eval/mean_coverage_rate"
        ),
        "decentralized_eval/target_episode_reward": record.get(
            "decentralized_eval/target_episode_reward"
        ),
    }


def append_event(event):
    event = {"wall_time": time.strftime("%Y-%m-%d %H:%M:%S"), **event}
    with EVENTS_JSONL.open("a", encoding="utf-8") as fp:
        fp.write(json.dumps(event, sort_keys=True) + "\n")


def write_summary(summary):
    tmp = SUMMARY_JSON.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(SUMMARY_JSON)


def main():
    MONITOR_DIR.mkdir(parents=True, exist_ok=True)
    seen_iterations = set()
    seen_eval_checkpoints = set()

    append_event({"event": "monitor_started", "trial_dir": str(TRIAL_DIR)})

    while True:
        result = read_last_jsonl(RESULT_JSON)
        joined = read_joined_records()

        if result:
            iteration = result.get("training_iteration")
            if iteration and iteration % 5 == 0 and iteration not in seen_iterations:
                seen_iterations.add(iteration)
                append_event({"event": "iteration_multiple_of_5", **pick_training_metrics(result)})

        for record in joined:
            checkpoint = record.get("checkpoint") or f"iter-{record.get('training_iteration')}"
            if checkpoint not in seen_eval_checkpoints:
                seen_eval_checkpoints.add(checkpoint)
                append_event({"event": "checkpoint_eval_joined", **pick_eval_metrics(record)})

        best_camera_coverage = None
        best_camera_target_suppression = None
        if joined:
            best_camera_coverage = max(
                joined,
                key=lambda r: r.get("decentralized_eval/mean_coverage_rate", float("-inf")),
            )
            best_camera_target_suppression = min(
                joined,
                key=lambda r: r.get("decentralized_eval/normalized_target_episode_reward", float("inf")),
            )

        write_summary(
            {
                "latest_training": pick_training_metrics(result) if result else None,
                "joined_eval_count": len(joined),
                "best_camera_coverage_eval": (
                    pick_eval_metrics(best_camera_coverage) if best_camera_coverage else None
                ),
                "best_camera_target_suppression_eval": (
                    pick_eval_metrics(best_camera_target_suppression)
                    if best_camera_target_suppression
                    else None
                ),
                "events_file": str(EVENTS_JSONL),
            }
        )

        if result and result.get("done"):
            append_event({"event": "training_done", **pick_training_metrics(result)})
            return

        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()
