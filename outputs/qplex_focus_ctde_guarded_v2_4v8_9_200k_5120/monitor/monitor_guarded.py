#!/usr/bin/env python3
import json
import pathlib
import time

BASE = pathlib.Path('/home/pavt1024/vsn/mate/outputs/qplex_focus_ctde_guarded_v2_4v8_9_200k_5120')
MONITOR_DIR = BASE / 'monitor'
EVENTS = MONITOR_DIR / 'events.jsonl'
SUMMARY = MONITOR_DIR / 'latest_summary.json'
POLL_SECONDS = 60


def find_trial_dir():
    candidates = sorted(BASE.glob('QPLEX_FOCUS/QPLEX_FOCUS_*/result.json'), key=lambda p: p.stat().st_mtime)
    return candidates[-1].parent if candidates else None


def read_jsonl(path):
    if not path.exists():
        return []
    rows = []
    with path.open('r', encoding='utf-8') as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def compact_training(row):
    if not row:
        return None
    stats = row.get('info', {}).get('learner', {}).get('default_policy', {}).get('learner_stats', {})
    keys = [
        'loss', 'td_loss', 'focus_belief_loss', 'focus_belief_pos_error_weighted',
        'focus_belief_baseline_pos_error_weighted', 'focus_belief_pos_error_improvement_weighted',
        'focus_belief_error_to_std_ratio_weighted', 'focus_belief_confidence_pos_error_corr',
        'focus_action_bias_eta_effective_mean', 'focus_action_bias_confidence_mean',
        'focus_action_distill_loss', 'focus_teacher_student_top1_agreement',
        'focus_teacher_q_top1_agreement', 'focus_teacher_student_kl',
    ]
    return {
        'training_iteration': row.get('training_iteration'),
        'timesteps_total': row.get('timesteps_total'),
        'episode_reward_mean': row.get('episode_reward_mean'),
        'episodes_total': row.get('episodes_total'),
        'done': row.get('done'),
        **{k: stats.get(k) for k in keys if k in stats},
    }


def compact_eval(row):
    if not row:
        return None
    keys = [
        'training_iteration', 'timesteps_total', 'episode_reward_mean', 'checkpoint_path',
        'decentralized_eval/mean_coverage_rate', 'decentralized_eval/mean_transport_rate',
        'decentralized_eval/normalized_target_episode_reward', 'decentralized_eval/target_episode_reward',
    ]
    out = {k: row.get(k) for k in keys if k in row}
    for key, value in row.items():
        if key.startswith('learner_stats/focus_belief_') or key in (
            'learner_stats/focus_action_bias_eta_effective_mean',
            'learner_stats/focus_action_bias_confidence_mean',
            'learner_stats/focus_teacher_student_top1_agreement',
            'learner_stats/focus_teacher_q_top1_agreement',
            'learner_stats/focus_teacher_student_kl',
        ):
            out[key] = value
    return out


def append(event):
    event = {'wall_time': time.strftime('%Y-%m-%d %H:%M:%S'), **event}
    with EVENTS.open('a', encoding='utf-8') as handle:
        handle.write(json.dumps(event, sort_keys=True) + '\n')


def main():
    MONITOR_DIR.mkdir(parents=True, exist_ok=True)
    seen_train = set()
    seen_eval = set()
    append({'event': 'monitor_started', 'base': str(BASE)})
    while True:
        trial = find_trial_dir()
        latest_training = None
        joined = []
        if trial:
            result_rows = read_jsonl(trial / 'result.json')
            latest = result_rows[-1] if result_rows else None
            latest_training = compact_training(latest)
            if latest:
                it = latest.get('training_iteration')
                if it and it % 5 == 0 and it not in seen_train:
                    seen_train.add(it)
                    append({'event': 'iteration_multiple_of_5', **compact_training(latest)})
            joined = [row for row in read_jsonl(trial / 'checkpoint_eval_results.jsonl') if row.get('status') == 'ok']
            for row in joined:
                key = row.get('checkpoint_path') or row.get('training_iteration')
                if key not in seen_eval:
                    seen_eval.add(key)
                    append({'event': 'checkpoint_eval_joined', **compact_eval(row)})

        best_cov = max(joined, key=lambda r: r.get('decentralized_eval/mean_coverage_rate', float('-inf'))) if joined else None
        best_sup = min(joined, key=lambda r: r.get('decentralized_eval/normalized_target_episode_reward', float('inf'))) if joined else None
        tmp = SUMMARY.with_suffix('.json.tmp')
        tmp.write_text(json.dumps({
            'trial_dir': str(trial) if trial else None,
            'latest_training': latest_training,
            'joined_eval_count': len(joined),
            'best_camera_coverage_eval': compact_eval(best_cov),
            'best_camera_target_suppression_eval': compact_eval(best_sup),
            'events_file': str(EVENTS),
        }, indent=2, sort_keys=True) + '\n', encoding='utf-8')
        tmp.replace(SUMMARY)
        if latest_training and latest_training.get('done'):
            append({'event': 'training_done', **latest_training})
            return
        time.sleep(POLL_SECONDS)


if __name__ == '__main__':
    main()
