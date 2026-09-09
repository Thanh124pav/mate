#!/usr/bin/env bash
# QPLEX-WM2 ablations: hyperparameters, target policy, transport evaluation.
# Environments: MATE 4v{2,4,8,16}-9. No main baselines or imagination runs.
set -Eeuo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"
GROUP="${1:-all}"
TIMESTEPS_TOTAL="${TIMESTEPS_TOTAL:-500000}"
BUFFER_CAPACITY="${BUFFER_CAPACITY:-200}"
NUM_WORKERS="${NUM_WORKERS:-3}"
NUM_ENVS_PER_WORKER="${NUM_ENVS_PER_WORKER:-8}"
EVALUATION_INTERVAL="${EVALUATION_INTERVAL:-5}"
WANDB_PROJECT="${WANDB_PROJECT:-mate-wm2-ablations}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/experiments/wm2_ablations_multi_env_server}"
SEEDS="${SEEDS:-0 1 2}"
ENVIRONMENTS="${ENVIRONMENTS:-MATE-4v2-9.yaml MATE-4v4-9.yaml MATE-4v8-9.yaml MATE-4v16-9.yaml}"
GPU_IDS="${GPU_IDS:-0}"
MAX_PARALLEL="${MAX_PARALLEL:-1}"
DRY_RUN="${DRY_RUN:-0}"

HYPER_VARIANTS=(lr_5e5 lr_3e4 wm_weight_01 wm_weight_10)
TARGET_VARIANTS=(
  target_greedy
  target_evasive_strength_025 target_evasive_strength_075
  target_evasive_range_025 target_evasive_range_075
  target_evasive_noise_025 target_evasive_noise_075
)
TRANSPORT_VARIANTS=(
  transport_penalty_010 transport_penalty_025
  transport_penalty_050 transport_penalty_100
)

explain() {
  cat <<'EOF'
Groups:
  hyper: lr={5e-5,3e-4} vs reference 1e-4; WM loss weight={0.1,1.0} vs 0.5.
  target: greedy, plus evasive strength/range/noise={0.25,0.75}; reference=0.5.
  transport: mean_transport_rate reward coefficient={-0.10,-0.25,-0.50,-1.0};
             coverage coefficient stays 1.0 and the main reference has no penalty.
  all: all 15 variants above (default).

All evaluations use greedy targets. Report coverage and mean_transport_rate together.
mean_transport_rate measures target cargo transport success/progress, not energy.
Defaults: 4 environments x 15 variants x 3 seeds = 180 runs, 500k steps each.

Examples:
  ./ablations.sh list
  SEEDS="0" DRY_RUN=1 ./ablations.sh all
  GPU_IDS="0 1 2 3" MAX_PARALLEL=4 ./ablations.sh all
  ENVIRONMENTS="MATE-4v8-9.yaml" ./ablations.sh target
EOF
}

if [[ "$GROUP" == list || "$GROUP" == help || "$GROUP" == --help ]]; then explain; exit 0; fi
case "$GROUP" in
  hyper) variants=("${HYPER_VARIANTS[@]}") ;;
  target) variants=("${TARGET_VARIANTS[@]}") ;;
  transport) variants=("${TRANSPORT_VARIANTS[@]}") ;;
  all) variants=("${HYPER_VARIANTS[@]}" "${TARGET_VARIANTS[@]}" "${TRANSPORT_VARIANTS[@]}") ;;
  *) echo "Choose: hyper, target, transport, all, list" >&2; exit 2 ;;
esac

activate_mate() {
  [[ "${CONDA_DEFAULT_ENV:-}" == mate ]] && return
  local conda_sh=""
  if command -v conda >/dev/null 2>&1; then
    conda_sh="$(conda info --base)/etc/profile.d/conda.sh"
  elif [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
    conda_sh="$HOME/miniconda3/etc/profile.d/conda.sh"
  elif [[ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]]; then
    conda_sh="$HOME/anaconda3/etc/profile.d/conda.sh"
  fi
  [[ -f "$conda_sh" ]] || { echo "Activate conda env mate first." >&2; exit 1; }
  source "$conda_sh"
  conda activate mate
}
activate_mate
export WANDB_MODE=online WANDB_SILENT=true
unset WANDB_DISABLED || true
mkdir -p "$OUTPUT_ROOT/logs"

read -r -a seeds <<< "$SEEDS"
read -r -a envs <<< "$ENVIRONMENTS"
read -r -a gpus <<< "$GPU_IDS"
[[ "$MAX_PARALLEL" =~ ^[1-9][0-9]*$ ]] || { echo "Invalid MAX_PARALLEL" >&2; exit 2; }
(( MAX_PARALLEL <= ${#gpus[@]} )) || { echo "Use at most one parallel run per GPU_ID." >&2; exit 2; }
for env in "${envs[@]}"; do
  [[ -f "$REPO_ROOT/mate/assets/$env" ]] || { echo "Missing mate/assets/$env" >&2; exit 2; }
done

is_complete() {
  python - "$OUTPUT_ROOT" "$1" "$2" "$3" "$TIMESTEPS_TOTAL" <<'PYCHECK'
import csv, sys
from pathlib import Path
root, env, variant, seed = Path(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4]
target = int(sys.argv[5])
slug = env.removeprefix('MATE-').removesuffix('.yaml').lower()
for run in sorted(root.glob(f'qplex_wm2__{slug}__{variant}__seed{seed}__*'), reverse=True):
    for file in run.glob('ray_results/**/progress.csv'):
        try:
            rows = list(csv.DictReader(file.open(encoding='utf-8')))
            if rows and int(float(rows[-1]['timesteps_total'])) >= target:
                raise SystemExit(0)
        except (KeyError, ValueError, OSError):
            pass
raise SystemExit(1)
PYCHECK
}

run_one() {
  local env="$1" variant="$2" seed="$3" gpu="$4"
  local slug="${env#MATE-}"; slug="${slug%.yaml}"
  local log="$OUTPUT_ROOT/logs/${slug}__${variant}__seed${seed}.log"
  local ray_tmp="${TMPDIR:-/tmp}/mate-wm2-${slug}-${variant}-seed${seed}"
  local cmd=(python -m scripts.run_wm2_ablation "$variant"
    --env "$env" --seed "$seed" --timesteps-total "$TIMESTEPS_TOTAL"
    --buffer-capacity "$BUFFER_CAPACITY" --num-workers "$NUM_WORKERS"
    --num-envs-per-worker "$NUM_ENVS_PER_WORKER" --num-gpus 1
    --evaluation-interval "$EVALUATION_INTERVAL" --project "$WANDB_PROJECT"
    --output-root "$OUTPUT_ROOT")
  if is_complete "$env" "$variant" "$seed"; then
    echo "SKIP completed: $env $variant seed=$seed"; return
  fi
  printf 'START env=%s variant=%s seed=%s gpu=%s\n' "$env" "$variant" "$seed" "$gpu"
  printf 'COMMAND CUDA_VISIBLE_DEVICES=%q ' "$gpu"; printf '%q ' "${cmd[@]}"; printf '\n'
  [[ "$DRY_RUN" == 1 ]] && return
  mkdir -p "$ray_tmp"
  CUDA_VISIBLE_DEVICES="$gpu" RAY_TMPDIR="$ray_tmp" "${cmd[@]}" 2>&1 | tee "$log"
}

pids=(); labels=(); job_index=0; failures=0
reap_first() {
  if ! wait "${pids[0]}"; then echo "FAILED: ${labels[0]}" >&2; ((failures+=1)); fi
  pids=("${pids[@]:1}"); labels=("${labels[@]:1}")
}

echo "variants=${#variants[@]} envs=${envs[*]} seeds=${seeds[*]} parallel=$MAX_PARALLEL"
for env in "${envs[@]}"; do
  for variant in "${variants[@]}"; do
    for seed in "${seeds[@]}"; do
      gpu="${gpus[$((job_index % ${#gpus[@]}))]}"
      run_one "$env" "$variant" "$seed" "$gpu" &
      pids+=("$!"); labels+=("$env $variant seed=$seed"); ((job_index+=1))
      (( ${#pids[@]} >= MAX_PARALLEL )) && reap_first
    done
  done
done
while (( ${#pids[@]} )); do reap_first; done
(( failures == 0 )) || { echo "$failures runs failed; see $OUTPUT_ROOT/logs" >&2; exit 1; }
echo "All requested runs completed or were already complete."
