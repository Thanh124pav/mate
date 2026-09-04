#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

if [[ -f /home/pavt1024/miniconda3/etc/profile.d/conda.sh ]]; then
    source /home/pavt1024/miniconda3/etc/profile.d/conda.sh
    conda activate mate
fi

TIMESTEPS_TOTAL="${TIMESTEPS_TOTAL:-120000}"
EVALUATION_INTERVAL="${EVALUATION_INTERVAL:-5}"
BUFFER_CAPACITY="${BUFFER_CAPACITY:-200}"
TARGET_AGENTS="${TARGET_AGENTS:-greedy evasive}"
WANDB_PROJECT="${WANDB_PROJECT:-mate}"
WANDB_GROUP_PREFIX="${WANDB_GROUP_PREFIX:-paper_hrl_preferred}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
LOG_ROOT="${LOG_ROOT:-$REPO_ROOT/logs/paper_camera_target_agents_$RUN_TAG}"

# This script is intended for paper runs, so W&B is enabled by default.
# Override WANDB_MODE before launching if you explicitly want offline logging.
export WANDB_DISABLED=false
export WANDB_MODE="${WANDB_MODE:-online}"

mkdir -p "$LOG_ROOT"

# Rule: use the HRL implementation when it exists for the algorithm/variant;
# run the flat/base implementation only when there is no corresponding HRL variant
# in the supported experiment set.
MODULES=(
    examples.hrl.qmix.camera
    examples.hrl.qmix_wm2.camera
    examples.hrl.spectra.camera
    examples.hrl.spectra_wm2.camera
    examples.hrl.mappo.camera
    examples.hrl.mappo_wm2.camera
    examples.hrl.ippo.camera
    examples.hrl.ippo_wm2.camera
    examples.hrl.qplex.camera
    examples.hrl.qplex_wm2.camera
    examples.hrl.duelmix.camera
    examples.hrl.duelmix_wm2.camera
    examples.i2c.camera
    examples.i2c_wm2.camera
    examples.tarmac.camera
    examples.tarmac_wm2.camera
)

{
    echo "run_tag=$RUN_TAG"
    echo "timesteps_total=$TIMESTEPS_TOTAL"
    echo "evaluation_interval=$EVALUATION_INTERVAL"
    echo "buffer_capacity=$BUFFER_CAPACITY"
    echo "target_agents=$TARGET_AGENTS"
    echo "wandb_project=$WANDB_PROJECT"
    echo "wandb_mode=$WANDB_MODE"
    echo "log_root=$LOG_ROOT"
    printf 'modules=%s\n' "${MODULES[*]}"
} | tee "$LOG_ROOT/config.log"

for target_agent in $TARGET_AGENTS; do
    for module in "${MODULES[@]}"; do
        run_id="${module//./_}_${target_agent}"
        group="${WANDB_GROUP_PREFIX}_${target_agent}_${RUN_TAG}"
        log_file="$LOG_ROOT/${run_id}.log"

        echo "[$(date --iso-8601=seconds)] START $module target=$target_agent group=$group" | tee -a "$LOG_ROOT/summary.log"
        module_path="${module#examples.}"
        train_file="examples/${module_path//./\/}/train.py"
        extra_args=()
        if grep -q -- '--buffer-capacity' "$train_file"; then
            extra_args+=(--buffer-capacity "$BUFFER_CAPACITY")
        fi

        python -m "${module}.train" \
            --target-agent "$target_agent" \
            --evaluation-interval "$EVALUATION_INTERVAL" \
            --timesteps-total "$TIMESTEPS_TOTAL" \
            --project "$WANDB_PROJECT" \
            --group "$group" \
            "${extra_args[@]}" \
            2>&1 | tee "$log_file"
        echo "[$(date --iso-8601=seconds)] DONE  $module target=$target_agent group=$group" | tee -a "$LOG_ROOT/summary.log"
    done
done
