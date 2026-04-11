#!/bin/bash
# Run QPLEX_WM and QPLEX_WM2 experiments (pure + HRL) on 4v5-0 and 4v8-9.
#
# Usage:
#   bash scripts/run_qplex_wm.sh              # run all
#   bash scripts/run_qplex_wm.sh 500000       # custom timesteps
#   nohup bash scripts/run_qplex_wm.sh &      # background
#
# Monitor: tail -f scripts/run_qplex_wm.log

set -e
cd "$(dirname "$0")/.."

source ~/miniconda3/etc/profile.d/conda.sh
conda activate mate

export PYTHONUNBUFFERED=1
export RAY_OBJECT_STORE_MEMORY=$((2*1024*1024*1024))
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

TIMESTEPS=${1:-102400}
LOG="scripts/run_qplex_wm.log"

echo "========================================" | tee -a "$LOG"
echo "QPLEX WM/SE experiments — $(date)"       | tee -a "$LOG"
echo "Timesteps: $TIMESTEPS"                    | tee -a "$LOG"
echo "========================================" | tee -a "$LOG"

run() {
    local MODULE=$1
    local ENV=$2
    local NAME="$MODULE ($ENV)"

    echo "" | tee -a "$LOG"
    echo ">>> [$NAME] Starting at $(date)" | tee -a "$LOG"

    ray stop --force 2>/dev/null || true
    sleep 2

    python -m "$MODULE" \
        --env "$ENV" \
        --timesteps-total "$TIMESTEPS" \
        --num-workers 3 \
        --num-envs-per-worker 4 \
        --num-gpus 0.25 \
        2>&1 | tee -a "$LOG"

    echo ">>> [$NAME] Finished at $(date)" | tee -a "$LOG"
}

# ============ WM1: Position Predictor (Transformer + MoE) ============
run "examples.qplex_wm.camera.train"       "MATE-4v5-0.yaml"
run "examples.qplex_wm.camera.train"       "MATE-4v8-9.yaml"
run "examples.hrl.qplex_wm.camera.train"   "MATE-4v5-0.yaml"
run "examples.hrl.qplex_wm.camera.train"   "MATE-4v8-9.yaml"

# ============ WM2: True World Model (RSSM Dreamer-style) ============
run "examples.qplex_wm2.camera.train"      "MATE-4v5-0.yaml"
run "examples.qplex_wm2.camera.train"      "MATE-4v8-9.yaml"
run "examples.hrl.qplex_wm2.camera.train"  "MATE-4v5-0.yaml"
run "examples.hrl.qplex_wm2.camera.train"  "MATE-4v8-9.yaml"

# ============ SE: Shared Encoder (auxiliary MSE on shared fc1) ============
run "examples.qplex_se.camera.train"        "MATE-4v5-0.yaml"
run "examples.qplex_se.camera.train"        "MATE-4v8-9.yaml"
run "examples.hrl.qplex_se.camera.train"    "MATE-4v5-0.yaml"
run "examples.hrl.qplex_se.camera.train"    "MATE-4v8-9.yaml"

echo "" | tee -a "$LOG"
echo "========================================" | tee -a "$LOG"
echo "All done — $(date)"                      | tee -a "$LOG"
echo "========================================" | tee -a "$LOG"
