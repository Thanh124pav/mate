#!/bin/bash
# Run experiments sequentially with Phase 1 + Phase 2 support for HiTMAC v2.
#
# Usage:
#   bash scripts/run_all.sh              # run all
#   nohup bash scripts/run_all.sh &      # run in background
#
# Monitor: tail -f scripts/run_all.log

cd "$(dirname "$0")/.."

source ~/miniconda3/etc/profile.d/conda.sh
conda activate mate

export PYTHONUNBUFFERED=1
export RAY_OBJECT_STORE_MEMORY=$((2*1024*1024*1024))

# Fix cuDNN version mismatch: prioritize conda env's cuDNN 8.9.2 over system cuda-12.8
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

TIMESTEPS=102400
LOG="scripts/run_all.log"
RESULTS="experiment_results"

echo "========================================" | tee -a "$LOG"
echo "Starting all experiments at $(date)"      | tee -a "$LOG"
echo "Timesteps per experiment: $TIMESTEPS"     | tee -a "$LOG"
echo "========================================" | tee -a "$LOG"

run_one() {
    local ALG=$1
    local ENV_YAML=$2
    local ENV_SHORT=$3
    local RUN_TYPE=$4
    local CONFIG_MODULE=$5
    local EXTRA_TIMESTEPS=${6:-$TIMESTEPS}
    local EXTRA_CONFIG=${7:-""}   # extra python code to inject into config

    local NAME="claude-${ENV_SHORT}-${ALG}"
    echo "" | tee -a "$LOG"
    echo ">>> [$NAME] Starting at $(date)" | tee -a "$LOG"

    # Stop any leftover Ray
    ray stop --force 2>/dev/null || true
    sleep 2

    python -u -c "
import os, sys, copy
sys.path.insert(0, '.')

import ray
ray.init(num_cpus=8, num_gpus=1)

from ray import tune
from ${CONFIG_MODULE} import config, make_env

exp_config = copy.deepcopy(config)
exp_config['env_config']['config'] = '${ENV_YAML}'
exp_config['num_workers'] = 3
exp_config['num_envs_per_worker'] = 4
exp_config['num_gpus'] = 0.25
exp_config['num_gpus_per_worker'] = 0
exp_config['num_cpus_for_driver'] = 1

# Adjust buffer for value decomposition
if '${ALG}' in ('duelmix', 'spectra'):
    exp_config['buffer_size'] = 50

# Extra config injection
${EXTRA_CONFIG}

cbs = []
try:
    from examples.utils import WandbLoggerCallback
    if WandbLoggerCallback.is_available():
        cbs.append(WandbLoggerCallback(project='claude-${ENV_SHORT}', group='${NAME}'))
        print('wandb enabled for ${NAME}')
except: pass

experiment = tune.Experiment(
    name='${NAME}',
    run='${RUN_TYPE}',
    config=exp_config,
    local_dir='${RESULTS}/${ALG}/${ENV_SHORT}',
    stop={'timesteps_total': ${EXTRA_TIMESTEPS}},
    checkpoint_freq=20,
    checkpoint_at_end=True,
    max_failures=3,
)

analysis = tune.run(experiment, metric='episode_reward_mean', mode='max', callbacks=cbs, verbose=2)

# Save latest checkpoint path for Phase 2
best_trial = analysis.get_best_trial('episode_reward_mean', 'max')
if best_trial and best_trial.checkpoint:
    ckpt_path = best_trial.checkpoint.value
    ckpt_file = os.path.join('${RESULTS}/${ALG}/${ENV_SHORT}', 'latest_checkpoint.txt')
    with open(ckpt_file, 'w') as f:
        f.write(ckpt_path)
    print(f'CHECKPOINT: {ckpt_path}')

print('DONE: ${NAME}')
ray.shutdown()
" 2>&1 | tee -a "$LOG"

    echo ">>> [$NAME] Finished at $(date)" | tee -a "$LOG"
}

# Helper: get latest checkpoint path for an algorithm/env
get_checkpoint() {
    local ALG=$1
    local ENV_SHORT=$2
    local CKPT_FILE="${RESULTS}/${ALG}/${ENV_SHORT}/latest_checkpoint.txt"
    if [ -f "$CKPT_FILE" ]; then
        cat "$CKPT_FILE"
    else
        # Fallback: find latest checkpoint dir
        find "${RESULTS}/${ALG}/${ENV_SHORT}" -name "checkpoint_*" -type d 2>/dev/null | sort | tail -1
    fi
}

# ============ DuelMIX ============
# run_one "duelmix" "MATE-4v5-0.yaml" "4v5-0" "DUELMIX" "examples.hrl.duelmix.camera.config"  # done
# run_one "duelmix" "MATE-4v8-9.yaml" "4v8-9" "DUELMIX" "examples.hrl.duelmix.camera.config"  # done

# ============ SPECTra ============
# run_one "spectra" "MATE-4v5-0.yaml" "4v5-0" "SPECTRA" "examples.hrl.spectra.camera.config"  # done
# run_one "spectra" "MATE-4v8-9.yaml" "4v8-9" "SPECTRA" "examples.hrl.spectra.camera.config"  # done

# ============ HiTMAC v2 — 4v5-0 (Phase 1 → Phase 2) ============
# run_one "hitmac_v2" "MATE-4v5-0.yaml" "4v5-0" "PPO" "examples.hitmac_v2.camera.config"  # paused
# run_one "hitmac_v2_coord" ...  # paused

# ============ HiTMAC v2 — 4v8-9 (Phase 1 → Phase 2) ============
# run_one "hitmac_v2" "MATE-4v8-9.yaml" "4v8-9" "PPO" "examples.hitmac_v2.camera.config"  # paused
# run_one "hitmac_v2_coord" ...  # paused

# ============ SMPE2 (train_batch_size=1024) ============
run_one "smpe2" "MATE-4v5-0.yaml" "4v5-0" "PPO" "examples.smpe2.camera.config"
run_one "smpe2" "MATE-4v8-9.yaml" "4v8-9" "PPO" "examples.smpe2.camera.config"

echo "" | tee -a "$LOG"
echo "========================================" | tee -a "$LOG"
echo "All experiments complete at $(date)"      | tee -a "$LOG"
echo "========================================" | tee -a "$LOG"
