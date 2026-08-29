#!/usr/bin/env bash
set -euo pipefail

if [[ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
    source "${HOME}/miniconda3/etc/profile.d/conda.sh"
    conda activate mate
elif [[ -f "${HOME}/Miniconda3/etc/profile.d/conda.sh" ]]; then
    source "${HOME}/Miniconda3/etc/profile.d/conda.sh"
    conda activate mate
fi

if [[ -n "${WANDB_API_KEY:-}" && -x "$(command -v wandb)" ]]; then
    wandb login --relogin "${WANDB_API_KEY}"
fi

python -m examples.hmvfe_focus.camera.train "$@"
