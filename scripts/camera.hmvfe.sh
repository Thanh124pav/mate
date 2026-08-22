#!/usr/bin/env bash
set -euo pipefail

if [[ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
    source "${HOME}/miniconda3/etc/profile.d/conda.sh"
    conda activate mate
elif [[ -f "${HOME}/Miniconda3/etc/profile.d/conda.sh" ]]; then
    source "${HOME}/Miniconda3/etc/profile.d/conda.sh"
    conda activate mate
fi

python -m examples.hmvfe.camera.train "$@"
