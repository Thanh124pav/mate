#!/usr/bin/env bash
set -euo pipefail
cd /home/pavt1024/vsn/mate
exec env PYTHONPATH=/home/pavt1024/vsn/mate WANDB_MODE=disabled /home/pavt1024/miniconda3/envs/mate/bin/python -u -m examples.qplex_focus.camera.train --local-dir /home/pavt1024/vsn/mate/outputs/qplex_focus_ctde_policy_kl_4v8_9_100k_5120_w2_b300 --timesteps-total 100000 --num-workers 2 --num-envs-per-worker 8 --buffer-capacity 300 --checkpoint-frequency 5 --decentralized-eval-episodes 5 --decentralized-eval-seed 0 --env-config MATE-4v8-9.yaml --seed 0
