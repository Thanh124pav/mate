# MATE: Multi-Agent Tracking Environment

MATE is a benchmark for cooperative and competitive multi-agent reinforcement
learning in distributed target coverage control. It models an asymmetric,
two-team, zero-sum stochastic game with partial observations and optional
intra-team communication.

- Documentation: <https://mate-gym.readthedocs.io>
- Paper: <https://openreview.net/forum?id=SyoUVEyzJbE>
- License: [MIT](LICENSE)

## Installation

The tested learning stack uses Python 3.9, Gym 0.21, and Ray 1.12. Create an
isolated environment from the repository root:

```bash
conda create -n mate python=3.9.23 -y
conda activate mate

python -m pip install "pip==23.2.1" "setuptools==65.5.0" "wheel==0.38.4"
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
```

For GPU training, install the PyTorch build matching the host CUDA driver. The
version in `requirements.txt` is the reproducible baseline and can be replaced
with a compatible CUDA wheel when newer hardware requires it.

This repository contains custom RLlib agents. Copy them into the Ray
installation used by the active environment:

```bash
RAY_AGENTS="$(python -c 'import site; print(site.getsitepackages()[0])')/ray/rllib/agents"
cp -a ray/rllib/agents/. "$RAY_AGENTS/"
```

The repository also vendors a partial `ray/` tree, so training entry points
must use installed Ray without letting the repository root shadow it. Create a
small runtime import overlay once:

```bash
mkdir -p .runtime_pythonpath
ln -sfn ../examples .runtime_pythonpath/examples
ln -sfn ../mate .runtime_pythonpath/mate
```

## Getting Started

Create the default environment and step both teams jointly:

```python
import mate

env = mate.make("MultiAgentTracking-v0")
env.seed(0)
camera_obs, target_obs = env.reset()
done = False

while not done:
    camera_action, target_action = env.action_space.sample()
    (camera_obs, target_obs), rewards, done, infos = env.step(
        (camera_action, target_action)
    )
```

MATE includes preset environments such as `MATE-4v2-9-v0`,
`MATE-4v8-9-v0`, `MATE-8v8-9-v0`, and `MATE-Navigation-v0`. A YAML/JSON
configuration can also be passed directly:

```python
env = mate.make("MultiAgentTracking-v0", config="MATE-4v8-9.yaml")
```

Wrappers support discrete actions, enhanced observations, auxiliary rewards,
single-team control, and communication constraints. See the
[documentation](https://mate-gym.readthedocs.io) for the complete API.

## Examples

Run a built-in evaluation:

```bash
python -m mate.evaluate --episodes 1 --config MATE-4v8-9.yaml

python -m mate.evaluate --episodes 1 --render-communication \
  --camera-agent examples.greedy:GreedyCameraAgent \
  --camera-kwargs '{"memory_period": 20}' \
  --target-agent examples.greedy:GreedyTargetAgent \
  --config MATE-4v8-9.yaml --seed 0
```

Run one value-factorization experiment from a directory that does not shadow
the installed Ray package:

```bash
REPO_ROOT="$PWD"
cd /tmp
PYTHONPATH="$REPO_ROOT/.runtime_pythonpath" \
python -m examples.facmac_focuss.camera \
  --env MATE-4v8-9.yaml \
  --timesteps-total 2000000 \
  --num-workers 4 \
  --num-envs-per-worker 2 \
  --num-gpus 0.5 \
  --evaluation-interval 50
```

For a sequence of long-running experiments, use the priority queue supervisor:

```bash
python scripts/run_value_factorization_2m_priority_queue.py \
  --output-dir outputs/value_factorization_2m \
  --api-key-file /path/to/wandb-token.txt \
  --algorithms facmac facmac_focuss resq resq_focuss \
  --num-workers 4 --num-envs-per-worker 2 --num-gpus 0.5
```

## Implemented Algorithms

The implementations and launchers live under [`examples/`](examples).

- Rule-based: Random, Naive, Greedy, and Heuristic.
- Independent/on-policy: IQL, IPPO, MAPPO, A3C, and MADDPG.
- Value factorization: QMIX, QPLEX, FACMAC, RESQ, RISKQ, QFREE, CONCAVEQ,
  DUELMIX, and their repository variants.
- FOCUSS variants: FACMAC_FOCUSS, RESQ_FOCUSS, RISKQ_FOCUSS, QFREE_FOCUSS,
  CONCAVEQ_FOCUSS, QPLEX_FOCUS, QMIX_FOCUS, MAPPO_FOCUS, and TarMAC_FOCUS.
- Communication: TarMAC and I2C.
- Population-based learning: Self-Play, Fictitious Self-Play, and PSRO-Nash.

Learning-based agents are built against the custom RLlib extensions in
[`ray/rllib/agents`](ray/rllib/agents) and are tested with Ray 1.12.x.

## Results

Representative deterministic evaluation results from the included MATE
experiments are shown below. Metrics can vary with seed and checkpoint.

### MATE-4v5-0

| Algorithm | Target episode reward | Mean transport rate | Mean coverage rate |
| --- | ---: | ---: | ---: |
| Greedy | +2317.80 | 28.973% | 75.021% |
| QPLEX_V2 | +2258.00 | 28.225% | 74.065% |
| QPLEX_WM2 | +2575.80 | 32.197% | 67.831% |
| QPLEX_SE | +2522.20 | 31.527% | 70.347% |

### MATE-4v8-9

| Algorithm | Target episode reward | Mean transport rate | Mean coverage rate |
| --- | ---: | ---: | ---: |
| Greedy | +4499.60 | 35.153% | 64.468% |
| QPLEX_V2 | +5694.60 | 44.489% | 52.661% |
| QPLEX_WM2 | +5019.00 | 39.211% | 59.137% |
| QPLEX_WM3 | +6592.60 | 51.505% | 45.708% |
| QPLEX_SE | +5526.20 | 43.173% | 55.348% |
| QPLEX_WM2_SE | +5271.50 | 41.184% | 55.682% |
| QPLEX_FOCUS | +5059.10 | 39.524% | 57.824% |

#### Provisional value-factorization sweep

The following seed-0 snapshot was collected on `MATE-4v8-9` on October 18,
2026. Runs were stopped at different budgets, so these numbers are useful for
tracking experiments but are **not** a controlled final comparison. Train
coverage is the last reported rolling mean; evaluation coverage is from the
latest deterministic evaluation checkpoint.

| Algorithm | Steps | Train coverage | Eval iteration | Eval coverage | Eval real coverage |
| --- | ---: | ---: | ---: | ---: | ---: |
| FACMAC | 650,411 | 54.44% | 100 | 53.42% | 47.00% |
| FACMAC_FOCUSS | 1,780,269 | 57.66% | 300 | 55.26% | 49.72% |
| RESQ | 2,005,541 | 53.53% | 300 | 42.81% | 41.30% |
| RESQ_FOCUSS | 1,712,494 | 60.39% | 250 | 59.29% | 56.09% |
| RISKQ | 1,060,237 | 49.81% | 150 | 39.78% | 38.20% |
| RISKQ_FOCUSS | 619,629 | 59.03% | 100 | 59.77% | 57.36% |
| QFREE | 1,161,518 | 54.30% | 150 | 51.93% | 45.04% |
| QFREE_FOCUSS | 835,089 | 58.87% | 100 | 62.18% | 56.31% |
| CONCAVEQ (pre-fix) | 577,288 | 38.56% | 50 | 41.13% | 36.95% |
| CONCAVEQ_FOCUSS (pre-fix) | 210,428 | 51.90% | - | - | - |

The two ConcaveQ rows predate the policy-value-mixer objective fix in this
revision and should not be used as final ConcaveQ results. Corrected runs
reached only 28,729 and 23,414 steps before the allocated MIG device failed;
their entropy remained stable, but the runs were too short for evaluation.
Experiment histories are available in the
[W&B project](https://wandb.ai/hust_edu_vn/mate-value-factorization-2m).

## Citation

If you use MATE in your work, please cite:

```bibtex
@inproceedings{pan2022mate,
  title     = {{MATE}: Benchmarking Multi-Agent Reinforcement Learning in Distributed Target Coverage Control},
  author    = {Xuehai Pan and Mickel Liu and Fangwei Zhong and Yaodong Yang and Song-Chun Zhu and Yizhou Wang},
  booktitle = {Thirty-sixth Conference on Neural Information Processing Systems Datasets and Benchmarks Track},
  year      = {2022},
  url       = {https://openreview.net/forum?id=SyoUVEyzJbE}
}
```
