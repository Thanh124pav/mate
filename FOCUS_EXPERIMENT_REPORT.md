# FOCUS experiment report

This report records the runs that were actually executed on
`MATE-4v8-9.yaml`. It does not claim a multi-seed reproduction of any paper.
All reported numbers are evaluation `coverage_rate`, not smoothed training
coverage.

## Algorithms with FOCUS integration

The experiment registry currently exposes FOCUS variants for:

- CTDE value-based: FACMAC, ResQ, RiskQ, QFree, ConcaveQ, QPLEX, QMIX, and
  DuelMIX.
- CTDE policy-based: MAPPO, TarMAC, and SMPE2.
- CTCE policy-based: HMVFE and HiTMAC.

The controlled result table below covers FACMAC, ResQ, RiskQ, QFree,
ConcaveQ, HMVFE, and HiTMAC. Other integrations were not rerun under this
exact 250k-step, seed-0 protocol and are therefore not mixed into the table.

## Training configuration

### Common environment and evaluation setup

- Environment: `MultiAgentTracking-v0` with `MATE-4v8-9.yaml`.
- Agents: four homogeneous camera agents against the greedy target policy
  (`seed=0`).
- Reward: dense cooperative reward with `coverage_rate` coefficient `1.0`
  and mean team reduction.
- Observations: partial local observations; no enhanced observations.
- HRL execution: hierarchical target-subset action, `multi_selection=true`,
  `frame_skip=5`, and horizon `500` environment steps.
- Training seed: `0` only. Consequently, the table has no cross-seed standard
  deviation or confidence interval.
- Main budget: `250,000` environment steps. FACMAC is only a completed 100k
  pilot. ConcaveQ was stopped at 149,732 steps after its evaluation coverage
  decreased and its loss magnitude continued to grow.
- Evaluation metric: `eval/mean_coverage_rate`. Value-based runs evaluate for
  10 episodes every 10 training iterations. CTCE runs evaluate for 10 episodes
  every 10 updates.
- The milestone columns use the evaluation nearest to 50k, 100k, 150k, 200k,
  and 250k steps. The exact evaluation step is included in each cell. The
  value-based schedule produced its last evaluation near 216k before training
  crossed 250k, so its 250k cells are left empty rather than duplicating the
  200k measurement.

### Value-based runs

- Framework: PyTorch; two rollout workers; four environments per worker;
  complete-episode batches; `train_batch_size=1024`;
  `timesteps_per_iteration=5120`; learning starts at 5,000 steps.
- Runtime replay capacity: 64 episodes. The QFree+FOCUS retry used capacity 32
  and CPU because the first GPU attempt failed with
  `CUBLAS_STATUS_EXECUTION_FAILED` on the 4 GB GPU.
- Common learner settings: recurrent hidden size 64, `gamma=0.99`, target
  network update frequency 200, and gradient clipping 10.
- Learning rates: FACMAC actor `2.5e-3` and critic `5e-4`; ResQ, RiskQ, and
  ConcaveQ `1e-3`; QFree `5e-4`.
- Algorithm-specific settings retained from their implementations: FACMAC
  TD(lambda) `0.8`; ResQ/RiskQ/ConcaveQ TD(lambda) `0.6`; RiskQ 32 quantiles
  with Wang risk parameter `0.75`; QFree transform/mixer hidden size 64;
  ConcaveQ concave hidden sizes `(64,64,64)` and unrestricted estimator hidden
  sizes `(64,64)`.

### FOCUS settings

- Credit coefficient `eta=0.1`; belief-loss coefficient `0.1`; belief learning
  rate `1e-4`.
- Three-step causal belief horizon with discount `0.7`; hidden size 256.
- Categorical position distribution with a 21-bin grid, maximum delta 400,
  and auxiliary position-loss coefficient `0.1`.
- Safeguard mode: zero initialization only. EMA gating and progressive
  blending were disabled in these runs.
- Future states are supervision labels only. They are not passed to action
  selection or evaluation-time execution.

### CTCE runs

- HMVFE/HiTMAC: four vector environments, CUDA, 250k steps, evaluation every
  10 updates with 10 episodes, `gamma=0.99`, GAE(lambda) `0.95`, learning rate
  `5e-4`, entropy coefficient `0.01`, value coefficient `0.5`, and gradient
  clipping 50.
- For +FOCUS, the causal current-state belief is used during both training and
  evaluation. Future visibility/position is used only as the supervised belief
  target.

## Evaluation coverage on MATE-4v8-9

`--` means that no valid evaluation exists at that milestone. “Mean eval” is
the arithmetic mean of all scheduled evaluation checkpoints in that run, not
the best checkpoint. Values are percentages.

| Algorithm | Variant | ~50k | ~100k | ~150k | ~200k | ~250k | Mean eval | Run status |
|---|---|---:|---:|---:|---:|---:|---:|---|
| FACMAC | Backbone | 52.11 (@54.1k) | 59.09 (@80.6k) | -- | -- | -- | 53.63 | 102,483-step pilot |
| FACMAC | +FOCUS | 50.00 (@53.6k) | 49.74 (@80.3k) | -- | -- | -- | 50.28 | 101,813-step pilot |
| ResQ | Backbone | 45.69 (@53.5k) | 51.71 (@107.2k) | 54.71 (@160.4k) | 53.12 (@214.3k) | -- | 51.31 | Complete, 252,225 steps |
| ResQ | +FOCUS | 52.95 (@53.8k) | 49.92 (@107.7k) | 59.60 (@161.5k) | 52.62 (@215.9k) | -- | 53.77 | Complete, 253,355 steps |
| RiskQ | Backbone | 52.89 (@54.1k) | 56.46 (@107.8k) | 56.76 (@161.7k) | 55.20 (@215.9k) | -- | 55.33 | Complete, 253,274 steps |
| RiskQ | +FOCUS | 53.14 (@53.7k) | 57.01 (@107.2k) | 61.50 (@161.0k) | 60.59 (@215.4k) | -- | 58.06 | Complete, 253,370 steps |
| QFree | Backbone | 53.05 (@53.9k) | 54.87 (@107.8k) | 56.10 (@161.0k) | 58.10 (@215.1k) | -- | 55.53 | Complete, 252,912 steps |
| QFree | +FOCUS | 59.40 (@53.4k) | 54.09 (@107.3k) | 52.68 (@161.0k) | 47.63 (@215.9k) | -- | 53.45 | Complete, 253,320 steps; CPU retry |
| ConcaveQ | Backbone | 40.02 (@53.1k) | 35.95 (@106.8k) | -- | -- | -- | 37.98 | Stopped at 149,732 steps |
| ConcaveQ | +FOCUS | -- | -- | -- | -- | -- | -- | Not run to evaluation |
| HMVFE | Backbone | 52.14 (@48k) | 41.53 (@100k) | 41.94 (@148k) | 45.72 (@200k) | 44.51 (@248k) | 45.49 | Complete, 250,000 steps |
| HMVFE | +FOCUS | 17.64 (@48k) | 17.64 (@100k) | 17.64 (@148k) | 17.64 (@200k) | 17.64 (@248k) | 21.28 | Complete, 250,000 steps |
| HiTMAC | Backbone | 59.52 (@48k) | 59.52 (@100k) | 59.52 (@148k) | 22.26 (@200k) | 17.64 (@248k) | 48.81 | Complete, 250,000 steps |
| HiTMAC | +FOCUS | 59.52 (@48k) | 59.52 (@100k) | 56.41 (@148k) | 19.47 (@200k) | 17.64 (@248k) | 48.84 | Complete, 250,000 steps |

## Honest summary

- RiskQ+FOCUS is the clearest positive result: mean evaluation improves by
  2.73 percentage points and the last scheduled evaluation improves by 5.39
  points.
- ResQ+FOCUS improves mean evaluation by 2.46 points, but its last evaluation
  is 0.50 points below the backbone. This is mixed rather than a consistent
  win.
- QFree+FOCUS is 2.08 points worse on mean evaluation and 10.47 points worse
  at the last evaluation.
- HMVFE+FOCUS collapses relative to HMVFE. HiTMAC+FOCUS is effectively tied on
  overall mean and does not improve the final evaluation; both HiTMAC runs
  collapse late.
- FACMAC is only a 100k pilot and FOCUS is worse by 3.35 points on the mean of
  its three evaluations. ConcaveQ has no valid backbone/+FOCUS comparison.
- These are single-seed engineering results. They are sufficient for smoke and
  trend validation, but not evidence of statistically reliable reproduction.
