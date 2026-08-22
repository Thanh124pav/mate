# HMVFE coordinator for MATE — variant D (Tier B: configurable critic)

Same algorithm and same MATE-extended 7-field observation as
[`hmvfe_mate_b/`](../hmvfe_mate_b), but the **state-value critic reduction is
configurable**. This is the "Tier B" experiment: the paper fixes the critic to a
parameter-free `max` over the per-pair interaction scores `z`; that is a lossy,
high-variance baseline that shares *all* parameters with the actor (pushing the
value up saturates one pair's sigmoid), which can distort the policy. Variant D
lets you swap it.

## `critic_reduction` (config / `--critic-reduction`)

| value | v^H = | notes |
|---|---|---|
| `max` | `max_ij(z_ij)` | paper default (Sec. 4.1.3), parameter-free, actor-coupled — **reproduces `hmvfe_mate_b` exactly** |
| `mean` | `mean_ij(z_ij)` | parameter-free, smoother baseline, still actor-coupled |
| `learned` (**default**) | `ValueHead(pool(e*, u*))` | a small MLP value head on the shared trunk (mean-pooled reweighted field embeddings), **decoupled** from the actor's `z` → lower-variance advantage baseline |

The learned head shares the embedding/MoE/FM **trunk** (so `value_coef` still
trains the representation) but has its **own output parameters**, so the value
target no longer forces the actor's `z` up. Input dim = `num_fields·d + num_fields`
(= 7·10 + 7 = 77 by default), MLP → `value_head_hidden` (128) → 1.

Everything else is identical to `hmvfe_mate_b` (7-field obs, `value_coef=0.5`,
fixed entropy, angle span 180°, `num_envs=8`, `gamma=0.99`, frozen executor,
~10M budget). No reward/executor/env change → still information-fair.

> Note: `critic_reduction` (Tier B) is orthogonal to `hmvfe_mate_c`'s Tier A
> tuning (entropy anneal + `value_coef=0.25` + angle span 90°). They can be
> combined later if both help.

## Run

Uses the repo-root `.venv`. **Default = learned critic**:

```bash
cd /Users/tranmanhcuong253/Workspace/Research/mate

./.venv/bin/python -m hmvfe_mate_d \
    --env-config MATE-4v8-9.yaml --seed 0 --total-env-steps 10000000 \
    --wandb-project mate-hmvfe --wandb-group benchmark-4v8-9 \
    --run-name hmvfe-d-seed0 --log-interval 10
```

Ablate the critic (same code, one flag):
```bash
./.venv/bin/python -m hmvfe_mate_d ... --critic-reduction max    # == hmvfe_mate_b
./.venv/bin/python -m hmvfe_mate_d ... --critic-reduction mean
./.venv/bin/python -m hmvfe_mate_d ... --critic-reduction learned  # default
```

Smoke test:
```bash
./.venv/bin/python -m hmvfe_mate_d \
    --total-env-steps 5000 --num-envs 4 --rollout-length 8 \
    --log-interval 1 --eval-interval 5 --eval-episodes 2 --wandb-mode disabled
```

## Suggested benchmark
Short A/B (1–2M) at matched seeds to isolate the critic choice:
- `--critic-reduction max`  → `hmvfe-d-max-seed{0,1,2}`  (= variant B baseline)
- `--critic-reduction mean` → `hmvfe-d-mean-seed{0,1,2}`
- `--critic-reduction learned` → `hmvfe-d-learned-seed{0,1,2}`

Curves overlay on `train/episode_coverage_rate` vs `environment_steps`. Target:
a smoother, faster climb past all-select ≈ 0.62 than `max`.

## Not included (possible next step)
Return / value-target normalization (PopArt-style) — the learned head already
absorbs the return scale, so it is deferred; add it only if the learned critic
still struggles with the large summed-reward returns.
