# QPLEX_V2 + FOCUS-learned

FOCUS adds a learned future-occupancy credit regularizer to QPLEX_V2 for
sparse-visibility multi-camera tracking.

For each training transition, the policy computes:

- QPLEX TD loss for `Q_tot`;
- a learned Gaussian occupancy belief for next target positions;
- a centralized predictive credit target `rho_i` computed from that belief;
- a cross-entropy loss that aligns normalized QPLEX `lambda_i` with `rho_i`.

The default implementation uses `belief_mode='learned'` and `horizon=3`. The
occupancy model predicts a Gaussian for each high-level future step
`h in {1, ..., H}` and is trained with Gaussian NLL on centralized future target
positions. In the HRL camera wrapper, one high-level step corresponds to
`frame_skip` primitive environment steps, so with `frame_skip=5` and `H=3`, the
belief targets are approximately `t+5`, `t+10`, and `t+15` primitive steps.
FOCUS approximates the predictive credit integral with deterministic sigma
points from these learned Gaussians and horizon weights `omega_h`.

Run:

```bash
python -m examples.hrl.qplex_focus.camera.train
```

Useful training stats:

- `td_loss`
- `focus_credit_loss`
- `focus_belief_loss`
- `focus_valid_ratio`
- `focus_mean_signal`
- `focus_rho_entropy`
- `focus_lambda_entropy`
