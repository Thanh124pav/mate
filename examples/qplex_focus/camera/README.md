# QPLEX_V2 + FOCUS-learned Camera Example

This is the low-level discrete-camera version of QPLEX_V2+FOCUS-learned.

Because the primitive camera action is not a target-selection bitmask, this
example sets `use_action_selection=False`. The HRL example in
`examples/hrl/qplex_focus/camera` is the closer match to the FOCUS credit
target in `PLAN.md`.

The occupancy belief is learned with `belief_mode='learned'` and `horizon=3`.
The model predicts a Gaussian for each future high-level step and is trained by
Gaussian NLL on centralized future target positions.

Run:

```bash
python -m examples.qplex_focus.camera.train
```
