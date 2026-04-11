"""HiTMAC v2: Hierarchical Task MAC with PPO Executors.

Matches the original paper (NeurIPS 2020, Xu et al.) by using on-policy
actor-critic (PPO/MAPPO) instead of QPLEX for executors.

Phase 1 — Train Executors:
    python -m examples.hitmac_v2.camera.train

Phase 2 — Train Coordinator (reuse existing HRL-MAPPO):
    python -m examples.hrl.mappo.camera.train
"""
