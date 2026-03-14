"""HiTMAC: Hierarchical Task MAC with QPLEX Executors for the MATE environment.

Theo paper (NeurIPS 2020, Sec 3.3 "Training Strategy"):

    Phase 1 — Train Executors (THIS PACKAGE):
        Dùng greedy heuristic thay coordinator. Executor (QPLEX) học track assigned targets.
        → python -m examples.hitmac.camera.train

    Phase 2 — Train Coordinator (reuse examples/hrl/mappo/):
        Executor được fix (geometric scripted executor = HierarchicalCamera.executor()).
        Coordinator (MAPPO) học assign targets tối ưu.
        → python -m examples.hrl.mappo.camera.train

Reference:
    Xu et al., "Learning Multi-Agent Coordination for Enhancing Target Coverage
    in Directional Sensor Networks", NeurIPS 2020.
    https://arxiv.org/abs/2010.13110
"""

from examples.hitmac import wrappers
from examples.hitmac.wrappers import (
    HiTMACWrapper,
    HiTMACCoordinatorWrapper,
    DiscreteCoordinatorSelection,
    HiTMACRoleWrapper,
)
