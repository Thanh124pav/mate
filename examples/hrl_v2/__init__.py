"""
Hierarchical Reinforcement Learning V2 for Multi-Agent Camera Tracking.

Architecture:
- High-level policy: Hard-coded greedy target assignment (can be replaced with learned policy)
- Low-level policy: Learned camera control using RL algorithms (QPLEX, MAPPO, A3C, etc.)

This is the inverse of the original HRL approach where:
- Original HRL: High-level learned (target selection), Low-level hard-coded (geometric control)
- HRL V2: High-level hard-coded (target assignment), Low-level learned (camera control)
"""

from examples.hrl_v2 import high_level, wrappers
from examples.hrl_v2.high_level import *
from examples.hrl_v2.wrappers import *

__all__ = high_level.__all__ + wrappers.__all__
