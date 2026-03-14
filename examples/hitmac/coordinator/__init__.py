"""HiTMAC Phase 2A — Coordinator training with fixed Phase 1 QPLEX Executor.

Mỗi thuật toán có thư mục riêng (mirror cấu trúc examples/hrl/):

  MAPPO coordinator:
      python -m examples.hitmac.coordinator.mappo.camera.train \\
          --executor-checkpoint examples/hitmac/camera/ray_results/HiTMAC-QPLEX/latest-checkpoint

  QPLEX_V2 coordinator:
      python -m examples.hitmac.coordinator.qplex_v2.camera.train \\
          --executor-checkpoint examples/hitmac/camera/ray_results/HiTMAC-QPLEX/latest-checkpoint
"""
