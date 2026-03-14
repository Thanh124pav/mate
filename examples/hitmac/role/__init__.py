"""HiTMAC Phase 2B — Role-Based Coordinator.

1 camera được chỉ định làm coordinator (single-agent PPO), n-1 cameras còn lại
là fixed QPLEX executors từ Phase 1.

Train:
    python -m examples.hitmac.role.camera.train \\
        --executor-checkpoint examples/hitmac/camera/ray_results/HiTMAC-QPLEX/latest-checkpoint
"""
