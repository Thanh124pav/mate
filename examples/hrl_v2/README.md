# Hierarchical Reinforcement Learning V2 (HRL V2)

## Overview

HRL V2 là một kiến trúc học tăng cường phân cấp **ngược lại** so với HRL gốc:

### So sánh HRL vs HRL V2

| Aspect | HRL (Original) | HRL V2 (New) |
|--------|----------------|--------------|
| **High-level Policy** | **Learned** (QPLEX/MAPPO) | **Hard-coded** (Greedy) |
| **High-level Task** | Target Selection | Target Assignment |
| **Low-level Policy** | **Hard-coded** (Geometric) | **Learned** (RL) |
| **Low-level Task** | Camera Control | Camera Control |
| **Action Space** | Discrete (target selection) | Continuous (camera angles) |
| **Complexity** | High-level decisions | Low-level control |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      HRL V2 ARCHITECTURE                     │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────────────────────────────────────────────┐  │
│  │         HIGH-LEVEL: Target Assignment                │  │
│  │                  (Hard-coded)                        │  │
│  ├──────────────────────────────────────────────────────┤  │
│  │  Input:  Global state (cameras + targets)           │  │
│  │  Output: Assignment matrix [cameras × targets]      │  │
│  │  Method: Greedy (distance/coverage/optimal)         │  │
│  │                                                      │  │
│  │  Options:                                            │  │
│  │  - GreedyDistanceAssigner: nearest target           │  │
│  │  - GreedyCoverageAssigner: maximize coverage        │  │
│  │  - LearnedAssigner: RL-based (future)               │  │
│  └──────────────────────────────────────────────────────┘  │
│                          ↓                                  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │         LOW-LEVEL: Camera Control                    │  │
│  │              (Learned via RL)                        │  │
│  ├──────────────────────────────────────────────────────┤  │
│  │  Input:  Camera obs + Assigned targets              │  │
│  │  Output: Camera actions [Δorientation, Δangle]      │  │
│  │  Method: RL (QPLEX, MAPPO, A3C, etc.)               │  │
│  │                                                      │  │
│  │  Observation:                                        │  │
│  │  {                                                   │  │
│  │    'obs': camera_observation,                       │  │
│  │    'assignment': [0, 1, 0, 1, 0]  # binary vector   │  │
│  │  }                                                   │  │
│  │                                                      │  │
│  │  Action: Continuous control                         │  │
│  │  [Δorientation, Δviewing_angle]                     │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Rationale

### Why HRL V2?

1. **Target Assignment is Easier to Hard-code**
   - Geometric algorithms work well (greedy distance, coverage optimization)
   - Well-studied problem with known solutions
   - Fast computation, no learning required

2. **Camera Control is Harder to Hard-code**
   - Complex dynamics (orientation + viewing angle)
   - Need to adapt to target movements
   - Constraints (FOV, range limits, obstacles)
   - Benefits from learning optimal control policies

3. **Complementary to Original HRL**
   - HRL: Good when target selection is strategic
   - HRL V2: Good when camera control is challenging
   - Can combine both approaches for ultimate flexibility

---

## Directory Structure

```
examples/hrl_v2/
├── __init__.py              # Package initialization
├── high_level.py            # High-level assignment strategies
├── wrappers.py              # HierarchicalCameraV2 wrapper
│
├── qplex/                   # QPLEX low-level learning
│   ├── __init__.py
│   └── camera/
│       ├── __init__.py
│       ├── __main__.py      # Evaluation script
│       ├── agent.py         # QPLEX agent implementation
│       ├── config.py        # Training configuration
│       └── train.py         # Training script
│
├── mappo/                   # MAPPO low-level learning
│   └── camera/
│       └── ...              # Similar structure
│
└── a3c/                     # A3C low-level learning
    └── camera/
        └── ...              # Similar structure
```

---

## Usage

### Training

Train low-level camera control policy using QPLEX:

```bash
python -m examples.hrl_v2.qplex.camera.train \
    --num-workers 14 \
    --num-envs-per-worker 8 \
    --timesteps-total 10000000
```

### Evaluation

Evaluate trained agent:

```bash
python -m examples.hrl_v2.qplex.camera \
    --checkpoint-path examples/hrl_v2/qplex/camera/ray_results/HRLv2-QPLEX-LowLevel/latest-checkpoint \
    --episodes 10 \
    --render
```

### Custom High-level Assigner

You can easily swap assignment strategies:

```python
from examples.hrl_v2.high_level import GreedyCoverageAssigner
from examples.hrl_v2.wrappers import HierarchicalCameraV2

# Use coverage-based assignment
env = HierarchicalCameraV2(
    base_env,
    assigner_class=GreedyCoverageAssigner,
    frame_skip=5,
)
```

---

## High-level Assignment Strategies

### 1. GreedyDistanceAssigner

Each camera assigned to nearest visible target.

**Pros:**
- Simple, fast
- Works well for distributed targets

**Cons:**
- May leave some targets uncovered
- No global optimization

### 2. GreedyCoverageAssigner

Iteratively assign to maximize coverage.

**Pros:**
- Better coverage distribution
- Considers global state

**Cons:**
- Slightly more computation
- Still greedy (not optimal)

### 3. MaxCoverageAssigner

Optimal assignment (can use Hungarian algorithm).

**Pros:**
- Theoretically optimal
- Best coverage

**Cons:**
- Higher computational cost
- May be overkill for simple scenarios

### 4. LearnedAssigner (Future)

Learn assignment policy via RL.

**Pros:**
- Can learn complex strategies
- Adapts to specific scenarios

**Cons:**
- Requires training
- More complex to implement

---

## Low-level Learning Algorithms

### Implemented

- **QPLEX**: Value decomposition with attention mechanism
  - Good for cooperative scenarios
  - Learns credit assignment

### To Implement

- **MAPPO**: Multi-Agent PPO
  - On-policy, stable
  - Good sample efficiency

- **A3C**: Asynchronous Actor-Critic
  - Fast, distributed
  - Good for continuous control

---

## Observation Space

Low-level policy receives augmented observations:

```python
{
    'obs': np.ndarray,        # Camera observation [obs_size]
                              # - Camera state (position, orientation, FOV)
                              # - Target positions (relative)
                              # - Obstacles
    
    'assignment': np.ndarray  # Binary assignment [num_targets]
                              # - 1 if target assigned, 0 otherwise
                              # - Tells policy which targets to track
}
```

---

## Action Space

Continuous camera control:

```python
action = [Δorientation, Δviewing_angle]

# Δorientation: Change in camera orientation (degrees)
# Δviewing_angle: Change in FOV (degrees)
```

---

## Reward Design

Same as original MATE environment:

- **Coverage Rate**: Percentage of targets being tracked
- **Dense Reward**: Incremental rewards for maintaining coverage
- **Shared Reward**: All cameras receive same reward (cooperation)

---

## Future Extensions

### 1. Hybrid Approach

Combine learned high-level + learned low-level:

```python
env = HierarchicalCameraV2(
    base_env,
    assigner_class=LearnedAssigner,  # Learn assignment
    assigner_kwargs={'checkpoint': 'path/to/hl_checkpoint'},
)
```

### 2. Meta-Learning

Learn to switch between assignment strategies:

- Easy scenarios: Greedy
- Complex scenarios: Learned
- Meta-policy decides which to use

### 3. Curriculum Learning

Progressive training:
1. Start with greedy assignment (easier)
2. Gradually introduce learned assignment
3. End-to-end fine-tuning

---

## Comparison with Original HRL

| Metric | HRL | HRL V2 |
|--------|-----|--------|
| Training Time | Medium | Medium |
| Sample Efficiency | High (discrete actions) | Medium (continuous) |
| Final Performance | Good | Good |
| Interpretability | High (geometric control) | Medium (learned control) |
| Flexibility | Medium | High (can swap assigners) |
| Transferability | High | High |

---

## Key Design Principles

1. **Modularity**: Easy to swap high-level assigners
2. **Simplicity**: Start with hard-coded, add learning later
3. **Flexibility**: Support multiple low-level algorithms
4. **Compatibility**: Works with existing MATE infrastructure
5. **Extensibility**: Easy to add new components

---

## Citation

If you use HRL V2 in your research, please cite:

```bibtex
@misc{mate_hrl_v2,
  title={Hierarchical Reinforcement Learning V2 for Multi-Agent Camera Tracking},
  author={Your Name},
  year={2026},
}
```

---

## License

Same as MATE project.

---

## Contact

For questions and suggestions, please open an issue on GitHub.
