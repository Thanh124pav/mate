"""HMVFE coordinator on the MATE environment (ray-free, top-level package).

Hierarchical Mixture of Vital Feature Experts (Son et al., JNCA 2026) plugged in
as the HiT-MAC *coordinator* replacement: a discretised look-up embedding +
Mixture-of-Experts "Vital Feature" reweighting + Factorization Machine head over
each (camera, target) pair, with a sigmoid per-pair selection actor and a
parameter-free max critic, trained with n-step actor-critic. It reuses the
proven ``hit_mac/`` MATE scaffold (env adapter, frozen geometric executor, A2C +
GAE trainer) so HMVFE is benchmarked *fairly* against the other MATE camera
algorithms -- only the coordinator network differs.

**Variant D (variant B + Tier B: configurable critic):** keeps ``hmvfe_mate_b``'s
7-field MATE-extended observation, but makes the state-value critic configurable
instead of the paper's fixed parameter-free ``max`` over interaction scores:
``critic_reduction`` in {``max`` (paper), ``mean``, ``learned`` (default)}. The
``learned`` option adds a small value head on the shared trunk (mean-pooled
reweighted field embeddings), **decoupled from the actor's per-pair scores**, to
give a lower-variance advantage baseline than ``max`` (which is lossy and, being
tied to the actor's ``z``, distorts the policy when its value is pushed up). The
paper reports ``max`` best on its own DSN env; this variant tests that on MATE.
Everything else = variant B (7-field obs, no Tier A tuning), so ``max`` here
reproduces ``hmvfe_mate_b`` exactly -> clean isolation of the critic choice.

It is a top-level package (not under ``examples/``) on purpose: importing
anything under ``examples`` pulls in RLlib/Ray via ``examples.utils``, whereas
this integration depends only on ``mate`` + ``torch``. The fairness-critical
low-level executor is vendored verbatim from ``examples/hrl/wrappers.py``
(see ``executor.py``); HMVFE's learned FM worker is intentionally dropped.

See ``README.md`` for the design rationale and fairness checklist.
"""

# Apply dependency compatibility shims (e.g. gym 0.23.1 RNG deepcopy) before
# anything touches mate / gym.
from hmvfe_mate_d import compat as _compat  # noqa: F401  (import for side effect)

from hmvfe_mate_d.config import HMVFEConfig, make_env
from hmvfe_mate_d.envs import MATECoordinatorEnv
from hmvfe_mate_d.models import HMVFECoordinator
from hmvfe_mate_d.observations import DiscretizedCoordinatorObservationBuilder


__all__ = [
    'HMVFEConfig',
    'make_env',
    'MATECoordinatorEnv',
    'HMVFECoordinator',
    'DiscretizedCoordinatorObservationBuilder',
]
