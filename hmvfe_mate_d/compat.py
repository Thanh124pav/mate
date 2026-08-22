"""Compatibility shims for the pinned dependency versions.

``gym==0.23.1`` ships a buggy ``RandomNumberGenerator._generator_ctor``: its
``__reduce__`` emits a two-element ``init_args`` tuple (``(name, ctor)``) but the
``_generator_ctor`` it points to only accepts a single positional argument. As a
result ``copy.deepcopy`` of any gym RNG raises::

    TypeError: _generator_ctor() takes from 0 to 1 positional arguments but 2 were given

MATE clones every rule-based agent via ``copy.deepcopy`` (``AgentBase.clone``),
so this breaks ``MultiCamera`` / agent spawning outright. We replace the static
method with a tolerant version that ignores the extra argument.

Importing :mod:`hmvfe_mate_d` applies the patch automatically (idempotent).
"""

from __future__ import annotations


_PATCHED = False


def patch_gym_seeding() -> None:
    """Make ``gym.utils.seeding.RandomNumberGenerator`` deep-copyable."""

    global _PATCHED  # pylint: disable=global-statement
    if _PATCHED:
        return

    try:
        from gym.utils import seeding
    except Exception:  # pylint: disable=broad-except
        return

    rng_cls = getattr(seeding, 'RandomNumberGenerator', None)
    if rng_cls is None:
        _PATCHED = True
        return

    def _generator_ctor(bit_generator_name='MT19937', *_args, **_kwargs):
        from numpy.random._pickle import BitGenerators

        if bit_generator_name not in BitGenerators:
            raise ValueError(f'{bit_generator_name} is not a known BitGenerator module.')
        return rng_cls(BitGenerators[bit_generator_name]())

    rng_cls._generator_ctor = staticmethod(_generator_ctor)
    _PATCHED = True


patch_gym_seeding()
