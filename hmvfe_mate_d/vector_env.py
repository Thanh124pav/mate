"""Synchronous vector of :class:`MATECoordinatorEnv` for batched on-policy A2C.

The original HiT-MAC decorrelated its on-policy data with several asynchronous
A3C workers. This single-process trainer instead steps ``num_envs`` independent
MATE coordinator envs *synchronously* (one ``step`` on each per macro-step) and
stacks their transitions, giving the learner a larger, decorrelated batch per
update -> much lower-variance gradients (the main fix for the jagged/unstable
curve). It does NOT speed up wall-clock for a fixed env-step budget (env
stepping is the bottleneck and the total number of steps is unchanged); the win
is gradient quality / stability.

Each sub-env gets a distinct seed (same task distribution, different layouts)
and, on the first reset, is advanced by a staggered offset so the envs do not
all hit the fixed horizon on the same macro-step (which would create
synchronized done-spikes). Sub-envs auto-reset on ``done`` and return the new
episode's first observation, matching the single-env trainer's reset-in-rollout
behaviour (GAE cuts the trace via the ``done`` flag).
"""

from __future__ import annotations

from typing import List

import numpy as np


__all__ = ['SyncVectorCoordinatorEnv']


class SyncVectorCoordinatorEnv:
    """Runs ``num_envs`` MATECoordinatorEnv instances in lock-step (one process)."""

    def __init__(self, config, num_envs: int, stagger: bool = True) -> None:
        from hmvfe_mate_d.config import make_env

        self.num_envs = int(num_envs)
        self.base_seed = int(config.seed)
        self.envs = []
        for i in range(self.num_envs):
            env = make_env(config)
            env.seed(self.base_seed + 1000 * (i + 1))  # distinct layouts per env
            self.envs.append(env)

        first = self.envs[0]
        self.feature_dim = first.feature_dim
        self.num_fields = first.num_fields
        self.table_size = first.table_size
        self.num_cameras = first.num_cameras
        self.num_targets = first.num_targets

        self._stagger = bool(stagger)
        self._episode_macro = max(1, int(config.horizon) // int(config.frame_skip))
        self._rng = np.random.RandomState(self.base_seed)

    def reset(self) -> np.ndarray:
        obs = [env.reset() for env in self.envs]
        if self._stagger and self.num_envs > 1:
            block = max(1, self._episode_macro // self.num_envs)
            for i, env in enumerate(self.envs):
                for _ in range(i * block):  # warm-up with random actions to desync phases
                    a = self._rng.randint(0, 2, size=(self.num_cameras, self.num_targets))
                    o, _, d, _ = env.step(a)
                    obs[i] = env.reset() if d else o
        return np.stack(obs, axis=0)  # [num_envs, N_cam, N_tgt, F]

    def step(self, actions):
        """``actions``: sequence of ``num_envs`` arrays shaped ``[N_cam, N_tgt]``."""

        obs_list: List[np.ndarray] = []
        rewards = np.zeros(self.num_envs, dtype=np.float64)
        dones = np.zeros(self.num_envs, dtype=bool)
        infos = []
        for i, env in enumerate(self.envs):
            o, r, d, info = env.step(actions[i])
            if d:
                o = env.reset()  # auto-reset; info still describes the finished episode
            obs_list.append(o)
            rewards[i] = r
            dones[i] = bool(d)
            infos.append(info)
        return np.stack(obs_list, axis=0), rewards, dones, infos

    def seed(self, seed: int) -> None:
        for i, env in enumerate(self.envs):
            env.seed(int(seed) + 1000 * (i + 1))

    def close(self) -> None:
        for env in self.envs:
            env.close()
