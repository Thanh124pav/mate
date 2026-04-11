"""SMPE2 environment wrapper: adds count-based intrinsic exploration rewards."""

import hashlib

import gym
import numpy as np

import mate


class SMPE2Wrapper(gym.Wrapper, metaclass=mate.WrapperMeta):
    """Adds intrinsic exploration rewards based on observation novelty.

    Uses SimHash on observations to compute count-based intrinsic rewards:
        r_intrinsic = 1 / sqrt(count(hash(obs)))
        r_total = r_extrinsic + beta * r_intrinsic

    This encourages agents to visit novel states, improving exploration
    in sparse-reward cooperative tasks.
    """

    def __init__(self, env, intrinsic_coeff=0.1, hash_dim=32, decay=0.999):
        assert isinstance(env, mate.MultiCamera), (
            f'{self.__class__.__name__} must wrap mate.MultiCamera. Got env = {env}.'
        )
        super().__init__(env)
        self.intrinsic_coeff = intrinsic_coeff
        self.hash_dim = hash_dim
        self.decay = decay

        # Random projection matrix for SimHash
        obs_dim = env.observation_space[0].shape[0]
        self._projection = np.random.randn(obs_dim, hash_dim).astype(np.float32)

        # Visit counts
        self._visit_counts = {}
        self._step_count = 0

    def reset(self, **kwargs):
        # Decay visit counts periodically to allow re-exploration
        if self._step_count > 0 and self.decay < 1.0:
            for k in self._visit_counts:
                self._visit_counts[k] *= self.decay
        return self.env.reset(**kwargs)

    def step(self, action):
        observations, rewards, dones, infos = self.env.step(action)
        self._step_count += 1

        # Add intrinsic rewards per camera
        rewards = np.array(rewards, dtype=np.float64)
        for c in range(self.num_cameras):
            hash_val = self._simhash(observations[c])
            self._visit_counts[hash_val] = self._visit_counts.get(hash_val, 0) + 1
            count = self._visit_counts[hash_val]
            intrinsic = 1.0 / np.sqrt(max(count, 1.0))
            rewards[c] += self.intrinsic_coeff * intrinsic
            infos[c]['intrinsic_reward'] = self.intrinsic_coeff * intrinsic

        return observations, rewards.tolist(), dones, infos

    def _simhash(self, obs):
        """Hash observation via random projection + binarization."""
        projected = obs.ravel().astype(np.float32) @ self._projection
        bits = (projected > 0).astype(np.uint8)
        return hashlib.md5(bits.tobytes()).hexdigest()
