"""Coordinator-level environment adapter for HiT-MAC on MATE.

``MATECoordinatorEnv`` plays the role that ``HierarchicalCamera`` +
``RLlibMultiAgentAPI`` play for the other HRL baselines, but collapsed into one
ray-free wrapper:

  * it consumes the camera team's joint observation (after ``MultiCamera ->
    RepeatedRewardIndividualDone -> AuxiliaryCameraRewards``),
  * exposes a target-centric ``[N_cam, N_tgt, F]`` observation and a single joint
    ``MultiBinary(N_cam * N_tgt)`` selection action,
  * runs the *vendored* geometric executor (identical to ``HierarchicalCamera``)
    for ``frame_skip`` low-level steps per coordinator decision,
  * returns a single shared scalar reward (mean coverage over the macro-step).

The macro-step loop mirrors ``HierarchicalCamera.step``: per low-level frame it
recomputes each camera's visible-target mask *from the observation*, tracks the
selected-and-visible targets, sums the per-camera reward fragments, and stops
early if the episode ends.
"""

from __future__ import annotations

from typing import Optional

import gym
import numpy as np
from gym import spaces

from hmvfe_mate_d.executor import joint_executor
from hmvfe_mate_d.observations import DiscretizedCoordinatorObservationBuilder


__all__ = ['MATECoordinatorEnv']


class MATECoordinatorEnv(gym.Env):
    """Single-agent coordinator view over a camera-team MATE env."""

    metadata = {'render.modes': ['human', 'rgb_array']}

    # info keys averaged over the macro-step fragment for logging
    _FRAGMENT_KEYS = ('coverage_rate', 'real_coverage_rate', 'mean_transport_rate', 'num_tracked')
    # info keys reported as their last value in the fragment
    _LAST_KEYS = ('num_delivered_cargoes',)

    def __init__(
        self,
        env,
        frame_skip: int = 5,
        horizon: int = 500,
        num_distance_bins: int = 16,
        num_angle_bins: int = 16,
        num_occlusion_bins: int = 8,
    ) -> None:
        import mate

        assert isinstance(env, mate.MultiCamera), (
            'MATECoordinatorEnv expects a single-team camera env (MultiCamera + '
            f'RepeatedRewardIndividualDone + AuxiliaryCameraRewards). Got env = {env!r}.'
        )
        super().__init__()

        self.env = env
        self.base_env = env.unwrapped
        self.num_cameras = env.num_cameras
        self.num_targets = env.num_targets
        self.num_obstacles = self.base_env.num_obstacles
        self.frame_skip = int(frame_skip)
        # Truncate episodes at the same env-step budget as the HRL baselines
        # (RLlib `horizon`), so episode length matches across algorithms.
        self.horizon = int(horizon)
        self._elapsed_env_steps = 0

        self.observation_builder = DiscretizedCoordinatorObservationBuilder(
            self.num_cameras,
            self.num_targets,
            self.num_obstacles,
            num_distance_bins=num_distance_bins,
            num_angle_bins=num_angle_bins,
            num_occlusion_bins=num_occlusion_bins,
        )
        self.num_fields = self.observation_builder.num_fields
        self.table_size = self.observation_builder.table_size
        # kept for backward-compat with the scaffold (not a continuous feature dim)
        self.feature_dim = self.num_fields

        self.observation_space = spaces.Box(
            low=0.0,
            high=float(self.table_size),
            shape=(self.num_cameras, self.num_targets, self.num_fields),
            dtype=np.float32,
        )
        self.action_space = spaces.MultiBinary(self.num_cameras * self.num_targets)

        self.target_view_mask_slice = self.observation_builder.slices['opponent_mask']
        self._joint_observation = None

    # -- gym API ---------------------------------------------------------------

    def seed(self, seed: Optional[int] = None):
        return self.env.seed(seed)

    def reset(self, **kwargs) -> np.ndarray:
        self._joint_observation = self.env.reset(**kwargs)
        self._elapsed_env_steps = 0
        return self._build_observation()

    def step(self, action):
        selection = np.asarray(action, dtype=np.int64).reshape(self.num_cameras, self.num_targets)
        selection = selection.astype(np.bool8)

        observations = self._joint_observation
        fragment_rewards = []
        fragment_infos = []
        dones = [False] * self.num_cameras

        for _ in range(self.frame_skip):
            view_mask = self._view_mask(observations)
            primitive = joint_executor(
                selection, view_mask, self.base_env.cameras, self.base_env.targets
            )
            observations, rewards, dones, infos = self.env.step(primitive)
            fragment_rewards.append(np.asarray(rewards, dtype=np.float64))
            fragment_infos.append(infos)
            if np.all(dones):
                break

        self._joint_observation = observations
        self._elapsed_env_steps += len(fragment_rewards)

        # cameras share the (mean-reduced) coverage reward; sum over the fragment
        # (matches HierarchicalCamera), then mean over cameras -> cooperative scalar.
        per_camera_return = np.sum(fragment_rewards, axis=0)
        reward = float(np.mean(per_camera_return))
        done = bool(np.all(dones)) or self._elapsed_env_steps >= self.horizon
        info = self._summarize(fragment_infos)

        return self._build_observation(), reward, done, info

    def render(self, mode='human'):
        return self.env.render(mode=mode)

    def close(self):
        return self.env.close()

    def global_state(self) -> np.ndarray:
        """Return the normalized privileged state for centralized training.

        The public observation remains the discretized partial view.  This
        method is intentionally separate so callers can make the CTDE boundary
        explicit: the actor never receives this value during action selection,
        while the critic/belief target may use it during training.
        """
        import mate

        state = self.base_env.state()
        return np.asarray(
            mate.normalize_observation(state, self.base_env.state_space),
            dtype=np.float32,
        )

    # -- helpers ---------------------------------------------------------------

    def _view_mask(self, observations) -> np.ndarray:
        joint = np.asarray(observations, dtype=np.float64)
        mask = joint[:, self.target_view_mask_slice].astype(np.bool8)
        return mask.reshape(self.num_cameras, self.num_targets)

    def action_masks(self) -> np.ndarray:
        """Per-(camera, target) visibility mask from the observation only."""

        return self._view_mask(self._joint_observation)

    def _build_observation(self) -> np.ndarray:
        return self.observation_builder.build(self._joint_observation)

    def _summarize(self, fragment_infos) -> dict:
        info = {}
        for key in self._FRAGMENT_KEYS:
            values = [
                np.mean([i[key] for i in step_infos if key in i])
                for step_infos in fragment_infos
                if any(key in i for i in step_infos)
            ]
            if values:
                info[key] = float(np.mean(values))
        for key in self._LAST_KEYS:
            last = fragment_infos[-1]
            values = [i[key] for i in last if key in i]
            if values:
                info[key] = float(np.mean(values))
        return info
