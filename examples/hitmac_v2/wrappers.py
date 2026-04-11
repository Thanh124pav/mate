"""HiTMAC v2 wrapper — PPO-compatible hierarchical task assignment.

Same hierarchical logic as HiTMAC (coordinator + executor) but designed for
PPO/MAPPO multi-agent training (per-agent API, no grouped agents).

Phase 1: HiTMACv2Wrapper — train executor with greedy coordinator.
Phase 2: HiTMACv2CoordinatorWrapper — train coordinator with frozen Phase 1 executor.
"""

import copy
import re

import gym
import numpy as np
from gym import spaces

import mate
from examples.utils import CustomMetricCallback, MetricCollector


__all__ = ['HiTMACv2Wrapper', 'HiTMACv2CoordinatorWrapper']


class _GreedyAssigner:
    """Distance-based greedy assignment: assign nearest target to each camera.

    Follows the original HiTMAC paper — each camera is assigned the closest
    target (by Euclidean distance), regardless of visibility. This allows
    the executor to learn to turn toward invisible but nearby targets.
    Ties are broken by least-covered count to balance load.
    """

    def __init__(self, num_cameras, num_targets, observation_slices):
        self.num_cameras = num_cameras
        self.num_targets = num_targets
        self.self_state_slice = observation_slices['self_state']
        self.opponent_states_slice = observation_slices['opponent_states_with_mask']

    def assign(self, base_observations):
        assignments = np.zeros((self.num_cameras, self.num_targets), dtype=np.bool8)
        target_coverage = np.zeros(self.num_targets, dtype=np.int32)

        # Extract all target positions: each target has [x, y, speed, is_loaded, mask]
        # stride = TARGET_STATE_DIM_PUBLIC + 1 = 5
        stride = 5  # TARGET_STATE_DIM_PUBLIC(4) + visibility_mask(1)

        for c in range(self.num_cameras):
            cam_xy = base_observations[c, self.self_state_slice][:2]
            opponent_block = base_observations[c, self.opponent_states_slice]

            # Compute distance to each target
            distances = np.full(self.num_targets, np.inf)
            for t in range(self.num_targets):
                target_xy = opponent_block[t * stride: t * stride + 2]
                distances[t] = np.sqrt(np.sum((cam_xy - target_xy) ** 2))

            # Assign nearest target, breaking ties by least coverage
            # Sort by (distance, coverage) to prefer close + under-covered
            order = np.lexsort((target_coverage, distances))
            nearest = order[0]
            assignments[c, nearest] = True
            target_coverage[nearest] += 1

        return assignments


class HiTMACv2Wrapper(gym.Wrapper, metaclass=mate.WrapperMeta):
    """Hierarchical Task MAC wrapper for PPO executors.

    Same logic as HiTMACWrapper but:
    - Works with per-agent API (no grouped agents)
    - Continuous action space (no DiscreteCamera required)
    - Observation augmented with task assignment bits

    Args:
        env: Environment after mate.MultiCamera + standard wrappers.
        coordinator_agents: List of coordinator agents (None → greedy fallback).
        coord_period: Steps between coordinator updates.
        frame_skip: Steps to repeat action.
    """

    INFO_KEYS = {
        'raw_reward': 'sum',
        'normalized_raw_reward': 'sum',
        re.compile(r'^auxiliary_reward(\w*)$'): 'sum',
        re.compile(r'^reward_coefficient(\w*)$'): 'mean',
        'coverage_rate': 'mean',
        'real_coverage_rate': 'mean',
        'mean_transport_rate': 'last',
        'num_delivered_cargoes': 'last',
        'num_tracked': 'mean',
        'num_assigned_targets': 'mean',
        'assignment_coverage_rate': 'mean',
    }

    def __init__(self, env, coordinator_agents=None, coord_period=5,
                 frame_skip=5, custom_metrics=None):
        assert isinstance(env, mate.MultiCamera), (
            f'{self.__class__.__name__} must wrap mate.MultiCamera. Got env = {env}.'
        )
        super().__init__(env)

        self.coordinator_agents = coordinator_agents
        self.coord_period = coord_period
        self.frame_skip = frame_skip

        self._greedy_assigner = None

        self.observation_slices = mate.camera_observation_slices_of(
            env.num_cameras, env.num_targets, env.num_obstacles
        )
        self.target_view_mask_slice = self.observation_slices['opponent_mask']

        # Augmented observation: base obs + num_targets assignment bits
        original_obs_space = env.observation_space[0]
        augmented_low = np.concatenate(
            [original_obs_space.low, np.zeros(env.num_targets, dtype=np.float32)]
        )
        augmented_high = np.concatenate(
            [original_obs_space.high, np.ones(env.num_targets, dtype=np.float32)]
        )
        self.camera_observation_space = spaces.Box(
            augmented_low, augmented_high, dtype=np.float32
        )
        self.observation_space = spaces.Tuple(
            (self.camera_observation_space,) * env.num_cameras
        )

        # Keep original action space (continuous for PPO)
        self.camera_action_space = env.camera_action_space
        self.action_space = env.action_space
        self.teammate_action_space = self.camera_action_space
        self.teammate_joint_action_space = self.camera_joint_action_space = self.action_space

        self.last_base_observations = None
        self.current_assignments = None
        self._episode_step = 0

        self.custom_metrics = custom_metrics or CustomMetricCallback.DEFAULT_CUSTOM_METRICS
        self.custom_metrics.update({
            'num_assigned_targets': 'mean',
            'assignment_coverage_rate': 'mean',
        })

    def load_config(self, config=None):
        self.env.load_config(config=config)
        self.__init__(
            self.env,
            coordinator_agents=self.coordinator_agents,
            coord_period=self.coord_period,
            frame_skip=self.frame_skip,
            custom_metrics=self.custom_metrics,
        )

    def reset(self, **kwargs):
        self._episode_step = 0
        self.last_base_observations = base_obs = self.env.reset(**kwargs)

        if self.coordinator_agents is not None:
            for c, agent in enumerate(self.coordinator_agents):
                agent.reset(base_obs[c])

        self.current_assignments = self._run_coordinator(base_obs)
        return self._augment_observations(base_obs)

    def step(self, action):
        action = np.asarray(action)

        fragment_rewards = []
        metric_collectors = (
            [MetricCollector(self.INFO_KEYS) for _ in range(self.num_cameras)]
            if self.frame_skip > 1 else []
        )

        base_obs = self.last_base_observations
        for _ in range(self.frame_skip):
            base_obs, rewards, dones, infos = self.env.step(action)
            self._episode_step += 1

            if self._episode_step % self.coord_period == 0:
                self.current_assignments = self._run_coordinator(base_obs)

            for c in range(self.num_cameras):
                assignment = self.current_assignments[c]
                visible_mask = base_obs[c, self.target_view_mask_slice].astype(np.bool8)
                num_assigned = int(assignment.sum())
                num_assigned_visible = int(np.logical_and(assignment, visible_mask).sum())
                infos[c]['num_assigned_targets'] = num_assigned
                infos[c]['assignment_coverage_rate'] = (
                    num_assigned_visible / max(1, num_assigned)
                )

            if self.frame_skip > 1:
                fragment_rewards.append(rewards)
                for collector, info in zip(metric_collectors, infos):
                    collector.add(info)

            if all(dones):
                break

        self.last_base_observations = base_obs

        if self.frame_skip > 1:
            rewards = np.sum(fragment_rewards, axis=0).tolist()
            for collector, info in zip(metric_collectors, infos):
                info.update(collector.collect())

        return self._augment_observations(base_obs), rewards, dones, infos

    def _run_coordinator(self, base_observations):
        assignments = np.zeros((self.num_cameras, self.num_targets), dtype=np.bool8)

        if self.coordinator_agents is not None:
            for c, agent in enumerate(self.coordinator_agents):
                agent.act(base_observations[c])
                if agent.last_selection is not None:
                    assignments[c] = agent.last_selection.astype(np.bool8)
        else:
            if self._greedy_assigner is None:
                self._greedy_assigner = _GreedyAssigner(
                    self.num_cameras, self.num_targets, self.observation_slices
                )
            assignments = self._greedy_assigner.assign(base_observations)

        return assignments

    def _augment_observations(self, base_observations):
        augmented = []
        for c in range(self.num_cameras):
            augmented.append(
                np.concatenate([
                    base_observations[c].ravel().astype(np.float32),
                    self.current_assignments[c].astype(np.float32),
                ])
            )
        return np.stack(augmented, axis=0)


# ------------------------------------------------------------------
# PPO Executor Helper (Phase 2: frozen Phase 1 executor)
# ------------------------------------------------------------------

class _PPOExecutor:
    """Loads Phase 1 PPO checkpoint and runs inference for all cameras.

    Used inside HiTMACv2CoordinatorWrapper to run the frozen executor
    while the coordinator is being trained.
    """

    def __init__(self, checkpoint_path, num_cameras, env_config):
        from examples.hitmac_v2.camera.config import config as _executor_config
        from examples.hitmac_v2.camera.config import make_env as _make_executor_env
        from examples.utils.rllib_policy import (
            load_checkpoint, get_preprocessor, DEFAULT_POLICY_ID, SHARED_POLICY_ID,
        )
        from ray.rllib.agents.ppo import PPOTorchPolicy

        _, worker, params = load_checkpoint(checkpoint_path)
        config = copy.deepcopy(params) if params is not None else copy.deepcopy(_executor_config)

        # Override env config to match current environment
        if env_config is not None:
            config['env_config'] = copy.deepcopy(env_config)

        with _make_executor_env(config.get('env_config', {})) as dummy_env:
            obs_space = dummy_env.observation_space
            act_space = dummy_env.action_space
            preprocessor = get_preprocessor(obs_space)
            policy = PPOTorchPolicy(
                preprocessor.observation_space,
                act_space,
                config=dict(config, num_gpus=0, num_gpus_per_worker=0),
            )

        if worker is not None:
            key = SHARED_POLICY_ID if SHARED_POLICY_ID in worker['state'] else DEFAULT_POLICY_ID
            # Only load model weights, skip optimizer state (causes numpy.object_ error)
            policy.set_weights(worker['state'][key]['weights'])

        self.policy = policy
        self.preprocessor = preprocessor
        self.num_cameras = num_cameras
        self.hidden_states = None
        self.normalize_actions = config.get('normalize_actions', True)

    def reset(self):
        self.hidden_states = [
            self.policy.get_initial_state() for _ in range(self.num_cameras)
        ]

    def run(self, base_observations, assignments):
        """Run frozen PPO executor for all cameras.

        Args:
            base_observations: np.ndarray [num_cameras, D] base obs.
            assignments: np.ndarray [num_cameras, num_targets] bool.

        Returns:
            actions: list[np.ndarray] — continuous action per camera.
        """
        from ray.rllib.utils.spaces import space_utils

        actions = []
        for c in range(self.num_cameras):
            augmented = np.concatenate([
                base_observations[c].ravel().astype(np.float32),
                assignments[c].astype(np.float32),
            ])
            # Pad to preprocessor space (same as RLlibMultiAgentCentralizedTraining)
            padded = np.zeros(
                shape=self.preprocessor.observation_space.shape,
                dtype=self.preprocessor.observation_space.dtype,
            )
            padded[:augmented.size] = augmented

            result = self.policy.compute_single_action(
                padded, state=self.hidden_states[c], explore=False
            )
            action, self.hidden_states[c], *_ = result

            if self.normalize_actions:
                action = space_utils.unsquash_action(action, self.policy.action_space_struct)

            actions.append(action)

        return actions


# ------------------------------------------------------------------
# Phase 2: HiTMACv2CoordinatorWrapper
# ------------------------------------------------------------------

class HiTMACv2CoordinatorWrapper(gym.Wrapper, metaclass=mate.WrapperMeta):
    """Phase 2 — Train MAPPO coordinator with frozen Phase 1 PPO executor.

    Coordinator (MAPPO) learns optimal target assignments. Executor (PPO, frozen
    from Phase 1) executes continuous camera actions based on [obs + assignment bits].

    Observation space: base env obs (NO task bits) — for coordinator.
    Action space: MultiDiscrete((2,)*num_targets) per camera — assignment bits.

    Args:
        env: Environment after mate.MultiCamera + standard wrappers.
        executor_checkpoint: Path to Phase 1 PPO checkpoint.
        executor_env_config: env_config dict used to build Phase 1 env for loading.
        frame_skip: Inner steps per coordinator decision.
    """

    INFO_KEYS = {
        'raw_reward': 'sum',
        'normalized_raw_reward': 'sum',
        re.compile(r'^auxiliary_reward(\w*)$'): 'sum',
        re.compile(r'^reward_coefficient(\w*)$'): 'mean',
        'coverage_rate': 'mean',
        'real_coverage_rate': 'mean',
        'mean_transport_rate': 'last',
        'num_delivered_cargoes': 'last',
        'num_tracked': 'mean',
        'num_selected_targets': 'mean',
        'num_valid_selected_targets': 'mean',
        'num_invalid_selected_targets': 'mean',
        'invalid_target_selection_rate': 'mean',
    }

    def __init__(self, env, executor_checkpoint, executor_env_config=None,
                 frame_skip=5, custom_metrics=None):
        assert isinstance(env, mate.MultiCamera), (
            f'{self.__class__.__name__} must wrap mate.MultiCamera. Got env = {env}.'
        )
        super().__init__(env)

        self.frame_skip = frame_skip
        self._executor_checkpoint = executor_checkpoint
        self._executor_env_config = executor_env_config

        # Load frozen Phase 1 PPO executor
        self._executor = _PPOExecutor(executor_checkpoint, env.num_cameras, executor_env_config)

        # Coordinator action space: binary assignment per target per camera
        self.camera_action_space = spaces.MultiDiscrete((2,) * env.num_targets)
        self.action_space = spaces.Tuple((self.camera_action_space,) * env.num_cameras)
        self.teammate_action_space = self.camera_action_space
        self.teammate_joint_action_space = self.camera_joint_action_space = self.action_space

        # Coordinator observation: base obs (no task bits)
        self.observation_slices = mate.camera_observation_slices_of(
            env.num_cameras, env.num_targets, env.num_obstacles
        )
        self.target_view_mask_slice = self.observation_slices['opponent_mask']

        self.last_observations = None

        self.custom_metrics = custom_metrics or CustomMetricCallback.DEFAULT_CUSTOM_METRICS
        self.custom_metrics.update({
            'num_selected_targets': 'mean',
            'num_valid_selected_targets': 'mean',
            'num_invalid_selected_targets': 'mean',
            'invalid_target_selection_rate': 'mean',
        })

    def load_config(self, config=None):
        self.env.load_config(config=config)
        self.__init__(
            self.env,
            executor_checkpoint=self._executor_checkpoint,
            executor_env_config=self._executor_env_config,
            frame_skip=self.frame_skip,
            custom_metrics=self.custom_metrics,
        )

    def reset(self, **kwargs):
        self.last_observations = obs = self.env.reset(**kwargs)
        self._executor.reset()
        return obs

    def step(self, action):
        # action: [num_cameras, num_targets] binary assignment bits from coordinator
        action = np.asarray(action, dtype=np.int64).reshape(
            self.num_cameras, self.num_targets
        ).astype(np.bool8)

        fragment_rewards = []
        metric_collectors = (
            [MetricCollector(self.INFO_KEYS) for _ in range(self.num_cameras)]
            if self.frame_skip > 1 else []
        )

        observations = self.last_observations
        for _ in range(self.frame_skip):
            # Executor takes base obs + assignments → continuous camera actions
            continuous_actions = self._executor.run(observations, action)
            observations, rewards, dones, infos = self.env.step(continuous_actions)

            for c in range(self.num_cameras):
                sel = action[c]
                mask = observations[c, self.target_view_mask_slice].astype(np.bool8)
                n_sel = int(sel.sum())
                n_valid = int(np.logical_and(sel, mask).sum())
                n_invalid = n_sel - n_valid
                infos[c]['num_selected_targets'] = n_sel
                infos[c]['num_valid_selected_targets'] = n_valid
                infos[c]['num_invalid_selected_targets'] = n_invalid
                infos[c]['invalid_target_selection_rate'] = n_invalid / max(1, n_sel)

            if self.frame_skip > 1:
                fragment_rewards.append(rewards)
                for collector, info in zip(metric_collectors, infos):
                    collector.add(info)

            if all(dones):
                break

        self.last_observations = observations
        if self.frame_skip > 1:
            rewards = np.sum(fragment_rewards, axis=0).tolist()
            for collector, info in zip(metric_collectors, infos):
                info.update(collector.collect())

        return observations, rewards, dones, infos
