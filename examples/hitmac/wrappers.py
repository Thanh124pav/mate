"""HiTMACWrapper — environment wrapper cho HiTMAC-QPLEX.

Cấu trúc phân cấp:
    High-level (Coordinator): Giao nhiệm vụ cho mỗi camera (targets nào cần theo dõi).
        • Trained: dùng HRLMAPPOCameraAgent đã được train từ examples/hrl/mappo/
        • Fallback: GreedyDistanceAssigner (khi không có checkpoint)
    Low-level (Executor): QPLEX agents học discrete camera control.
        • Nhận observation gốc ghép thêm task-assignment bits (flat concat).
        • Action space: kế thừa DiscreteCamera từ env bên dưới.

Wrapper chain bên ngoài (xem camera/config.py):
    mate.DiscreteCamera → mate.MultiCamera → RelativeCoords → Rescale
    → RepeatedReward → AuxReward → HiTMACWrapper → RLlibMultiAgentAPI
    → RLlibMultiAgentCentralizedTraining → env.with_agent_groups()
"""

import copy
import re

import gym
import numpy as np
from gym import spaces

import mate
from examples.utils import CustomMetricCallback, MetricCollector


__all__ = [
    'HiTMACWrapper',
    'HiTMACCoordinatorWrapper',
    'DiscreteCoordinatorSelection',
    'HiTMACRoleWrapper',
]


class HiTMACWrapper(gym.Wrapper, metaclass=mate.WrapperMeta):
    """Hierarchical Task MAC wrapper với QPLEX executors.

    Args:
        env: Môi trường sau khi đã wrap với mate.MultiCamera + wrappers chuẩn.
             DiscreteCamera phải được áp dụng TRƯỚC MultiCamera.
        coordinator_agents: Danh sách HRLMAPPOCameraAgent (1 per camera).
            Nếu None → dùng GreedyDistanceAssigner làm fallback.
        coord_period: Số env steps giữa các lần coordinator cập nhật task.
        frame_skip: Số lần lặp lại action trong mỗi high-level step.
        custom_metrics: Metrics bổ sung cho CustomMetricCallback.
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

    def __init__(self, env, coordinator_agents=None, coord_period=5, frame_skip=5,
                 custom_metrics=None):
        assert isinstance(env, mate.MultiCamera), (
            f'{self.__class__.__name__} phải được wrap bên ngoài mate.MultiCamera. '
            f'Got env = {env}.'
        )
        super().__init__(env)

        self.coordinator_agents = coordinator_agents
        self.coord_period = coord_period
        self.frame_skip = frame_skip

        # Greedy fallback (dùng khi coordinator_agents is None)
        self._greedy_assigner = None

        # Observation slices để trích xuất visibility mask
        self.observation_slices = mate.camera_observation_slices_of(
            env.num_cameras, env.num_targets, env.num_obstacles
        )
        self.target_view_mask_slice = self.observation_slices['opponent_mask']

        # Observation space: ghép thêm num_targets bits vào cuối mỗi camera obs
        original_obs_space = env.observation_space[0]  # Box(D,)
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

        # Action space: giữ nguyên (DiscreteCamera đã được áp dụng bên dưới)
        self.camera_action_space = env.camera_action_space
        self.action_space = env.action_space
        self.teammate_action_space = self.camera_action_space
        self.teammate_joint_action_space = self.camera_joint_action_space = self.action_space

        # State
        self.last_base_observations = None
        self.current_assignments = None  # [num_cameras, num_targets] bool
        self._env_step = 0  # global env step (không reset mỗi episode)
        self._episode_step = 0  # step trong episode hiện tại

        self.custom_metrics = custom_metrics or CustomMetricCallback.DEFAULT_CUSTOM_METRICS
        self.custom_metrics.update(
            {
                'num_assigned_targets': 'mean',
                'assignment_coverage_rate': 'mean',
            }
        )

    def load_config(self, config=None):
        self.env.load_config(config=config)
        self.__init__(
            self.env,
            coordinator_agents=self.coordinator_agents,
            coord_period=self.coord_period,
            frame_skip=self.frame_skip,
            custom_metrics=self.custom_metrics,
        )

    # ------------------------------------------------------------------
    # Reset / Step
    # ------------------------------------------------------------------

    def reset(self, **kwargs):
        self._episode_step = 0
        self.last_base_observations = base_obs = self.env.reset(**kwargs)

        # Reset coordinator agents
        if self.coordinator_agents is not None:
            for c, agent in enumerate(self.coordinator_agents):
                agent.reset(base_obs[c])

        # Khởi tạo assignments ban đầu
        self.current_assignments = self._run_coordinator(base_obs)

        return self._augment_observations(base_obs)

    def step(self, action):
        action = np.asarray(action)

        fragment_rewards = []
        metric_collectors = (
            [MetricCollector(self.INFO_KEYS) for _ in range(self.num_cameras)]
            if self.frame_skip > 1
            else []
        )

        base_obs = self.last_base_observations

        for _ in range(self.frame_skip):
            base_obs, rewards, dones, infos = self.env.step(action)
            self._episode_step += 1

            # Cập nhật coordinator task mỗi coord_period bước
            if self._episode_step % self.coord_period == 0:
                self.current_assignments = self._run_coordinator(base_obs)

            # Thêm assignment metrics vào infos
            for c in range(self.num_cameras):
                assignment = self.current_assignments[c]
                visible_mask = base_obs[c, self.target_view_mask_slice].astype(np.bool8)
                num_assigned = int(assignment.sum())
                num_assigned_and_visible = int(np.logical_and(assignment, visible_mask).sum())
                infos[c]['num_assigned_targets'] = num_assigned
                infos[c]['assignment_coverage_rate'] = (
                    num_assigned_and_visible / max(1, num_assigned)
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

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_coordinator(self, base_observations):
        """Chạy coordinator để lấy task assignment.

        Returns:
            assignments: np.ndarray [num_cameras, num_targets] bool
        """
        assignments = np.zeros(
            (self.num_cameras, self.num_targets), dtype=np.bool8
        )

        if self.coordinator_agents is not None:
            # Dùng MAPPO coordinator đã load từ checkpoint
            for c, agent in enumerate(self.coordinator_agents):
                # act() trả về continuous camera action (bỏ qua),
                # nhưng cập nhật agent.last_selection → target assignment bits
                agent.act(base_observations[c])
                if agent.last_selection is not None:
                    assignments[c] = agent.last_selection.astype(np.bool8)
        else:
            # Greedy fallback: mỗi camera assign target gần nhất trong tầm nhìn
            if self._greedy_assigner is None:
                self._greedy_assigner = _GreedyAssigner(
                    self.num_cameras, self.num_targets, self.target_view_mask_slice
                )
            assignments = self._greedy_assigner.assign(base_observations)

        return assignments

    def _augment_observations(self, base_observations):
        """Ghép assignment bits vào cuối mỗi camera observation.

        Returns:
            augmented: np.ndarray [num_cameras, D + num_targets]
        """
        augmented = []
        for c in range(self.num_cameras):
            augmented.append(
                np.concatenate(
                    [
                        base_observations[c].ravel().astype(np.float32),
                        self.current_assignments[c].astype(np.float32),
                    ]
                )
            )
        return np.stack(augmented, axis=0)


# ------------------------------------------------------------------
# Greedy Fallback Assigner
# ------------------------------------------------------------------

class _GreedyAssigner:
    """Gán target gần nhất và có thể nhìn thấy cho mỗi camera.

    Dùng làm fallback khi không có coordinator checkpoint.
    """

    def __init__(self, num_cameras, num_targets, target_view_mask_slice):
        self.num_cameras = num_cameras
        self.num_targets = num_targets
        self.target_view_mask_slice = target_view_mask_slice

    def assign(self, base_observations):
        assignments = np.zeros((self.num_cameras, self.num_targets), dtype=np.bool8)
        target_coverage = np.zeros(self.num_targets, dtype=np.int32)

        for c in range(self.num_cameras):
            visible = base_observations[c, self.target_view_mask_slice].astype(np.bool8)
            if visible.any():
                visible_indices = np.where(visible)[0]
                # Ưu tiên target ít được cover nhất
                least_covered = visible_indices[np.argmin(target_coverage[visible_indices])]
                assignments[c, least_covered] = True
                target_coverage[least_covered] += 1
            else:
                # Không nhìn thấy ai: assign target ít cover nhất (theo round-robin)
                least_covered_idx = int(np.argmin(target_coverage))
                assignments[c, least_covered_idx] = True
                target_coverage[least_covered_idx] += 1

        return assignments


# ------------------------------------------------------------------
# QPLEX Executor Helper (dùng trong Phase 2 wrappers)
# ------------------------------------------------------------------

class _QplexExecutor:
    """Loads Phase 1 QPLEX checkpoint và chạy inference cho tất cả cameras.

    Được dùng bên trong HiTMACCoordinatorWrapper và HiTMACRoleWrapper để
    chạy executor fixed (không trainable) trong khi coordinator được train.
    """

    def __init__(self, checkpoint_path, num_cameras):
        from examples.hitmac.camera.config import config as _executor_config
        from examples.hitmac.camera.config import make_env as _make_executor_env
        from examples.utils.rllib_policy import load_checkpoint, get_preprocessor, DEFAULT_POLICY_ID
        from ray.rllib.agents.qplex_v2.qplex_policy import QPLEXTorchPolicy

        _, worker, params = load_checkpoint(checkpoint_path)
        config = copy.deepcopy(params) if params is not None else copy.deepcopy(_executor_config)
        config['mixer'] = 'qplex_v2'

        env_config = config.get('env_config', {})
        with _make_executor_env(env_config) as dummy_env:
            grouped_obs_space = dummy_env.observation_space
            preprocessor = get_preprocessor(grouped_obs_space)
            policy = QPLEXTorchPolicy(
                grouped_obs_space,
                dummy_env.action_space,
                config=dict(config, num_gpus=0, num_gpus_per_worker=0),
            )

        if worker is not None:
            key = 'camera' if 'camera' in worker['state'] else DEFAULT_POLICY_ID
            policy.set_state(worker['state'][key])

        self.policy = policy
        self.preprocessor = preprocessor
        self.num_cameras = num_cameras
        self.hidden_states = None

    def reset(self):
        self.hidden_states = [
            self.policy.get_initial_state() for _ in range(self.num_cameras)
        ]

    def run(self, base_observations, assignments):
        """Chạy QPLEX executor cho tất cả cameras.

        Args:
            base_observations: np.ndarray [num_cameras, D] đã preprocessed.
            assignments: np.ndarray [num_cameras, num_targets] bool.

        Returns:
            discrete_actions: list[int] — discrete action index per camera.
        """
        discrete_actions = []
        for c in range(self.num_cameras):
            augmented = np.concatenate([
                base_observations[c].ravel().astype(np.float32),
                assignments[c].astype(np.float32),
            ])
            # Pad đến grouped obs space (cùng cách RLlibGroupedPolicyMixIn làm)
            padded = np.zeros(
                shape=self.preprocessor.observation_space.shape,
                dtype=self.preprocessor.observation_space.dtype,
            )
            padded[:augmented.size] = augmented

            results = self.policy.compute_single_action(
                padded, state=self.hidden_states[c], explore=False
            )
            joint_action, self.hidden_states[c], *_ = results
            discrete_actions.append(int(joint_action[0]))

        return discrete_actions


# ------------------------------------------------------------------
# Phương án A: HiTMACCoordinatorWrapper
# ------------------------------------------------------------------

class HiTMACCoordinatorWrapper(gym.Wrapper, metaclass=mate.WrapperMeta):
    """Phase 2A — Train MAPPO coordinator với fixed Phase 1 QPLEX executor.

    Coordinator (MAPPO) học assign targets tối ưu. Executor (QPLEX, fixed từ
    Phase 1) thực thi bằng cách chọn discrete camera action dựa trên [obs + bits].

    Observation space: base env obs (KHÔNG có task bits) — cho coordinator.
    Action space: MultiDiscrete((2,)*num_targets) per camera — assignment bits.

    Args:
        env: Môi trường sau khi đã wrap với mate.MultiCamera + wrappers chuẩn.
             DiscreteCamera phải được áp dụng TRƯỚC MultiCamera.
        executor_checkpoint: Path đến Phase 1 QPLEX checkpoint.
        frame_skip: Số inner steps per coordinator decision.
        custom_metrics: Metrics bổ sung cho CustomMetricCallback.
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

    def __init__(self, env, executor_checkpoint, frame_skip=5, custom_metrics=None):
        assert isinstance(env, mate.MultiCamera), (
            f'{self.__class__.__name__} phải được wrap bên ngoài mate.MultiCamera. '
            f'Got env = {env}.'
        )
        super().__init__(env)

        self.frame_skip = frame_skip
        self._executor_checkpoint = executor_checkpoint

        # Load Phase 1 QPLEX executor (fixed, không train)
        self._executor = _QplexExecutor(executor_checkpoint, env.num_cameras)

        # Action space: MultiDiscrete assignment bits per camera
        self.camera_action_space = spaces.MultiDiscrete((2,) * env.num_targets)
        self.action_mask_space = spaces.MultiBinary(2 * env.num_targets)
        self.action_space = spaces.Tuple((self.camera_action_space,) * env.num_cameras)
        self.teammate_action_space = self.camera_action_space
        self.teammate_joint_action_space = self.camera_joint_action_space = self.action_space

        # Observation space: base obs (không augment task bits)
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
            frame_skip=self.frame_skip,
            custom_metrics=self.custom_metrics,
        )

    def reset(self, **kwargs):
        self.last_observations = obs = self.env.reset(**kwargs)
        self._executor.reset()
        return obs

    def step(self, action):
        # action: [num_cameras, num_targets] binary assignment bits
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
            discrete_actions = self._executor.run(observations, action)
            observations, rewards, dones, infos = self.env.step(discrete_actions)

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

    def action_mask(self, observation):
        target_view_mask = observation[self.target_view_mask_slice].ravel().astype(np.bool8)
        action_mask = np.repeat(target_view_mask, repeats=2)
        action_mask[::2] = True
        return action_mask


# ------------------------------------------------------------------
# DiscreteCoordinatorSelection — QPLEX dùng Discrete action space
# ------------------------------------------------------------------

class DiscreteCoordinatorSelection(gym.ActionWrapper, metaclass=mate.WrapperMeta):
    """Flatten MultiDiscrete assignment space → Discrete cho QPLEX coordinator training.

    Wrapper này cần thiết khi train coordinator bằng QPLEX (thay vì MAPPO):
    QPLEX yêu cầu Discrete action space (grouped), trong khi HiTMACCoordinatorWrapper
    expose MultiDiscrete((2,)*num_targets) per camera.

    Ví dụ với 8 targets: MultiDiscrete([2]*8) → Discrete(256).
    """

    def __init__(self, env):
        super().__init__(env)
        assert isinstance(env, HiTMACCoordinatorWrapper), (
            f'{self.__class__.__name__} cần được wrap bên ngoài HiTMACCoordinatorWrapper. '
            f'Got env = {env}.'
        )
        from examples.hrl.wrappers import MultiDiscrete2DiscreteActionMapper

        self._action_mapper = MultiDiscrete2DiscreteActionMapper(
            original_space=env.camera_action_space  # MultiDiscrete((2,)*num_targets)
        )

        self.camera_action_space = self._action_mapper.space  # Discrete(2^num_targets)
        self.action_mask_space = self._action_mapper.mask_space
        self.action_space = spaces.Tuple(
            (self.camera_action_space,) * env.num_cameras
        )
        self.teammate_action_space = self.camera_action_space
        self.teammate_joint_action_space = self.camera_joint_action_space = self.action_space

    def action(self, action):
        """Discrete(2^T)^N → MultiDiscrete((2,)*T)^N."""
        return self._action_mapper.multi_discrete_action_batched(action)

    def reverse_action(self, action):
        """MultiDiscrete((2,)*T)^N → Discrete(2^T)^N."""
        return self._action_mapper.discrete_action_batched(action)

    def action_mask(self, observation):
        action_mask = self.env.action_mask(observation)
        return self._action_mapper.discrete_action_mask(action_mask)


# ------------------------------------------------------------------
# Phương án B: HiTMACRoleWrapper
# ------------------------------------------------------------------

class HiTMACRoleWrapper(gym.Wrapper, metaclass=mate.WrapperMeta):
    """Phase 2B — Train single coordinator camera với fixed Phase 1 QPLEX executors.

    Một camera được chỉ định làm coordinator (coordinator_idx). Camera này có
    enhanced observation (global view) và quyết định assignment cho TẤT CẢ cameras.
    Các cameras còn lại (và cả coordinator) thực thi bằng Phase 1 QPLEX executor.

    Training là single-agent: chỉ coordinator policy được train với PPO.

    Observation space: coordinator camera's obs — 1D Box.
    Action space: MultiDiscrete((2,) * num_cameras * num_targets).

    Args:
        env: Môi trường sau mate.MultiCamera + wrappers chuẩn.
             DiscreteCamera phải được áp dụng TRƯỚC MultiCamera.
             EnhancedObservation nên được áp dụng trước MultiCamera.
        executor_checkpoint: Path đến Phase 1 QPLEX checkpoint.
        coordinator_idx: Index của camera làm coordinator (default: 0).
        frame_skip: Số inner steps per coordinator decision.
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
    }

    def __init__(self, env, executor_checkpoint, coordinator_idx=0, frame_skip=5):
        assert isinstance(env, mate.MultiCamera), (
            f'{self.__class__.__name__} phải được wrap bên ngoài mate.MultiCamera. '
            f'Got env = {env}.'
        )
        super().__init__(env)

        self.coordinator_idx = coordinator_idx
        self.frame_skip = frame_skip
        self._executor_checkpoint = executor_checkpoint

        # Load Phase 1 QPLEX executor (fixed)
        self._executor = _QplexExecutor(executor_checkpoint, env.num_cameras)

        # Action space: assignments cho tất cả cameras (flattened)
        self.action_space = spaces.MultiDiscrete((2,) * (env.num_cameras * env.num_targets))

        # Observation space: coordinator camera's obs (1 camera)
        self.observation_space = env.observation_space[coordinator_idx]

        self.observation_slices = mate.camera_observation_slices_of(
            env.num_cameras, env.num_targets, env.num_obstacles
        )
        self.last_observations = None

    def load_config(self, config=None):
        self.env.load_config(config=config)
        self.__init__(
            self.env,
            executor_checkpoint=self._executor_checkpoint,
            coordinator_idx=self.coordinator_idx,
            frame_skip=self.frame_skip,
        )

    def reset(self, **kwargs):
        joint_obs = self.env.reset(**kwargs)
        self.last_observations = joint_obs
        self._executor.reset()
        return joint_obs[self.coordinator_idx]

    def step(self, action):
        # action: flat [num_cameras * num_targets] → reshape to [num_cameras, num_targets]
        action = np.asarray(action, dtype=np.int64).reshape(
            self.num_cameras, self.num_targets
        ).astype(np.bool8)

        fragment_rewards = []
        metric_collectors = (
            [MetricCollector(self.INFO_KEYS) for _ in range(self.num_cameras)]
            if self.frame_skip > 1 else []
        )

        joint_obs = self.last_observations
        for _ in range(self.frame_skip):
            discrete_actions = self._executor.run(joint_obs, action)
            joint_obs, rewards, dones, infos = self.env.step(discrete_actions)

            for c in range(self.num_cameras):
                infos[c]['num_assigned_targets'] = int(action[c].sum())

            if self.frame_skip > 1:
                fragment_rewards.append(rewards)
                for collector, info in zip(metric_collectors, infos):
                    collector.add(info)

            if all(dones):
                break

        self.last_observations = joint_obs
        if self.frame_skip > 1:
            rewards = np.sum(fragment_rewards, axis=0).tolist()
            for collector, info in zip(metric_collectors, infos):
                info.update(collector.collect())

        coord_obs = joint_obs[self.coordinator_idx]
        coord_reward = float(rewards[self.coordinator_idx])
        coord_done = bool(dones[self.coordinator_idx])
        coord_info = infos[self.coordinator_idx]

        return coord_obs, coord_reward, coord_done, coord_info
