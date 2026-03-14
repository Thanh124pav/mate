"""HiTMACRoleCameraAgent — inference agent cho HiTMAC Phase 2B.

Tại inference:
    - Camera coordinator_idx: chạy MAPPO coordinator policy để assign targets.
    - Tất cả cameras (bao gồm coordinator): chạy QPLEX executor với assigned targets.

Note:
    Tại inference, mỗi camera chạy theo vai trò của mình:
    - Coordinator camera: quyết định toàn bộ assignments rồi nhận assignment của chính mình.
    - Executor cameras: nhận assignment từ coordinator rồi thực thi QPLEX.
    Vì inference cần 1 agent per camera, ta dùng HiTMACRoleCameraAgent cho camera
    coordinator_idx và HiTMACExecutorCameraAgent (wrapper đơn giản) cho các camera khác.
    Hoặc đơn giản hơn: HiTMACRoleCameraAgent.spawn() trả về đúng loại agent cho mỗi camera.

Sử dụng:
    agent = HiTMACRoleCameraAgent(
        checkpoint_path='examples/hitmac/role/camera/ray_results/.../checkpoint',
        executor_checkpoint_path='examples/hitmac/camera/ray_results/.../checkpoint',
        coordinator_idx=0,
    )
    agents = agent.spawn(env.num_cameras)
"""

import copy

import numpy as np
from ray.rllib.agents.ppo import PPOTorchPolicy

import mate
from examples.hitmac.role.camera.config import config as _config
from examples.hitmac.role.camera.config import make_env as _make_env
from examples.hitmac.wrappers import _QplexExecutor
from examples.utils import RLlibPolicyMixIn


class HiTMACRoleCameraAgent(RLlibPolicyMixIn, mate.CameraAgentBase):
    """HiTMAC Role-Based Camera Agent (Phase 2B).

    Camera coordinator_idx chạy MAPPO coordinator. Tất cả cameras chạy QPLEX executor.
    Tại inference, coordinator_agent được chia sẻ giữa tất cả cameras (via spawn).

    Args:
        config: Config dict cho MAPPO coordinator.
        checkpoint_path: Checkpoint của coordinator (Phase 2B output).
        executor_checkpoint_path: Checkpoint của QPLEX executor (Phase 1 output).
        coordinator_idx: Index của camera là coordinator.
        make_env: Factory function.
        seed: Random seed.
    """

    POLICY_CLASS = PPOTorchPolicy
    DEFAULT_CONFIG = copy.deepcopy(_config)

    def __init__(
        self,
        config=None,
        checkpoint_path=None,
        executor_checkpoint_path=None,
        coordinator_idx=0,
        make_env=_make_env,
        seed=None,
    ):
        super().__init__(
            config=config,
            checkpoint_path=checkpoint_path,
            make_env=make_env,
            seed=seed,
        )

        env_config = self.config.get('env_config', {})
        self.frame_skip = env_config.get('frame_skip', 5)
        self.coordinator_idx = coordinator_idx

        self._executor_checkpoint_path = executor_checkpoint_path

        # Shared executor (1 instance, run for 1 camera at a time per agent)
        self._executor = None
        if executor_checkpoint_path is not None:
            self._executor = _QplexExecutor(executor_checkpoint_path, num_cameras=1)

        # Shared coordinator assignment: coordinator camera sets this, executors read it
        # Tại inference với spawn(), cần 1 coordinator agent và n-1 executor agents
        self._shared_assignments = None   # [num_cameras, num_targets] — set by coordinator
        self._is_coordinator = False       # set after spawn() via camera_idx

        self.last_selection = None
        self.last_action = None

    def clone(self):
        return self.__class__(
            config=self.config,
            checkpoint_path=self.checkpoint_path,
            executor_checkpoint_path=self._executor_checkpoint_path,
            coordinator_idx=self.coordinator_idx,
            make_env=self.make_env,
            seed=self.np_random.randint(np.iinfo(int).max),
        )

    def spawn(self, num_cameras):
        """Tạo danh sách agents cho evaluation.

        Returns:
            agents[coordinator_idx]: HiTMACRoleCameraAgent (coordinator policy + executor)
            agents[other]: _HiTMACExecutorAgent (executor only, nhận assignments từ coordinator)
        """
        agents = []
        # Shared assignment buffer: coordinator sets, executors read
        shared_state = {'assignments': None}

        coordinator = self.clone()
        coordinator._is_coordinator = True
        coordinator._shared_state = shared_state
        if coordinator._executor is not None:
            coordinator._executor.reset()

        for c in range(num_cameras):
            if c == self.coordinator_idx:
                agents.append(coordinator)
            else:
                exec_agent = _HiTMACExecutorAgent(
                    executor_checkpoint_path=self._executor_checkpoint_path,
                    coordinator_idx=self.coordinator_idx,
                    camera_idx=c,
                    shared_state=shared_state,
                    seed=self.np_random.randint(np.iinfo(int).max),
                )
                agents.append(exec_agent)

        return agents

    def reset(self, observation):
        super().reset(observation)
        if self._executor is not None:
            self._executor.reset()
        self.last_selection = np.zeros(self.num_targets, dtype=np.bool8)
        self._shared_assignments = None
        self.last_action = None

    def act(self, observation, info=None, deterministic=None):
        self.state, observation, info, messages = self.check_inputs(observation, info)

        # --- Coordinator: MAPPO policy quyết định assignments cho TẤT CẢ cameras ---
        if self.episode_step % self.frame_skip == 0:
            flat_action, self.hidden_state = self.compute_single_action(
                observation,
                state=self.hidden_state,
                info=info,
                deterministic=deterministic,
            )
            # flat_action: [num_cameras * num_targets] → [num_cameras, num_targets]
            all_assignments = np.asarray(flat_action, dtype=np.bool8).reshape(
                self.num_cameras, self.num_targets
            )
            self.last_selection = all_assignments[self.coordinator_idx]

            # Chia sẻ assignments cho các executor agents qua shared_state
            if hasattr(self, '_shared_state'):
                self._shared_state['assignments'] = all_assignments

        # --- Executor: QPLEX chọn discrete action dựa trên obs + coordinator's assignment ---
        if self._executor is not None:
            obs_2d = observation[np.newaxis, :].astype(np.float32)
            sel_2d = self.last_selection[np.newaxis, :]
            discrete_actions = self._executor.run(obs_2d, sel_2d)
            discrete_action = discrete_actions[0]

            discrete_levels = self.config.get('env_config', {}).get('discrete_levels', 5)
            action_grid = mate.DiscreteCamera.discrete_action_grid(levels=discrete_levels)
            self.last_action = self.action_space.high * action_grid[discrete_action]
        else:
            from examples.hrl.wrappers import HierarchicalCamera
            target_states, tracked_bits = self.get_all_opponent_states(observation)
            self.last_action = HierarchicalCamera.executor(
                self.state,
                target_states,
                target_selection_bits=self.last_selection,
                target_view_mask=tracked_bits,
            )

        return self.last_action


class _HiTMACExecutorAgent(mate.CameraAgentBase):
    """Executor-only agent dùng Phase 1 QPLEX và nhận assignment từ coordinator.

    Được tạo bởi HiTMACRoleCameraAgent.spawn() cho các non-coordinator cameras.
    """

    def __init__(
        self,
        executor_checkpoint_path,
        coordinator_idx,
        camera_idx,
        shared_state,
        seed=None,
    ):
        super().__init__(seed=seed)
        self.coordinator_idx = coordinator_idx
        self.camera_idx = camera_idx
        self._shared_state = shared_state
        self._executor_checkpoint_path = executor_checkpoint_path

        self._executor = None
        if executor_checkpoint_path is not None:
            self._executor = _QplexExecutor(executor_checkpoint_path, num_cameras=1)

        self._current_assignment = None
        self.last_action = None

    def reset(self, observation):
        super().reset(observation)
        if self._executor is not None:
            self._executor.reset()
        self._current_assignment = np.zeros(self.num_targets, dtype=np.bool8)
        self.last_action = None

    def act(self, observation, info=None, deterministic=None):
        self.state, observation, info, messages = self.check_inputs(observation, info)

        # Lấy assignment từ coordinator (qua shared_state)
        assignments = self._shared_state.get('assignments')
        if assignments is not None:
            self._current_assignment = assignments[self.camera_idx].astype(np.bool8)

        if self._executor is not None:
            obs_2d = observation[np.newaxis, :].astype(np.float32)
            sel_2d = self._current_assignment[np.newaxis, :]
            discrete_actions = self._executor.run(obs_2d, sel_2d)
            discrete_action = discrete_actions[0]

            action_grid = mate.DiscreteCamera.discrete_action_grid(levels=5)
            self.last_action = self.action_space.high * action_grid[discrete_action]
        else:
            self.last_action = self.action_space.low

        return self.last_action
