"""HiTMACCoordinatorCameraAgent — inference agent cho HiTMAC Phase 2A.

Kết hợp:
    - MAPPO Coordinator (từ examples/hitmac/coordinator/): giao task mỗi coord_period bước.
      Sử dụng coordinator_agent.last_selection để lấy task bits.
    - QPLEX Executor (Phase 1, fixed): chọn discrete camera action dựa trên [obs, task_bits].

Sử dụng:
    agent = HiTMACCoordinatorCameraAgent(
        checkpoint_path='examples/hitmac/coordinator/camera/ray_results/.../checkpoint',
        executor_checkpoint_path='examples/hitmac/camera/ray_results/.../checkpoint',
    )
"""

import copy

import numpy as np
from ray.rllib.agents.ppo import PPOTorchPolicy

import mate
from examples.hitmac.coordinator.mappo.camera.config import config as _config
from examples.hitmac.coordinator.mappo.camera.config import make_env as _make_env
from examples.hitmac.wrappers import _QplexExecutor
from examples.utils import RLlibPolicyMixIn


class HiTMACCoordinatorCameraAgent(RLlibPolicyMixIn, mate.CameraAgentBase):
    """HiTMAC Coordinator Camera Agent (Phase 2A): MAPPO coordinator + QPLEX executor.

    Args:
        config: Config dict cho MAPPO coordinator.
        checkpoint_path: Checkpoint của MAPPO coordinator (Phase 2A output).
        executor_checkpoint_path: Checkpoint của QPLEX executor (Phase 1 output).
        make_env: Factory function để lấy obs/action spaces.
        seed: Random seed.
    """

    POLICY_CLASS = PPOTorchPolicy
    DEFAULT_CONFIG = copy.deepcopy(_config)

    def __init__(
        self,
        config=None,
        checkpoint_path=None,
        executor_checkpoint_path=None,
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

        # Lưu để clone() truyền lại
        self._executor_checkpoint_path = executor_checkpoint_path

        # Load QPLEX executor từ Phase 1 checkpoint
        self._executor = None
        if executor_checkpoint_path is not None:
            self._executor = _QplexExecutor(executor_checkpoint_path, num_cameras=1)

        # State per episode
        self.last_selection = None
        self.last_action = None

    def clone(self):
        return self.__class__(
            config=self.config,
            checkpoint_path=self.checkpoint_path,
            executor_checkpoint_path=self._executor_checkpoint_path,
            make_env=self.make_env,
            seed=self.np_random.randint(np.iinfo(int).max),
        )

    def reset(self, observation):
        super().reset(observation)
        if self._executor is not None:
            self._executor.reset()
        self.last_selection = np.zeros(self.num_targets, dtype=np.bool8)
        self.last_action = None

    def act(self, observation, info=None, deterministic=None):
        self.state, observation, info, messages = self.check_inputs(observation, info)

        # --- Step 1: MAPPO Coordinator cập nhật assignment ---
        if self.episode_step % self.frame_skip == 0:
            selection, self.hidden_state = self.compute_single_action(
                observation,
                state=self.hidden_state,
                info=info,
                deterministic=deterministic,
            )
            self.last_selection = np.asarray(selection, dtype=np.bool8)

        # --- Step 2: QPLEX Executor chọn discrete action ---
        if self._executor is not None:
            # _QplexExecutor expects [num_cameras, D] and [num_cameras, num_targets]
            obs_2d = observation[np.newaxis, :].astype(np.float32)
            sel_2d = self.last_selection[np.newaxis, :]
            discrete_actions = self._executor.run(obs_2d, sel_2d)
            discrete_action = discrete_actions[0]

            # Lấy normalized action grid từ executor config để convert discrete → continuous
            discrete_levels = self.config.get('env_config', {}).get('discrete_levels', 5)
            action_grid = mate.DiscreteCamera.discrete_action_grid(levels=discrete_levels)
            self.last_action = self.action_space.high * action_grid[discrete_action]
        else:
            # Fallback: geometric executor (như HRL-MAPPO)
            from examples.hrl.wrappers import HierarchicalCamera
            target_states, tracked_bits = self.get_all_opponent_states(observation)
            self.last_action = HierarchicalCamera.executor(
                self.state,
                target_states,
                target_selection_bits=self.last_selection,
                target_view_mask=tracked_bits,
            )

        return self.last_action
