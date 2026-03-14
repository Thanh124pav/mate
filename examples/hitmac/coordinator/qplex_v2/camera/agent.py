"""HiTMACQPLEXV2CoordinatorCameraAgent — inference agent cho HiTMAC Phase 2A QPLEX_V2.

Kết hợp:
    - QPLEX_V2 Coordinator: Discrete(2^num_targets) → multi-select assignment bits.
      Được train với HiTMACCoordinatorWrapper + DiscreteCoordinatorSelection.
    - QPLEX Executor (Phase 1, fixed): chọn discrete camera action dựa trên [obs, task_bits].

Cấu trúc giống HRLQPLEXV2CameraAgent nhưng bước 2 (executor) dùng QPLEX thay vì
geometric scripted executor.

Sử dụng:
    agent = HiTMACQPLEXV2CoordinatorCameraAgent(
        checkpoint_path='examples/hitmac/coordinator/qplex_v2/camera/ray_results/.../checkpoint',
        executor_checkpoint_path='examples/hitmac/camera/ray_results/.../checkpoint',
    )
"""

import copy

import numpy as np
from gym import spaces
from ray.rllib.agents.qplex_v2.qplex_policy import QPLEXTorchPolicy

import mate
from examples.hitmac.coordinator.qplex_v2.camera.config import config as _config
from examples.hitmac.coordinator.qplex_v2.camera.config import make_env as _make_env
from examples.hitmac.wrappers import _QplexExecutor
from examples.hrl.wrappers import MultiDiscrete2DiscreteActionMapper
from examples.utils import RLlibGroupedPolicyMixIn


class HiTMACQPLEXV2CoordinatorCameraAgent(RLlibGroupedPolicyMixIn, mate.CameraAgentBase):
    """HiTMAC Phase 2A Camera Agent: QPLEX_V2 coordinator + QPLEX executor.

    Args:
        config: Config dict cho QPLEX_V2 coordinator.
        checkpoint_path: Checkpoint của QPLEX_V2 coordinator (Phase 2A output).
        executor_checkpoint_path: Checkpoint của QPLEX executor (Phase 1 output).
        make_env: Factory function.
        seed: Random seed.
    """

    POLICY_CLASS = QPLEXTorchPolicy
    DEFAULT_CONFIG = copy.deepcopy(_config)

    def get_policy(self):
        # Sync mixer name từ POLICY_CLASS module để tránh stale params.pkl override.
        policy_module = self.POLICY_CLASS.__module__
        expected_mixer = 'qplex_v2' if 'qplex_v2' in policy_module else 'qplex'
        if self.config.get('mixer') != expected_mixer:
            self.config['mixer'] = expected_mixer
        return super().get_policy()

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

        self._executor_checkpoint_path = executor_checkpoint_path

        # Load Phase 1 QPLEX executor (fixed)
        self._executor = None
        if executor_checkpoint_path is not None:
            self._executor = _QplexExecutor(executor_checkpoint_path, num_cameras=1)

        # Action mapper: Discrete(2^num_targets) → MultiDiscrete((2,)*num_targets)
        self.last_selection = None
        self.last_action = None
        self._action_mapper = None  # initialized in reset()

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

        # Khởi tạo action mapper sau khi biết num_targets
        original_space = spaces.MultiDiscrete((2,) * self.num_targets)
        self._action_mapper = MultiDiscrete2DiscreteActionMapper(original_space=original_space)

        self.last_selection = np.zeros(self.num_targets, dtype=np.bool8)
        self.last_action = None

    def act(self, observation, info=None, deterministic=None):
        self.state, observation, info, messages = self.check_inputs(observation, info)

        # --- Step 1: QPLEX_V2 Coordinator → target assignment bits ---
        if self.episode_step % self.frame_skip == 0:
            # compute_single_action (grouped): obs → Discrete(2^num_targets) index
            discrete_selection, self.hidden_state = self.compute_single_action(
                observation,
                state=self.hidden_state,
                info=info,
                deterministic=deterministic,
            )
            # Decode: Discrete index → MultiDiscrete binary bits
            self.last_selection = self._action_mapper.multi_discrete_action(
                discrete_selection
            ).astype(np.bool8)

        # --- Step 2: QPLEX Executor → discrete camera action → continuous ---
        if self._executor is not None:
            obs_2d = observation[np.newaxis, :].astype(np.float32)
            sel_2d = self.last_selection[np.newaxis, :]
            discrete_actions = self._executor.run(obs_2d, sel_2d)
            discrete_action = discrete_actions[0]

            discrete_levels = self.config.get('env_config', {}).get('discrete_levels', 5)
            action_grid = mate.DiscreteCamera.discrete_action_grid(levels=discrete_levels)
            self.last_action = self.action_space.high * action_grid[discrete_action]
        else:
            # Geometric fallback
            from examples.hrl.wrappers import HierarchicalCamera
            target_states, tracked_bits = self.get_all_opponent_states(observation)
            self.last_action = HierarchicalCamera.executor(
                self.state,
                target_states,
                target_selection_bits=self.last_selection,
                target_view_mask=tracked_bits,
            )

        return self.last_action
