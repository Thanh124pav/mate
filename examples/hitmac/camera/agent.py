"""HiTMACQPLEXCameraAgent — inference agent cho HiTMAC-QPLEX.

Kết hợp:
    - MAPPO Coordinator (từ examples/hrl/mappo/): giao task mỗi coord_period bước.
      Sử dụng coordinator_agent.last_selection để lấy task bits (không dùng
      continuous action từ executor geometry).
    - QPLEX Executor: chọn discrete camera action dựa trên [obs, task_bits].

Sử dụng:
    agent = HiTMACQPLEXCameraAgent(
        checkpoint_path='examples/hitmac/camera/ray_results/.../checkpoint',
        coordinator_checkpoint_path='examples/hrl/mappo/camera/ray_results/.../checkpoint',
    )
"""

import copy

import numpy as np
from ray.rllib.agents.qplex_v2.qplex_policy import QPLEXTorchPolicy

import mate
from examples.hitmac.camera.config import config as _config
from examples.hitmac.camera.config import make_env as _make_env
from examples.hitmac.wrappers import _GreedyAssigner
from examples.hrl.wrappers import MultiDiscrete2DiscreteActionMapper
from examples.utils import RLlibGroupedPolicyMixIn


class HiTMACQPLEXCameraAgent(RLlibGroupedPolicyMixIn, mate.CameraAgentBase):
    """HiTMAC Camera Agent: MAPPO coordinator + QPLEX executor tại inference.

    Args:
        config: Config dict cho QPLEX executor (mặc định: config từ camera/config.py).
        checkpoint_path: Checkpoint của QPLEX executor (Phase 2 output).
        coordinator_checkpoint_path: Checkpoint của MAPPO coordinator (Phase 1 output).
            Nếu None → dùng greedy fallback.
        make_env: Factory function tạo env (để lấy obs/action spaces).
        seed: Random seed.
    """

    POLICY_CLASS = QPLEXTorchPolicy
    DEFAULT_CONFIG = copy.deepcopy(_config)

    def get_policy(self):
        # params.pkl trong checkpoint có thể lưu mixer name cũ không khớp với
        # POLICY_CLASS hiện tại. Suy mixer name từ module path của POLICY_CLASS
        # để đồng bộ, tránh assert fail trong qplex_policy.py.
        policy_module = self.POLICY_CLASS.__module__  # e.g. 'ray.rllib.agents.qplex_v2.qplex_policy'
        expected_mixer = 'qplex_v2' if 'qplex_v2' in policy_module else 'qplex'
        if self.config.get('mixer') != expected_mixer:
            self.config['mixer'] = expected_mixer
        return super().get_policy()

    def __init__(
        self,
        config=None,
        checkpoint_path=None,
        coordinator_checkpoint_path=None,
        coordinator_agent_class=None,
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
        self.coord_period = env_config.get('coord_period', 5)
        self.discrete_levels = env_config.get('discrete_levels', 5)

        # Lưới action cho DiscreteCamera
        self.normalized_action_grid = mate.DiscreteCamera.discrete_action_grid(
            levels=self.discrete_levels
        )

        # Lưu để clone() truyền lại cho các bản sao
        self._coordinator_checkpoint_path = coordinator_checkpoint_path
        self._coordinator_agent_class = coordinator_agent_class

        # Coordinator
        self.coordinator_agent = None
        if coordinator_checkpoint_path is not None:
            self._init_coordinator(coordinator_checkpoint_path, coordinator_agent_class)

        # Greedy fallback (khởi tạo lazily tại reset)
        self._greedy_assigner = None

        # State per episode
        self.current_assignment = None  # [num_targets] bool
        self.last_action = None
        self._observation_slices = None

    @staticmethod
    def _detect_coordinator_class(checkpoint_path):
        """Auto-detect coordinator agent class từ params.pkl của checkpoint.

        Phân biệt dựa vào key 'mixer':
          - Có 'mixer'  → QPLEX-based coordinator  → HRLQPLEXV2CameraAgent
          - Không có    → PPO/MAPPO coordinator     → HRLMAPPOCameraAgent
        """
        from examples.utils import load_checkpoint

        _, _, params = load_checkpoint(checkpoint_path)
        if params is not None and 'mixer' in params:
            from examples.hrl.qplex_v2.camera.agent import HRLQPLEXV2CameraAgent
            return HRLQPLEXV2CameraAgent

        from examples.hrl.mappo.camera.agent import HRLMAPPOCameraAgent
        return HRLMAPPOCameraAgent

    def clone(self):
        # Override để truyền coordinator info — base clone() chỉ truyền executor params.
        return self.__class__(
            config=self.config,
            checkpoint_path=self.checkpoint_path,
            coordinator_checkpoint_path=self._coordinator_checkpoint_path,
            coordinator_agent_class=self._coordinator_agent_class,
            make_env=self.make_env,
            seed=self.np_random.randint(np.iinfo(int).max),
        )

    def _init_coordinator(self, coordinator_checkpoint_path, coordinator_agent_class=None):
        """Load coordinator từ checkpoint.

        Args:
            coordinator_checkpoint_path: path đến checkpoint.
            coordinator_agent_class: class của coordinator agent.
                Nếu None → auto-detect từ params.pkl của checkpoint.
        """
        if coordinator_agent_class is None:
            coordinator_agent_class = self._detect_coordinator_class(
                coordinator_checkpoint_path
            )

        self.coordinator_agent = coordinator_agent_class(
            checkpoint_path=coordinator_checkpoint_path,
            seed=self.np_random.randint(2**31),
        )
        # HiTMAC tự quản lý timing bằng coord_period — coordinator không nên
        # skip frames nội bộ, phải trả last_selection mỗi lần được gọi.
        self.coordinator_agent.frame_skip = 1

    def reset(self, observation):
        super().reset(observation)

        # Reset coordinator
        if self.coordinator_agent is not None:
            self.coordinator_agent.reset(observation)

        # Khởi tạo greedy fallback (lazily)
        if self.coordinator_agent is None and self._greedy_assigner is None:
            obs_slices = mate.camera_observation_slices_of(
                self.num_cameras, self.num_targets, num_obstacles=0
            )
            self._greedy_assigner = _GreedyAssigner(
                num_cameras=self.num_cameras,
                num_targets=self.num_targets,
                target_view_mask_slice=obs_slices['opponent_mask'],
            )
            self._observation_slices = obs_slices

        self.current_assignment = np.zeros(self.num_targets, dtype=np.bool8)
        self.last_action = None

    def act(self, observation, info=None, deterministic=None):
        self.state, observation, info, messages = self.check_inputs(observation, info)

        # --- Step 1: Cập nhật task assignment từ coordinator ---
        if self.episode_step % self.coord_period == 0:
            self.current_assignment = self._get_coordinator_assignment(observation)

        # --- Step 2: QPLEX executor chọn discrete camera action ---
        if self.episode_step % self.frame_skip == 0:
            # Preprocess raw observation FIRST (RelativeCoordinates + RescaledObservation),
            # THEN concatenate task bits — ensures task bits are NOT transformed.
            preprocessed_obs = self.preprocess_raw_observation(
                observation.ravel().astype(np.float32)
            )
            augmented_obs = np.concatenate(
                [preprocessed_obs.ravel(), self.current_assignment.astype(np.float32)]
            )

            # Temporarily disable preprocessing flags so compute_single_action does not
            # apply coordinate/rescale transforms to the already-preprocessed augmented obs.
            need_convert = self.need_convert_coordinates
            need_rescale = self.need_rescale_observation
            self.need_convert_coordinates = False
            self.need_rescale_observation = False
            try:
                discrete_action, self.hidden_state = self.compute_single_action(
                    augmented_obs,
                    state=self.hidden_state,
                    info=info,
                    deterministic=deterministic,
                )
            finally:
                self.need_convert_coordinates = need_convert
                self.need_rescale_observation = need_rescale

            # Chuyển discrete index → continuous camera action
            self.last_action = (
                self.action_space.high * self.normalized_action_grid[discrete_action]
            )

        return self.last_action

    def _get_coordinator_assignment(self, observation):
        """Lấy task assignment từ coordinator hoặc greedy fallback.

        Returns:
            assignment: np.ndarray [num_targets] bool
        """
        if self.coordinator_agent is not None:
            # Gọi MAPPO coordinator; lấy last_selection (bits) trước khi
            # executor geometry chuyển sang continuous action
            self.coordinator_agent.act(observation)
            if self.coordinator_agent.last_selection is not None:
                return self.coordinator_agent.last_selection.astype(np.bool8)

        # Greedy fallback: assign target gần nhất trong tầm nhìn
        if self._greedy_assigner is not None and self._observation_slices is not None:
            # Wrap observation thành dạng multi-camera để greedy assigner xử lý
            obs_2d = observation[np.newaxis, :]  # [1, D]
            single_assignments = self._greedy_assigner.assign(obs_2d)
            return single_assignments[0]  # [num_targets]

        # Fallback tuyệt đối: không assign gì
        return np.zeros(self.num_targets, dtype=np.bool8)
