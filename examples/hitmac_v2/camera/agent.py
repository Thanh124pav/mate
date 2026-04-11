"""HiTMACv2PPOCameraAgent — inference agent for HiTMAC v2."""

import copy

import numpy as np
from ray.rllib.agents.ppo import PPOTorchPolicy

import mate
from examples.hitmac_v2.camera.config import config as _config
from examples.hitmac_v2.camera.config import make_env as _make_env
from examples.hitmac_v2.wrappers import _GreedyAssigner
from examples.utils import RLlibPolicyMixIn


class HiTMACv2PPOCameraAgent(RLlibPolicyMixIn, mate.CameraAgentBase):
    """HiTMAC v2 Camera Agent: PPO executor + greedy/trained coordinator.

    At inference:
    - Every coord_period steps: coordinator assigns targets (greedy or MAPPO).
    - Every frame_skip steps: PPO executor selects continuous camera action
      given augmented observation [obs | assignment_bits].
    """

    POLICY_CLASS = PPOTorchPolicy
    DEFAULT_CONFIG = copy.deepcopy(_config)

    def __init__(self, config=None, checkpoint_path=None,
                 coordinator_checkpoint_path=None, make_env=_make_env, seed=None):
        super().__init__(
            config=config,
            checkpoint_path=checkpoint_path,
            make_env=make_env,
            seed=seed,
        )
        env_config = self.config.get('env_config', {})
        self.frame_skip = env_config.get('frame_skip', 5)
        self.coord_period = env_config.get('coord_period', 5)

        self._coordinator_checkpoint_path = coordinator_checkpoint_path
        self.coordinator_agent = None
        if coordinator_checkpoint_path is not None:
            self._init_coordinator(coordinator_checkpoint_path)

        self._greedy_assigner = None
        self._observation_slices = None
        self.current_assignment = None
        self.last_action = None

    def _init_coordinator(self, checkpoint_path):
        from examples.hrl.mappo.camera.agent import HRLMAPPOCameraAgent
        self.coordinator_agent = HRLMAPPOCameraAgent(
            checkpoint_path=checkpoint_path,
            seed=self.np_random.randint(2**31),
        )
        self.coordinator_agent.frame_skip = 1

    def reset(self, observation):
        super().reset(observation)

        if self.coordinator_agent is not None:
            self.coordinator_agent.reset(observation)

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

        # Step 1: Update task assignment from coordinator
        if self.episode_step % self.coord_period == 0:
            self.current_assignment = self._get_assignment(observation)

        # Step 2: PPO executor selects action with augmented obs
        if self.episode_step % self.frame_skip == 0:
            augmented_obs = np.concatenate([
                observation.ravel().astype(np.float32),
                self.current_assignment.astype(np.float32),
            ])

            action, self.hidden_state = self.compute_single_action(
                augmented_obs,
                state=self.hidden_state,
                info=info,
                deterministic=deterministic,
            )
            self.last_action = action

        return self.last_action

    def _get_assignment(self, observation):
        if self.coordinator_agent is not None:
            self.coordinator_agent.act(observation)
            if self.coordinator_agent.last_selection is not None:
                return self.coordinator_agent.last_selection.astype(np.bool8)

        if self._greedy_assigner is not None:
            obs_2d = observation[np.newaxis, :]
            single_assignments = self._greedy_assigner.assign(obs_2d)
            return single_assignments[0]

        return np.zeros(self.num_targets, dtype=np.bool8)
