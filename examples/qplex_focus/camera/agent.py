import copy
import sys
from pathlib import Path

import numpy as np


from ray.rllib.agents.qplex_focus.qplex_policy import (
    QPLEXFocusTorchPolicy,
    _extract_target_positions_normalized,
    _local_visible_target_positions,
    _mac,
    _normalized_target_pos_to_world,
)
import mate
from examples.qplex_focus.camera.config import config as _config
from examples.qplex_focus.camera.config import make_env as _make_env
from examples.utils import RLlibGroupedPolicyMixIn


class QPLEXFocusCameraAgent(RLlibGroupedPolicyMixIn, mate.CameraAgentBase):
    """QPLEX+FOCUS Camera Agent

    A wrapper for the trained RLlib policy.

    Note:
        The agent always produces a primitive continuous action. If the RLlib policy is trained with
        discrete actions, the output action will be converted to primitive continuous action.
    """

    POLICY_CLASS = QPLEXFocusTorchPolicy
    DEFAULT_CONFIG = copy.deepcopy(_config)

    def __init__(
        self,
        config=None,
        checkpoint_path=None,
        make_env=_make_env,
        seed=None,
        decentralized_execution=True,
        decentralized_fallback_mode='off',
        fallback_margin=0.0,
        decentralized_local_belief_guide=True,
        local_belief_guide_eta=None,
        student_action_bias_eta=None,
        local_belief_action_eta=None,
    ):
        super().__init__(
            config=config, checkpoint_path=checkpoint_path, make_env=make_env, seed=seed
        )

        self.decentralized_execution = bool(decentralized_execution)
        self.decentralized_fallback_mode = str(decentralized_fallback_mode).lower()
        self.fallback_margin = float(fallback_margin)
        self.decentralized_local_belief_guide = bool(decentralized_local_belief_guide)
        self.local_belief_guide_eta = (
            None if local_belief_guide_eta is None else float(local_belief_guide_eta)
        )
        self.student_action_bias_eta = (
            None if student_action_bias_eta is None else float(student_action_bias_eta)
        )
        self.local_belief_action_eta = (
            None if local_belief_action_eta is None else float(local_belief_action_eta)
        )
        if self.decentralized_execution:
            focus_config = self.config.setdefault('focus', {})
            self.config['explore'] = False
            self.policy.config['explore'] = False
            focus_config['action_bias_eta'] = 0.0
            self.policy.focus_config['action_bias_eta'] = 0.0
            focus_config['student_action_bias_enabled'] = self.decentralized_local_belief_guide
            self.policy.focus_config[
                'student_action_bias_enabled'
            ] = self.decentralized_local_belief_guide
            focus_config['local_belief_action_guide_enabled'] = self.decentralized_local_belief_guide
            self.policy.focus_config[
                'local_belief_action_guide_enabled'
            ] = self.decentralized_local_belief_guide
            fallback_eta = self.local_belief_guide_eta
            student_eta = self.student_action_bias_eta
            if student_eta is None:
                student_eta = fallback_eta
            belief_eta = self.local_belief_action_eta
            if belief_eta is None:
                belief_eta = fallback_eta
            if student_eta is not None:
                focus_config['student_action_bias_eta'] = student_eta
                self.policy.focus_config['student_action_bias_eta'] = student_eta
            if belief_eta is not None:
                focus_config['local_belief_action_eta'] = belief_eta
                self.policy.focus_config['local_belief_action_eta'] = belief_eta

        self.greedy_fallback = None
        if self.decentralized_execution and self.decentralized_fallback_mode != 'off':
            self.greedy_fallback = mate.agents.GreedyCameraAgent(seed=seed)

        self.frame_skip = self.config.get('env_config', {}).get('frame_skip', 1)
        self.discrete_levels = self.config.get('env_config', {}).get('discrete_levels', None)
        assert self.discrete_levels is not None, 'QPLEX only supports discrete actions.'
        self.normalized_action_grid = mate.DiscreteCamera.discrete_action_grid(
            levels=self.discrete_levels
        )

        self.last_action = None


    def belief_diagnostics(self, observation, normalized_global_state):
        """Evaluate local belief predictions against the centralized state label."""
        policy = getattr(self, 'policy', None)
        local_belief_model = getattr(policy, 'local_belief_model', None)
        if policy is None or local_belief_model is None or self.hidden_state is None:
            return {}

        # pylint: disable-next=import-outside-toplevel
        import torch

        single_agent_observation = np.asarray(observation).ndim == 1
        preprocessed = self.preprocess_observation(observation)
        obs_np, _, state_np = policy._unpack_observation([preprocessed])  # pylint: disable=protected-access
        if normalized_global_state is not None:
            state_np = np.asarray(normalized_global_state, dtype=np.float32).reshape(1, -1)
        if state_np is None:
            return {}

        device = getattr(policy, 'device', None)
        obs_tensor = torch.as_tensor(obs_np, dtype=torch.float, device=device)
        hidden = [
            torch.as_tensor(np.asarray(s), dtype=torch.float, device=device).reshape(
                1, policy.n_agents, -1
            )
            for s in self.hidden_state
        ]
        state_tensor = torch.as_tensor(state_np.reshape(1, 1, -1), dtype=torch.float, device=device)

        local_belief_model.eval()
        with torch.no_grad():
            _, hiddens = _mac(policy.model, obs_tensor, hidden)
            mean, std = local_belief_model(hiddens[0])
            target = _extract_target_positions_normalized(
                state_tensor, policy.n_agents, int(policy.focus_config.get('n_targets', 8))
            )
            if target.dim() == mean.dim() and target.size(1) == 1:
                target = target.squeeze(1)
            target = target.unsqueeze(-3).expand_as(mean)
            pred_world = _normalized_target_pos_to_world(mean)
            target_world = _normalized_target_pos_to_world(target)
            error = torch.linalg.norm(pred_world - target_world, dim=-1)
            observed_pos, visible_mask = _local_visible_target_positions(
                obs_tensor, int(policy.focus_config.get('n_targets', 8))
            )
            if single_agent_observation:
                error = error[:, :1]
                std = std[:, :1]
                target_world = target_world[:, :1]
                if observed_pos is not None:
                    observed_pos = observed_pos[:, :1]
                if visible_mask is not None:
                    visible_mask = visible_mask[:, :1]

            diagnostics = {
                'belief_local_pos_error': error.mean().item(),
                'belief_local_std_world': (std * 2000.0).mean().item(),
            }
            if visible_mask is not None:
                visible_mask = visible_mask.to(device=error.device, dtype=torch.bool)
                diagnostics['belief_visible_target_ratio'] = visible_mask.float().mean().item()
                diagnostics['belief_visible_step_ratio'] = visible_mask.any(dim=-1).float().mean().item()
                if visible_mask.any():
                    diagnostics['belief_local_visible_pos_error'] = error[visible_mask].mean().item()
                invisible_mask = ~visible_mask
                if invisible_mask.any():
                    diagnostics['belief_local_invisible_pos_error'] = error[invisible_mask].mean().item()
                if observed_pos is not None and visible_mask.any():
                    observed_error = torch.linalg.norm(
                        observed_pos.to(device=target_world.device, dtype=target_world.dtype)
                        - target_world,
                        dim=-1,
                    )
                    diagnostics['belief_visible_obs_pos_error'] = observed_error[visible_mask].mean().item()
            return diagnostics

    def reset(self, observation):
        super().reset(observation)
        if self.greedy_fallback is not None:
            self.greedy_fallback.reset(observation)

        self.last_action = None

    def observe(self, observation, info=None):
        super().observe(observation, info)
        if self.greedy_fallback is not None:
            self.greedy_fallback.observe(observation, info)

    def send_responses(self):
        if self.greedy_fallback is not None:
            return self.greedy_fallback.send_responses()
        return super().send_responses()

    def receive_responses(self, messages):
        super().receive_responses(messages)
        if self.greedy_fallback is not None:
            self.greedy_fallback.receive_responses(messages)

    def _should_use_greedy_fallback(self, policy_action, greedy_action):
        if self.greedy_fallback is None:
            return False
        if self.decentralized_fallback_mode == 'greedy':
            return True
        if self.decentralized_fallback_mode != 'auto':
            return False
        if policy_action is None:
            return True
        # Conservative auto mode: if the learned action barely moves the camera
        # while greedy sees a concrete local target, prefer the local heuristic.
        return np.linalg.norm(greedy_action) > np.linalg.norm(policy_action) + self.fallback_margin

    def act(self, observation, info=None, deterministic=None):
        self.state, observation, info, messages = self.check_inputs(observation, info)
        if self.decentralized_execution:
            deterministic = True
            assert self.policy.config.get('explore') is False
            assert float(self.policy.focus_config.get('action_bias_eta', 0.0)) == 0.0
            assert self.policy.focus_config.get('student_action_bias_enabled', False) == (
                self.decentralized_local_belief_guide
            )
        elif deterministic is None:
            deterministic = True

        if self.episode_step % self.frame_skip == 0:
            if self.greedy_fallback is not None and self.decentralized_fallback_mode == 'greedy':
                self.last_action = self.greedy_fallback.act(
                    observation, info=info, deterministic=deterministic
                )
                return self.last_action

            policy_action, self.hidden_state = self.compute_single_action(
                observation, state=self.hidden_state, info=info, deterministic=deterministic
            )

            if self.normalized_action_grid is not None:
                # Convert discretized action to primitive continuous action
                policy_action = self.action_space.high * self.normalized_action_grid[policy_action]

            greedy_action = None
            if self.greedy_fallback is not None:
                greedy_action = self.greedy_fallback.act(
                    observation, info=info, deterministic=deterministic
                )

            if self._should_use_greedy_fallback(policy_action, greedy_action):
                self.last_action = greedy_action
            else:
                self.last_action = policy_action

        return self.last_action
