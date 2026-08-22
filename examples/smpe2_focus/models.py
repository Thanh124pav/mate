import numpy as np
import torch
from ray.rllib.models import ModelCatalog
from ray.rllib.policy.sample_batch import SampleBatch

from examples.smpe2.models import SMPE2Model
from ray.rllib.agents.qplex_focus.qplex_policy import (
    CAMERA_STATE_DIM_PRIVATE,
    PRESERVED_DIM,
    TARGET_STATE_DIM_PRIVATE,
    LearnedOccupancyModel,
    _extract_camera_fov,
    _extract_target_positions,
    resolve_focus_config,
)
from ray.rllib.agents.focus_common import (
    FOCUS_CONFIDENCE,
    FOCUS_GAIN,
    FOCUS_RHO,
    FOCUS_RHO_ENTROPY,
    FOCUS_TOTAL_GAIN,
    FOCUS_VALID,
    FOCUS_WEIGHT,
    FocusResponsibilityEngine,
    MAPPOFocusAdapter,
)


class SMPE2FocusModel(SMPE2Model):
    """SMPE2 plus FOCUS counterfactual responsibility weighting."""

    def __init__(self, obs_space, action_space, num_outputs, model_config, name, **kwargs):
        super().__init__(obs_space, action_space, num_outputs, model_config, name, **kwargs)

        self.n_action_choices = int(getattr(self.action_space, "n", self.action_dim))
        custom_model_config = model_config.get("custom_model_config", {})
        self.focus_config = resolve_focus_config(
            {
                "focus": custom_model_config.get("focus", {}),
                "env_config": custom_model_config.get("env_config", {}),
            }
        )
        self.n_focus_agents = int(self.focus_config.get("n_agents", 4))
        self.focus_model = None
        self.focus_engine = None
        self.focus_adapter = None
        if self.focus_config.get("enabled", False):
            self.focus_model = LearnedOccupancyModel(
                self.global_state_dim,
                self.n_focus_agents,
                int(self.focus_config.get("n_targets", 8)),
                horizon=int(self.focus_config.get("horizon", 3)),
                hidden_dim=int(self.focus_config.get("belief_hidden_dim", 256)),
                max_delta=float(self.focus_config.get("belief_max_delta", 400.0)),
                min_std=float(self.focus_config.get("belief_min_std", 25.0)),
            )
            self.focus_engine = FocusResponsibilityEngine(
                self.focus_model,
                self.n_focus_agents,
                self.n_action_choices,
                self.focus_config,
            )
            self.focus_adapter = MAPPOFocusAdapter(
                self.focus_config.get(
                    "policy_eta",
                    self.focus_config.get("focus_policy_eta", 0.0),
                )
            )
        self._focus_stats = {}
        self._last_action_bias_stats = {}
        levels = int(round(self.n_action_choices ** 0.5))
        if levels * levels == self.n_action_choices:
            action_grid = np.stack(
                np.meshgrid(
                    np.linspace(-1.0, 1.0, num=levels),
                    np.linspace(-1.0, 1.0, num=levels),
                ),
                axis=-1,
            ).reshape(-1, 2)
        else:
            action_grid = np.zeros((self.n_action_choices, 2), dtype=np.float32)
        self.register_buffer(
            "focus_action_grid",
            torch.as_tensor(action_grid, dtype=torch.float32),
            persistent=False,
        )

    def forward_rnn(self, inputs, state, seq_lens):
        action_out, new_state = super().forward_rnn(inputs, state, seq_lens)
        eta = float(self.focus_config.get("action_bias_eta", 0.0))
        if eta > 0.0:
            action_bias = self._focus_action_bias(inputs)
            if action_bias is not None and action_bias.shape == action_out.shape:
                action_out = action_out + eta * action_bias.to(action_out.dtype)
                valid = torch.isfinite(action_bias)
                if valid.any():
                    self._last_action_bias_stats = {
                        "focus/action_bias_mean": action_bias[valid].mean().detach(),
                        "focus/action_bias_std": action_bias[valid].std(unbiased=False).detach(),
                        "focus/action_bias_eta": torch.tensor(eta, device=action_out.device),
                    }
        return action_out, new_state

    def _focus_action_bias(self, inputs):
        if self.focus_action_grid.size(0) != self.n_action_choices:
            return None
        if self.global_state_slice.stop > inputs.size(-1):
            return None
        n_targets = int(self.focus_config.get("n_targets", 8))
        required_dim = PRESERVED_DIM + self.n_focus_agents * CAMERA_STATE_DIM_PRIVATE
        required_dim += n_targets * TARGET_STATE_DIM_PRIVATE
        if self.global_state_dim < required_dim:
            return None

        local_obs = inputs[..., self.local_obs_slice]
        global_state = inputs[..., self.global_state_slice]
        agent_index = local_obs[..., 3].round().long().clamp(0, self.n_focus_agents - 1)
        cam_pos, cam_orient, cam_range, cam_half_angle = _extract_camera_fov(
            global_state, self.n_focus_agents
        )
        target_pos = _extract_target_positions(global_state, self.n_focus_agents, n_targets)

        gather_xy = agent_index.view(*agent_index.shape, 1, 1).expand(-1, -1, 1, 2)
        own_pos = cam_pos.gather(2, gather_xy).squeeze(2)
        gather_scalar = agent_index.unsqueeze(-1)
        own_orient = cam_orient.gather(2, gather_scalar).squeeze(-1)
        own_range = cam_range.gather(2, gather_scalar).squeeze(-1).clamp_min(1.0)
        own_half = cam_half_angle.gather(2, gather_scalar).squeeze(-1).clamp_min(1e-3)

        grid = self.focus_action_grid.to(device=inputs.device, dtype=inputs.dtype)
        rotation_step = float(self.focus_config.get("action_bias_rotation_step", 5.0))
        zooming_step = float(self.focus_config.get("action_bias_zooming_step", 2.5))
        min_angle = float(self.focus_config.get("action_bias_min_viewing_angle", 30.0))
        max_angle = float(self.focus_config.get("action_bias_max_viewing_angle", 180.0))
        max_range = float(self.focus_config.get("action_bias_max_sight_range", 1500.0))

        delta_orient = grid[:, 0].view(1, 1, -1) * (rotation_step * np.pi / 180.0)
        orient = own_orient.unsqueeze(-1) + delta_orient
        current_angle = (own_half * 2.0 * 180.0 / np.pi).clamp(min=min_angle, max=max_angle)
        next_angle = (current_angle.unsqueeze(-1) + grid[:, 1].view(1, 1, -1) * zooming_step)
        next_angle = next_angle.clamp(min=min_angle, max=max_angle)
        area_product = own_range.square().unsqueeze(-1) * current_angle.unsqueeze(-1)
        next_range = torch.sqrt(area_product / next_angle.clamp_min(1e-3)).clamp(1.0, max_range)
        next_half = next_angle * (np.pi / 180.0) / 2.0

        rel = target_pos.unsqueeze(2) - own_pos.unsqueeze(2).unsqueeze(3)
        dist = torch.sqrt(rel.square().sum(dim=-1) + 1e-8)
        target_angle = torch.atan2(rel[..., 1], rel[..., 0])
        angle_delta = torch.atan2(
            torch.sin(target_angle - orient.unsqueeze(-1)),
            torch.cos(target_angle - orient.unsqueeze(-1)),
        ).abs()

        angular_temp = float(self.focus_config.get("action_bias_angular_temp", 0.12))
        range_temp = float(self.focus_config.get("action_bias_range_temp", 100.0))
        angular_score = torch.sigmoid((next_half.unsqueeze(-1) - angle_delta) / angular_temp)
        range_score = torch.sigmoid((next_range.unsqueeze(-1) - dist) / range_temp)
        target_weight = torch.ones_like(dist)
        if self.focus_config.get("action_bias_real_only", True):
            weights = []
            target_start = PRESERVED_DIM + self.n_focus_agents * CAMERA_STATE_DIM_PRIVATE
            for j in range(n_targets):
                weights.append(global_state[..., target_start + j * TARGET_STATE_DIM_PRIVATE + 3])
            target_weight = torch.stack(weights, dim=-1).clamp(0.0, 1.0).unsqueeze(2)

        score = (angular_score * range_score * target_weight).sum(dim=-1)
        centered = score - score.mean(dim=-1, keepdim=True)
        scale = centered.std(dim=-1, keepdim=True, unbiased=False).clamp_min(1e-4)
        bias = centered / scale
        clip = float(self.focus_config.get("action_bias_clip", 2.0))
        return bias.clamp(-clip, clip)

    def _add_default_focus_fields(self, sample_batch):
        count = len(sample_batch)
        sample_batch[FOCUS_RHO] = np.full(
            count, 1.0 / max(self.n_focus_agents, 1), dtype=np.float32
        )
        sample_batch[FOCUS_WEIGHT] = np.ones(count, dtype=np.float32)
        sample_batch[FOCUS_VALID] = np.zeros(count, dtype=np.bool_)
        sample_batch[FOCUS_CONFIDENCE] = np.zeros(count, dtype=np.float32)
        sample_batch[FOCUS_GAIN] = np.zeros(count, dtype=np.float32)
        sample_batch[FOCUS_RHO_ENTROPY] = np.zeros(count, dtype=np.float32)
        sample_batch[FOCUS_TOTAL_GAIN] = np.zeros(count, dtype=np.float32)
        return sample_batch

    @staticmethod
    def _unwrap_other_batch(value):
        if isinstance(value, tuple) and len(value) == 2:
            return value[1]
        return value

    @staticmethod
    def _agent_id_to_index(agent_id):
        try:
            return int(str(agent_id).rsplit("_", 1)[-1])
        except (TypeError, ValueError):
            return None

    def add_focus_to_trajectory(self, policy, sample_batch, other_agent_batches=None, episode=None):
        del policy, episode
        self._add_default_focus_fields(sample_batch)
        if self.focus_engine is None or self.focus_adapter is None:
            return sample_batch
        if other_agent_batches is None:
            return sample_batch
        required = [
            SampleBatch.EPS_ID,
            SampleBatch.AGENT_INDEX,
            SampleBatch.ACTIONS,
            SampleBatch.CUR_OBS,
            SampleBatch.NEXT_OBS,
        ]
        if any(key not in sample_batch for key in required):
            return sample_batch

        other_entries = []
        other_agent_indices = set()
        for agent_id, value in other_agent_batches.items():
            batch = self._unwrap_other_batch(value)
            agent_index = self._agent_id_to_index(agent_id)
            if batch is not None and all(key in batch for key in required[:3]):
                other_entries.append((agent_index, batch))
                if agent_index is not None:
                    other_agent_indices.add(agent_index)

        expected_agent_indices = set(range(self.n_focus_agents))
        current_agent_index = None
        missing_agent_indices = expected_agent_indices - other_agent_indices
        if len(missing_agent_indices) == 1 and len(other_entries) == self.n_focus_agents - 1:
            current_agent_index = next(iter(missing_agent_indices))

        if len(other_entries) != self.n_focus_agents - 1:
            return sample_batch

        all_batches = [(current_agent_index, sample_batch), *other_entries]
        use_timestep_key = all(SampleBatch.T in batch for _, batch in all_batches)
        if not use_timestep_key and any(len(batch) != len(sample_batch) for _, batch in all_batches):
            return sample_batch

        def row_key(batch, row):
            env_ids = batch.get(SampleBatch.ENV_ID)
            env_value = int(env_ids[row]) if env_ids is not None else 0
            step_value = int(batch[SampleBatch.T][row]) if use_timestep_key else row
            return int(batch[SampleBatch.EPS_ID][row]), env_value, step_value

        rows_by_key = {}
        for agent_index_hint, batch in all_batches:
            for row in range(len(batch)):
                key = row_key(batch, row)
                if agent_index_hint is None:
                    agent_index = int(batch[SampleBatch.AGENT_INDEX][row])
                else:
                    agent_index = int(agent_index_hint)
                rows_by_key.setdefault(key, {})[agent_index] = (batch, row)

        row_indices = []
        agent_indices = []
        states = []
        next_states = []
        joint_actions = []
        for row in range(len(sample_batch)):
            members = rows_by_key.get(row_key(sample_batch, row), {})
            if len(members) != self.n_focus_agents:
                continue
            expected = set(range(self.n_focus_agents))
            if set(members) != expected:
                continue

            actions = []
            for agent_index in range(self.n_focus_agents):
                batch, member_row = members[agent_index]
                actions.append(batch[SampleBatch.ACTIONS][member_row])
            row_indices.append(row)
            if current_agent_index is None:
                agent_indices.append(int(sample_batch[SampleBatch.AGENT_INDEX][row]))
            else:
                agent_indices.append(int(current_agent_index))
            states.append(sample_batch[SampleBatch.CUR_OBS][row, self.global_state_slice])
            next_states.append(sample_batch[SampleBatch.NEXT_OBS][row, self.global_state_slice])
            joint_actions.append(actions)

        if not row_indices:
            return sample_batch

        device = next(self.focus_engine.parameters()).device
        state_tensor = torch.as_tensor(
            np.asarray(states, dtype=np.float32), dtype=torch.float32, device=device
        ).unsqueeze(0)
        next_state_tensor = torch.as_tensor(
            np.asarray(next_states, dtype=np.float32), dtype=torch.float32, device=device
        ).unsqueeze(0)
        action_tensor = torch.as_tensor(
            np.asarray(joint_actions), dtype=torch.long, device=device
        ).unsqueeze(0)
        valid_mask = torch.ones((1, len(row_indices)), dtype=torch.float32, device=device)

        with torch.no_grad():
            focus = self.focus_engine(
                global_state=state_tensor,
                joint_actions=action_tensor,
                valid_mask=valid_mask,
                next_global_state=next_state_tensor,
            )
            confidence = (
                focus.confidence
                if self.focus_config.get(
                    "use_confidence",
                    self.focus_config.get("focus_use_confidence", True),
                )
                else None
            )
            weights = self.focus_adapter.responsibility_to_weight(
                focus.rho, self.n_focus_agents, confidence=confidence
            )

        rows_np = np.asarray(row_indices, dtype=np.int64)
        agents_np = np.asarray(agent_indices, dtype=np.int64)
        rho_np = focus.rho.squeeze(0).detach().cpu().numpy()
        gain_np = focus.gains.squeeze(0).detach().cpu().numpy()
        valid_np = focus.valid.squeeze(0).detach().cpu().numpy().astype(np.bool_)
        confidence_np = focus.confidence.squeeze(0).detach().cpu().numpy()
        weight_np = weights.squeeze(0).detach().cpu().numpy()
        eps = float(self.focus_config.get("eps", 1e-8))
        rho_entropy_np = -(rho_np * np.log(rho_np + eps)).sum(axis=-1)
        total_gain_np = focus.total_gain.squeeze(0).detach().cpu().numpy()

        sample_batch[FOCUS_RHO][rows_np] = rho_np[np.arange(len(rows_np)), agents_np]
        sample_batch[FOCUS_GAIN][rows_np] = gain_np[np.arange(len(rows_np)), agents_np]
        sample_batch[FOCUS_CONFIDENCE][rows_np] = confidence_np
        sample_batch[FOCUS_RHO_ENTROPY][rows_np] = rho_entropy_np
        sample_batch[FOCUS_TOTAL_GAIN][rows_np] = total_gain_np
        sample_batch[FOCUS_VALID][rows_np] = valid_np
        sample_batch[FOCUS_WEIGHT][rows_np] = np.where(
            valid_np,
            weight_np[np.arange(len(rows_np)), agents_np],
            1.0,
        ).astype(np.float32)
        return sample_batch

    def custom_loss(self, policy_loss, loss_inputs):
        policy_loss = super().custom_loss(policy_loss, loss_inputs)
        if self.focus_model is None:
            return policy_loss
        obs = loss_inputs[SampleBatch.CUR_OBS].float()
        if obs.size(-1) != self.flat_obs_dim or SampleBatch.SEQ_LENS not in loss_inputs:
            return policy_loss
        seq_lens = loss_inputs[SampleBatch.SEQ_LENS].long()
        B = seq_lens.numel()
        if B == 0:
            return policy_loss
        T = obs.shape[0] // B
        if T < 2:
            return policy_loss

        global_state = obs[:, self.global_state_slice].reshape(B, T, self.global_state_dim)
        state = global_state[:, :-1, :]
        future_state = global_state[:, 1:, :]
        per_horizon_losses, per_horizon_valid, *_ = self.focus_model.nll(state, future_state)
        discount = float(self.focus_config.get("horizon_discount", 0.9))
        weights = torch.tensor(
            [discount ** h for h in range(len(per_horizon_losses))],
            dtype=global_state.dtype,
            device=global_state.device,
        )
        weights = weights / (weights.sum() + float(self.focus_config.get("eps", 1e-8)))
        time_valid = torch.arange(T - 1, device=global_state.device).unsqueeze(0) < (
            seq_lens - 1
        ).clamp_min(0).unsqueeze(1)
        if SampleBatch.AGENT_INDEX in loss_inputs:
            agent_index = loss_inputs[SampleBatch.AGENT_INDEX].long().reshape(B, T)[:, :-1]
            time_valid = time_valid & (agent_index == 0)

        belief_terms = []
        for h, (nll_per_step, valid_h) in enumerate(zip(per_horizon_losses, per_horizon_valid)):
            valid_h = valid_h & time_valid
            if valid_h.any():
                belief_terms.append(weights[h] * nll_per_step[valid_h].mean())
        reference_loss = policy_loss[0] if isinstance(policy_loss, list) else policy_loss
        belief_loss = (
            torch.stack(belief_terms).sum()
            if belief_terms
            else torch.zeros_like(reference_loss)
        )
        beta = float(self.focus_config.get("beta_belief", 0.01))
        self._focus_stats = {
            "focus/belief_loss": belief_loss.detach(),
            "focus/beta_belief": torch.tensor(beta, device=belief_loss.device),
        }
        if isinstance(policy_loss, list):
            return [loss + beta * belief_loss for loss in policy_loss]
        return policy_loss + beta * belief_loss

    def metrics(self):
        return {**self._focus_stats, **self._last_action_bias_stats}


ModelCatalog.register_custom_model("SMPE2FocusModel", SMPE2FocusModel)
