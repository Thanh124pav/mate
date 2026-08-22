from collections import OrderedDict

import numpy as np
from gym import spaces
from ray.rllib.evaluation.postprocessing import Postprocessing
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.recurrent_net import RecurrentNetwork as TorchRNN
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.framework import try_import_torch

from examples.utils import SimpleRNN, get_space_flat_size, orthogonal_initializer
from ray.rllib.agents.qplex_focus.qplex_policy import LearnedOccupancyModel, resolve_focus_config
from ray.rllib.agents.focus_utils import focus_action_q_bias
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


torch, nn = try_import_torch()


class MAPPOModel(TorchRNN, nn.Module):
    def __init__(
        self,
        obs_space,
        action_space,
        num_outputs,
        model_config,
        name,
        # Extra MAPPOModel arguments
        actor_hiddens=None,
        actor_hidden_activation='tanh',
        critic_hiddens=None,
        critic_hidden_activation='tanh',
        lstm_cell_size=256,
        **kwargs,
    ):
        if actor_hiddens is None:
            actor_hiddens = [256, 256]

        if critic_hiddens is None:
            critic_hiddens = [256, 256]

        nn.Module.__init__(self)
        super().__init__(obs_space, action_space, num_outputs, model_config, name)

        assert hasattr(obs_space, 'original_space') and isinstance(
            obs_space.original_space, spaces.Dict
        )
        original_space = obs_space.original_space
        self.local_obs_space = original_space['obs']
        self.global_state_space = original_space['state']
        if 'action_mask' in original_space.spaces:
            self.action_mask_space = original_space['action_mask']
            self.has_action_mask = True
        else:
            self.action_mask_space = None
            self.has_action_mask = False

        self.flat_obs_dim = get_space_flat_size(self.obs_space)
        self.space_dims = OrderedDict(
            [(key, get_space_flat_size(subspace)) for key, subspace in original_space.items()]
        )
        indices = np.cumsum([0, *self.space_dims.values()])
        self.flat_obs_slices = OrderedDict(
            [
                (key, slice(indices[i], indices[i + 1]))
                for i, key in enumerate(self.space_dims.keys())
            ]
        )

        self.local_obs_dim = self.space_dims['obs']
        self.local_obs_slice = self.flat_obs_slices['obs']
        self.global_state_dim = self.space_dims['state']
        self.global_state_slice = self.flat_obs_slices['state']

        self.action_dim = get_space_flat_size(self.action_space)
        self.n_action_choices = int(getattr(self.action_space, 'n', self.action_dim))

        if self.has_action_mask:
            self.action_mask_slice = self.flat_obs_slices['action_mask']
            assert self.space_dims['action_mask'] == num_outputs
        else:
            self.action_mask_slice = None

        self.actor_hiddens = actor_hiddens or []
        self.critic_hiddens = critic_hiddens or list(self.actor_hiddens)
        self.actor_hidden_activation = actor_hidden_activation
        self.critic_hidden_activation = critic_hidden_activation
        self.lstm_cell_size = lstm_cell_size

        self.actor = SimpleRNN(
            name='actor',
            input_dim=self.local_obs_dim,
            hidden_dims=self.actor_hiddens,
            cell_size=self.lstm_cell_size,
            output_dim=num_outputs,
            activation=self.actor_hidden_activation,
            output_activation=None,
            hidden_weight_initializer=orthogonal_initializer(scale=1.0),
            output_weight_initializer=orthogonal_initializer(scale=0.01),
        )

        self.critic = SimpleRNN(
            name='critic',
            input_dim=self.global_state_dim,
            hidden_dims=self.critic_hiddens,
            cell_size=self.lstm_cell_size,
            output_dim=1,
            activation=self.critic_hidden_activation,
            output_activation=None,
            hidden_weight_initializer=orthogonal_initializer(scale=1.0),
            output_weight_initializer=orthogonal_initializer(scale=1.0),
        )
        custom_model_config = model_config.get("custom_model_config", {})
        self.focus_config = resolve_focus_config({
            "focus": custom_model_config.get("focus", {}),
            "env_config": custom_model_config.get("env_config", {}),
        })
        self.n_focus_agents = int(self.focus_config.get("n_agents", 4))
        self.belief_state_enabled = bool(self.focus_config.get("belief_state_enabled", False))
        self.belief_state_loss = str(self.focus_config.get("belief_state_loss", "mse")).lower()
        self.belief_state_detach_for_prior = bool(
            self.focus_config.get("belief_state_detach_for_prior", True)
        )
        actor_feature_dim = (self.actor_hiddens[-1] if self.actor_hiddens else self.local_obs_dim)
        actor_feature_dim += self.lstm_cell_size
        belief_state_hidden_dim = int(
            self.focus_config.get("belief_state_hidden_dim", self.lstm_cell_size)
        )
        self.belief_state_head = None
        if self.belief_state_enabled:
            self.belief_state_head = nn.Sequential(
                nn.Linear(actor_feature_dim, belief_state_hidden_dim),
                nn.Tanh(),
                nn.Linear(belief_state_hidden_dim, self.global_state_dim),
                nn.Tanh(),
            )
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
        self._last_action_logits = None
        self._last_belief_state = None

    def get_initial_state(self):
        return [*self.actor.get_initial_state(), *self.critic.get_initial_state()]

    def forward_rnn(self, inputs, state, seq_lens):
        assert inputs.size(-1) == self.flat_obs_dim

        local_obs = inputs[..., self.local_obs_slice]
        actor_state_in = state[:2]
        action_out, actor_state_out = self.actor(local_obs, actor_state_in)

        self._last_belief_state = None
        if self.belief_state_head is not None:
            belief_state = self.belief_state_head(self.actor.last_features)
            self._last_belief_state = belief_state.reshape(-1, belief_state.size(-1))

        if self.has_action_mask:
            action_mask = inputs[..., self.action_mask_slice].clamp(min=0.0, max=1.0)
            inf_mask = torch.log(action_mask).clamp_min(min=torch.finfo(action_out.dtype).min)
            action_out = action_out + inf_mask

        if self.focus_config.get("action_prior_enabled", False) and self.focus_config.get(
            "action_prior_apply_to_logits", False
        ):
            flat_inputs = inputs.reshape(-1, inputs.size(-1))
            flat_logits = action_out.reshape(-1, action_out.size(-1))
            selected_bias = self._selected_focus_action_bias(
                flat_inputs,
                flat_logits,
                self._last_belief_state,
            )
            if selected_bias is not None:
                action_out = action_out + selected_bias.reshape_as(action_out)

        self._last_action_logits = action_out.reshape(-1, action_out.size(-1))

        global_state = inputs[..., self.global_state_slice]
        critic_state_in = state[2:]
        _, critic_state_out = self.critic(global_state, critic_state_in, features_only=True)

        return action_out, [*actor_state_out, *critic_state_out]

    def value_function(self):
        assert self.critic.last_features is not None, 'must call forward() first'

        return self.critic.output(self.critic.last_features).reshape(-1)

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
        valid_mask = torch.ones(
            (1, len(row_indices)), dtype=torch.float32, device=device
        )

        with torch.no_grad():
            focus = self.focus_engine(
                global_state=state_tensor,
                joint_actions=action_tensor,
                valid_mask=valid_mask,
                next_global_state=next_state_tensor,
            )
            confidence = focus.confidence if self.focus_config.get("use_confidence", self.focus_config.get("focus_use_confidence", True)) else None
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

    def _flat_sequence_mask(self, obs, loss_inputs):
        if SampleBatch.SEQ_LENS not in loss_inputs:
            return torch.ones(obs.shape[0], dtype=torch.bool, device=obs.device)
        seq_lens = loss_inputs[SampleBatch.SEQ_LENS].long().to(obs.device)
        B = seq_lens.numel()
        if B == 0 or obs.shape[0] % B != 0:
            return torch.ones(obs.shape[0], dtype=torch.bool, device=obs.device)
        T = obs.shape[0] // B
        return (
            torch.arange(T, device=obs.device).unsqueeze(0) < seq_lens.unsqueeze(1)
        ).reshape(-1)

    def _belief_state_loss(self, obs, loss_inputs, reference_loss):
        coeff = float(
            self.focus_config.get(
                "belief_state_coeff",
                self.focus_config.get("beta_belief", 0.0),
            )
        )
        prediction = self._last_belief_state
        if (
            coeff <= 0.0
            or prediction is None
            or prediction.shape[0] != obs.shape[0]
            or obs.size(-1) != self.flat_obs_dim
        ):
            return torch.zeros_like(reference_loss), None

        target = obs[:, self.global_state_slice].to(prediction.device, dtype=prediction.dtype)
        if self.belief_state_loss in ("smooth_l1", "huber"):
            per_dim_loss = torch.nn.functional.smooth_l1_loss(
                prediction,
                target,
                reduction="none",
            )
        else:
            per_dim_loss = (prediction - target).square()
        per_row_loss = per_dim_loss.mean(dim=-1)
        valid = self._flat_sequence_mask(obs, loss_inputs)
        if valid.any():
            belief_state_loss = per_row_loss[valid].mean()
            belief_state_mae = (prediction.detach() - target).abs().mean(dim=-1)[valid].mean()
        else:
            belief_state_loss = torch.zeros_like(reference_loss)
            belief_state_mae = torch.zeros_like(reference_loss)
        stats = {
            "focus_belief_state_loss": belief_state_loss.detach(),
            "focus_belief_state_coeff": torch.tensor(
                coeff,
                dtype=belief_state_loss.dtype,
                device=belief_state_loss.device,
            ),
            "focus_belief_state_mae": belief_state_mae.detach(),
        }
        return belief_state_loss, stats

    def _selected_focus_action_bias(self, obs, logits, global_state):
        if global_state is None or logits is None or obs.size(-1) != self.flat_obs_dim:
            return None
        if global_state.shape[0] != obs.shape[0] or logits.shape[0] != obs.shape[0]:
            return None
        prior_config = dict(self.focus_config)
        prior_config["action_bias_eta"] = float(
            self.focus_config.get(
                "action_prior_bias_eta",
                self.focus_config.get("action_bias_eta", 1.0),
            )
        )
        if self.belief_state_detach_for_prior:
            global_state = global_state.detach()
        q_values = logits.new_zeros((obs.shape[0], self.n_focus_agents, self.n_action_choices))
        action_bias = focus_action_q_bias(
            q_values,
            global_state,
            prior_config,
            self.n_focus_agents,
            self.n_action_choices,
        )
        if action_bias is None:
            return None
        agent_index = obs[:, self.local_obs_slice.start + 3].round().long()
        agent_index = agent_index.clamp(0, self.n_focus_agents - 1).to(logits.device)
        rows = torch.arange(obs.shape[0], device=logits.device)
        selected_bias = action_bias[rows, agent_index]
        eta = float(self.focus_config.get("action_prior_logit_eta", 1.0))
        return eta * selected_bias.to(dtype=logits.dtype, device=logits.device)

    def _focus_action_prior_loss(self, obs, loss_inputs, reference_loss):
        if not self.focus_config.get("action_prior_enabled", False):
            return torch.zeros_like(reference_loss), None
        coeff = float(self.focus_config.get("action_prior_coeff", 0.0))
        logits = self._last_action_logits
        if coeff <= 0.0 or logits is None or logits.shape[0] != obs.shape[0]:
            return torch.zeros_like(reference_loss), None
        if obs.size(-1) != self.flat_obs_dim:
            return torch.zeros_like(reference_loss), None

        prior_config = dict(self.focus_config)
        prior_config["action_bias_eta"] = float(
            self.focus_config.get(
                "action_prior_bias_eta",
                self.focus_config.get("action_bias_eta", 1.0),
            )
        )
        q_values = logits.new_zeros((obs.shape[0], self.n_focus_agents, self.n_action_choices))
        state_source = str(self.focus_config.get("action_prior_state_source", "belief")).lower()
        if state_source in ("belief", "predicted", "prediction", "local_belief"):
            global_state = self._last_belief_state
            if global_state is None or global_state.shape[0] != obs.shape[0]:
                return torch.zeros_like(reference_loss), None
            if self.belief_state_detach_for_prior:
                global_state = global_state.detach()
        else:
            global_state = obs[:, self.global_state_slice]
        action_bias = focus_action_q_bias(
            q_values,
            global_state,
            prior_config,
            self.n_focus_agents,
            self.n_action_choices,
        )
        if action_bias is None:
            return torch.zeros_like(reference_loss), None

        if SampleBatch.AGENT_INDEX in loss_inputs:
            agent_index = loss_inputs[SampleBatch.AGENT_INDEX].long().to(obs.device)
        else:
            agent_index = obs[:, self.local_obs_slice.start + 3].round().long()
        agent_index = agent_index.clamp(0, self.n_focus_agents - 1)
        rows = torch.arange(obs.shape[0], device=obs.device)
        selected_bias = action_bias[rows, agent_index]

        temperature = max(float(self.focus_config.get("action_prior_temperature", 1.0)), 1e-6)
        target_probs = torch.softmax((selected_bias / temperature).detach(), dim=-1)
        if self.has_action_mask:
            action_mask = obs[:, self.action_mask_slice].clamp(min=0.0, max=1.0)
            target_probs = target_probs * action_mask
            target_probs = target_probs / target_probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)

        log_probs = torch.log_softmax(logits, dim=-1)
        per_row_loss = -(target_probs * log_probs).sum(dim=-1)
        valid = self._flat_sequence_mask(obs, loss_inputs)
        if self.focus_config.get("action_prior_use_focus_valid", False) and FOCUS_VALID in loss_inputs:
            valid = valid & loss_inputs[FOCUS_VALID].bool().to(obs.device)

        weights = torch.ones_like(per_row_loss)
        if self.focus_config.get("action_prior_use_confidence", True) and FOCUS_CONFIDENCE in loss_inputs:
            weights = weights * loss_inputs[FOCUS_CONFIDENCE].float().to(obs.device).detach().clamp(0.0, 1.0)

        positive_fraction = torch.zeros_like(reference_loss)
        if self.focus_config.get("action_prior_use_positive_advantage", True) and Postprocessing.ADVANTAGES in loss_inputs:
            advantages = loss_inputs[Postprocessing.ADVANTAGES].float().to(obs.device).detach()
            if advantages.shape == per_row_loss.shape:
                centered_advantages = advantages
                if self.focus_config.get("action_prior_center_advantage", True) and valid.any():
                    centered_advantages = centered_advantages - centered_advantages[valid].mean()
                positive_advantages = centered_advantages.clamp_min(0.0)
                if valid.any():
                    positive_fraction = (positive_advantages[valid] > 0.0).float().mean()
                    positive_scale = positive_advantages[valid].mean()
                else:
                    positive_scale = positive_advantages.mean()
                positive_scale = positive_scale.clamp_min(1e-6)
                advantage_weights = (positive_advantages / positive_scale).clamp(
                    max=float(self.focus_config.get("action_prior_advantage_clip", 5.0))
                )
                floor = float(self.focus_config.get("action_prior_weight_floor", 0.05))
                weights = weights * (floor + advantage_weights)

        if valid.any():
            valid_weights = weights[valid]
            denom = valid_weights.sum().clamp_min(1e-8)
            prior_loss = (per_row_loss[valid] * valid_weights).sum() / denom
            target_entropy = -(target_probs[valid] * torch.log(target_probs[valid] + 1e-8)).sum(dim=-1).mean()
            weight_mean = valid_weights.mean()
        else:
            prior_loss = torch.zeros_like(reference_loss)
            target_entropy = torch.zeros_like(reference_loss)
            weight_mean = torch.zeros_like(reference_loss)

        stats = {
            "focus_action_prior_loss": prior_loss.detach(),
            "focus_action_prior_coeff": torch.tensor(coeff, dtype=prior_loss.dtype, device=prior_loss.device),
            "focus_action_prior_target_entropy": target_entropy.detach(),
            "focus_action_prior_weight_mean": weight_mean.detach(),
            "focus_action_prior_positive_fraction": positive_fraction.detach(),
        }
        return prior_loss, stats

    def custom_loss(self, policy_loss, loss_inputs):
        reference_loss = policy_loss[0] if isinstance(policy_loss, list) else policy_loss
        obs = loss_inputs[SampleBatch.CUR_OBS].float()
        belief_loss = torch.zeros_like(reference_loss)
        beta = float(self.focus_config.get("beta_belief", 0.01))

        if (
            beta > 0.0
            and self.focus_model is not None
            and obs.size(-1) == self.flat_obs_dim
            and SampleBatch.SEQ_LENS in loss_inputs
        ):
            seq_lens = loss_inputs[SampleBatch.SEQ_LENS].long()
            B = seq_lens.numel()
            if B > 0 and obs.shape[0] % B == 0:
                T = obs.shape[0] // B
                if T >= 2:
                    global_state = obs[:, self.global_state_slice].reshape(B, T, self.global_state_dim)
                    state = global_state[:, :-1, :]
                    future_state = global_state[:, 1:, :]
                    per_horizon_losses, per_horizon_valid, *_ = self.focus_model.nll(
                        state, future_state
                    )
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
                    if belief_terms:
                        belief_loss = torch.stack(belief_terms).sum()

        prior_loss, prior_stats = self._focus_action_prior_loss(obs, loss_inputs, reference_loss)
        prior_coeff = float(self.focus_config.get("action_prior_coeff", 0.0))
        belief_state_loss, belief_state_stats = self._belief_state_loss(
            obs,
            loss_inputs,
            reference_loss,
        )
        belief_state_coeff = float(
            self.focus_config.get(
                "belief_state_coeff",
                self.focus_config.get("beta_belief", 0.0),
            )
        )
        self._focus_stats = {
            "focus/belief_loss": belief_loss.detach(),
            "focus/beta_belief": torch.tensor(beta, device=belief_loss.device),
        }
        self.tower_stats["focus_belief_loss"] = belief_loss.detach()
        self.tower_stats["focus_beta_belief"] = torch.tensor(beta, device=belief_loss.device)
        if belief_state_stats is not None:
            self.tower_stats.update(belief_state_stats)
            self._focus_stats.update(
                {
                    "focus/belief_state_loss": belief_state_stats[
                        "focus_belief_state_loss"
                    ],
                    "focus/belief_state_coeff": belief_state_stats[
                        "focus_belief_state_coeff"
                    ],
                    "focus/belief_state_mae": belief_state_stats[
                        "focus_belief_state_mae"
                    ],
                }
            )
        if prior_stats is not None:
            self.tower_stats.update(prior_stats)
            self._focus_stats.update(
                {
                    "focus/action_prior_loss": prior_stats["focus_action_prior_loss"],
                    "focus/action_prior_coeff": prior_stats["focus_action_prior_coeff"],
                    "focus/action_prior_target_entropy": prior_stats[
                        "focus_action_prior_target_entropy"
                    ],
                }
            )

        auxiliary_loss = (
            beta * belief_loss
            + belief_state_coeff * belief_state_loss
            + prior_coeff * prior_loss
        )
        if isinstance(policy_loss, list):
            return [loss + auxiliary_loss for loss in policy_loss]
        return policy_loss + auxiliary_loss

    def metrics(self):
        return self._focus_stats


ModelCatalog.register_custom_model('MAPPOModel', MAPPOModel)
