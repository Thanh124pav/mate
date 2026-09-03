from collections import OrderedDict

import numpy as np
from gym import spaces
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.recurrent_net import RecurrentNetwork as TorchRNN
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.framework import try_import_torch

from examples.utils import SimpleRNN, get_space_flat_size
from examples.wm2_utils import (
    DEFAULT_WM2_CONFIG,
    discrete_action_count,
    encode_local_obs,
    make_local_world_model,
    wm2_auxiliary_loss,
    world_model_feature_dim,
)
from ray.rllib.agents.qplex_focus.qplex_policy import LearnedOccupancyModel, resolve_focus_config
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


class MessageAggregator(nn.Module):
    def __init__(self, key_dim, value_dim, hidden_dim):
        super().__init__()

        self.key_dim = self.query_dim = key_dim
        self.value_dim = value_dim
        self.message_dim = self.key_dim + self.value_dim

        self.hidden_dim = hidden_dim

        self.query_predictor = nn.Linear(
            in_features=self.hidden_dim, out_features=self.query_dim, bias=False
        )
        self.scale = 1.0 / np.sqrt(self.query_dim)

    def forward(self, messages, hidden_states):
        # fmt: off
        assert messages.ndim == 3       # (B, T, Na * Dm)
        assert hidden_states.ndim == 3  # (B, T, Dh)
        B, T, joint_message_dim = messages.shape

        messages = messages.view(B, T, -1, self.message_dim)                   # (B, T, Na, Dm)
        keys, values = messages.split([self.key_dim, self.value_dim], dim=-1)  # (B, T, Na, *)

        queries = self.query_predictor(hidden_states)  # (B, T, Dq)
        queries = queries.unsqueeze(dim=-2)            # (B, T, 1, Dq)

        attns = self.scale * torch.matmul(queries, keys.transpose(-1, -2))  # (B, T, 1, Na)
        attns = attns.softmax(dim=-1)                                       # (B, T, 1, Na)
        outputs = torch.matmul(attns, values)                               # (B, T, 1, Dv)
        outputs = outputs.squeeze(dim=-2)                                   # (B, T, Dv)
        # fmt: on

        return outputs


class TarMACWM2Model(TorchRNN, nn.Module):
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
        # Extra TarMACWM2Model arguments
        message_key_dim=32,
        message_value_dim=32,
        critic_use_global_state=True,
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
        assert isinstance(action_space, spaces.Dict) and tuple(action_space.keys())[-1] == 'message'
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
        self.joint_message_dim = self.space_dims['messages']
        self.joint_message_slice = self.flat_obs_slices['messages']

        self.action_space_dims = OrderedDict(
            [(key, get_space_flat_size(subspace)) for key, subspace in self.action_space.items()]
        )
        self.action_dim = self.action_space_dims['action']
        self.message_dim = self.action_space_dims['message']
        self.n_action_choices = int(getattr(self.action_space['action'], 'n', self.action_dim))
        custom_model_config = model_config.get('custom_model_config', {})
        self.wm2_config = {**DEFAULT_WM2_CONFIG, **custom_model_config.get('world_model_v2', {})}
        self.wm2_feature_dim = world_model_feature_dim(self.wm2_config)
        n_wm2_actions = discrete_action_count(self.action_space)
        if self.wm2_feature_dim > 0 and n_wm2_actions is None:
            raise ValueError('TarMACWM2Model requires a discrete physical action space for WM2.')
        self.world_model_v2 = make_local_world_model(
            self.local_obs_dim,
            self.global_state_dim,
            n_wm2_actions or 1,
            self.wm2_config,
        )
        self.wm2_n_actions = n_wm2_actions or 1
        self.wm2_loss_weight = float(self.wm2_config.get('wm_loss_weight', 0.5))
        self._wm2_stats = {}
        self.message_key_dim = self.message_query_dim = message_key_dim
        self.message_value_dim = message_value_dim
        assert self.message_dim == self.message_key_dim + self.message_value_dim
        assert self.joint_message_dim % self.message_dim == 0
        self.num_agents = self.joint_message_dim // self.message_dim

        if self.has_action_mask:
            self.action_mask_slice = self.flat_obs_slices['action_mask']
            assert self.space_dims['action_mask'] == num_outputs - self.message_dim
        else:
            self.action_mask_slice = None

        self.actor_hiddens = actor_hiddens or []
        self.critic_hiddens = critic_hiddens or list(self.actor_hiddens)
        self.actor_hidden_activation = actor_hidden_activation
        self.critic_hidden_activation = critic_hidden_activation
        self.lstm_cell_size = lstm_cell_size

        self.message_aggregator = MessageAggregator(
            key_dim=self.message_key_dim,
            value_dim=self.message_value_dim,
            hidden_dim=self.lstm_cell_size * 2,
        )

        self.actor = SimpleRNN(
            name='actor',
            input_dim=self.local_obs_dim + self.message_value_dim + self.wm2_feature_dim,
            hidden_dims=self.actor_hiddens,
            cell_size=self.lstm_cell_size,
            output_dim=num_outputs,
            activation=self.actor_hidden_activation,
            output_activation=None,
        )

        self.critic_use_global_state = critic_use_global_state
        critic_obs_input_dim = (
            self.global_state_dim if self.critic_use_global_state else self.local_obs_dim
        )
        self.critic = SimpleRNN(
            name='critic',
            input_dim=critic_obs_input_dim + self.joint_message_dim + self.wm2_feature_dim,
            hidden_dims=self.critic_hiddens,
            cell_size=self.lstm_cell_size,
            output_dim=1,
            activation=self.critic_hidden_activation,
            output_activation=None,
        )

        custom_model_config = model_config.get('custom_model_config', {})
        self.focus_config = resolve_focus_config({
            'focus': custom_model_config.get('focus', {}),
            'env_config': custom_model_config.get('env_config', {}),
        })
        self.n_focus_agents = int(self.focus_config.get('n_agents', self.num_agents))
        self.focus_model = None
        self.focus_engine = None
        self.focus_adapter = None
        if self.focus_config.get('enabled', False):
            self.focus_model = LearnedOccupancyModel(
                self.global_state_dim,
                self.n_focus_agents,
                int(self.focus_config.get('n_targets', 8)),
                horizon=int(self.focus_config.get('horizon', 3)),
                hidden_dim=int(self.focus_config.get('belief_hidden_dim', 256)),
                max_delta=float(self.focus_config.get('belief_max_delta', 400.0)),
                min_std=float(self.focus_config.get('belief_min_std', 25.0)),
            )
            self.focus_engine = FocusResponsibilityEngine(
                self.focus_model,
                self.n_focus_agents,
                self.n_action_choices,
                self.focus_config,
            )
            self.focus_adapter = MAPPOFocusAdapter(
                self.focus_config.get(
                    'policy_eta',
                    self.focus_config.get('focus_policy_eta', 0.0),
                )
            )
        self._focus_stats = {}

    def get_initial_state(self):
        return [*self.actor.get_initial_state(), *self.critic.get_initial_state()]

    def forward_rnn(self, inputs, state, seq_lens):
        # fmt: off
        assert inputs.ndim == 3  # (B, T, *)
        B, T, flat_obs_dim = inputs.shape
        assert flat_obs_dim == self.flat_obs_dim

        local_obs = inputs[..., self.local_obs_slice]     # (B, T, Do)
        wm2_feature = encode_local_obs(self.world_model_v2, local_obs)
        messages = inputs[..., self.joint_message_slice]  # (B, T, Na * Dm)

        action_out_list = []
        message_out_list = []
        hidden_states = state[:2]
        for t in range(T):
            aggregated_message = self.message_aggregator(messages[:, t:t + 1],                       # (B, 1, Dm)
                                                         torch.cat(hidden_states, dim=-1).unsqueeze(dim=1))
            local_obs_with_message = torch.cat((local_obs[:, t:t + 1], aggregated_message, wm2_feature[:, t:t + 1]), dim=-1)  # (B, 1, Do + Dm)
            actor_out, hidden_states = self.actor(local_obs_with_message, hidden_states)             # (B, 1, Da + Dm)
            action_out, message_out = actor_out.split([self.action_dim, self.message_dim], dim=-1)   # (B, T, *)
            message_out = message_out.tanh()  # squash messages to [-1., +1.]
            action_out_list.append(action_out)
            message_out_list.append(message_out)
        actor_state_out = hidden_states
        action_out = torch.cat(action_out_list, dim=1)    # (B, T, Da)
        message_out = torch.cat(message_out_list, dim=1)  # (B, T, Dm)

        if self.has_action_mask:
            action_mask = inputs[..., self.action_mask_slice].clamp(min=0.0, max=1.0)
            inf_mask = torch.log(action_mask).clamp_min(min=torch.finfo(action_out.dtype).min)
            action_out = action_out + inf_mask

        if self.critic_use_global_state:
            global_state = inputs[..., self.global_state_slice]
            critic_inputs = torch.cat((global_state, messages, wm2_feature), dim=-1)
        else:
            critic_inputs = torch.cat((local_obs, messages, wm2_feature), dim=-1)
        critic_state_in = state[2:]
        _, critic_state_out = self.critic(critic_inputs, critic_state_in, features_only=True)

        action_out = torch.cat((action_out, message_out), dim=-1)  # (B, 1, Da + Dm)
        # fmt: on
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
            return int(str(agent_id).rsplit('_', 1)[-1])
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _physical_action_from_batch(batch, row):
        actions = batch[SampleBatch.ACTIONS]
        if isinstance(actions, dict):
            action = actions['action'][row]
        else:
            action = actions[row]
            if isinstance(action, dict):
                action = action['action']
            elif isinstance(action, (tuple, list)):
                action = action[0]
            else:
                action_array = np.asarray(action)
                if action_array.shape != () and action_array.size > 1:
                    action = action_array.reshape(-1)[0]
        return int(np.asarray(action).reshape(()))

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
                actions.append(self._physical_action_from_batch(batch, member_row))
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
            confidence = focus.confidence if self.focus_config.get('use_confidence', self.focus_config.get('focus_use_confidence', True)) else None
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
        eps = float(self.focus_config.get('eps', 1e-8))
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
        reference_loss = policy_loss[0] if isinstance(policy_loss, list) else policy_loss
        wm2_loss, wm2_stats = wm2_auxiliary_loss(
            self.world_model_v2,
            loss_inputs,
            reference_loss,
            self.flat_obs_dim,
            self.local_obs_slice,
            self.local_obs_dim,
            self.global_state_slice,
            self.global_state_dim,
            self.wm2_n_actions,
            coeff=self.wm2_loss_weight,
        )
        self.tower_stats.update(wm2_stats)
        self._wm2_stats = {f'wm2/{key}': value for key, value in wm2_stats.items()}
        if self.focus_model is None:
            if isinstance(policy_loss, list):
                return [loss + wm2_loss for loss in policy_loss]
            return policy_loss + wm2_loss
        obs = loss_inputs[SampleBatch.CUR_OBS].float()
        if obs.size(-1) != self.flat_obs_dim or SampleBatch.SEQ_LENS not in loss_inputs:
            if isinstance(policy_loss, list):
                return [loss + wm2_loss for loss in policy_loss]
            return policy_loss + wm2_loss
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
        per_horizon_losses, per_horizon_valid, *_ = self.focus_model.nll(
            state, future_state
        )
        discount = float(self.focus_config.get('horizon_discount', 0.9))
        weights = torch.tensor(
            [discount ** h for h in range(len(per_horizon_losses))],
            dtype=global_state.dtype,
            device=global_state.device,
        )
        weights = weights / (weights.sum() + float(self.focus_config.get('eps', 1e-8)))
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
        belief_loss = (
            torch.stack(belief_terms).sum()
            if belief_terms
            else torch.zeros_like(reference_loss)
        )
        beta = float(self.focus_config.get('beta_belief', 0.01))
        self._focus_stats = {
            'focus/belief_loss': belief_loss.detach(),
            'focus/beta_belief': torch.tensor(beta, device=belief_loss.device),
        }
        if isinstance(policy_loss, list):
            return [loss + beta * belief_loss + wm2_loss for loss in policy_loss]
        return policy_loss + beta * belief_loss + wm2_loss

    def metrics(self):
        stats = {
            'num_in_comm_edges': self.num_agents,
        }
        stats.update(self._focus_stats)
        stats.update(self._wm2_stats)
        return stats


ModelCatalog.register_custom_model('TarMACWM2Model', TarMACWM2Model)
