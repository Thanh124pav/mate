from collections import OrderedDict

import numpy as np
from gym import spaces
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.recurrent_net import RecurrentNetwork as TorchRNN
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.framework import try_import_torch

from examples.utils import SimpleRNN, get_space_flat_size, orthogonal_initializer
from ray.rllib.agents.qplex_focus.qplex_policy import LearnedOccupancyModel, resolve_focus_config


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
        self.focus_model = None
        if self.focus_config.get("enabled", False):
            self.focus_model = LearnedOccupancyModel(
                self.global_state_dim,
                int(self.focus_config.get("n_agents", 4)),
                int(self.focus_config.get("n_targets", 8)),
                horizon=int(self.focus_config.get("horizon", 3)),
                hidden_dim=int(self.focus_config.get("belief_hidden_dim", 256)),
                max_delta=float(self.focus_config.get("belief_max_delta", 400.0)),
                min_std=float(self.focus_config.get("belief_min_std", 25.0)),
            )
        self._focus_stats = {}

    def get_initial_state(self):
        return [*self.actor.get_initial_state(), *self.critic.get_initial_state()]

    def forward_rnn(self, inputs, state, seq_lens):
        assert inputs.size(-1) == self.flat_obs_dim

        local_obs = inputs[..., self.local_obs_slice]
        actor_state_in = state[:2]
        action_out, actor_state_out = self.actor(local_obs, actor_state_in)

        if self.has_action_mask:
            action_mask = inputs[..., self.action_mask_slice].clamp(min=0.0, max=1.0)
            inf_mask = torch.log(action_mask).clamp_min(min=torch.finfo(action_out.dtype).min)
            action_out = action_out + inf_mask

        global_state = inputs[..., self.global_state_slice]
        critic_state_in = state[2:]
        _, critic_state_out = self.critic(global_state, critic_state_in, features_only=True)

        return action_out, [*actor_state_out, *critic_state_out]

    def value_function(self):
        assert self.critic.last_features is not None, 'must call forward() first'

        return self.critic.output(self.critic.last_features).reshape(-1)

    def custom_loss(self, policy_loss, loss_inputs):
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
        per_horizon_losses, per_horizon_valid, _, _ = self.focus_model.nll(
            state, future_state
        )
        discount = float(self.focus_config.get("horizon_discount", 0.9))
        weights = torch.tensor(
            [discount ** h for h in range(len(per_horizon_losses))],
            dtype=global_state.dtype,
            device=global_state.device,
        )
        weights = weights / (weights.sum() + float(self.focus_config.get("eps", 1e-8)))
        belief_terms = []
        for h, (nll_per_step, valid_h) in enumerate(zip(per_horizon_losses, per_horizon_valid)):
            time_valid = torch.arange(T - 1, device=global_state.device).unsqueeze(0) < (seq_lens - 1).clamp_min(0).unsqueeze(1)
            valid_h = valid_h & time_valid
            if valid_h.any():
                belief_terms.append(weights[h] * nll_per_step[valid_h].mean())
        belief_loss = torch.stack(belief_terms).sum() if belief_terms else torch.zeros_like(policy_loss)
        beta = float(self.focus_config.get("beta_belief", 0.01))
        self._focus_stats = {
            "focus_belief_loss": belief_loss.detach(),
            "focus_beta_belief": torch.tensor(beta, device=belief_loss.device),
        }
        return policy_loss + beta * belief_loss

    def metrics(self):
        return self._focus_stats


ModelCatalog.register_custom_model('MAPPOModel', MAPPOModel)
