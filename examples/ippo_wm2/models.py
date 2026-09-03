from collections import OrderedDict

import numpy as np
from gym import spaces
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.recurrent_net import RecurrentNetwork as TorchRNN
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.framework import try_import_torch

from examples.utils import SimpleRNN, get_space_flat_size, orthogonal_initializer
from examples.wm2_utils import (
    DEFAULT_WM2_CONFIG,
    discrete_action_count,
    encode_local_obs,
    make_local_world_model,
    wm2_auxiliary_loss,
    world_model_feature_dim,
)


torch, nn = try_import_torch()


class IPPOLocalWM2Model(TorchRNN, nn.Module):
    def __init__(
        self,
        obs_space,
        action_space,
        num_outputs,
        model_config,
        name,
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

        original_space = getattr(obs_space, 'original_space', obs_space)
        self.has_action_mask = isinstance(original_space, spaces.Dict) and 'action_mask' in original_space.spaces
        if isinstance(original_space, spaces.Dict):
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
            self.action_mask_slice = self.flat_obs_slices.get('action_mask')
        else:
            self.space_dims = OrderedDict([('obs', get_space_flat_size(original_space))])
            self.flat_obs_slices = OrderedDict([('obs', slice(0, self.space_dims['obs']))])
            self.local_obs_dim = self.space_dims['obs']
            self.local_obs_slice = self.flat_obs_slices['obs']
            self.action_mask_slice = None

        self.flat_obs_dim = get_space_flat_size(self.obs_space)
        self.action_dim = get_space_flat_size(self.action_space)
        self.n_action_choices = int(getattr(self.action_space, 'n', self.action_dim))
        if self.has_action_mask:
            assert self.space_dims['action_mask'] == num_outputs

        custom_model_config = model_config.get('custom_model_config', {})
        self.wm2_config = {**DEFAULT_WM2_CONFIG, **custom_model_config.get('world_model_v2', {})}
        self.wm2_feature_dim = world_model_feature_dim(self.wm2_config)
        n_wm2_actions = discrete_action_count(self.action_space)
        if self.wm2_feature_dim > 0 and n_wm2_actions is None:
            raise ValueError('IPPOLocalWM2Model requires a discrete action space for WM2.')
        self.world_model_v2 = make_local_world_model(
            self.local_obs_dim,
            self.local_obs_dim,
            n_wm2_actions or 1,
            self.wm2_config,
        )
        self.wm2_n_actions = n_wm2_actions or 1
        self.wm2_loss_weight = float(self.wm2_config.get('wm_loss_weight', 0.5))
        self._wm2_stats = {}

        self.actor = SimpleRNN(
            name='actor',
            input_dim=self.local_obs_dim + self.wm2_feature_dim,
            hidden_dims=actor_hiddens or [],
            cell_size=lstm_cell_size,
            output_dim=num_outputs,
            activation=actor_hidden_activation,
            output_activation=None,
            hidden_weight_initializer=orthogonal_initializer(scale=1.0),
            output_weight_initializer=orthogonal_initializer(scale=0.01),
        )
        self.critic = SimpleRNN(
            name='critic',
            input_dim=self.local_obs_dim + self.wm2_feature_dim,
            hidden_dims=critic_hiddens or [],
            cell_size=lstm_cell_size,
            output_dim=1,
            activation=critic_hidden_activation,
            output_activation=None,
            hidden_weight_initializer=orthogonal_initializer(scale=1.0),
            output_weight_initializer=orthogonal_initializer(scale=1.0),
        )

    def get_initial_state(self):
        return [*self.actor.get_initial_state(), *self.critic.get_initial_state()]

    def forward_rnn(self, inputs, state, seq_lens):
        assert inputs.size(-1) == self.flat_obs_dim
        local_obs = inputs[..., self.local_obs_slice]
        wm2_feature = encode_local_obs(self.world_model_v2, local_obs)
        model_input = torch.cat((local_obs, wm2_feature), dim=-1)

        action_out, actor_state_out = self.actor(model_input, state[:2])
        if self.has_action_mask:
            action_mask = inputs[..., self.action_mask_slice].clamp(min=0.0, max=1.0)
            inf_mask = torch.log(action_mask).clamp_min(min=torch.finfo(action_out.dtype).min)
            action_out = action_out + inf_mask

        _, critic_state_out = self.critic(model_input, state[2:], features_only=True)
        return action_out, [*actor_state_out, *critic_state_out]

    def value_function(self):
        assert self.critic.last_features is not None, 'must call forward() first'
        return self.critic.output(self.critic.last_features).reshape(-1)

    def custom_loss(self, policy_loss, loss_inputs):
        reference_loss = policy_loss[0] if isinstance(policy_loss, list) else policy_loss
        wm2_loss, wm2_stats = wm2_auxiliary_loss(
            self.world_model_v2,
            loss_inputs,
            reference_loss,
            self.flat_obs_dim,
            self.local_obs_slice,
            self.local_obs_dim,
            self.local_obs_slice,
            self.local_obs_dim,
            self.wm2_n_actions,
            coeff=self.wm2_loss_weight,
        )
        self.tower_stats.update(wm2_stats)
        self._wm2_stats = {f'wm2/{key}': value for key, value in wm2_stats.items()}
        if isinstance(policy_loss, list):
            return [loss + wm2_loss for loss in policy_loss]
        return policy_loss + wm2_loss

    def metrics(self):
        return self._wm2_stats


ModelCatalog.register_custom_model('IPPOLocalWM2Model', IPPOLocalWM2Model)