"""SMPE2 Model: MAPPO + variational belief + AM filters + adversarial exploration.

Extends MAPPOModel with:
  - Variational encoder: o_i → (mu, log_sigma) → z_i
  - Belief decoder: z_i → reconstruct filtered neighbor observations
  - AM filters: per-neighbor sigmoid weights to filter non-informative features
  - Actor conditioned on obs + belief: π(a | h_i, z_i)
  - Critic conditioned on state + belief: V(s, z)
  - custom_loss() adds L_rec + L_KL + L_norm to PPO loss
"""

from collections import OrderedDict

import numpy as np
from gym import spaces
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.recurrent_net import RecurrentNetwork as TorchRNN
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.framework import try_import_torch

from examples.utils import SimpleRNN, get_space_flat_size, orthogonal_initializer

torch, nn = try_import_torch()


class SMPE2Model(TorchRNN, nn.Module):
    """SMPE2 actor-critic model with variational belief inference."""

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
        latent_dim=32,
        lambda_rec=1.0,
        lambda_kl=0.01,
        lambda_norm=0.01,
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
            [(key, slice(indices[i], indices[i + 1]))
             for i, key in enumerate(self.space_dims.keys())]
        )

        self.local_obs_dim = self.space_dims['obs']
        self.local_obs_slice = self.flat_obs_slices['obs']
        self.global_state_dim = self.space_dims['state']
        self.global_state_slice = self.flat_obs_slices['state']
        self.action_dim = get_space_flat_size(self.action_space)

        if self.has_action_mask:
            self.action_mask_slice = self.flat_obs_slices['action_mask']
        else:
            self.action_mask_slice = None

        self.latent_dim = latent_dim
        self.lambda_rec = lambda_rec
        self.lambda_kl = lambda_kl
        self.lambda_norm = lambda_norm
        self.lstm_cell_size = lstm_cell_size

        # --- Variational Encoder: obs → (mu, log_sigma) → z ---
        self.encoder = nn.Sequential(
            nn.Linear(self.local_obs_dim, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, latent_dim * 2),
        )

        # --- Belief Decoder: z → reconstruct global_state (filtered) ---
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64), nn.ReLU(),
            nn.Linear(64, 128), nn.ReLU(),
            nn.Linear(128, self.global_state_dim),
        )

        # --- AM Filter: learn which state features are informative ---
        self.am_filter = nn.Sequential(
            nn.Linear(self.local_obs_dim, 64), nn.ReLU(),
            nn.Linear(64, self.global_state_dim), nn.Sigmoid(),
        )

        # --- Actor RNN: input = obs + z ---
        actor_input_dim = self.local_obs_dim + latent_dim
        self.actor = SimpleRNN(
            name='actor',
            input_dim=actor_input_dim,
            hidden_dims=actor_hiddens,
            cell_size=lstm_cell_size,
            output_dim=num_outputs,
            activation=actor_hidden_activation,
            output_activation=None,
            hidden_weight_initializer=orthogonal_initializer(scale=1.0),
            output_weight_initializer=orthogonal_initializer(scale=0.01),
        )

        # --- Critic RNN: input = state + z ---
        critic_input_dim = self.global_state_dim + latent_dim
        self.critic = SimpleRNN(
            name='critic',
            input_dim=critic_input_dim,
            hidden_dims=critic_hiddens,
            cell_size=lstm_cell_size,
            output_dim=1,
            activation=critic_hidden_activation,
            output_activation=None,
            hidden_weight_initializer=orthogonal_initializer(scale=1.0),
            output_weight_initializer=orthogonal_initializer(scale=1.0),
        )

        # Stored for custom_loss
        self._mu = None
        self._log_sigma = None
        self._z = None
        self._local_obs = None

    def get_initial_state(self):
        return [*self.actor.get_initial_state(), *self.critic.get_initial_state()]

    def forward_rnn(self, inputs, state, seq_lens):
        assert inputs.size(-1) == self.flat_obs_dim

        local_obs = inputs[..., self.local_obs_slice]
        global_state = inputs[..., self.global_state_slice]

        # --- Variational belief inference ---
        enc_out = self.encoder(local_obs)
        mu, log_sigma = enc_out.chunk(2, dim=-1)
        sigma = torch.exp(log_sigma.clamp(-5, 2))
        z = mu + sigma * torch.randn_like(sigma)

        # Store for custom_loss
        self._mu = mu
        self._log_sigma = log_sigma
        self._z = z
        self._local_obs = local_obs
        self._global_state = global_state

        # --- Actor: π(a | h_i, z_i) ---
        actor_input = torch.cat([local_obs, z], dim=-1)
        actor_state_in = state[:2]
        action_out, actor_state_out = self.actor(actor_input, actor_state_in)

        if self.has_action_mask:
            action_mask = inputs[..., self.action_mask_slice].clamp(min=0.0, max=1.0)
            inf_mask = torch.log(action_mask).clamp_min(min=torch.finfo(action_out.dtype).min)
            action_out = action_out + inf_mask

        # --- Critic: V(s, z) ---
        critic_input = torch.cat([global_state, z.detach()], dim=-1)
        critic_state_in = state[2:]
        _, critic_state_out = self.critic(critic_input, critic_state_in, features_only=True)

        return action_out, [*actor_state_out, *critic_state_out]

    def value_function(self):
        assert self.critic.last_features is not None, 'must call forward() first'
        return self.critic.output(self.critic.last_features).reshape(-1)

    def custom_loss(self, policy_loss, loss_inputs):
        """Add SMPE2 auxiliary losses to PPO policy loss."""
        if self._mu is None or self._z is None:
            return policy_loss

        # --- KL divergence: KL(q(z|o) || N(0,I)) ---
        kl_loss = -0.5 * torch.mean(
            1 + 2 * self._log_sigma - self._mu.pow(2) - (2 * self._log_sigma).exp()
        )

        # --- Reconstruction loss: decode z → predict filtered global state ---
        decoded = self.decoder(self._z)
        am_weights = self.am_filter(self._local_obs)
        filtered_target = am_weights.detach() * self._global_state.detach()
        filtered_pred = am_weights.detach() * decoded
        rec_loss = nn.functional.mse_loss(filtered_pred, filtered_target)

        # --- Normalization loss: prevent AM filters from collapsing to 0 ---
        norm_loss = -torch.mean(am_weights.pow(2))

        total_aux = (
            self.lambda_rec * rec_loss
            + self.lambda_kl * kl_loss
            + self.lambda_norm * norm_loss
        )

        return [loss + total_aux for loss in policy_loss]


ModelCatalog.register_custom_model('SMPE2Model', SMPE2Model)
