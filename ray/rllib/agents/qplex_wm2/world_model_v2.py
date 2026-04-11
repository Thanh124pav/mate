"""Dreamer-style RSSM World Model for Multi-Agent QPLEX (WM2).

A true world model that conditions on actions and models full environment dynamics:
  (state_t, action_t) → (state_{t+1}, reward_{t+1})

Architecture (RSSM — Recurrent State-Space Model):
  - ObservationEncoder: per-agent obs → aggregated embedding
  - TransitionModel (RSSM): (latent_t, action_t) → latent_{t+1}
      - Prior: predict next latent without observations (imagination)
      - Posterior: refine prediction using actual observations (training)
  - StateDecoder: latent → reconstructed global state
  - RewardPredictor: latent → predicted reward

Latent state = [mean, std, stoch, deter]:
  - stoch: sampled stochastic component (captures uncertainty)
  - deter: deterministic GRU hidden state (captures temporal structure)
  - feature = cat(stoch, deter) — used for all downstream tasks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as td
import numpy as np

# MATE observation layout constants
PRESERVED_DIM = 13
CAMERA_STATE_DIM_PRIVATE = 9
TARGET_STATE_DIM_PRIVATE = 14


class ObservationEncoder(nn.Module):
    """Encode per-agent observations into a joint embedding.

    Per-agent MLP + mean-pool across agents → single embedding vector.
    """

    def __init__(self, obs_size, n_agents, embed_dim=128, hidden_dim=128):
        super().__init__()
        self.n_agents = n_agents
        self.agent_encoder = nn.Sequential(
            nn.Linear(obs_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim),
            nn.ReLU(),
        )

    def forward(self, obs):
        """
        Args:
            obs: [B, n_agents, obs_size]

        Returns:
            embed: [B, embed_dim]
        """
        B, C, _ = obs.shape
        agent_embeds = self.agent_encoder(obs.reshape(B * C, -1))  # [B*C, embed_dim]
        agent_embeds = agent_embeds.reshape(B, C, -1)               # [B, C, embed_dim]
        return agent_embeds.mean(dim=1)  # [B, embed_dim]


class ActionEmbedding(nn.Module):
    """Embed discrete joint actions into continuous vectors.

    Each agent's action is embedded separately, then concatenated.
    """

    def __init__(self, n_actions, n_agents, action_embed_dim=16):
        super().__init__()
        self.n_agents = n_agents
        self.embed = nn.Embedding(n_actions, action_embed_dim)
        self.output_dim = n_agents * action_embed_dim

    def forward(self, actions):
        """
        Args:
            actions: [B, n_agents] (long)

        Returns:
            joint_embed: [B, n_agents * action_embed_dim]
        """
        embedded = self.embed(actions)  # [B, n_agents, action_embed_dim]
        return embedded.reshape(actions.shape[0], -1)


class TransitionModel(nn.Module):
    """RSSM: Recurrent State-Space Model.

    Maintains a latent state = [mean, std, stoch, deter].
    Two pathways:
      - Prior (img_step): predict next latent from (prev_latent, action) — no observation
      - Posterior (obs_step): refine with actual observation embedding
    """

    def __init__(self, action_dim, embed_dim, stoch_dim=32, deter_dim=128, hidden_dim=128):
        super().__init__()
        self.stoch_dim = stoch_dim
        self.deter_dim = deter_dim
        self.hidden_dim = hidden_dim

        self.prior_fc1 = nn.Linear(stoch_dim + action_dim, hidden_dim)
        self.gru = nn.GRUCell(hidden_dim, deter_dim)
        self.prior_fc2 = nn.Linear(deter_dim, hidden_dim)
        self.prior_fc3 = nn.Linear(hidden_dim, 2 * stoch_dim)

        self.post_fc1 = nn.Linear(deter_dim + embed_dim, hidden_dim)
        self.post_fc2 = nn.Linear(hidden_dim, 2 * stoch_dim)

    @property
    def feature_dim(self):
        return self.stoch_dim + self.deter_dim

    def get_initial_state(self, batch_size, device):
        return [
            torch.zeros(batch_size, self.stoch_dim, device=device),
            torch.zeros(batch_size, self.stoch_dim, device=device),
            torch.zeros(batch_size, self.stoch_dim, device=device),
            torch.zeros(batch_size, self.deter_dim, device=device),
        ]

    def get_feature(self, state):
        return torch.cat([state[2], state[3]], dim=-1)

    def img_step(self, prev_state, action_embed):
        prev_stoch = prev_state[2]
        prev_deter = prev_state[3]

        x = F.elu(self.prior_fc1(torch.cat([prev_stoch, action_embed], dim=-1)))
        deter = self.gru(x, prev_deter)
        x = F.elu(self.prior_fc2(deter))
        stats = self.prior_fc3(x)
        mean, log_std = stats.chunk(2, dim=-1)
        std = F.softplus(log_std) + 0.1
        stoch = mean + std * torch.randn_like(std)

        return [mean, std, stoch, deter]

    def obs_step(self, prev_state, action_embed, obs_embed):
        prior = self.img_step(prev_state, action_embed)
        prior_deter = prior[3]

        x = F.elu(self.post_fc1(torch.cat([prior_deter, obs_embed], dim=-1)))
        stats = self.post_fc2(x)
        post_mean, post_log_std = stats.chunk(2, dim=-1)
        post_std = F.softplus(post_log_std) + 0.1
        post_stoch = post_mean + post_std * torch.randn_like(post_std)

        posterior = [post_mean, post_std, post_stoch, prior_deter]
        return posterior, prior

    def observe(self, obs_embeds, action_embeds, initial_state=None):
        B, T, _ = obs_embeds.shape

        if initial_state is None:
            state = self.get_initial_state(B, obs_embeds.device)
        else:
            state = initial_state

        post_list = [[] for _ in range(4)]
        prior_list = [[] for _ in range(4)]

        for t in range(T):
            posterior, prior = self.obs_step(
                state, action_embeds[:, t], obs_embeds[:, t]
            )
            state = posterior
            for i in range(4):
                post_list[i].append(posterior[i])
                prior_list[i].append(prior[i])

        posteriors = [torch.stack(x, dim=1) for x in post_list]
        priors = [torch.stack(x, dim=1) for x in prior_list]
        return posteriors, priors

    def imagine(self, action_embeds, initial_state):
        B, H, _ = action_embeds.shape
        state = initial_state
        states_list = [[] for _ in range(4)]

        for h in range(H):
            state = self.img_step(state, action_embeds[:, h])
            for i in range(4):
                states_list[i].append(state[i])

        return [torch.stack(x, dim=1) for x in states_list]


class StateDecoder(nn.Module):
    """Decode latent feature → reconstructed global state."""

    def __init__(self, feature_dim, state_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim),
        )

    def forward(self, feature):
        return self.net(feature)


class RewardPredictor(nn.Module):
    """Predict reward from latent feature."""

    def __init__(self, feature_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, feature):
        return self.net(feature).squeeze(-1)


class LatentWorldModel(nn.Module):
    """Complete RSSM-based world model for multi-agent QPLEX (WM2).

    Combines: ObservationEncoder + ActionEmbedding + TransitionModel +
              StateDecoder + RewardPredictor
    """

    def __init__(self, obs_size, state_dim, n_agents, n_actions,
                 stoch_dim=32, deter_dim=128, hidden_dim=128,
                 action_embed_dim=16, embed_dim=128,
                 imagination_horizon=5, kl_coeff=1.0, free_nats=1.0):
        super(LatentWorldModel, self).__init__()

        self.obs_size = obs_size
        self.state_dim = state_dim
        self.n_agents = n_agents
        self.n_actions = n_actions
        self.stoch_dim = stoch_dim
        self.deter_dim = deter_dim
        self.imagination_horizon = imagination_horizon
        self.kl_coeff = kl_coeff
        self.free_nats = free_nats

        self.feature_dim = stoch_dim + deter_dim

        self.obs_encoder = ObservationEncoder(obs_size, n_agents, embed_dim, hidden_dim)
        self.action_embed = ActionEmbedding(n_actions, n_agents, action_embed_dim)
        self.transition = TransitionModel(
            self.action_embed.output_dim, embed_dim, stoch_dim, deter_dim, hidden_dim,
        )
        self.state_decoder = StateDecoder(self.feature_dim, state_dim, hidden_dim)
        self.reward_predictor = RewardPredictor(self.feature_dim, hidden_dim)

    def get_initial_state(self, batch_size, device):
        return self.transition.get_initial_state(batch_size, device)

    def compute_loss(self, obs, actions, state, rewards, mask, return_posteriors=False):
        """Compute world model losses: reconstruction + reward + KL.

        Args:
            obs: [B, T, n_agents, obs_size]
            actions: [B, T, n_agents] (long)
            state: [B, T, state_dim]
            rewards: [B, T, n_agents]
            mask: [B, T]
            return_posteriors: if True, also return posteriors for imagination

        Returns:
            wm_loss, features [B,T,feature_dim], stats, (posteriors if requested)
        """
        B, T = obs.shape[0], obs.shape[1]

        obs_flat = obs.reshape(B * T, self.n_agents, self.obs_size)
        obs_embeds = self.obs_encoder(obs_flat).reshape(B, T, -1)

        actions_flat = actions.reshape(B * T, self.n_agents)
        action_embeds = self.action_embed(actions_flat).reshape(B, T, -1)

        posteriors, priors = self.transition.observe(obs_embeds, action_embeds)

        features = torch.cat([posteriors[2], posteriors[3]], dim=-1)

        features_flat = features.reshape(B * T, -1)
        state_pred = self.state_decoder(features_flat).reshape(B, T, -1)
        recon_loss = ((state_pred - state.detach()) ** 2).mean(dim=-1)
        recon_loss = (recon_loss * mask).sum() / mask.sum().clamp(min=1)

        reward_pred = self.reward_predictor(features_flat).reshape(B, T)
        reward_target = rewards.mean(dim=-1)
        reward_loss = ((reward_pred - reward_target.detach()) ** 2)
        reward_loss = (reward_loss * mask).sum() / mask.sum().clamp(min=1)

        post_dist = td.Normal(posteriors[0], posteriors[1])
        prior_dist = td.Normal(priors[0], priors[1])
        kl = td.kl_divergence(post_dist, prior_dist).sum(dim=-1)
        kl = torch.clamp(kl, min=self.free_nats)
        kl_loss = (kl * mask).sum() / mask.sum().clamp(min=1)

        wm_loss = recon_loss + reward_loss + self.kl_coeff * kl_loss

        stats = {
            "wm_recon_loss": recon_loss.item(),
            "wm_reward_loss": reward_loss.item(),
            "wm_kl_loss": kl_loss.item(),
            "wm_total_loss": wm_loss.item(),
        }

        if return_posteriors:
            return wm_loss, features, stats, posteriors
        return wm_loss, features, stats

    def encode_obs(self, obs):
        """Encode observations into a feature vector (snapshot, no temporal dynamics).

        Args:
            obs: [B, n_agents, obs_size]

        Returns:
            feature: [B, feature_dim]
        """
        B = obs.shape[0]
        embed = self.obs_encoder(obs)
        init_state = self.get_initial_state(B, obs.device)
        zero_action = torch.zeros(B, self.action_embed.output_dim, device=obs.device)
        posterior, _ = self.transition.obs_step(init_state, zero_action, embed)
        return self.transition.get_feature(posterior)

    def extract_camera_positions(self, state):
        positions = []
        for i in range(self.n_agents):
            start = PRESERVED_DIM + i * CAMERA_STATE_DIM_PRIVATE
            positions.append(state[:, :, start:start + 2])
        return torch.stack(positions, dim=2)
