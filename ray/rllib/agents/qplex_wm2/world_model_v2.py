"""Dreamer-style RSSM World Model for Multi-Agent QPLEX.

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
        # Encode each agent's observation
        B, C, _ = obs.shape
        agent_embeds = self.agent_encoder(obs.reshape(B * C, -1))  # [B*C, embed_dim]
        agent_embeds = agent_embeds.reshape(B, C, -1)               # [B, C, embed_dim]
        # Aggregate: mean pool across agents
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

        # Prior pathway: (stoch + action) → hidden → GRU → hidden → (mean, std)
        self.prior_fc1 = nn.Linear(stoch_dim + action_dim, hidden_dim)
        self.gru = nn.GRUCell(hidden_dim, deter_dim)
        self.prior_fc2 = nn.Linear(deter_dim, hidden_dim)
        self.prior_fc3 = nn.Linear(hidden_dim, 2 * stoch_dim)

        # Posterior pathway: (deter + embed) → hidden → (mean, std)
        self.post_fc1 = nn.Linear(deter_dim + embed_dim, hidden_dim)
        self.post_fc2 = nn.Linear(hidden_dim, 2 * stoch_dim)

    @property
    def feature_dim(self):
        return self.stoch_dim + self.deter_dim

    def get_initial_state(self, batch_size, device):
        """Returns initial latent state [mean, std, stoch, deter]."""
        return [
            torch.zeros(batch_size, self.stoch_dim, device=device),  # mean
            torch.zeros(batch_size, self.stoch_dim, device=device),  # std
            torch.zeros(batch_size, self.stoch_dim, device=device),  # stoch
            torch.zeros(batch_size, self.deter_dim, device=device),  # deter
        ]

    def get_feature(self, state):
        """Extract feature vector from latent state.

        Args:
            state: list of [mean, std, stoch, deter]

        Returns:
            feature: [B, stoch_dim + deter_dim]
        """
        return torch.cat([state[2], state[3]], dim=-1)

    def img_step(self, prev_state, action_embed):
        """Prior transition: predict next state from (prev_state, action) — no observation.

        Used during imagination rollouts.

        Args:
            prev_state: [mean, std, stoch, deter] each [B, dim]
            action_embed: [B, action_dim]

        Returns:
            prior_state: [mean, std, stoch, deter]
        """
        prev_stoch = prev_state[2]
        prev_deter = prev_state[3]

        x = F.elu(self.prior_fc1(torch.cat([prev_stoch, action_embed], dim=-1)))
        deter = self.gru(x, prev_deter)
        x = F.elu(self.prior_fc2(deter))
        stats = self.prior_fc3(x)
        mean, log_std = stats.chunk(2, dim=-1)
        std = F.softplus(log_std) + 0.1  # ensure positive std
        stoch = mean + std * torch.randn_like(std)

        return [mean, std, stoch, deter]

    def obs_step(self, prev_state, action_embed, obs_embed):
        """Posterior transition: refine prior with observation.

        Used during training.

        Args:
            prev_state: [mean, std, stoch, deter]
            action_embed: [B, action_dim]
            obs_embed: [B, embed_dim]

        Returns:
            posterior: [mean, std, stoch, deter]
            prior: [mean, std, stoch, deter]
        """
        # First compute prior
        prior = self.img_step(prev_state, action_embed)
        prior_deter = prior[3]

        # Then refine with observation → posterior
        x = F.elu(self.post_fc1(torch.cat([prior_deter, obs_embed], dim=-1)))
        stats = self.post_fc2(x)
        post_mean, post_log_std = stats.chunk(2, dim=-1)
        post_std = F.softplus(post_log_std) + 0.1
        post_stoch = post_mean + post_std * torch.randn_like(post_std)

        posterior = [post_mean, post_std, post_stoch, prior_deter]
        return posterior, prior

    def observe(self, obs_embeds, action_embeds, initial_state=None):
        """Roll through a sequence with observations (training mode).

        Args:
            obs_embeds: [B, T, embed_dim]
            action_embeds: [B, T, action_dim]
            initial_state: [mean, std, stoch, deter] or None

        Returns:
            posteriors: list of 4 tensors, each [B, T, dim]
            priors: list of 4 tensors, each [B, T, dim]
        """
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

        posteriors = [torch.stack(x, dim=1) for x in post_list]   # each [B, T, dim]
        priors = [torch.stack(x, dim=1) for x in prior_list]
        return posteriors, priors

    def imagine(self, action_embeds, initial_state):
        """Roll forward using prior only — no observations (imagination mode).

        Args:
            action_embeds: [B, H, action_dim]
            initial_state: [mean, std, stoch, deter]

        Returns:
            imagined_states: list of 4 tensors, each [B, H, dim]
        """
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
        return self.net(feature).squeeze(-1)  # [B]


class LatentWorldModel(nn.Module):
    """Complete RSSM-based world model for multi-agent QPLEX.

    Combines: ObservationEncoder + ActionEmbedding + TransitionModel + StateDecoder + RewardPredictor

    Args:
        obs_size: per-agent observation dimension
        state_dim: global state dimension
        n_agents: number of agents (cameras)
        n_actions: number of discrete actions per agent
        stoch_dim: stochastic latent dimension
        deter_dim: deterministic (GRU) latent dimension
        hidden_dim: MLP hidden dimension
        action_embed_dim: action embedding dimension
        embed_dim: observation embedding dimension
        imagination_horizon: H steps for imagination rollout
        kl_coeff: KL divergence loss coefficient
        free_nats: minimum KL (free bits trick)
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

        # Feature dimension (= augmentation dimension for obs and state)
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

    # -----------------------------------------------------------------
    # Training: compute all world model losses
    # -----------------------------------------------------------------

    def compute_loss(self, obs, actions, state, rewards, mask, return_posteriors=False):
        """Compute world model losses: reconstruction + reward + KL.

        Args:
            obs: [B, T, n_agents, obs_size] — local observations
            actions: [B, T, n_agents] — discrete actions (long)
            state: [B, T, state_dim] — global state (reconstruction target)
            rewards: [B, T, n_agents] — actual rewards (mean across agents for reward target)
            mask: [B, T] — valid timestep mask
            return_posteriors: if True, also return posteriors list for imagination rollout

        Returns:
            wm_loss: scalar — total world model loss
            features: [B, T, feature_dim] — posterior features (for obs/state augmentation)
            stats: dict — individual loss components for logging
            posteriors: (only if return_posteriors=True) list of 4 tensors [B, T, dim]
        """
        B, T = obs.shape[0], obs.shape[1]

        # Encode observations: [B, T, n_agents, obs_size] → [B, T, embed_dim]
        obs_flat = obs.reshape(B * T, self.n_agents, self.obs_size)
        obs_embeds = self.obs_encoder(obs_flat).reshape(B, T, -1)

        # Embed actions: [B, T, n_agents] → [B, T, action_dim]
        actions_flat = actions.reshape(B * T, self.n_agents)
        action_embeds = self.action_embed(actions_flat).reshape(B, T, -1)

        # RSSM observe: roll through sequence with posterior
        posteriors, priors = self.transition.observe(obs_embeds, action_embeds)

        # Extract features from posterior: [B, T, feature_dim]
        features = torch.cat([posteriors[2], posteriors[3]], dim=-1)

        # --- Reconstruction loss: decode latent → global state ---
        features_flat = features.reshape(B * T, -1)
        state_pred = self.state_decoder(features_flat).reshape(B, T, -1)
        recon_loss = ((state_pred - state.detach()) ** 2).mean(dim=-1)  # [B, T]
        recon_loss = (recon_loss * mask).sum() / mask.sum().clamp(min=1)

        # --- Reward prediction loss ---
        reward_pred = self.reward_predictor(features_flat).reshape(B, T)
        reward_target = rewards.mean(dim=-1)  # average across agents → [B, T]
        reward_loss = ((reward_pred - reward_target.detach()) ** 2)  # [B, T]
        reward_loss = (reward_loss * mask).sum() / mask.sum().clamp(min=1)

        # --- KL divergence: posterior || prior ---
        post_dist = td.Normal(posteriors[0], posteriors[1])
        prior_dist = td.Normal(priors[0], priors[1])
        kl = td.kl_divergence(post_dist, prior_dist).sum(dim=-1)  # [B, T]
        kl = torch.clamp(kl, min=self.free_nats)  # free nats trick
        kl_loss = (kl * mask).sum() / mask.sum().clamp(min=1)

        # --- Total world model loss ---
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

    # -----------------------------------------------------------------
    # Imagination rollout
    # -----------------------------------------------------------------

    def imagine_rollout(self, start_features, action_sequence):
        """Roll forward H steps in imagination (prior only, no observations).

        Args:
            start_features: posteriors at time t — list of [mean, std, stoch, deter] each [B, dim]
            action_sequence: [B, H, n_agents] — actions for H future steps (long)

        Returns:
            imagined_features: [B, H, feature_dim]
            imagined_rewards: [B, H]
        """
        B, H, _ = action_sequence.shape

        # Embed future actions
        actions_flat = action_sequence.reshape(B * H, self.n_agents)
        action_embeds = self.action_embed(actions_flat).reshape(B, H, -1)

        # Imagine H steps forward
        imagined_states = self.transition.imagine(action_embeds, start_features)

        # Extract features and predict rewards
        imag_features = torch.cat([imagined_states[2], imagined_states[3]], dim=-1)  # [B, H, feat]
        imag_rewards = self.reward_predictor(
            imag_features.reshape(B * H, -1)
        ).reshape(B, H)

        return imag_features, imag_rewards

    # -----------------------------------------------------------------
    # Inference: simple encoding (no temporal RSSM state)
    # -----------------------------------------------------------------

    def encode_obs(self, obs):
        """Encode observations into a feature vector (for inference-time augmentation).

        Uses the observation encoder + a zero-state prior as a simple feature.
        No temporal RSSM dynamics — just a snapshot encoding.

        Args:
            obs: [B, n_agents, obs_size]

        Returns:
            feature: [B, feature_dim]
        """
        B = obs.shape[0]
        embed = self.obs_encoder(obs)  # [B, embed_dim]

        # Use posterior pathway with zero RSSM state and zero action
        init_state = self.get_initial_state(B, obs.device)
        zero_action = torch.zeros(B, self.action_embed.output_dim, device=obs.device)
        posterior, _ = self.transition.obs_step(init_state, zero_action, embed)
        return self.transition.get_feature(posterior)  # [B, feature_dim]

    # -----------------------------------------------------------------
    # Camera position extraction (for reward shaping)
    # -----------------------------------------------------------------

    def extract_camera_positions(self, state):
        """Extract camera (x, y) from global state.

        Args:
            state: [B, T, state_dim]

        Returns:
            [B, T, n_cameras, 2]
        """
        positions = []
        for i in range(self.n_agents):
            start = PRESERVED_DIM + i * CAMERA_STATE_DIM_PRIVATE
            positions.append(state[:, :, start:start + 2])
        return torch.stack(positions, dim=2)

    def extract_target_positions(self, state):
        """Extract target (x, y) from global state.

        Args:
            state: [B, T, state_dim]

        Returns:
            [B, T, n_targets, 2] where n_targets is inferred from state_dim
        """
        target_start = PRESERVED_DIM + self.n_agents * CAMERA_STATE_DIM_PRIVATE
        # Infer n_targets from state dimensions
        remaining = self.state_dim - target_start
        # After targets: obstacles + extra. Approximate by dividing by TARGET_STATE_DIM_PRIVATE
        n_targets = 0
        idx = target_start
        positions = []
        while idx + TARGET_STATE_DIM_PRIVATE <= self.state_dim:
            positions.append(state[:, :, idx:idx + 2])
            idx += TARGET_STATE_DIM_PRIVATE
            n_targets += 1
            if n_targets >= 20:  # safety cap
                break
        if not positions:
            return torch.zeros(state.shape[0], state.shape[1], 0, 2, device=state.device)
        return torch.stack(positions, dim=2)
