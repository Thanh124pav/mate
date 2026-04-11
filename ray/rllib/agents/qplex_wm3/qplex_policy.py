"""QPLEX_WM3: Distributional QPLEX + Stochastic World Model + Shared Encoder.

Combines:
  1. DFAC-style distributional Q-values (quantile regression) with QPLEX decomposition
  2. Stochastic world model (Gaussian mixture) predicting probability regions
  3. Shared encoder: auxiliary NLL loss backprops through GRU+fc1
  4. Probability region coverage as reward shaping signal

Architecture:
  obs → fc1 (shared) → GRU → hidden_state
                                   │
                   ┌───────────────┼────────────────┐
                   │               │                │
            fc2 (distributional)   │     StochasticPredictionHead
                   │               │         (Gaussian mixture)
                   │               │                │
         [n_actions × N_quantiles] │    means, stds, weights per target
                   │               │                │
           Quantile Huber loss     │         NLL loss → gradient → fc1
                   │               │
            DuelMixerV2            │
          (per-quantile mixing)    │
                                   └──→ probability region coverage
                                        (reward shaping)

Metrics: td_loss, prediction_nll, coverage_mean, loss
"""

from gym.spaces import Tuple, Discrete, Dict
import logging
import numpy as np
import tree
import torch.nn.functional as F
from argparse import Namespace

import ray
from ray.rllib.agents.qplex_v2.mixers import DuelMixerV2
from ray.rllib.agents.qplex_v2.qplex_policy import (
    _validate, _mac, _drop_agent_dim, _add_agent_dim, adjust_args,
)
from .model import DistributionalRNNModel
from .world_model_v3 import (
    StochasticPredictionHead,
    compute_probability_coverage,
    extract_target_positions,
    extract_camera_positions,
    extract_camera_fov,
    _denorm_target_pos,
)

from ray.rllib.env.multi_agent_env import ENV_STATE
from ray.rllib.env.wrappers.group_agents_wrapper import GROUP_REWARDS
from ray.rllib.models.torch.torch_action_dist import TorchCategorical
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.rnn_sequencing import chop_into_sequences
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.models.catalog import ModelCatalog
from ray.rllib.models.modelv2 import _unpack_obs
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.metrics.learner_info import LEARNER_STATS_KEY
from ray.rllib.utils.annotations import override

torch, nn = try_import_torch(error=True)

logger = logging.getLogger(__name__)


# =========================================================================
# Distributional MAC helpers
# =========================================================================

def _mac_distributional(model, obs, h, n_quantiles):
    """Forward pass returning distributional Q-values.

    Returns:
        q_dist: [B, n_agents, n_actions, n_quantiles]
        h: hidden states
    """
    B, n_agents = obs.size(0), obs.size(1)
    q_flat, h = _mac(model, obs, h)
    # q_flat: [B, n_agents, n_actions * n_quantiles]
    n_actions = q_flat.shape[-1] // n_quantiles
    q_dist = q_flat.reshape(B, n_agents, n_actions, n_quantiles)
    return q_dist, h


def _unroll_mac_distributional(model, obs_tensor, n_quantiles):
    """Unroll distributional MAC over time, also returns hidden states.

    Returns:
        mac_out: [B, T, n_agents, n_actions, n_quantiles]
        hiddens: [B, T, n_agents, h_size]
    """
    B, T, n_agents = obs_tensor.shape[:3]
    mac_out = []
    hiddens_out = []
    h = [s.expand([B, n_agents, -1]) for s in model.get_initial_state()]
    for t in range(T):
        q, h = _mac_distributional(model, obs_tensor[:, t], h, n_quantiles)
        mac_out.append(q)
        hiddens_out.append(h[0])
    mac_out = torch.stack(mac_out, dim=1)
    hiddens_out = torch.stack(hiddens_out, dim=1)
    return mac_out, hiddens_out


def quantile_huber_loss(predictions, targets, taus, kappa=1.0):
    """Quantile Huber loss (QR-DQN).

    Args:
        predictions: [B, N_q] — predicted quantile values
        targets: [B, N_q] — target quantile values
        taus: [N_q] — quantile fractions
        kappa: float — Huber loss threshold

    Returns:
        loss: [B] — per-sample loss
    """
    # Pairwise delta: [B, N_q (pred), N_q (target)]
    delta = targets.unsqueeze(-2) - predictions.unsqueeze(-1)

    # Huber loss
    huber = torch.where(
        delta.abs() <= kappa,
        0.5 * delta ** 2,
        kappa * (delta.abs() - 0.5 * kappa),
    )

    # Quantile weighting (ensure taus on same device as delta)
    quantile_weight = (taus.to(delta.device).view(1, -1, 1) - (delta < 0).float()).abs()

    # Loss: mean over target quantiles, mean over pred quantiles
    loss = (quantile_weight * huber).sum(dim=-1).mean(dim=-1)  # [B]
    return loss


# =========================================================================
# Distributional Mixer wrapper
# =========================================================================

def _mix_per_quantile(mixer, agent_qs, states, actions_onehot=None,
                      max_action_vals=None, is_v=False, n_quantiles=8):
    """Run DuelMixerV2 per-quantile by expanding quantiles into batch dimension.

    Args:
        agent_qs: [B_T, n_agents, N_q]
        states: [B_T, state_dim]
        actions_onehot: [B_T, n_agents, n_actions] or None
        max_action_vals: [B_T, n_agents, N_q] or None
        is_v: bool
        n_quantiles: int

    Returns:
        mixed: [B_T, N_q, 1]
    """
    B_T, C, N_q = agent_qs.shape

    # Expand: [B_T, C, N_q] → [B_T*N_q, C]
    q_flat = agent_qs.permute(0, 2, 1).reshape(B_T * N_q, C)
    state_flat = states.unsqueeze(1).expand(-1, N_q, -1).reshape(B_T * N_q, -1)

    if is_v:
        result = mixer(q_flat, state_flat, is_v=True)  # [B_T*N_q, 1, 1]
    else:
        act_flat = actions_onehot.unsqueeze(1).expand(-1, N_q, -1, -1).reshape(
            B_T * N_q, C, -1
        )
        max_q_flat = max_action_vals.permute(0, 2, 1).reshape(B_T * N_q, C)
        result = mixer(q_flat, state_flat, act_flat, max_q_flat, is_v=False)

    return result.reshape(B_T, N_q, 1)


# =========================================================================
# Loss
# =========================================================================

class QPLEXWM3Loss(nn.Module):
    """Distributional QPLEX loss + stochastic world model auxiliary loss."""

    def __init__(
        self,
        model, target_model,
        mixer, target_mixer,
        pred_head,
        n_agents, n_actions, n_targets, n_quantiles,
        double_q=True, gamma=0.99,
        aux_loss_weight=0.1,
        coverage_bonus_coeff=0.05,
    ):
        nn.Module.__init__(self)
        self.model = model
        self.target_model = target_model
        self.mixer = mixer
        self.target_mixer = target_mixer
        self.pred_head = pred_head
        self.n_agents = n_agents
        self.n_actions = n_actions
        self.n_targets = n_targets
        self.n_quantiles = n_quantiles
        self.double_q = double_q
        self.gamma = gamma
        self.aux_loss_weight = aux_loss_weight
        self.coverage_bonus_coeff = coverage_bonus_coeff

        # Fixed quantile fractions: τ_i = (i + 0.5) / N
        self.register_buffer(
            "taus", torch.arange(0, n_quantiles, dtype=torch.float32).add(0.5).div(n_quantiles)
        )

    def forward(
        self,
        rewards, actions, terminated, mask,
        obs, next_obs, action_mask, next_action_mask,
        state=None, next_state=None,
    ):
        if state is None and next_state is None:
            state = obs
            next_state = next_obs
        elif (state is None) != (next_state is None):
            raise ValueError("Expected either neither or both of state/next_state.")

        B, T = obs.shape[0], obs.shape[1]
        N_q = self.n_quantiles

        # =================================================================
        # 1. Distributional Q-values + hidden states
        # =================================================================
        mac_out, hiddens = _unroll_mac_distributional(self.model, obs, N_q)
        # mac_out: [B, T, C, n_actions, N_q], hiddens: [B, T, C, h_size]

        # Gather Q-quantiles for taken actions: [B, T, C, N_q]
        actions_exp = actions.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, 1, N_q)
        chosen_q_dist = torch.gather(mac_out, dim=3, index=actions_exp).squeeze(3)

        # Mean Q for action selection and max Q
        q_mean = mac_out.mean(dim=-1)  # [B, T, C, n_actions]
        ignore_action = (action_mask == 0) & (mask == 1).unsqueeze(-1)
        q_mean_masked = q_mean.clone().detach()
        q_mean_masked[ignore_action] = -np.inf
        max_action_idx = q_mean_masked.argmax(dim=3)  # [B, T, C]

        # Max action quantiles: [B, T, C, N_q]
        max_idx_exp = max_action_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, 1, N_q)
        max_action_q_dist = torch.gather(mac_out.detach(), 3, max_idx_exp).squeeze(3)

        # =================================================================
        # 2. Target Q-quantiles
        # =================================================================
        target_mac_out = _unroll_mac_distributional(self.target_model, next_obs, N_q)[0]
        ignore_tp1 = (next_action_mask == 0) & (mask == 1).unsqueeze(-1)
        target_q_mean = target_mac_out.mean(dim=-1)
        target_q_mean[ignore_tp1] = -np.inf

        if self.double_q:
            policy_mac_next = _unroll_mac_distributional(self.model, next_obs, N_q)[0]
            policy_q_mean_next = policy_mac_next.mean(dim=-1)
            policy_q_mean_next[ignore_tp1] = -np.inf
            cur_max_actions = policy_q_mean_next.argmax(dim=3)
        else:
            cur_max_actions = target_q_mean.argmax(dim=3)

        # Target quantiles for selected actions: [B, T, C, N_q]
        cma_exp = cur_max_actions.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, 1, N_q)
        target_q_dist = torch.gather(target_mac_out, 3, cma_exp).squeeze(3)

        # =================================================================
        # 3. Mix per-quantile through DuelMixerV2
        # =================================================================
        chosen_mixed = torch.zeros(B, T, N_q, device=obs.device)
        target_mixed = torch.zeros(B, T, N_q, device=obs.device)

        if self.mixer is not None:
            # Reshape for mixer: [B*T, C, N_q]
            cq = chosen_q_dist.reshape(B * T, self.n_agents, N_q)
            mq = max_action_q_dist.reshape(B * T, self.n_agents, N_q)
            st = state.reshape(B * T, -1)

            v_chosen = _mix_per_quantile(self.mixer, cq, st, is_v=True, n_quantiles=N_q)
            actions_oh = F.one_hot(actions, num_classes=self.n_actions).reshape(
                B * T, self.n_agents, self.n_actions
            )
            a_chosen = _mix_per_quantile(
                self.mixer, cq, st, actions_oh, mq, is_v=False, n_quantiles=N_q
            )
            chosen_mixed = (v_chosen + a_chosen).squeeze(-1).reshape(B, T, N_q)

            tq = target_q_dist.reshape(B * T, self.n_agents, N_q)
            ns = next_state.reshape(B * T, -1)
            v_target = _mix_per_quantile(
                self.target_mixer, tq, ns, is_v=True, n_quantiles=N_q
            )
            cma_oh = F.one_hot(cur_max_actions, num_classes=self.n_actions).reshape(
                B * T, self.n_agents, self.n_actions
            )
            a_target = _mix_per_quantile(
                self.target_mixer, tq, ns, cma_oh, tq, is_v=False, n_quantiles=N_q
            )
            target_mixed = (v_target + a_target).squeeze(-1).reshape(B, T, N_q)

        # =================================================================
        # 4. Stochastic World Model (shared encoder auxiliary loss)
        # =================================================================
        C, H = self.n_agents, hiddens.shape[-1]
        hiddens_flat = hiddens.reshape(B * T * C, H)

        # Labels: next target positions from next_state
        labels = extract_target_positions(next_state, self.n_agents, self.n_targets)
        labels_flat = labels.reshape(B * T, self.n_targets, 2)
        labels_per_agent = labels_flat.unsqueeze(1).expand(-1, C, -1, -1).reshape(
            B * T * C, self.n_targets, 2
        )

        # NLL loss (gradient flows through shared fc1+GRU)
        nll = self.pred_head.nll_loss(hiddens_flat, labels_per_agent)  # [B*T*C]
        wm_mask = mask[:, :, 0]
        nll_per_step = nll.reshape(B, T, C).mean(dim=2)  # [B, T]
        aux_loss = (nll_per_step * wm_mask).sum() / wm_mask.sum().clamp(min=1)

        # =================================================================
        # 5. Probability region coverage (reward shaping)
        # =================================================================
        with torch.no_grad():
            # Get mode means and weights from prediction head (per agent)
            means_all, stds_all, weights_all = self.pred_head(hiddens_flat.detach())
            # means_all: [B*T*C, n_targets, K, 2], weights_all: [B*T*C, n_targets, K]
            K = means_all.shape[2]

            # Aggregate mode means across agents: average per mode
            means_per_agent = means_all.reshape(B, T, C, self.n_targets, K, 2)
            weights_per_agent = weights_all.reshape(B, T, C, self.n_targets, K)
            # Average means and weights across cameras
            means_agg = means_per_agent.mean(dim=2)    # [B, T, n_targets, K, 2]
            weights_agg = weights_per_agent.mean(dim=2)  # [B, T, n_targets, K]
            weights_agg = weights_agg / weights_agg.sum(dim=-1, keepdim=True).clamp(min=1e-8)

            # Denormalize mode means to real coordinates (predictions are in normalized space,
            # trained against normalized global state)
            means_denorm = (means_agg + 1.0) / 2.0 * 4000.0 - 2000.0  # [-1,1] → [-2000,2000]

            # Extract denormalized camera FOV
            cam_pos, cam_orient, cam_sr, cam_ha = extract_camera_fov(state, self.n_agents)

            coverage_list = []
            for t_idx in range(T):
                cov = compute_probability_coverage(
                    cam_pos[:, t_idx], cam_orient[:, t_idx],
                    cam_sr[:, t_idx], cam_ha[:, t_idx],
                    means_denorm[:, t_idx],    # [B, n_targets, K, 2]
                    weights_agg[:, t_idx],     # [B, n_targets, K]
                )  # [B, n_targets]
                coverage_list.append(cov)
            coverage = torch.stack(coverage_list, dim=1)  # [B, T, n_targets]
            coverage_bonus = coverage.mean(dim=-1, keepdim=True)  # [B, T, 1]

        # =================================================================
        # 6. TD targets with shaped rewards + quantile Huber loss
        # =================================================================
        reward_mean = rewards.mean(dim=-1)  # [B, T] — average across agents
        shaped_reward = reward_mean + self.coverage_bonus_coeff * coverage_bonus.squeeze(-1)

        # Target quantiles: r + γ * (1-done) * target_quantile
        terminated_mean = terminated.mean(dim=-1)  # [B, T]
        target_quantiles = (
            shaped_reward.unsqueeze(-1)
            + self.gamma * (1 - terminated_mean).unsqueeze(-1) * target_mixed
        ).detach()  # [B, T, N_q]

        # Quantile Huber loss
        pred_flat = chosen_mixed.reshape(B * T, N_q)
        target_flat = target_quantiles.reshape(B * T, N_q)
        qh_loss = quantile_huber_loss(pred_flat, target_flat, self.taus)  # [B*T]
        qh_loss = qh_loss.reshape(B, T)
        td_loss = (qh_loss * wm_mask).sum() / wm_mask.sum().clamp(min=1)

        # =================================================================
        # 7. Total loss
        # =================================================================
        total_loss = td_loss + self.aux_loss_weight * aux_loss

        # For logging: compute masked TD error analog
        td_error = (chosen_mixed.mean(dim=-1) - target_quantiles.mean(dim=-1))
        mask_2d = wm_mask
        masked_td_error = td_error * mask_2d

        stats = {
            "td_loss": td_loss.item(),
            "prediction_nll": aux_loss.item(),
            "coverage_mean": coverage_bonus.mean().item(),
        }

        return total_loss, stats, mask_2d, masked_td_error, chosen_mixed.mean(dim=-1), target_quantiles.mean(dim=-1)


# =========================================================================
# Policy
# =========================================================================

class QPLEXWM3TorchPolicy(Policy):
    """Distributional QPLEX + Stochastic World Model + Shared Encoder."""

    def __init__(self, obs_space, action_space, config):
        _validate(obs_space, action_space)
        config = dict(ray.rllib.agents.qplex_wm3.qplex.DEFAULT_CONFIG, **config)

        self.args = Namespace(**config)
        self.args = adjust_args(self.args)
        self.framework = "torch"
        super().__init__(obs_space, action_space, config)
        self.n_agents = len(obs_space.original_space.spaces)
        config["model"]["n_agents"] = self.n_agents
        self.n_actions = action_space.spaces[0].n
        self.h_size = config["model"]["lstm_cell_size"]
        self.has_env_global_state = False
        self.has_action_mask = False
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

        agent_obs_space = obs_space.original_space.spaces[0]
        if isinstance(agent_obs_space, Dict):
            space_keys = set(agent_obs_space.spaces.keys())
            if "obs" not in space_keys:
                raise ValueError("Dict obs space must have subspace labeled `obs`")
            self.obs_size = _get_size(agent_obs_space.spaces["obs"])
            if "action_mask" in space_keys:
                mask_shape = tuple(agent_obs_space.spaces["action_mask"].shape)
                if mask_shape != (self.n_actions,):
                    raise ValueError("Action mask shape must be {}, got {}".format(
                        (self.n_actions,), mask_shape))
                self.has_action_mask = True
            if ENV_STATE in space_keys:
                self.env_global_state_shape = _get_size(agent_obs_space.spaces[ENV_STATE])
                self.has_env_global_state = True
            else:
                self.env_global_state_shape = (self.obs_size, self.n_agents)
            config["model"]["full_obs_space"] = agent_obs_space
            agent_obs_space = agent_obs_space.spaces["obs"]
        else:
            self.obs_size = _get_size(agent_obs_space)
            self.env_global_state_shape = (self.obs_size, self.n_agents)

        # =====================================================================
        # Config
        # =====================================================================
        wm3_config = config.get("wm3", {})
        self.n_targets = wm3_config.get("n_targets", 8)
        self.n_quantiles = wm3_config.get("n_quantiles", 8)
        config["model"]["n_quantiles"] = self.n_quantiles

        # =====================================================================
        # Distributional RNN model (same obs_size — no augmentation)
        # =====================================================================
        self.model = ModelCatalog.get_model_v2(
            agent_obs_space, action_space.spaces[0], self.n_actions,
            config["model"], framework="torch", name="model",
            default_model=DistributionalRNNModel,
        ).to(self.device)

        self.target_model = ModelCatalog.get_model_v2(
            agent_obs_space, action_space.spaces[0], self.n_actions,
            config["model"], framework="torch", name="target_model",
            default_model=DistributionalRNNModel,
        ).to(self.device)

        self.exploration = self._create_exploration()

        # =====================================================================
        # Stochastic prediction head (shared encoder auxiliary task)
        # =====================================================================
        self.pred_head = StochasticPredictionHead(
            hidden_dim=self.h_size,
            n_targets=self.n_targets,
            n_modes=wm3_config.get("n_modes", 3),
            pred_hidden=wm3_config.get("pred_hidden", 128),
        ).to(self.device)

        # =====================================================================
        # Mixer — SAME state_dim as QPLEX_V2 (no augmentation)
        # =====================================================================
        self.mixer = DuelMixerV2(
            self.args, self.n_agents, self.n_actions, self.env_global_state_shape,
            config['mixing_embed_dim'], self.args.ffn_hidden_dim, self.args.num_kernel,
        ).to(self.device)
        self.target_mixer = DuelMixerV2(
            self.args, self.n_agents, self.n_actions, self.env_global_state_shape,
            config['mixing_embed_dim'], self.args.ffn_hidden_dim, self.args.num_kernel,
        ).to(self.device)
        assert config['mixer'] == 'qplex_wm3'

        self.cur_epsilon = 1.0
        self.update_target()

        # =====================================================================
        # Optimizer
        # =====================================================================
        self.params = list(self.model.parameters())
        self.params += list(self.mixer.parameters())
        self.params += list(self.pred_head.parameters())

        self.loss = QPLEXWM3Loss(
            self.model, self.target_model,
            self.mixer, self.target_mixer,
            self.pred_head,
            self.n_agents, self.n_actions, self.n_targets, self.n_quantiles,
            self.config["double_q"], self.config["gamma"],
            aux_loss_weight=wm3_config.get("aux_loss_weight", 0.1),
            coverage_bonus_coeff=wm3_config.get("coverage_bonus_coeff", 0.05),
        )

        from torch.optim import RMSprop
        self.optimiser = RMSprop(
            params=self.params,
            lr=config["lr"],
            alpha=config["optim_alpha"],
            eps=config["optim_eps"],
        )

    # -----------------------------------------------------------------
    # Actions (use mean Q for action selection)
    # -----------------------------------------------------------------

    @override(Policy)
    def compute_actions(
        self, obs_batch, state_batches=None, prev_action_batch=None,
        prev_reward_batch=None, info_batch=None, episodes=None,
        explore=None, timestep=None, **kwargs
    ):
        explore = explore if explore is not None else self.config["explore"]
        obs_batch, action_mask, _ = self._unpack_observation(obs_batch)

        with torch.no_grad():
            obs_tensor = torch.as_tensor(obs_batch, dtype=torch.float, device=self.device)
            q_dist, hiddens = _mac_distributional(
                self.model, obs_tensor,
                [torch.as_tensor(np.array(s), dtype=torch.float, device=self.device)
                 for s in state_batches],
                self.n_quantiles,
            )
            # q_dist: [B, n_agents, n_actions, N_q]
            # Use mean Q for action selection
            q_mean = q_dist.mean(dim=-1)  # [B, n_agents, n_actions]

            avail = torch.as_tensor(action_mask, dtype=torch.float, device=self.device)
            q_mean[avail == 0.0] = -float("inf")
            q_mean_folded = torch.reshape(q_mean, [-1] + list(q_mean.shape)[2:])

            if timestep is None:
                timestep = int(1e9)
            actions, _ = self.exploration.get_exploration_action(
                action_distribution=TorchCategorical(q_mean_folded),
                timestep=timestep, explore=explore,
            )
            actions = torch.reshape(actions, list(q_mean.shape)[:-1]).cpu().numpy()
            hiddens = [s.cpu().numpy() for s in hiddens]

        return tuple(actions.transpose([1, 0])), hiddens, {}

    @override(Policy)
    def compute_log_likelihoods(self, actions, obs_batch, state_batches=None,
                                prev_action_batch=None, prev_reward_batch=None):
        obs_batch, action_mask, _ = self._unpack_observation(obs_batch)
        return np.zeros(obs_batch.size()[0])

    # -----------------------------------------------------------------
    # Training
    # -----------------------------------------------------------------

    @override(Policy)
    def learn_on_batch(self, samples):
        obs_batch, action_mask, env_global_state = self._unpack_observation(
            samples[SampleBatch.CUR_OBS])
        (next_obs_batch, next_action_mask, next_env_global_state,
         ) = self._unpack_observation(samples[SampleBatch.NEXT_OBS])
        group_rewards = self._get_group_rewards(samples[SampleBatch.INFOS])

        input_list = [
            group_rewards, action_mask, next_action_mask,
            samples[SampleBatch.ACTIONS], samples[SampleBatch.DONES],
            obs_batch, next_obs_batch,
        ]
        if self.has_env_global_state:
            input_list.extend([env_global_state, next_env_global_state])

        output_list, _, seq_lens = chop_into_sequences(
            episode_ids=samples[SampleBatch.EPS_ID],
            unroll_ids=samples[SampleBatch.UNROLL_ID],
            agent_indices=samples[SampleBatch.AGENT_INDEX],
            feature_columns=input_list,
            state_columns=[],
            max_seq_len=self.config["model"]["max_seq_len"],
            dynamic_max=True,
        )

        if self.has_env_global_state:
            (rew, action_mask, next_action_mask, act, dones, obs, next_obs,
             env_global_state, next_env_global_state) = output_list
        else:
            (rew, action_mask, next_action_mask, act, dones, obs, next_obs) = output_list

        B, T = len(seq_lens), max(seq_lens)

        def to_batches(arr, dtype):
            new_shape = [B, T] + list(arr.shape[1:])
            return torch.as_tensor(np.reshape(arr, new_shape), dtype=dtype, device=self.device)

        rewards = to_batches(rew, torch.float)
        actions = to_batches(act, torch.long)
        obs = to_batches(obs, torch.float).reshape([B, T, self.n_agents, self.obs_size])
        action_mask = to_batches(action_mask, torch.float)
        next_obs = to_batches(next_obs, torch.float).reshape(
            [B, T, self.n_agents, self.obs_size])
        next_action_mask = to_batches(next_action_mask, torch.float)
        if self.has_env_global_state:
            env_global_state = to_batches(env_global_state, torch.float)
            next_env_global_state = to_batches(next_env_global_state, torch.float)

        terminated = to_batches(dones, torch.float).unsqueeze(2).expand(B, T, self.n_agents)

        filled = np.reshape(
            np.tile(np.arange(T, dtype=np.float32), B), [B, T]
        ) < np.expand_dims(seq_lens, 1)
        mask = (
            torch.as_tensor(filled, dtype=torch.float, device=self.device)
            .unsqueeze(2).expand(B, T, self.n_agents)
        )

        (total_loss, stats, mask_2d, masked_td_error, chosen_q_mean, target_q_mean,
         ) = self.loss(
            rewards, actions, terminated, mask,
            obs, next_obs, action_mask, next_action_mask,
            env_global_state, next_env_global_state,
        )

        self.optimiser.zero_grad()
        total_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.params, self.config["grad_norm_clipping"])
        self.optimiser.step()

        mask_elems = mask_2d.sum().item()
        stats.update({
            "loss": total_loss.item(),
            "grad_norm": grad_norm if isinstance(grad_norm, float) else grad_norm.item(),
            "td_error_abs": masked_td_error.abs().sum().item() / max(mask_elems, 1),
            "q_taken_mean": (chosen_q_mean * mask_2d).sum().item() / max(mask_elems, 1),
            "target_mean": (target_q_mean * mask_2d).sum().item() / max(mask_elems, 1),
        })
        return {LEARNER_STATS_KEY: stats}

    # -----------------------------------------------------------------
    # State management
    # -----------------------------------------------------------------

    @override(Policy)
    def get_initial_state(self):
        return [s.expand([self.n_agents, -1]).cpu().numpy()
                for s in self.model.get_initial_state()]

    @override(Policy)
    def get_weights(self):
        return {
            "model": self._cpu_dict(self.model.state_dict()),
            "target_model": self._cpu_dict(self.target_model.state_dict()),
            "mixer": self._cpu_dict(self.mixer.state_dict()),
            "target_mixer": self._cpu_dict(self.target_mixer.state_dict()),
            "pred_head": self._cpu_dict(self.pred_head.state_dict()),
        }

    @override(Policy)
    def set_weights(self, weights):
        self.model.load_state_dict(self._device_dict(weights["model"]))
        self.target_model.load_state_dict(self._device_dict(weights["target_model"]))
        self.mixer.load_state_dict(self._device_dict(weights["mixer"]))
        self.target_mixer.load_state_dict(self._device_dict(weights["target_mixer"]))
        if "pred_head" in weights and weights["pred_head"] is not None:
            self.pred_head.load_state_dict(self._device_dict(weights["pred_head"]))

    @override(Policy)
    def get_state(self):
        state = self.get_weights()
        state["cur_epsilon"] = self.cur_epsilon
        return state

    @override(Policy)
    def set_state(self, state):
        self.set_weights(state)
        self.set_epsilon(state["cur_epsilon"])

    def update_target(self):
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_mixer.load_state_dict(self.mixer.state_dict())

    def set_epsilon(self, epsilon):
        self.cur_epsilon = epsilon

    def _get_group_rewards(self, info_batch):
        return np.array([info.get(GROUP_REWARDS, [0.0] * self.n_agents) for info in info_batch])

    def _device_dict(self, state_dict):
        return {k: torch.as_tensor(v, device=self.device) for k, v in state_dict.items()}

    @staticmethod
    def _cpu_dict(state_dict):
        return {k: v.cpu().detach().numpy() for k, v in state_dict.items()}

    def _unpack_observation(self, obs_batch):
        unpacked = _unpack_obs(
            np.array(obs_batch, dtype=np.float32),
            self.observation_space.original_space, tensorlib=np)
        if isinstance(unpacked[0], dict):
            assert "obs" in unpacked[0]
            unpacked_obs = [np.concatenate(tree.flatten(u["obs"]), 1) for u in unpacked]
        else:
            unpacked_obs = unpacked
        obs = np.concatenate(unpacked_obs, axis=1).reshape(
            [len(obs_batch), self.n_agents, self.obs_size])
        if self.has_action_mask:
            action_mask = np.concatenate(
                [o["action_mask"] for o in unpacked], axis=1
            ).reshape([len(obs_batch), self.n_agents, self.n_actions])
        else:
            action_mask = np.ones(
                [len(obs_batch), self.n_agents, self.n_actions], dtype=np.float32)
        if self.has_env_global_state:
            state = np.concatenate(tree.flatten(unpacked[0][ENV_STATE]), 1)
        else:
            state = None
        return obs, action_mask, state


def _get_size(obs_space):
    from ray.rllib.models.preprocessors import get_preprocessor
    return get_preprocessor(obs_space)(obs_space).size
