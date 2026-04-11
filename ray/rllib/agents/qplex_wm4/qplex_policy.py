"""QPLEX + Dreamer-style World Model (WM4) Policy.

WM4 extends WM2 with full Dreamer imagination:
  - ObsDecoder: global latent → per-agent obs (policy-in-loop during imagination)
  - Obs reconstruction loss in world model training
  - Dreamer rollout: Q-network selects actions at each imagined step
  - Imagination reconstruction loss: 1-step imagined state vs real next_state

Training flow:
  1. RSSM observe() — encode sequence with posterior (conditioned on real obs)
  2. World model loss — reconstruction + reward prediction + KL + obs reconstruction
  3. Dreamer imagination — H steps: ObsDecoder → Q-network → Prior transition
  4. QPLEX loss — obs/state augmented with latent features
"""

from gym.spaces import Tuple, Discrete, Dict, Box
import logging
import numpy as np
import tree
import torch.nn.functional as F
from argparse import Namespace

import ray
from ray.rllib.agents.qplex_v2.mixers import DuelMixerV2
from ray.rllib.agents.qplex_v2.model import RNNModel, _get_size
from ray.rllib.agents.qplex_v2.qplex_policy import (
    _validate, _mac, _unroll_mac, _drop_agent_dim, _add_agent_dim, adjust_args,
)
from .world_model_v4 import LatentWorldModel, PRESERVED_DIM, CAMERA_STATE_DIM_PRIVATE

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


def _ema_update(ema_model, model, decay=0.995):
    """Exponential moving average update: ema = decay * ema + (1-decay) * model."""
    with torch.no_grad():
        for ema_p, p in zip(ema_model.parameters(), model.parameters()):
            ema_p.data.mul_(decay).add_(p.data, alpha=1.0 - decay)


class QPLEXWM4Loss(nn.Module):
    """QPLEX loss with Dreamer-style RSSM world model (WM4).

    World model provides:
      - Latent features for obs/state augmentation (via EMA world model for stability)
      - Reconstruction + KL + reward prediction + obs reconstruction auxiliary losses
      - Dreamer imagination rollout with policy-in-loop (ObsDecoder → Q-network)
      - 1-step imagination reconstruction loss vs real next_state
    """

    def __init__(
        self,
        model, target_model,
        mixer, target_mixer,
        world_model, ema_world_model,
        n_agents, n_actions, h_size,
        double_q=True, gamma=0.99,
        wm_loss_weight=0.5,
        reward_bonus_coeff=0.1,
        reward_bonus_scale=0.5,
        imagination_horizon=5,
        use_imagination_targets=False,
        imagination_loss_weight=0.1,
        ema_decay=0.995,
    ):
        nn.Module.__init__(self)
        self.model = model
        self.target_model = target_model
        self.mixer = mixer
        self.target_mixer = target_mixer
        self.world_model = world_model
        self.ema_world_model = ema_world_model
        self.n_agents = n_agents
        self.n_actions = n_actions
        self.h_size = h_size          # GRU hidden size — used for zero hidden state in imagination
        self.double_q = double_q
        self.gamma = gamma
        self.wm_loss_weight = wm_loss_weight
        self.reward_bonus_coeff = reward_bonus_coeff
        self.reward_bonus_scale = reward_bonus_scale
        self.imagination_horizon = imagination_horizon
        self.use_imagination_targets = use_imagination_targets
        self.imagination_loss_weight = imagination_loss_weight
        self.ema_decay = ema_decay

    def _compute_reward_bonus(self, state_decoded, state_real):
        """Reward bonus based on how well the world model reconstructs the state."""
        recon_error = ((state_decoded - state_real) ** 2).mean(dim=-1, keepdim=True)
        bonus = torch.exp(-recon_error / self.reward_bonus_scale)
        return bonus  # [B, T, 1]

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
        wm_mask = mask[:, :, 0]

        # =================================================================
        # 1. World Model: RSSM observe + losses
        # =================================================================
        wm_loss, features, wm_stats, posteriors = self.world_model.compute_loss(
            obs, actions, state, rewards, wm_mask, return_posteriors=True
        )

        # =================================================================
        # 2. EMA World Model: stable latent features for augmentation
        # =================================================================
        with torch.no_grad():
            _, ema_features, _ = self.ema_world_model.compute_loss(
                obs, actions, state, rewards, wm_mask
            )
            feature_det = ema_features

            feature_expanded = feature_det.unsqueeze(2).expand(-1, -1, self.n_agents, -1)
            obs_aug = torch.cat([obs, feature_expanded], dim=-1)

            next_obs_flat = next_obs.reshape(B * T, self.n_agents, -1)
            next_feature = self.ema_world_model.encode_obs(next_obs_flat)
            next_feature = next_feature.reshape(B, T, -1)
            next_feature_expanded = next_feature.unsqueeze(2).expand(-1, -1, self.n_agents, -1)
            next_obs_aug = torch.cat([next_obs, next_feature_expanded], dim=-1)

        # =================================================================
        # 3. State Augmentation
        # =================================================================
        aug_state = torch.cat([state, feature_det], dim=-1)
        aug_next_state = torch.cat([next_state, next_feature], dim=-1)

        # =================================================================
        # 4. Reward Shaping
        # =================================================================
        state_decoded = self.world_model.state_decoder(
            features.detach().reshape(B * T, -1)
        ).reshape(B, T, -1)
        reward_bonus = self._compute_reward_bonus(state_decoded, state.detach())
        shaped_rewards = rewards + self.reward_bonus_coeff * reward_bonus.expand_as(rewards)

        # =================================================================
        # 5. QPLEX Q-value computation
        # =================================================================
        mac_out = _unroll_mac(self.model, obs_aug)

        chosen_action_qvals = torch.gather(
            mac_out, dim=3, index=actions.unsqueeze(3)
        ).squeeze(3)
        ignore_action = (action_mask == 0) & (mask == 1).unsqueeze(-1)
        x_mac_out = mac_out.clone().detach()
        x_mac_out[ignore_action] = -np.inf
        max_action_vals, _ = x_mac_out.max(dim=3)

        target_mac_out = _unroll_mac(self.target_model, next_obs_aug)
        ignore_action_tp1 = (next_action_mask == 0) & (mask == 1).unsqueeze(-1)
        target_mac_out[ignore_action_tp1] = -np.inf

        if self.double_q:
            mac_out_tp1 = _unroll_mac(self.model, next_obs_aug)
            mac_out_tp1[ignore_action_tp1] = -np.inf
            cur_max_actions = mac_out_tp1.argmax(dim=3, keepdim=True)
            target_max_qvals = torch.gather(target_mac_out, 3, cur_max_actions).squeeze(3)
        else:
            target_max_qvals = target_mac_out.max(dim=3)[0]

        assert target_max_qvals.min().item() != -np.inf

        # =================================================================
        # 6. Mix with augmented state
        # =================================================================
        if self.mixer is not None:
            ans_chosen = self.mixer(chosen_action_qvals, aug_state, is_v=True)
            actions_onehot = F.one_hot(actions, num_classes=self.n_actions)
            ans_adv = self.mixer(
                chosen_action_qvals, aug_state, actions_onehot,
                max_action_vals=max_action_vals, is_v=False
            )
            chosen_action_qvals = ans_chosen + ans_adv

            target_chosen = self.target_mixer(target_max_qvals, aug_next_state, is_v=True)
            cur_max_actions_onehot = F.one_hot(cur_max_actions, num_classes=self.n_actions)
            target_adv = self.target_mixer(
                target_max_qvals, aug_next_state, cur_max_actions_onehot,
                target_max_qvals, is_v=False
            )
            target_max_qvals = target_chosen + target_adv

        # =================================================================
        # 7. TD targets + Dreamer imagination (if enabled)
        # =================================================================
        targets = shaped_rewards + self.gamma * (1 - terminated) * target_max_qvals
        imag_recon_loss = torch.tensor(0.0, device=obs.device)

        if self.use_imagination_targets:
            H = self.imagination_horizon
            BT = B * T

            # Flatten posteriors: [B, T, dim] → [B*T, dim]
            imag_state = [p.reshape(BT, -1).detach() for p in posteriors]
            act_flat = actions.reshape(BT, self.n_agents)

            imag_rewards_list = []
            imag_decoded_states = []

            for h in range(H):
                # ── Get current latent feature ───────────────────────────
                imag_feature = self.world_model.transition.get_feature(imag_state)

                # ── Decode obs from latent (ObsDecoder) ──────────────────
                with torch.no_grad():
                    imag_obs = self.world_model.obs_decoder(imag_feature)  # [BT, n_agents, obs_size]

                # ── Select action via Q-network (greedy, zero hidden) ────
                if h == 0:
                    step_actions = act_flat  # use real action at first step
                else:
                    feat_exp = imag_feature.detach().unsqueeze(1).expand(-1, self.n_agents, -1)
                    obs_aug_imag = torch.cat([imag_obs, feat_exp], dim=-1)
                    h_zero = [torch.zeros(BT, self.n_agents, self.h_size, device=obs.device)]
                    with torch.no_grad():
                        q_imag, _ = _mac(self.model, obs_aug_imag, h_zero)
                    step_actions = q_imag.argmax(dim=-1)  # [BT, n_agents]

                # ── Prior step: (z_t, a_t) → z_{t+1}^imag ───────────────
                action_embed = self.world_model.action_embed(step_actions)
                imag_state = self.world_model.transition.img_step(imag_state, action_embed)

                # ── Predict reward + decode state ─────────────────────────
                imag_feature_next = self.world_model.transition.get_feature(imag_state)
                imag_reward = self.world_model.reward_predictor(imag_feature_next)
                imag_state_decoded = self.world_model.state_decoder(imag_feature_next)

                imag_rewards_list.append(imag_reward)
                imag_decoded_states.append(imag_state_decoded)

            # ── Discounted H-step imagined return ─────────────────────────
            imag_rewards_tensor = torch.stack(imag_rewards_list, dim=1)  # [BT, H]
            gammas = torch.pow(
                torch.tensor(self.gamma, dtype=torch.float, device=obs.device),
                torch.arange(H, device=obs.device, dtype=torch.float),
            )
            imag_return = (imag_rewards_tensor * gammas.unsqueeze(0)).sum(dim=1).reshape(B, T)

            term_mean = terminated.mean(dim=-1)
            imag_td_targets = (
                imag_return
                + (self.gamma ** H) * (1 - term_mean) * target_max_qvals.detach()
            )

            targets = (
                (1.0 - self.imagination_loss_weight) * targets
                + self.imagination_loss_weight * imag_td_targets
            )

            # ── Reconstruction loss: z_{t+1}^imag vs state_{t+1}^real ────
            # Only use first imagination step — h>1 has no real ground truth
            real_next_state = next_state.reshape(BT, -1).detach()
            first_imag_state = imag_decoded_states[0]                          # [BT, state_dim]
            imag_recon = ((first_imag_state - real_next_state) ** 2).mean(dim=-1)
            imag_recon_loss = imag_recon.reshape(B, T)
            wm_mask_2d = mask[:, :, 0]
            imag_recon_loss = (imag_recon_loss * wm_mask_2d).sum() / wm_mask_2d.sum().clamp(min=1)

        td_error = chosen_action_qvals - targets.detach()
        mask = mask.expand_as(td_error)
        masked_td_error = td_error * mask

        td_loss = (masked_td_error ** 2).sum() / mask.sum()
        total_loss = (
            td_loss
            + self.wm_loss_weight * wm_loss
            + self.imagination_loss_weight * imag_recon_loss
        )

        stats = {
            "td_loss": td_loss.item(),
            "reward_bonus_mean": reward_bonus.mean().item(),
            "imag_recon_loss": imag_recon_loss.item(),
            **wm_stats,
        }

        return total_loss, stats, mask, masked_td_error, chosen_action_qvals, targets


class QPLEXWM4TorchPolicy(Policy):
    """QPLEX + RSSM World Model (WM4) policy with Dreamer imagination.

    The RNN model receives obs augmented with RSSM latent features.
    When use_imagination_targets=True, a Dreamer-style rollout uses the
    ObsDecoder + Q-network to select actions at each imagined step.
    """

    def __init__(self, obs_space, action_space, config):
        _validate(obs_space, action_space)
        config = dict(ray.rllib.agents.qplex_wm4.qplex.DEFAULT_CONFIG, **config)

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
        # World Model (RSSM + ObsDecoder)
        # =====================================================================
        wm_config = config.get("world_model_v4", {})
        state_dim = int(np.prod(self.env_global_state_shape))
        self.stoch_dim = wm_config.get("stoch_dim", 32)
        self.deter_dim = wm_config.get("deter_dim", 128)

        self.world_model = LatentWorldModel(
            obs_size=self.obs_size,
            state_dim=state_dim,
            n_agents=self.n_agents,
            n_actions=self.n_actions,
            stoch_dim=self.stoch_dim,
            deter_dim=self.deter_dim,
            hidden_dim=wm_config.get("hidden_dim", 128),
            action_embed_dim=wm_config.get("action_embed_dim", 16),
            embed_dim=wm_config.get("embed_dim", 128),
            imagination_horizon=wm_config.get("imagination_horizon", 5),
            kl_coeff=wm_config.get("kl_coeff", 1.0),
            free_nats=wm_config.get("free_nats", 1.0),
        ).to(self.device)

        feature_dim = self.world_model.feature_dim

        import copy
        self.ema_world_model = copy.deepcopy(self.world_model)
        for p in self.ema_world_model.parameters():
            p.requires_grad = False
        self.ema_decay = wm_config.get("ema_decay", 0.995)

        # =====================================================================
        # Agent RNN model — augmented obs: obs_size + feature_dim
        # =====================================================================
        self.augmented_obs_size = self.obs_size + feature_dim
        augmented_agent_obs_space = Box(
            low=-np.inf * np.ones(self.augmented_obs_size, dtype=np.float32),
            high=np.inf * np.ones(self.augmented_obs_size, dtype=np.float32),
            dtype=np.float32,
        )

        self.model = ModelCatalog.get_model_v2(
            augmented_agent_obs_space, action_space.spaces[0], self.n_actions,
            config["model"], framework="torch", name="model", default_model=RNNModel,
        ).to(self.device)

        self.target_model = ModelCatalog.get_model_v2(
            augmented_agent_obs_space, action_space.spaces[0], self.n_actions,
            config["model"], framework="torch", name="target_model", default_model=RNNModel,
        ).to(self.device)

        self.exploration = self._create_exploration()

        # =====================================================================
        # Mixer — augmented state: state_dim + feature_dim
        # =====================================================================
        augmented_state_dim = state_dim + feature_dim
        augmented_state_shape = (augmented_state_dim,)

        self.mixer = DuelMixerV2(
            self.args, self.n_agents, self.n_actions, augmented_state_shape,
            config['mixing_embed_dim'], self.args.ffn_hidden_dim, self.args.num_kernel,
        ).to(self.device)
        self.target_mixer = DuelMixerV2(
            self.args, self.n_agents, self.n_actions, augmented_state_shape,
            config['mixing_embed_dim'], self.args.ffn_hidden_dim, self.args.num_kernel,
        ).to(self.device)
        assert config['mixer'] == 'qplex_wm4'

        self.cur_epsilon = 1.0
        self.update_target()

        # =====================================================================
        # Optimizer
        # =====================================================================
        self.params = list(self.model.parameters())
        self.params += list(self.mixer.parameters())
        self.params += list(self.world_model.parameters())

        self.loss = QPLEXWM4Loss(
            self.model, self.target_model,
            self.mixer, self.target_mixer,
            self.world_model, self.ema_world_model,
            self.n_agents, self.n_actions, self.h_size,
            self.config["double_q"], self.config["gamma"],
            wm_loss_weight=wm_config.get("wm_loss_weight", 0.5),
            reward_bonus_coeff=wm_config.get("reward_bonus_coeff", 0.1),
            reward_bonus_scale=wm_config.get("reward_bonus_scale", 0.5),
            imagination_horizon=wm_config.get("imagination_horizon", 5),
            use_imagination_targets=wm_config.get("use_imagination_targets", False),
            imagination_loss_weight=wm_config.get("imagination_loss_weight", 0.1),
            ema_decay=self.ema_decay,
        )

        from torch.optim import RMSprop
        self.optimiser = RMSprop(
            params=self.params,
            lr=config["lr"],
            alpha=config["optim_alpha"],
            eps=config["optim_eps"],
        )

    # -----------------------------------------------------------------
    # Inference
    # -----------------------------------------------------------------

    def _augment_obs_inference(self, obs_tensor):
        B = obs_tensor.shape[0]
        feature = self.ema_world_model.encode_obs(obs_tensor)
        feature_expanded = feature.unsqueeze(1).expand(-1, self.n_agents, -1)
        return torch.cat([obs_tensor, feature_expanded], dim=-1)

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
            augmented_obs = self._augment_obs_inference(obs_tensor)

            q_values, hiddens = _mac(
                self.model, augmented_obs,
                [torch.as_tensor(np.array(s), dtype=torch.float, device=self.device)
                 for s in state_batches],
            )
            avail = torch.as_tensor(action_mask, dtype=torch.float, device=self.device)
            masked_q_values = q_values.clone()
            masked_q_values[avail == 0.0] = -float("inf")
            masked_q_values_folded = torch.reshape(
                masked_q_values, [-1] + list(masked_q_values.shape)[2:]
            )
            if timestep is None:
                timestep = int(1e9)
            actions, _ = self.exploration.get_exploration_action(
                action_distribution=TorchCategorical(masked_q_values_folded),
                timestep=timestep, explore=explore,
            )
            actions = torch.reshape(actions, list(masked_q_values.shape)[:-1]).cpu().numpy()
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

        (total_loss, stats, mask, masked_td_error, chosen_action_qvals, targets,
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

        _ema_update(self.ema_world_model, self.world_model, self.ema_decay)

        mask_elems = mask.sum().item()
        stats.update({
            "loss": total_loss.item(),
            "grad_norm": grad_norm if isinstance(grad_norm, float) else grad_norm.item(),
            "td_error_abs": masked_td_error.abs().sum().item() / mask_elems,
            "q_taken_mean": (chosen_action_qvals * mask).sum().item() / mask_elems,
            "target_mean": (targets * mask).sum().item() / mask_elems,
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
            "world_model": self._cpu_dict(self.world_model.state_dict()),
            "ema_world_model": self._cpu_dict(self.ema_world_model.state_dict()),
        }

    @override(Policy)
    def set_weights(self, weights):
        self.model.load_state_dict(self._device_dict(weights["model"]))
        self.target_model.load_state_dict(self._device_dict(weights["target_model"]))
        self.mixer.load_state_dict(self._device_dict(weights["mixer"]))
        self.target_mixer.load_state_dict(self._device_dict(weights["target_mixer"]))
        if "world_model" in weights and weights["world_model"] is not None:
            self.world_model.load_state_dict(self._device_dict(weights["world_model"]))
        if "ema_world_model" in weights and weights["ema_world_model"] is not None:
            self.ema_world_model.load_state_dict(self._device_dict(weights["ema_world_model"]))

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
