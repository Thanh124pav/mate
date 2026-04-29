"""DuelMix + RSSM World Model (WM2) Policy."""

from argparse import Namespace
from gym.spaces import Dict, Discrete, Tuple
import logging
import numpy as np
import tree
import torch.nn.functional as F

import ray
from ray.rllib.agents.duelmix.duelmix import DEFAULT_CONFIG as DUELMIX_DEFAULT_CONFIG
from ray.rllib.agents.duelmix.duelmix_policy import (
    _drop_agent_dim,
    _mac_duelmix,
    _unroll_mac_duelmix,
    _validate,
)
from ray.rllib.agents.duelmix.mixers import DuelMixMixer
from ray.rllib.agents.duelmix.model import DualStreamRNNModel, _get_size
from ray.rllib.agents.duelmix_wm2.world_model_v2 import LatentWorldModel
from ray.rllib.env.multi_agent_env import ENV_STATE
from ray.rllib.env.wrappers.group_agents_wrapper import GROUP_REWARDS
from ray.rllib.models.catalog import ModelCatalog
from ray.rllib.models.modelv2 import _unpack_obs
from ray.rllib.models.torch.torch_action_dist import TorchCategorical
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.rnn_sequencing import chop_into_sequences
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.metrics.learner_info import LEARNER_STATS_KEY


torch, nn = try_import_torch(error=True)

logger = logging.getLogger(__name__)


def _adjust_args(args):
    defaults = {
        "target_update_interval": 200,
        "agent_output_type": "q",
        "double_q": True,
        "hypernet_embed": 64,
        "adv_hypernet_layers": 2,
        "adv_hypernet_embed": 64,
        "ffn_hidden_dim": 64,
        "num_kernel": 5,
        "is_minus_one": True,
    }
    for key, value in defaults.items():
        if not hasattr(args, key):
            setattr(args, key, value)
    return args


class DuelMixWM2Loss(nn.Module):
    def __init__(
        self,
        model, target_model,
        mixer, target_mixer,
        world_model, ema_world_model,
        n_agents, n_actions,
        double_q=True, gamma=0.99,
        wm_loss_weight=0.5,
        reward_bonus_coeff=0.1,
        reward_bonus_scale=0.5,
        imagination_horizon=5,
        use_imagination_targets=False,
        imagination_loss_weight=0.1,
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
        self.double_q = double_q
        self.gamma = gamma
        self.wm_loss_weight = wm_loss_weight
        self.reward_bonus_coeff = reward_bonus_coeff
        self.reward_bonus_scale = reward_bonus_scale
        self.imagination_horizon = imagination_horizon
        self.use_imagination_targets = use_imagination_targets
        self.imagination_loss_weight = imagination_loss_weight

    def _compute_reward_bonus(self, state_decoded, state_real):
        recon_error = ((state_decoded - state_real) ** 2).mean(dim=-1, keepdim=True)
        return torch.exp(-recon_error / self.reward_bonus_scale)

    def forward(
        self,
        rewards, actions, terminated, mask,
        obs, next_obs, action_mask, next_action_mask,
        state=None, next_state=None,
    ):
        if state is None and next_state is None:
            state = obs.reshape(obs.shape[0], obs.shape[1], -1)
            next_state = next_obs.reshape(next_obs.shape[0], next_obs.shape[1], -1)
        elif (state is None) != (next_state is None):
            raise ValueError("Expected either neither or both of state/next_state.")

        if state.ndim == 4:
            state = state.reshape(state.shape[0], state.shape[1], -1)
        if next_state.ndim == 4:
            next_state = next_state.reshape(next_state.shape[0], next_state.shape[1], -1)

        B, T = obs.shape[0], obs.shape[1]
        wm_mask = mask[:, :, 0]

        wm_loss, features, wm_stats, posteriors = self.world_model.compute_loss(
            obs, actions, state, rewards, wm_mask, return_posteriors=True
        )

        with torch.no_grad():
            _, ema_features, _ = self.ema_world_model.compute_loss(
                obs, actions, state, rewards, wm_mask
            )
            feature_det = ema_features

            obs_aug = torch.cat([
                obs,
                feature_det.unsqueeze(2).expand(-1, -1, self.n_agents, -1),
            ], dim=-1)

            next_obs_flat = next_obs.reshape(B * T, self.n_agents, -1)
            next_feature = self.ema_world_model.encode_obs(next_obs_flat).reshape(B, T, -1)
            next_obs_aug = torch.cat([
                next_obs,
                next_feature.unsqueeze(2).expand(-1, -1, self.n_agents, -1),
            ], dim=-1)

        aug_state = torch.cat([state, feature_det], dim=-1)
        aug_next_state = torch.cat([next_state, next_feature], dim=-1)

        state_decoded = self.world_model.state_decoder(features.detach().reshape(B * T, -1)).reshape(B, T, -1)
        reward_bonus = self._compute_reward_bonus(state_decoded, state.detach())
        shaped_rewards = rewards + self.reward_bonus_coeff * reward_bonus.expand_as(rewards)

        all_v, all_a = _unroll_mac_duelmix(self.model, obs_aug)
        chosen_v = all_v.squeeze(3)
        chosen_a = torch.gather(all_a, dim=3, index=actions.unsqueeze(3)).squeeze(3)

        x_a = all_a.clone().detach()
        x_a[(action_mask == 0) & (mask == 1).unsqueeze(-1)] = -np.inf
        max_a_vals = x_a.max(dim=3)[0]

        target_v, target_a = _unroll_mac_duelmix(self.target_model, next_obs_aug)
        target_q = target_v + target_a
        ignore_tp1 = (next_action_mask == 0) & (mask == 1).unsqueeze(-1)
        target_q[ignore_tp1] = -np.inf

        if self.double_q:
            mac_v_tp1, mac_a_tp1 = _unroll_mac_duelmix(self.model, next_obs_aug)
            mac_q_tp1 = mac_v_tp1 + mac_a_tp1
            mac_q_tp1[ignore_tp1] = -np.inf
            cur_max_actions = mac_q_tp1.argmax(dim=3, keepdim=True)
            target_max_v = target_v.squeeze(3)
        else:
            cur_max_actions = target_q.argmax(dim=3, keepdim=True)
            target_max_v = target_v.squeeze(3)

        v_tot = self.mixer(chosen_v, states=aug_state, is_v=True)
        actions_onehot = F.one_hot(actions, num_classes=self.n_actions)
        a_tot = self.mixer(
            chosen_v,
            agent_as=chosen_a,
            states=aug_state,
            actions=actions_onehot,
            max_action_advs=max_a_vals,
            is_v=False,
        )
        chosen_q_tot = v_tot + a_tot

        target_q_tot = self.target_mixer(target_max_v, states=aug_next_state, is_v=True)
        targets = shaped_rewards.mean(dim=-1, keepdim=True) + self.gamma * (1 - terminated.mean(dim=-1, keepdim=True)) * target_q_tot

        if self.use_imagination_targets:
            H = self.imagination_horizon
            BT = B * T
            imag_state = [p.reshape(BT, -1).detach() for p in posteriors]
            act_flat = actions.reshape(BT, self.n_agents)

            imag_rewards_list = []
            for _ in range(H):
                action_embed = self.world_model.action_embed(act_flat)
                imag_state = self.world_model.transition.img_step(imag_state, action_embed)
                imag_feature = self.world_model.transition.get_feature(imag_state)
                imag_reward = self.world_model.reward_predictor(imag_feature)
                imag_rewards_list.append(imag_reward)

            imag_rewards_tensor = torch.stack(imag_rewards_list, dim=1)
            gammas = torch.pow(
                torch.tensor(self.gamma, dtype=torch.float, device=obs.device),
                torch.arange(H, device=obs.device, dtype=torch.float),
            )
            imag_return = (imag_rewards_tensor * gammas.unsqueeze(0)).sum(dim=1).reshape(B, T, 1)
            term_mean = terminated.mean(dim=-1, keepdim=True)
            imag_td_targets = imag_return + (self.gamma ** H) * (1 - term_mean) * target_q_tot.detach()
            targets = (
                (1.0 - self.imagination_loss_weight) * targets
                + self.imagination_loss_weight * imag_td_targets
            )

        td_error = chosen_q_tot - targets.detach()
        mask_2d = mask[:, :, :1]
        masked_td_error = td_error * mask_2d
        td_loss = (masked_td_error ** 2).sum() / mask_2d.sum().clamp(min=1)
        total_loss = td_loss + self.wm_loss_weight * wm_loss

        stats = {
            "td_loss": td_loss.item(),
            "reward_bonus_mean": reward_bonus.mean().item(),
            **wm_stats,
        }
        return total_loss, stats, mask_2d, masked_td_error, chosen_q_tot, targets


class DuelMixWM2TorchPolicy(Policy):
    def __init__(self, obs_space, action_space, config):
        _validate(obs_space, action_space)
        config = dict(DUELMIX_DEFAULT_CONFIG, **config)
        self.args = Namespace(**config)
        self.args = _adjust_args(self.args)
        self.framework = "torch"
        super().__init__(obs_space, action_space, config)
        self.n_agents = len(obs_space.original_space.spaces)
        config["model"]["n_agents"] = self.n_agents
        self.n_actions = action_space.spaces[0].n
        self.has_env_global_state = False
        self.has_action_mask = False
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        agent_obs_space = obs_space.original_space.spaces[0]
        if isinstance(agent_obs_space, Dict):
            space_keys = set(agent_obs_space.spaces.keys())
            if "obs" not in space_keys:
                raise ValueError("Dict obs space must have subspace labeled `obs`")
            self.obs_size = _get_size(agent_obs_space.spaces["obs"])
            if "action_mask" in space_keys:
                mask_shape = tuple(agent_obs_space.spaces["action_mask"].shape)
                if mask_shape != (self.n_actions,):
                    raise ValueError(f"Action mask shape must be {(self.n_actions,)}, got {mask_shape}")
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

        wm_config = config.get("world_model_v2", {})
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

        self.augmented_obs_size = self.obs_size + feature_dim
        augmented_agent_obs_space = torch.zeros(self.augmented_obs_size)
        from gym.spaces import Box
        augmented_agent_obs_space = Box(
            low=-np.inf * np.ones(self.augmented_obs_size, dtype=np.float32),
            high=np.inf * np.ones(self.augmented_obs_size, dtype=np.float32),
            dtype=np.float32,
        )

        self.model = ModelCatalog.get_model_v2(
            augmented_agent_obs_space, action_space.spaces[0], self.n_actions,
            config["model"], framework="torch", name="model", default_model=DualStreamRNNModel,
        ).to(self.device)
        self.target_model = ModelCatalog.get_model_v2(
            augmented_agent_obs_space, action_space.spaces[0], self.n_actions,
            config["model"], framework="torch", name="target_model", default_model=DualStreamRNNModel,
        ).to(self.device)

        self.exploration = self._create_exploration()

        augmented_state_dim = state_dim + feature_dim
        augmented_state_shape = (augmented_state_dim,)
        self.mixer = DuelMixMixer(
            self.args, self.n_agents, self.n_actions, augmented_state_shape,
            config["mixing_embed_dim"], self.args.ffn_hidden_dim, self.args.num_kernel,
        ).to(self.device)
        self.target_mixer = DuelMixMixer(
            self.args, self.n_agents, self.n_actions, augmented_state_shape,
            config["mixing_embed_dim"], self.args.ffn_hidden_dim, self.args.num_kernel,
        ).to(self.device)
        assert config["mixer"] == "duelmix_wm2"

        self.cur_epsilon = 1.0
        self.update_target()

        self.params = list(self.model.parameters()) + list(self.mixer.parameters()) + list(self.world_model.parameters())
        self.loss = DuelMixWM2Loss(
            self.model,
            self.target_model,
            self.mixer,
            self.target_mixer,
            self.world_model,
            self.ema_world_model,
            self.n_agents,
            self.n_actions,
            self.config["double_q"],
            self.config["gamma"],
            wm_loss_weight=wm_config.get("wm_loss_weight", 0.5),
            reward_bonus_coeff=wm_config.get("reward_bonus_coeff", 0.1),
            reward_bonus_scale=wm_config.get("reward_bonus_scale", 0.5),
            imagination_horizon=wm_config.get("imagination_horizon", 5),
            use_imagination_targets=wm_config.get("use_imagination_targets", False),
            imagination_loss_weight=wm_config.get("imagination_loss_weight", 0.1),
        )

        from torch.optim import RMSprop
        self.optimiser = RMSprop(
            params=self.params,
            lr=config["lr"],
            alpha=config["optim_alpha"],
            eps=config["optim_eps"],
        )

    @override(Policy)
    def compute_actions(
        self, obs_batch, state_batches=None, prev_action_batch=None,
        prev_reward_batch=None, info_batch=None, episodes=None,
        explore=None, timestep=None, **kwargs,
    ):
        explore = explore if explore is not None else self.config["explore"]
        obs_batch, action_mask, _ = self._unpack_observation(obs_batch)
        with torch.no_grad():
            obs_tensor = torch.as_tensor(obs_batch, dtype=torch.float, device=self.device)
            feature = self.ema_world_model.encode_obs(obs_tensor)
            augmented_obs = torch.cat([
                obs_tensor,
                feature.unsqueeze(1).expand(-1, self.n_agents, -1),
            ], dim=-1)
            v_vals, a_vals, hiddens = _mac_duelmix(
                self.model,
                augmented_obs,
                [torch.as_tensor(np.array(s), dtype=torch.float, device=self.device) for s in state_batches],
            )
            q_values = v_vals + a_vals
            avail = torch.as_tensor(action_mask, dtype=torch.float, device=self.device)
            masked_q = q_values.clone()
            masked_q[avail == 0.0] = -float("inf")
            masked_q_flat = torch.reshape(masked_q, [-1] + list(masked_q.shape)[2:])
            if timestep is None:
                timestep = int(1e9)
            actions, _ = self.exploration.get_exploration_action(
                action_distribution=TorchCategorical(masked_q_flat),
                timestep=timestep, explore=explore,
            )
            actions = torch.reshape(actions, list(masked_q.shape)[:-1]).cpu().numpy()
            hiddens = [s.cpu().numpy() for s in hiddens]
        return tuple(actions.transpose([1, 0])), hiddens, {}

    @override(Policy)
    def compute_log_likelihoods(self, actions, obs_batch, state_batches=None,
                                prev_action_batch=None, prev_reward_batch=None):
        obs_batch, _, _ = self._unpack_observation(obs_batch)
        return np.zeros(obs_batch.shape[0])

    @override(Policy)
    def learn_on_batch(self, samples):
        obs_batch, action_mask, env_global_state = self._unpack_observation(samples[SampleBatch.CUR_OBS])
        next_obs_batch, next_action_mask, next_env_global_state = self._unpack_observation(
            samples[SampleBatch.NEXT_OBS]
        )
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
            rew, action_mask, next_action_mask, act, dones, obs, next_obs, env_global_state, next_env_global_state = output_list
        else:
            rew, action_mask, next_action_mask, act, dones, obs, next_obs = output_list

        if len(seq_lens) == 0:
            return {}

        B, T = len(seq_lens), max(seq_lens)

        def to_batches(arr, dtype):
            new_shape = [B, T] + list(arr.shape[1:])
            return torch.as_tensor(np.reshape(arr, new_shape), dtype=dtype, device=self.device)

        rewards = to_batches(rew, torch.float)
        actions = to_batches(act, torch.long)
        obs = to_batches(obs, torch.float).reshape([B, T, self.n_agents, self.obs_size])
        action_mask = to_batches(action_mask, torch.float)
        next_obs = to_batches(next_obs, torch.float).reshape([B, T, self.n_agents, self.obs_size])
        next_action_mask = to_batches(next_action_mask, torch.float)
        if self.has_env_global_state:
            env_global_state = to_batches(env_global_state, torch.float)
            next_env_global_state = to_batches(next_env_global_state, torch.float)

        terminated = to_batches(dones, torch.float).unsqueeze(2).expand(B, T, self.n_agents)
        filled = np.reshape(np.tile(np.arange(T, dtype=np.float32), B), [B, T]) < np.expand_dims(seq_lens, 1)
        mask = torch.as_tensor(filled, dtype=torch.float, device=self.device).unsqueeze(2).expand(B, T, self.n_agents)

        total_loss, stats, mask_out, masked_td_error, chosen_q, targets = self.loss(
            rewards,
            actions,
            terminated,
            mask,
            obs,
            next_obs,
            action_mask,
            next_action_mask,
            env_global_state if self.has_env_global_state else None,
            next_env_global_state if self.has_env_global_state else None,
        )

        self.optimiser.zero_grad()
        total_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.params, self.config["grad_norm_clipping"])
        self.optimiser.step()

        mask_elems = mask_out.sum().item()
        learner_stats = {
            "loss": total_loss.item(),
            "grad_norm": grad_norm if isinstance(grad_norm, float) else grad_norm.item(),
            "td_error_abs": masked_td_error.abs().sum().item() / max(mask_elems, 1.0),
            "q_taken_mean": (chosen_q * mask_out).sum().item() / max(mask_elems, 1.0),
            "target_mean": (targets * mask_out).sum().item() / max(mask_elems, 1.0),
            **stats,
        }
        return {LEARNER_STATS_KEY: learner_stats}

    @override(Policy)
    def get_initial_state(self):
        return [s.expand([self.n_agents, -1]).cpu().numpy() for s in self.model.get_initial_state()]

    @override(Policy)
    def get_weights(self):
        return {
            "model": self._cpu_dict(self.model.state_dict()),
            "target_model": self._cpu_dict(self.target_model.state_dict()),
            "mixer": self._cpu_dict(self.mixer.state_dict()),
            "target_mixer": self._cpu_dict(self.target_mixer.state_dict()),
        }

    @override(Policy)
    def set_weights(self, weights):
        self.model.load_state_dict(self._device_dict(weights["model"]))
        self.target_model.load_state_dict(self._device_dict(weights["target_model"]))
        self.mixer.load_state_dict(self._device_dict(weights["mixer"]))
        self.target_mixer.load_state_dict(self._device_dict(weights["target_mixer"]))

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
            self.observation_space.original_space, tensorlib=np,
        )
        if isinstance(unpacked[0], dict):
            assert "obs" in unpacked[0]
            unpacked_obs = [np.concatenate(tree.flatten(u["obs"]), 1) for u in unpacked]
        else:
            unpacked_obs = unpacked

        obs = np.concatenate(unpacked_obs, axis=1).reshape([len(obs_batch), self.n_agents, self.obs_size])
        if self.has_action_mask:
            action_mask = np.concatenate([o["action_mask"] for o in unpacked], axis=1).reshape(
                [len(obs_batch), self.n_agents, self.n_actions]
            )
        else:
            action_mask = np.ones([len(obs_batch), self.n_agents, self.n_actions], dtype=np.float32)
        if self.has_env_global_state:
            state = np.concatenate(tree.flatten(unpacked[0][ENV_STATE]), 1)
        else:
            state = None
        return obs, action_mask, state
