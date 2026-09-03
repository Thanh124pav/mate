"""QMIX + RSSM World Model (WM2) Policy."""

import copy
import logging
from gym.spaces import Box, Dict, Discrete, Tuple

import numpy as np
import tree

import ray
from ray.rllib.agents.qmix.mixers import QMixer, VDNMixer
from ray.rllib.agents.qmix.model import RNNModel, _get_size
from ray.rllib.agents.qplex_wm2.world_model_v2 import LatentWorldModel
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


def _ema_update(ema_model, model, decay=0.995):
    with torch.no_grad():
        for ema_p, p in zip(ema_model.parameters(), model.parameters()):
            ema_p.data.mul_(decay).add_(p.data, alpha=1.0 - decay)


class QMixWM2Loss(nn.Module):
    def __init__(
        self,
        model,
        target_model,
        mixer,
        target_mixer,
        world_model,
        ema_world_model,
        n_agents,
        n_actions,
        double_q=True,
        gamma=0.99,
        wm_loss_weight=0.5,
        reward_bonus_coeff=0.1,
        reward_bonus_scale=0.5,
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

    def _compute_reward_bonus(self, state_decoded, state_real):
        recon_error = ((state_decoded - state_real) ** 2).mean(dim=-1, keepdim=True)
        return torch.exp(-recon_error / self.reward_bonus_scale)

    def forward(
        self,
        rewards,
        actions,
        terminated,
        mask,
        obs,
        next_obs,
        action_mask,
        next_action_mask,
        state=None,
        next_state=None,
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
        wm_loss, features, wm_stats = self.world_model.compute_loss(
            obs, actions, state, rewards, wm_mask
        )

        with torch.no_grad():
            _, ema_features, _ = self.ema_world_model.compute_loss(
                obs, actions, state, rewards, wm_mask
            )
            obs_aug = torch.cat(
                [obs, ema_features.unsqueeze(2).expand(-1, -1, self.n_agents, -1)], dim=-1
            )
            next_obs_flat = next_obs.reshape(B * T, self.n_agents, -1)
            next_features = self.ema_world_model.encode_obs(next_obs_flat).reshape(B, T, -1)
            next_obs_aug = torch.cat(
                [next_obs, next_features.unsqueeze(2).expand(-1, -1, self.n_agents, -1)],
                dim=-1,
            )
            state_aug = torch.cat([state, ema_features], dim=-1)
            next_state_aug = torch.cat([next_state, next_features], dim=-1)

        state_decoded = self.world_model.state_decoder(
            features.detach().reshape(B * T, -1)
        ).reshape(B, T, -1)
        reward_bonus = self._compute_reward_bonus(state_decoded, state.detach())
        shaped_rewards = rewards + self.reward_bonus_coeff * reward_bonus.expand_as(rewards)

        mac_out = _unroll_mac(self.model, obs_aug)
        chosen_action_qvals = torch.gather(
            mac_out, dim=3, index=actions.unsqueeze(3)
        ).squeeze(3)

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

        if self.mixer is not None:
            chosen_action_qvals = self.mixer(chosen_action_qvals, state_aug)
            target_max_qvals = self.target_mixer(target_max_qvals, next_state_aug)

        targets = shaped_rewards + self.gamma * (1 - terminated) * target_max_qvals
        td_error = chosen_action_qvals - targets.detach()
        mask = mask.expand_as(td_error)
        masked_td_error = td_error * mask
        td_loss = (masked_td_error ** 2).sum() / mask.sum().clamp(min=1)
        total_loss = td_loss + self.wm_loss_weight * wm_loss

        stats = {
            "td_loss": td_loss.item(),
            "reward_bonus_mean": reward_bonus.mean().item(),
            **wm_stats,
        }
        return total_loss, stats, mask, masked_td_error, chosen_action_qvals, targets


class QMixWM2TorchPolicy(Policy):
    """QMIX with RSSM world-model latent features."""

    def __init__(self, obs_space, action_space, config):
        _validate(obs_space, action_space)
        config = dict(ray.rllib.agents.qmix_wm2.qmix.DEFAULT_CONFIG, **config)
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
        else:
            self.obs_size = _get_size(agent_obs_space)
            self.env_global_state_shape = (self.obs_size, self.n_agents)

        wm_config = config.get("world_model_v2", {})
        state_dim = int(np.prod(self.env_global_state_shape))
        self.world_model = LatentWorldModel(
            obs_size=self.obs_size,
            state_dim=state_dim,
            n_agents=self.n_agents,
            n_actions=self.n_actions,
            stoch_dim=wm_config.get("stoch_dim", 32),
            deter_dim=wm_config.get("deter_dim", 128),
            hidden_dim=wm_config.get("hidden_dim", 128),
            action_embed_dim=wm_config.get("action_embed_dim", 16),
            embed_dim=wm_config.get("embed_dim", 128),
            imagination_horizon=wm_config.get("imagination_horizon", 5),
            kl_coeff=wm_config.get("kl_coeff", 1.0),
            free_nats=wm_config.get("free_nats", 1.0),
        ).to(self.device)
        self.ema_world_model = copy.deepcopy(self.world_model)
        for p in self.ema_world_model.parameters():
            p.requires_grad = False
        self.ema_decay = wm_config.get("ema_decay", 0.995)
        feature_dim = self.world_model.feature_dim

        augmented_agent_obs_space = Box(
            low=-np.inf * np.ones(self.obs_size + feature_dim, dtype=np.float32),
            high=np.inf * np.ones(self.obs_size + feature_dim, dtype=np.float32),
            dtype=np.float32,
        )
        augmented_state_shape = (state_dim + feature_dim,)

        self.model = ModelCatalog.get_model_v2(
            augmented_agent_obs_space,
            action_space.spaces[0],
            self.n_actions,
            config["model"],
            framework="torch",
            name="model",
            default_model=RNNModel,
        ).to(self.device)
        self.target_model = ModelCatalog.get_model_v2(
            augmented_agent_obs_space,
            action_space.spaces[0],
            self.n_actions,
            config["model"],
            framework="torch",
            name="target_model",
            default_model=RNNModel,
        ).to(self.device)
        self.exploration = self._create_exploration()

        if config["mixer"] is None:
            self.mixer = None
            self.target_mixer = None
        elif config["mixer"] == "qmix_wm2":
            self.mixer = QMixer(self.n_agents, augmented_state_shape, config["mixing_embed_dim"]).to(self.device)
            self.target_mixer = QMixer(self.n_agents, augmented_state_shape, config["mixing_embed_dim"]).to(self.device)
        elif config["mixer"] == "vdn_wm2":
            self.mixer = VDNMixer().to(self.device)
            self.target_mixer = VDNMixer().to(self.device)
        else:
            raise ValueError(f"Unknown mixer type {config['mixer']}")

        self.cur_epsilon = 1.0
        self.update_target()

        self.params = list(self.model.parameters()) + list(self.world_model.parameters())
        if self.mixer:
            self.params += list(self.mixer.parameters())
        self.loss = QMixWM2Loss(
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
        self,
        obs_batch,
        state_batches=None,
        prev_action_batch=None,
        prev_reward_batch=None,
        info_batch=None,
        episodes=None,
        explore=None,
        timestep=None,
        **kwargs,
    ):
        explore = explore if explore is not None else self.config["explore"]
        obs_batch, action_mask, _ = self._unpack_observation(obs_batch)
        with torch.no_grad():
            obs_tensor = torch.as_tensor(obs_batch, dtype=torch.float, device=self.device)
            feature = self.ema_world_model.encode_obs(obs_tensor)
            obs_tensor = torch.cat(
                [obs_tensor, feature.unsqueeze(1).expand(-1, self.n_agents, -1)], dim=-1
            )
            q_values, hiddens = _mac(
                self.model,
                obs_tensor,
                [torch.as_tensor(np.array(s), dtype=torch.float, device=self.device) for s in state_batches],
            )
            avail = torch.as_tensor(action_mask, dtype=torch.float, device=self.device)
            masked_q_values = q_values.clone()
            masked_q_values[avail == 0.0] = -float("inf")
            masked_q_values_folded = torch.reshape(masked_q_values, [-1] + list(masked_q_values.shape)[2:])
            if timestep is None:
                timestep = int(1e9)
            actions, _ = self.exploration.get_exploration_action(
                action_distribution=TorchCategorical(masked_q_values_folded),
                timestep=timestep,
                explore=explore,
            )
            actions = torch.reshape(actions, list(masked_q_values.shape)[:-1]).cpu().numpy()
            hiddens = [s.cpu().numpy() for s in hiddens]
        return tuple(actions.transpose([1, 0])), hiddens, {}

    @override(Policy)
    def compute_log_likelihoods(self, actions, obs_batch, state_batches=None,
                                prev_action_batch=None, prev_reward_batch=None):
        return np.zeros(len(obs_batch))

    @override(Policy)
    def learn_on_batch(self, samples):
        obs_batch, action_mask, env_global_state = self._unpack_observation(samples[SampleBatch.CUR_OBS])
        next_obs_batch, next_action_mask, next_env_global_state = self._unpack_observation(samples[SampleBatch.NEXT_OBS])
        group_rewards = self._get_group_rewards(samples[SampleBatch.INFOS])

        input_list = [
            group_rewards,
            action_mask,
            next_action_mask,
            samples[SampleBatch.ACTIONS],
            samples[SampleBatch.DONES],
            obs_batch,
            next_obs_batch,
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

        B, T = len(seq_lens), max(seq_lens)

        def to_batches(arr, dtype):
            return torch.as_tensor(np.reshape(arr, [B, T] + list(arr.shape[1:])), dtype=dtype, device=self.device)

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

        loss_out, wm_stats, mask, masked_td_error, chosen_action_qvals, targets = self.loss(
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
        loss_out.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.params, self.config["grad_norm_clipping"])
        self.optimiser.step()
        _ema_update(self.ema_world_model, self.world_model, self.ema_decay)

        mask_elems = mask.sum().item()
        stats = {
            "loss": loss_out.item(),
            "grad_norm": grad_norm if isinstance(grad_norm, float) else grad_norm.item(),
            "td_error_abs": masked_td_error.abs().sum().item() / mask_elems,
            "q_taken_mean": (chosen_action_qvals * mask).sum().item() / mask_elems,
            "target_mean": (targets * mask).sum().item() / mask_elems,
        }
        stats.update(wm_stats)
        return {LEARNER_STATS_KEY: stats}

    @override(Policy)
    def get_initial_state(self):
        return [s.expand([self.n_agents, -1]).cpu().numpy() for s in self.model.get_initial_state()]

    @override(Policy)
    def get_weights(self):
        return {
            "model": self._cpu_dict(self.model.state_dict()),
            "target_model": self._cpu_dict(self.target_model.state_dict()),
            "mixer": self._cpu_dict(self.mixer.state_dict()) if self.mixer else None,
            "target_mixer": self._cpu_dict(self.target_mixer.state_dict()) if self.mixer else None,
            "world_model": self._cpu_dict(self.world_model.state_dict()),
            "ema_world_model": self._cpu_dict(self.ema_world_model.state_dict()),
        }

    @override(Policy)
    def set_weights(self, weights):
        self.model.load_state_dict(self._device_dict(weights["model"]))
        self.target_model.load_state_dict(self._device_dict(weights["target_model"]))
        if weights["mixer"] is not None:
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
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())
        logger.debug("Updated target networks")

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
            self.observation_space.original_space,
            tensorlib=np,
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


def _validate(obs_space, action_space):
    if not hasattr(obs_space, "original_space") or not isinstance(obs_space.original_space, Tuple):
        raise ValueError(f"Obs space must be a Tuple, got {obs_space}.")
    if not isinstance(action_space, Tuple):
        raise ValueError(f"Action space must be a Tuple, got {action_space}.")
    if not isinstance(action_space.spaces[0], Discrete):
        raise ValueError(f"QMIX_WM2 requires discrete action space, got {action_space.spaces[0]}")


def _mac(model, obs, h):
    B, n_agents = obs.size(0), obs.size(1)
    if not isinstance(obs, dict):
        obs = {"obs": obs}
    obs_flat = {k: v.reshape([B * n_agents] + list(v.shape)[2:]) for k, v in obs.items()}
    h_flat = [s.reshape([B * n_agents, -1]) for s in h]
    q_flat, h_flat = model(obs_flat, h_flat, None)
    return q_flat.reshape([B, n_agents, -1]), [s.reshape([B, n_agents, -1]) for s in h_flat]


def _unroll_mac(model, obs_tensor):
    B, T, n_agents = obs_tensor.size(0), obs_tensor.size(1), obs_tensor.size(2)
    mac_out = []
    h = [s.expand([B, n_agents, -1]) for s in model.get_initial_state()]
    for t in range(T):
        q, h = _mac(model, obs_tensor[:, t], h)
        mac_out.append(q)
    return torch.stack(mac_out, dim=1)
