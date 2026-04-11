"""SPECTra policy: QMIX-style DQN with SPECTra model and ST-HyperNet mixer."""

from gym.spaces import Tuple, Discrete, Dict
import logging
import numpy as np
import tree
import torch.nn.functional as F
from argparse import Namespace

import ray
from .mixers import SPECTraMixer
from .model import SPECTraRNNModel, _get_size
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


class SPECTraLoss(nn.Module):
    """Standard DQN loss with SPECTra mixer (simpler than QPLEX — no V/A split)."""

    def __init__(self, model, target_model, mixer, target_mixer,
                 n_agents, n_actions, double_q=True, gamma=0.99):
        nn.Module.__init__(self)
        self.model = model
        self.target_model = target_model
        self.mixer = mixer
        self.target_mixer = target_mixer
        self.n_agents = n_agents
        self.n_actions = n_actions
        self.double_q = double_q
        self.gamma = gamma

    def forward(self, rewards, actions, terminated, mask,
                obs, next_obs, action_mask, next_action_mask,
                state=None, next_state=None):
        if state is None and next_state is None:
            state = obs
            next_state = next_obs

        # Current Q-values
        mac_out = _unroll_mac(self.model, obs)
        chosen_qs = torch.gather(mac_out, dim=3, index=actions.unsqueeze(3)).squeeze(3)

        # Target Q-values
        target_mac_out = _unroll_mac(self.target_model, next_obs)
        ignore_tp1 = (next_action_mask == 0) & (mask == 1).unsqueeze(-1)
        target_mac_out[ignore_tp1] = -np.inf

        if self.double_q:
            mac_out_tp1 = _unroll_mac(self.model, next_obs)
            mac_out_tp1[ignore_tp1] = -np.inf
            cur_max_actions = mac_out_tp1.argmax(dim=3, keepdim=True)
            target_max_qs = torch.gather(target_mac_out, 3, cur_max_actions).squeeze(3)
        else:
            target_max_qs = target_mac_out.max(dim=3)[0]

        # Mix
        chosen_q_tot = self.mixer(chosen_qs, state)
        target_q_tot = self.target_mixer(target_max_qs, next_state)

        # TD targets
        targets = rewards + self.gamma * (1 - terminated) * target_q_tot
        td_error = chosen_q_tot - targets.detach()
        mask = mask.expand_as(td_error)
        masked_td_error = td_error * mask
        loss = (masked_td_error ** 2).sum() / mask.sum()

        return loss, mask, masked_td_error, chosen_q_tot, targets


class SPECTraTorchPolicy(Policy):
    """SPECTra policy with SAQA agent model and ST-HyperNet mixer."""

    def __init__(self, obs_space, action_space, config):
        _validate(obs_space, action_space)
        config = dict(ray.rllib.agents.spectra.spectra.DEFAULT_CONFIG, **config)

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
                raise ValueError("Dict obs space must have 'obs' subspace.")
            self.obs_size = _get_size(agent_obs_space.spaces["obs"])
            if "action_mask" in space_keys:
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

        self.model = ModelCatalog.get_model_v2(
            agent_obs_space, action_space.spaces[0], self.n_actions,
            config["model"], framework="torch", name="model",
            default_model=SPECTraRNNModel,
        ).to(self.device)

        self.target_model = ModelCatalog.get_model_v2(
            agent_obs_space, action_space.spaces[0], self.n_actions,
            config["model"], framework="torch", name="target_model",
            default_model=SPECTraRNNModel,
        ).to(self.device)

        self.exploration = self._create_exploration()

        # SPECTra mixer
        mixing_dim = config.get("mixing_embed_dim", 32)
        embed_dim = config.get("st_hypernet_embed", 64)
        n_heads = config.get("n_attention_heads", 4)
        self.mixer = SPECTraMixer(
            self.n_agents, self.env_global_state_shape,
            mixing_dim, embed_dim, n_heads,
        ).to(self.device)
        self.target_mixer = SPECTraMixer(
            self.n_agents, self.env_global_state_shape,
            mixing_dim, embed_dim, n_heads,
        ).to(self.device)

        self.cur_epsilon = 1.0
        self.update_target()

        self.params = list(self.model.parameters()) + list(self.mixer.parameters())
        self.loss = SPECTraLoss(
            self.model, self.target_model, self.mixer, self.target_mixer,
            self.n_agents, self.n_actions, config["double_q"], config["gamma"],
        )
        from torch.optim import RMSprop
        self.optimiser = RMSprop(
            params=self.params, lr=config["lr"],
            alpha=config.get("optim_alpha", 0.99),
            eps=config.get("optim_eps", 1e-5),
        )

    @override(Policy)
    def compute_actions(self, obs_batch, state_batches=None, prev_action_batch=None,
                        prev_reward_batch=None, info_batch=None, episodes=None,
                        explore=None, timestep=None, **kwargs):
        explore = explore if explore is not None else self.config["explore"]
        obs_batch, action_mask, _ = self._unpack_observation(obs_batch)

        with torch.no_grad():
            q_values, hiddens = _mac(
                self.model,
                torch.as_tensor(obs_batch, dtype=torch.float, device=self.device),
                [torch.as_tensor(np.array(s), dtype=torch.float, device=self.device)
                 for s in state_batches],
            )
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
        return np.zeros(len(obs_batch))

    @override(Policy)
    def learn_on_batch(self, samples):
        obs_batch, action_mask, env_global_state = self._unpack_observation(
            samples[SampleBatch.CUR_OBS]
        )
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
            rew, action_mask, next_action_mask, act, dones, obs, next_obs, \
                env_global_state, next_env_global_state = output_list
        else:
            rew, action_mask, next_action_mask, act, dones, obs, next_obs = output_list

        B, T = len(seq_lens), max(seq_lens)

        def to_batches(arr, dtype):
            return torch.as_tensor(
                np.reshape(arr, [B, T] + list(arr.shape[1:])),
                dtype=dtype, device=self.device,
            )

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

        loss_out, mask, masked_td_error, chosen_q, targets = self.loss(
            rewards, actions, terminated, mask, obs, next_obs,
            action_mask, next_action_mask,
            env_global_state if self.has_env_global_state else None,
            next_env_global_state if self.has_env_global_state else None,
        )

        self.optimiser.zero_grad()
        loss_out.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.params, self.config["grad_norm_clipping"])
        self.optimiser.step()

        mask_elems = mask.sum().item()
        return {LEARNER_STATS_KEY: {
            "loss": loss_out.item(),
            "grad_norm": grad_norm if isinstance(grad_norm, float) else grad_norm.item(),
            "td_error_abs": masked_td_error.abs().sum().item() / mask_elems,
            "q_taken_mean": (chosen_q * mask).sum().item() / mask_elems,
            "target_mean": (targets * mask).sum().item() / mask_elems,
        }}

    @override(Policy)
    def get_initial_state(self):
        return [s.expand([self.n_agents, -1]).cpu().numpy()
                for s in self.model.get_initial_state()]

    @override(Policy)
    def get_weights(self):
        return {
            "model": {k: v.cpu().detach().numpy() for k, v in self.model.state_dict().items()},
            "target_model": {k: v.cpu().detach().numpy() for k, v in self.target_model.state_dict().items()},
            "mixer": {k: v.cpu().detach().numpy() for k, v in self.mixer.state_dict().items()},
            "target_mixer": {k: v.cpu().detach().numpy() for k, v in self.target_mixer.state_dict().items()},
        }

    @override(Policy)
    def set_weights(self, weights):
        dev = lambda sd: {k: torch.as_tensor(v, device=self.device) for k, v in sd.items()}
        self.model.load_state_dict(dev(weights["model"]))
        self.target_model.load_state_dict(dev(weights["target_model"]))
        self.mixer.load_state_dict(dev(weights["mixer"]))
        self.target_mixer.load_state_dict(dev(weights["target_mixer"]))

    @override(Policy)
    def get_state(self):
        state = self.get_weights()
        state["cur_epsilon"] = self.cur_epsilon
        return state

    @override(Policy)
    def set_state(self, state):
        self.set_weights(state)
        self.cur_epsilon = state["cur_epsilon"]

    def update_target(self):
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_mixer.load_state_dict(self.mixer.state_dict())

    def set_epsilon(self, epsilon):
        self.cur_epsilon = epsilon

    def _get_group_rewards(self, info_batch):
        return np.array([info.get(GROUP_REWARDS, [0.0] * self.n_agents) for info in info_batch])

    def _unpack_observation(self, obs_batch):
        unpacked = _unpack_obs(
            np.array(obs_batch, dtype=np.float32),
            self.observation_space.original_space, tensorlib=np,
        )
        if isinstance(unpacked[0], dict):
            unpacked_obs = [np.concatenate(tree.flatten(u["obs"]), 1) for u in unpacked]
        else:
            unpacked_obs = unpacked
        obs = np.concatenate(unpacked_obs, axis=1).reshape(
            [len(obs_batch), self.n_agents, self.obs_size]
        )
        if self.has_action_mask:
            action_mask = np.concatenate(
                [o["action_mask"] for o in unpacked], axis=1
            ).reshape([len(obs_batch), self.n_agents, self.n_actions])
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
        raise ValueError(f"SPECTra requires discrete action space, got {action_space.spaces[0]}")


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
