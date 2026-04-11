"""DuelMIX policy: dual-stream Q-learning with separate V/A mixing."""

from gym.spaces import Tuple, Discrete, Dict
import logging
import numpy as np
import tree
import torch.nn.functional as F
from argparse import Namespace

import ray
from .mixers import DuelMixMixer
from .model import DualStreamRNNModel, _get_size
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


class DuelMixLoss(nn.Module):
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
        elif (state is None) != (next_state is None):
            raise ValueError("Expected both or neither state/next_state.")

        # --- Current model outputs ---
        all_v, all_a = _unroll_mac_duelmix(self.model, obs)
        # all_v: [B, T, n_agents, 1], all_a: [B, T, n_agents, n_actions]

        # Q = V + A for action selection masking
        all_q = all_v + all_a  # [B, T, n_agents, n_actions]

        # Chosen action values
        chosen_a = torch.gather(all_a, dim=3, index=actions.unsqueeze(3)).squeeze(3)
        chosen_v = all_v.squeeze(3)  # [B, T, n_agents]

        # Max advantage values (for centering)
        ignore_action = (action_mask == 0) & (mask == 1).unsqueeze(-1)
        x_a = all_a.clone().detach()
        x_a[ignore_action] = -np.inf
        max_a_vals = x_a.max(dim=3)[0]  # [B, T, n_agents]

        # --- Target model outputs ---
        target_v, target_a = _unroll_mac_duelmix(self.target_model, next_obs)
        target_q = target_v + target_a

        # Mask unavailable actions for t+1
        ignore_tp1 = (next_action_mask == 0) & (mask == 1).unsqueeze(-1)
        target_q[ignore_tp1] = -np.inf

        if self.double_q:
            # Use current model for action selection
            mac_v_tp1, mac_a_tp1 = _unroll_mac_duelmix(self.model, next_obs)
            mac_q_tp1 = mac_v_tp1 + mac_a_tp1
            mac_q_tp1[ignore_tp1] = -np.inf
            cur_max_actions = mac_q_tp1.argmax(dim=3, keepdim=True)
            target_max_v = target_v.squeeze(3)  # [B, T, n_agents]
        else:
            cur_max_actions = target_q.argmax(dim=3, keepdim=True)
            target_max_v = target_v.squeeze(3)

        assert target_max_v.min().item() != -np.inf

        # --- Mix ---
        # Current: Q_tot = V_tot + A_tot
        v_tot = self.mixer(chosen_v, states=state, is_v=True)
        actions_onehot = F.one_hot(actions, num_classes=self.n_actions)
        a_tot = self.mixer(
            chosen_v, agent_as=chosen_a, states=state,
            actions=actions_onehot, max_action_advs=max_a_vals, is_v=False,
        )
        chosen_q_tot = v_tot + a_tot

        # Target: at argmax action, advantage = 0, so Q_tot = V_tot
        target_v_tot = self.target_mixer(target_max_v, states=next_state, is_v=True)
        target_q_tot = target_v_tot  # A_tot = 0 at argmax

        # --- TD loss ---
        targets = rewards + self.gamma * (1 - terminated) * target_q_tot
        td_error = chosen_q_tot - targets.detach()
        mask = mask.expand_as(td_error)
        masked_td_error = td_error * mask
        loss = (masked_td_error ** 2).sum() / mask.sum()

        return loss, mask, masked_td_error, chosen_q_tot, targets


def _adjust_args(args):
    defaults = {
        'target_update_interval': 200,
        'agent_output_type': 'q',
        'double_q': True,
        'hypernet_embed': 64,
        'adv_hypernet_layers': 2,
        'adv_hypernet_embed': 64,
        'ffn_hidden_dim': 64,
        'num_kernel': 5,
        'is_minus_one': True,
    }
    for k, v in defaults.items():
        if not hasattr(args, k):
            setattr(args, k, v)
    return args


class DuelMixTorchPolicy(Policy):
    """DuelMIX policy with dual-stream model and separate V/A mixing."""

    def __init__(self, obs_space, action_space, config):
        _validate(obs_space, action_space)
        config = dict(ray.rllib.agents.duelmix.duelmix.DEFAULT_CONFIG, **config)

        self.args = Namespace(**config)
        self.args = _adjust_args(self.args)
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

        self.model = ModelCatalog.get_model_v2(
            agent_obs_space, action_space.spaces[0], self.n_actions,
            config["model"], framework="torch", name="model",
            default_model=DualStreamRNNModel,
        ).to(self.device)

        self.target_model = ModelCatalog.get_model_v2(
            agent_obs_space, action_space.spaces[0], self.n_actions,
            config["model"], framework="torch", name="target_model",
            default_model=DualStreamRNNModel,
        ).to(self.device)

        self.exploration = self._create_exploration()

        # Mixer
        self.mixer = DuelMixMixer(
            self.args, self.n_agents, self.n_actions,
            self.env_global_state_shape, config['mixing_embed_dim'],
            self.args.ffn_hidden_dim, self.args.num_kernel,
        ).to(self.device)
        self.target_mixer = DuelMixMixer(
            self.args, self.n_agents, self.n_actions,
            self.env_global_state_shape, config['mixing_embed_dim'],
            self.args.ffn_hidden_dim, self.args.num_kernel,
        ).to(self.device)

        self.cur_epsilon = 1.0
        self.update_target()

        self.params = list(self.model.parameters()) + list(self.mixer.parameters())
        self.loss = DuelMixLoss(
            self.model, self.target_model, self.mixer, self.target_mixer,
            self.n_agents, self.n_actions, self.config["double_q"], self.config["gamma"],
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
            v_vals, a_vals, hiddens = _mac_duelmix(
                self.model,
                torch.as_tensor(obs_batch, dtype=torch.float, device=self.device),
                [torch.as_tensor(np.array(s), dtype=torch.float, device=self.device)
                 for s in state_batches],
            )
            # Q = V + A for action selection
            q_values = v_vals + a_vals  # [B, n_agents, n_actions]
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
        stats = {
            "loss": loss_out.item(),
            "grad_norm": grad_norm if isinstance(grad_norm, float) else grad_norm.item(),
            "td_error_abs": masked_td_error.abs().sum().item() / mask_elems,
            "q_taken_mean": (chosen_q * mask).sum().item() / mask_elems,
            "target_mean": (targets * mask).sum().item() / mask_elems,
        }
        return {LEARNER_STATS_KEY: stats}

    @override(Policy)
    def get_initial_state(self):
        return [
            s.expand([self.n_agents, -1]).cpu().numpy()
            for s in self.model.get_initial_state()
        ]

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
        return np.array(
            [info.get(GROUP_REWARDS, [0.0] * self.n_agents) for info in info_batch]
        )

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

        obs = np.concatenate(unpacked_obs, axis=1).reshape(
            [len(obs_batch), self.n_agents, self.obs_size]
        )
        if self.has_action_mask:
            action_mask = np.concatenate(
                [o["action_mask"] for o in unpacked], axis=1
            ).reshape([len(obs_batch), self.n_agents, self.n_actions])
        else:
            action_mask = np.ones(
                [len(obs_batch), self.n_agents, self.n_actions], dtype=np.float32
            )
        if self.has_env_global_state:
            state = np.concatenate(tree.flatten(unpacked[0][ENV_STATE]), 1)
        else:
            state = None
        return obs, action_mask, state


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _validate(obs_space, action_space):
    if not hasattr(obs_space, "original_space") or not isinstance(obs_space.original_space, Tuple):
        raise ValueError(
            f"Obs space must be a Tuple, got {obs_space}. "
            "Use MultiAgentEnv.with_agent_groups() to group agents for DuelMIX."
        )
    if not isinstance(action_space, Tuple):
        raise ValueError(f"Action space must be a Tuple, got {action_space}.")
    if not isinstance(action_space.spaces[0], Discrete):
        raise ValueError(f"DuelMIX requires discrete action space, got {action_space.spaces[0]}")
    if len({str(x) for x in obs_space.original_space.spaces}) > 1:
        raise ValueError(f"Grouped agent observations must be homogeneous, got {obs_space.original_space.spaces}")
    if len({str(x) for x in action_space.spaces}) > 1:
        raise ValueError(f"Grouped agent actions must be homogeneous, got {action_space.spaces}")


def _mac_duelmix(model, obs, h):
    """Forward pass returning separate V and A values.

    Returns:
        v_vals: [B, n_agents, 1]
        a_vals: [B, n_agents, n_actions]
        h: list of [B, n_agents, h_size]
    """
    B, n_agents = obs.size(0), obs.size(1)
    if not isinstance(obs, dict):
        obs = {"obs": obs}
    obs_flat = {k: _drop_agent_dim(v) for k, v in obs.items()}
    h_flat = [s.reshape([B * n_agents, -1]) for s in h]
    out, h_flat = model(obs_flat, h_flat, None)
    v = out["v"].reshape([B, n_agents, 1])
    a = out["a"].reshape([B, n_agents, -1])
    return v, a, [s.reshape([B, n_agents, -1]) for s in h_flat]


def _unroll_mac_duelmix(model, obs_tensor):
    """Unroll over time, returning V and A trajectories."""
    B, T, n_agents = obs_tensor.size(0), obs_tensor.size(1), obs_tensor.size(2)
    all_v, all_a = [], []
    h = [s.expand([B, n_agents, -1]) for s in model.get_initial_state()]
    for t in range(T):
        v, a, h = _mac_duelmix(model, obs_tensor[:, t], h)
        all_v.append(v)
        all_a.append(a)
    return torch.stack(all_v, dim=1), torch.stack(all_a, dim=1)


def _drop_agent_dim(T):
    shape = list(T.shape)
    B, n_agents = shape[0], shape[1]
    return T.reshape([B * n_agents] + shape[2:])
