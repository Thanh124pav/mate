"""QPLEX + Shared Encoder (SE) Policy.

Architecture identical to QPLEX_V2 — same obs_size, same state_dim, same mixer.
Only difference: a PredictionHead branches off the RNN hidden state to predict
next target positions. MSE loss backprops through the shared GRU+fc1, improving
the representation used by Q-values.

                    obs (126 dims)
                         │
                      fc1 (shared) ──────────────────┐
                         │                            │
                        GRU                           │
                         │                            │
                   hidden_state                       │
                    │          │                       │
                  fc2        PredictionHead            │
                    │          │                       │
               Q-values   predicted_pos               │
                    │          │                       │
                 TD loss    MSE loss ──► gradient ──► fc1, GRU
                    │                                 │
                    └──────── gradient ───────────────┘

Guarantee: worst case = QPLEX_V2 (if MSE gradient is useless, model ignores it).
Best case: better representation → better Q-values.
"""

from gym.spaces import Tuple, Discrete, Dict
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

# MATE constants for target position extraction from global state
PRESERVED_DIM = 13
CAMERA_STATE_DIM_PRIVATE = 9
TARGET_STATE_DIM_PRIVATE = 14


class PredictionHead(nn.Module):
    """Small MLP that predicts next target positions from RNN hidden state.

    Each agent independently predicts absolute target positions.
    All agents share this head and predict the same targets.
    """

    def __init__(self, hidden_dim, n_targets, pred_hidden=128):
        super().__init__()
        self.n_targets = n_targets
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, pred_hidden),
            nn.ReLU(),
            nn.Linear(pred_hidden, n_targets * 2),
        )

    def forward(self, hidden):
        """
        Args:
            hidden: [B, hidden_dim]
        Returns:
            predictions: [B, n_targets, 2]
        """
        return self.net(hidden).view(-1, self.n_targets, 2)


def _extract_target_positions(state, n_agents, n_targets):
    """Extract absolute target (x, y) from env global state.

    Args:
        state: [B, T, state_dim]
        n_agents: number of cameras
        n_targets: number of targets

    Returns:
        positions: [B, T, n_targets, 2]
    """
    target_start = PRESERVED_DIM + n_agents * CAMERA_STATE_DIM_PRIVATE
    positions = []
    for i in range(n_targets):
        start = target_start + i * TARGET_STATE_DIM_PRIVATE
        positions.append(state[:, :, start:start + 2])
    return torch.stack(positions, dim=2)


def _unroll_mac_with_hidden(model, obs_tensor):
    """Like _unroll_mac but also returns hidden states at each timestep.

    Returns:
        mac_out: [B, T, n_agents, n_actions]
        hiddens: [B, T, n_agents, h_size]
    """
    B = obs_tensor.size(0)
    T = obs_tensor.size(1)
    n_agents = obs_tensor.size(2)

    mac_out = []
    hiddens_out = []
    h = [s.expand([B, n_agents, -1]) for s in model.get_initial_state()]
    for t in range(T):
        q, h = _mac(model, obs_tensor[:, t], h)
        mac_out.append(q)
        hiddens_out.append(h[0])  # GRU hidden state [B, n_agents, h_size]

    mac_out = torch.stack(mac_out, dim=1)
    hiddens_out = torch.stack(hiddens_out, dim=1)
    return mac_out, hiddens_out


class QPLEXSELoss(nn.Module):
    """QPLEX loss + shared encoder MSE auxiliary loss.

    Standard QPLEX_V2 loss, plus MSE prediction loss that backprops
    through the shared RNN hidden state → fc1.
    No augmentation of obs or state.
    """

    def __init__(
        self,
        model, target_model,
        mixer, target_mixer,
        pred_head,
        n_agents, n_actions, n_targets,
        double_q=True, gamma=0.99,
        aux_loss_weight=0.1,
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
        self.double_q = double_q
        self.gamma = gamma
        self.aux_loss_weight = aux_loss_weight

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

        # =================================================================
        # Q-values + hidden states (shared encoder)
        # =================================================================
        mac_out, hiddens = _unroll_mac_with_hidden(self.model, obs)
        # mac_out: [B, T, n_agents, n_actions]
        # hiddens: [B, T, n_agents, h_size]

        chosen_action_qvals = torch.gather(
            mac_out, dim=3, index=actions.unsqueeze(3)
        ).squeeze(3)
        ignore_action = (action_mask == 0) & (mask == 1).unsqueeze(-1)
        x_mac_out = mac_out.clone().detach()
        x_mac_out[ignore_action] = -np.inf
        max_action_vals, _ = x_mac_out.max(dim=3)

        # Target Q-values (no hidden states needed — use fast _unroll_mac)
        target_mac_out = _unroll_mac(self.target_model, next_obs)
        ignore_action_tp1 = (next_action_mask == 0) & (mask == 1).unsqueeze(-1)
        target_mac_out[ignore_action_tp1] = -np.inf

        if self.double_q:
            mac_out_tp1 = _unroll_mac(self.model, next_obs)
            mac_out_tp1[ignore_action_tp1] = -np.inf
            cur_max_actions = mac_out_tp1.argmax(dim=3, keepdim=True)
            target_max_qvals = torch.gather(target_mac_out, 3, cur_max_actions).squeeze(3)
        else:
            target_max_qvals = target_mac_out.max(dim=3)[0]

        assert target_max_qvals.min().item() != -np.inf

        # =================================================================
        # Mix (original state — no augmentation)
        # =================================================================
        if self.mixer is not None:
            ans_chosen = self.mixer(chosen_action_qvals, state, is_v=True)
            actions_onehot = F.one_hot(actions, num_classes=self.n_actions)
            ans_adv = self.mixer(
                chosen_action_qvals, state, actions_onehot,
                max_action_vals=max_action_vals, is_v=False
            )
            chosen_action_qvals = ans_chosen + ans_adv

            target_chosen = self.target_mixer(target_max_qvals, next_state, is_v=True)
            cur_max_actions_onehot = F.one_hot(cur_max_actions, num_classes=self.n_actions)
            target_adv = self.target_mixer(
                target_max_qvals, next_state, cur_max_actions_onehot,
                target_max_qvals, is_v=False
            )
            target_max_qvals = target_chosen + target_adv

        # =================================================================
        # TD loss (identical to QPLEX_V2)
        # =================================================================
        targets = rewards + self.gamma * (1 - terminated) * target_max_qvals
        td_error = chosen_action_qvals - targets.detach()
        mask = mask.expand_as(td_error)
        masked_td_error = td_error * mask
        td_loss = (masked_td_error ** 2).sum() / mask.sum()

        # =================================================================
        # Auxiliary MSE loss (prediction head on shared hidden states)
        # =================================================================
        # Each agent predicts next absolute target positions
        C, H = self.n_agents, hiddens.shape[-1]
        pred = self.pred_head(hiddens.reshape(B * T * C, H))   # [B*T*C, n_targets, 2]
        pred = pred.reshape(B, T, C, self.n_targets, 2)

        # Labels: absolute next target positions from next_state
        labels = _extract_target_positions(next_state, self.n_agents, self.n_targets)
        labels = labels.unsqueeze(2).expand_as(pred)  # [B, T, C, n_targets, 2]

        mse = ((pred - labels.detach()) ** 2).mean(dim=(-1, -2))  # [B, T, C]
        mse = mse.mean(dim=2)  # [B, T] — average across agents
        wm_mask = mask[:, :, 0]
        aux_loss = (mse * wm_mask).sum() / wm_mask.sum().clamp(min=1)

        # =================================================================
        # Total loss
        # =================================================================
        total_loss = td_loss + self.aux_loss_weight * aux_loss

        return total_loss, td_loss, aux_loss, mask, masked_td_error, chosen_action_qvals, targets


class QPLEXSETorchPolicy(Policy):
    """QPLEX + Shared Encoder policy.

    Identical architecture to QPLEX_V2. Only addition: PredictionHead on hidden states.
    MSE auxiliary loss improves shared representation.
    """

    def __init__(self, obs_space, action_space, config):
        _validate(obs_space, action_space)
        config = dict(ray.rllib.agents.qplex_se.qplex.DEFAULT_CONFIG, **config)

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
        # Agent RNN model — SAME as QPLEX_V2 (no augmentation)
        # =====================================================================
        self.model = ModelCatalog.get_model_v2(
            agent_obs_space, action_space.spaces[0], self.n_actions,
            config["model"], framework="torch", name="model", default_model=RNNModel,
        ).to(self.device)

        self.target_model = ModelCatalog.get_model_v2(
            agent_obs_space, action_space.spaces[0], self.n_actions,
            config["model"], framework="torch", name="target_model", default_model=RNNModel,
        ).to(self.device)

        self.exploration = self._create_exploration()

        # =====================================================================
        # Prediction Head (auxiliary task on shared hidden states)
        # =====================================================================
        se_config = config.get("shared_encoder", {})
        self.n_targets = se_config.get("n_targets", 8)
        self.aux_loss_weight = se_config.get("aux_loss_weight", 0.1)

        self.pred_head = PredictionHead(
            hidden_dim=self.h_size,
            n_targets=self.n_targets,
            pred_hidden=se_config.get("pred_hidden", 128),
        ).to(self.device)

        # =====================================================================
        # Mixer — SAME as QPLEX_V2 (original state_dim, no augmentation)
        # =====================================================================
        self.mixer = DuelMixerV2(
            self.args, self.n_agents, self.n_actions, self.env_global_state_shape,
            config['mixing_embed_dim'], self.args.ffn_hidden_dim, self.args.num_kernel,
        ).to(self.device)
        self.target_mixer = DuelMixerV2(
            self.args, self.n_agents, self.n_actions, self.env_global_state_shape,
            config['mixing_embed_dim'], self.args.ffn_hidden_dim, self.args.num_kernel,
        ).to(self.device)
        assert config['mixer'] == 'qplex_se'

        self.cur_epsilon = 1.0
        self.update_target()

        # =====================================================================
        # Optimizer (model + mixer + prediction head)
        # =====================================================================
        self.params = list(self.model.parameters())
        self.params += list(self.mixer.parameters())
        self.params += list(self.pred_head.parameters())

        self.loss = QPLEXSELoss(
            self.model, self.target_model,
            self.mixer, self.target_mixer,
            self.pred_head,
            self.n_agents, self.n_actions, self.n_targets,
            self.config["double_q"], self.config["gamma"],
            self.aux_loss_weight,
        )

        from torch.optim import RMSprop
        self.optimiser = RMSprop(
            params=self.params,
            lr=config["lr"],
            alpha=config["optim_alpha"],
            eps=config["optim_eps"],
        )

    # -----------------------------------------------------------------
    # Actions (identical to QPLEX_V2 — no augmentation)
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
            q_values, hiddens = _mac(
                self.model,
                torch.as_tensor(obs_batch, dtype=torch.float, device=self.device),
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

        (total_loss, td_loss, aux_loss,
         mask, masked_td_error, chosen_action_qvals, targets,
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

        mask_elems = mask.sum().item()
        stats = {
            "loss": total_loss.item(),
            "td_loss": td_loss.item(),
            "prediction_mse": aux_loss.item(),
            "grad_norm": grad_norm if isinstance(grad_norm, float) else grad_norm.item(),
            "td_error_abs": masked_td_error.abs().sum().item() / mask_elems,
            "q_taken_mean": (chosen_action_qvals * mask).sum().item() / mask_elems,
            "target_mean": (targets * mask).sum().item() / mask_elems,
        }
        return {LEARNER_STATS_KEY: stats}

    # -----------------------------------------------------------------
    # State management (identical to QPLEX_V2)
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
