"""QPLEX + Greedy Behavioral Cloning Distillation Policy.

Extends QPLEX_V2 with an auxiliary cross-entropy BC loss that distills
the greedy camera heuristic into the learned Q-network.

Training loss:
    L = L_TD  +  greedy_bc_coeff * L_BC

    L_BC = CrossEntropy(Q(s,·) / temperature, greedy_action)

Greedy action (per agent, from observation):
    - Find the nearest *visible* target in the agent's local observation.
    - Encode as discrete index: select-only-that-target bit-vector.
    - If no target visible → action 0 (no-op / no selection).

Observation layout assumed (after RelativeCoordinates + RescaledObservation):
    [0:13]   preserved data
    [13:22]  self state
    [22:62]  opponent (target) states with mask  — 5 values per target:
                 obs[22 + t*5 : 22 + t*5 + 2]   → (x, y) relative position
                 obs[22 + t*5 + 4]               → visibility mask (0 or 1)
    ...
"""

import math
import logging
import numpy as np
import tree
import torch
import torch.nn as nn
import torch.nn.functional as F
from argparse import Namespace

import ray
from ray.rllib.agents.qplex_v2.mixers import DuelMixerV2
from ray.rllib.agents.qplex_v2.model import RNNModel, _get_size
from ray.rllib.agents.qplex_v2.qplex_policy import (
    _validate, _mac, _unroll_mac, adjust_args,
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

# ---------------------------------------------------------------------------
# Observation layout constants (mate.camera_observation_slices_of)
# ---------------------------------------------------------------------------
_OPP_OBS_START = 22        # start of opponent_states_with_mask block
_OPP_STATE_STRIDE = 5      # TARGET_STATE_DIM_PUBLIC(4) + mask(1)
_OPP_MASK_OFFSET = 4       # offset within each target block for visibility mask


# ---------------------------------------------------------------------------
# Greedy action helper
# ---------------------------------------------------------------------------

def compute_greedy_actions(obs: torch.Tensor, n_actions: int, device: torch.device,
                           memory_period: int = 25) -> torch.Tensor:
    """Compute memory-augmented greedy action for each agent across a sequence.

    Mirrors GreedyCameraAgent's time2forget mechanism: a target that was visible
    within the last `memory_period` steps is still trackable using its last known
    relative position, even when currently invisible.

    Greedy rule: select the nearest *trackable* target (visible OR recently seen).
    If no trackable target exists, return action 0 (no-op).

    NOTE: positions stored in memory are relative to the camera at the time they
    were observed. They become stale as camera and target move, but still provide
    a useful directional hint — consistent with how the GRU should learn to behave.

    Args:
        obs:           [B, T, n_agents, obs_size] float tensor
        n_actions:     total discrete actions = 2^n_targets
        device:        torch device
        memory_period: steps to remember a target after it leaves field of view

    Returns:
        greedy_actions: [B, T, n_agents] long tensor
    """
    n_targets = int(round(math.log2(n_actions)))
    strides = torch.tensor(
        [2 ** (n_targets - 1 - t) for t in range(n_targets)],
        dtype=torch.long, device=device,
    )  # [n_targets]

    B, T, C, obs_size = obs.shape
    obs_d = obs.detach()

    # Extract positions [B, T, C, n_targets, 2] and visibility [B, T, C, n_targets]
    pos = torch.stack(
        [obs_d[:, :, :, _OPP_OBS_START + t * _OPP_STATE_STRIDE:
                        _OPP_OBS_START + t * _OPP_STATE_STRIDE + 2]
         for t in range(n_targets)],
        dim=3,
    )  # [B, T, C, n_targets, 2]

    vis = torch.stack(
        [obs_d[:, :, :, _OPP_OBS_START + t * _OPP_STATE_STRIDE + _OPP_MASK_OFFSET]
         for t in range(n_targets)],
        dim=3,
    ) > 0.5  # [B, T, C, n_targets] bool

    # Memory state (mirrors GreedyCameraAgent.time2forget)
    last_pos = torch.zeros(B, C, n_targets, 2, device=device)
    time_since_seen = torch.full(
        (B, C, n_targets), fill_value=memory_period, dtype=torch.long, device=device
    )

    greedy_actions = torch.zeros(B, T, C, dtype=torch.long, device=device)

    for t in range(T):
        vis_t = vis[:, t]    # [B, C, n_targets]
        pos_t = pos[:, t]    # [B, C, n_targets, 2]

        # Update memory: reset counter for newly visible targets
        time_since_seen = (time_since_seen + 1).clamp(max=memory_period)
        time_since_seen[vis_t] = 0
        last_pos[vis_t] = pos_t[vis_t]

        # A target is trackable if seen within memory_period steps
        trackable = time_since_seen < memory_period   # [B, C, n_targets]

        dist = last_pos.norm(dim=-1).clone()          # [B, C, n_targets]
        dist[~trackable] = float('inf')

        nearest = dist.argmin(dim=-1)                 # [B, C]
        has_trackable = trackable.any(dim=-1)         # [B, C]

        bits = torch.zeros(B, C, n_targets, dtype=torch.long, device=device)
        bits.scatter_(2, nearest.unsqueeze(2), 1)
        bits[~has_trackable] = 0

        greedy_actions[:, t] = (bits * strides).sum(dim=-1)  # [B, C]

    return greedy_actions


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

class QPLEXDistillLoss(nn.Module):
    """QPLEX TD loss + greedy behavioral-cloning cross-entropy loss."""

    def __init__(
        self,
        model,
        target_model,
        mixer,
        target_mixer,
        n_agents,
        n_actions,
        double_q=True,
        gamma=0.99,
        greedy_bc_coeff=0.1,
        greedy_bc_temperature=1.0,
    ):
        nn.Module.__init__(self)
        self.model = model
        self.target_model = target_model
        self.mixer = mixer
        self.target_mixer = target_mixer
        self.n_agents = n_agents
        self.n_actions = n_actions
        self.double_q = double_q
        self.gamma = gamma
        self.greedy_bc_coeff = greedy_bc_coeff
        self.greedy_bc_temperature = greedy_bc_temperature

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
        greedy_actions=None,
    ):
        """
        Args:
            rewards:          [B, T, n_agents]
            actions:          [B, T, n_agents]  long
            terminated:       [B, T, n_agents]
            mask:             [B, T, n_agents]
            obs:              [B, T, n_agents, obs_size]
            next_obs:         [B, T, n_agents, obs_size]
            action_mask:      [B, T, n_agents, n_actions]
            next_action_mask: [B, T, n_agents, n_actions]
            state:            [B, T, state_dim]  (optional)
            next_state:       [B, T, state_dim]  (optional)
            greedy_actions:   [B, T, n_agents]  long  (optional)
        """
        if state is None and next_state is None:
            state = obs
            next_state = next_obs
        elif (state is None) != (next_state is None):
            raise ValueError("Expected either neither or both of state/next_state.")

        B, T = obs.shape[0], obs.shape[1]

        # -----------------------------------------------------------------
        # 1. Compute per-agent Q-values
        # -----------------------------------------------------------------
        mac_out = _unroll_mac(self.model, obs)          # [B, T, n_agents, n_actions]

        chosen_action_qvals = torch.gather(
            mac_out, dim=3, index=actions.unsqueeze(3)
        ).squeeze(3)                                    # [B, T, n_agents]

        ignore_action = (action_mask == 0) & (mask == 1).unsqueeze(-1)
        x_mac_out = mac_out.clone().detach()
        x_mac_out[ignore_action] = -np.inf
        max_action_vals, max_action_index = x_mac_out.max(dim=3)
        max_action_index = max_action_index.detach().unsqueeze(3)

        # -----------------------------------------------------------------
        # 2. Target Q-values (Double-Q)
        # -----------------------------------------------------------------
        target_mac_out = _unroll_mac(self.target_model, next_obs)  # [B, T, n_a, n_act]

        ignore_action_tp1 = (next_action_mask == 0) & (mask == 1).unsqueeze(-1)
        target_mac_out[ignore_action_tp1] = -np.inf

        if self.double_q:
            mac_out_tp1 = _unroll_mac(self.model, next_obs)
            mac_out_tp1[ignore_action_tp1] = -np.inf
            cur_max_actions = mac_out_tp1.argmax(dim=3, keepdim=True)
            target_max_qvals = torch.gather(target_mac_out, 3, cur_max_actions).squeeze(3)
        else:
            target_max_qvals = target_mac_out.max(dim=3)[0]

        assert target_max_qvals.min().item() != -np.inf, (
            "target_max_qvals contains a masked action; "
            "there may be a state with no valid actions."
        )

        # -----------------------------------------------------------------
        # 3. Mix
        # -----------------------------------------------------------------
        if self.mixer is not None:
            ans_chosen = self.mixer(chosen_action_qvals, state, is_v=True)
            actions_onehot = F.one_hot(actions, num_classes=self.n_actions)
            ans_adv = self.mixer(
                chosen_action_qvals, state, actions_onehot,
                max_action_vals=max_action_vals, is_v=False,
            )
            chosen_action_qvals = ans_chosen + ans_adv

            target_chosen = self.target_mixer(target_max_qvals, next_state, is_v=True)
            cur_max_actions_onehot = F.one_hot(cur_max_actions, num_classes=self.n_actions)
            target_adv = self.target_mixer(
                target_max_qvals, next_state, cur_max_actions_onehot,
                target_max_qvals, is_v=False,
            )
            target_max_qvals = target_chosen + target_adv

        # -----------------------------------------------------------------
        # 4. TD loss
        # -----------------------------------------------------------------
        targets = rewards + self.gamma * (1 - terminated) * target_max_qvals
        td_error = chosen_action_qvals - targets.detach()
        mask = mask.expand_as(td_error)
        masked_td_error = td_error * mask
        td_loss = (masked_td_error ** 2).sum() / mask.sum()

        # -----------------------------------------------------------------
        # 5. Greedy BC loss  (cross-entropy treating Q as logits)
        # -----------------------------------------------------------------
        bc_loss = torch.tensor(0.0, device=obs.device)
        if greedy_actions is not None and self.greedy_bc_coeff > 0.0:
            # mac_out: [B, T, n_agents, n_actions] — same as computed above
            scaled_q = mac_out / self.greedy_bc_temperature  # [B, T, C, n_actions]
            bc_raw = F.cross_entropy(
                scaled_q.reshape(-1, self.n_actions),   # [B*T*C, n_actions]
                greedy_actions.reshape(-1),             # [B*T*C]
                reduction='none',
            ).reshape(B, T, self.n_agents)              # [B, T, C]
            bc_loss = (bc_raw * mask).sum() / mask.sum().clamp(min=1)

        total_loss = td_loss + self.greedy_bc_coeff * bc_loss

        stats = {
            "td_loss": td_loss.item(),
            "bc_loss": bc_loss.item(),
        }
        return total_loss, stats, mask, masked_td_error, chosen_action_qvals, targets


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------

class QPLEXDistillTorchPolicy(Policy):
    """QPLEX + Greedy Distillation.

    Identical to QPLEX_V2 except:
      - Also computes greedy discrete actions from the replay-buffer observations.
      - Adds cross-entropy BC loss against greedy actions.
    """

    def __init__(self, obs_space, action_space, config):
        _validate(obs_space, action_space)
        import ray.rllib.agents.qplex_distill.qplex as _qd
        config = dict(_qd.DEFAULT_CONFIG, **config)

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
        if isinstance(agent_obs_space, type({}).__class__):
            pass  # unreachable placeholder
        from gym.spaces import Dict as GymDict
        if isinstance(agent_obs_space, GymDict):
            space_keys = set(agent_obs_space.spaces.keys())
            if "obs" not in space_keys:
                raise ValueError("Dict obs space must have subspace labeled `obs`")
            self.obs_size = _get_size(agent_obs_space.spaces["obs"])
            if "action_mask" in space_keys:
                mask_shape = tuple(agent_obs_space.spaces["action_mask"].shape)
                if mask_shape != (self.n_actions,):
                    raise ValueError(
                        "Action mask shape must be {}, got {}".format(
                            (self.n_actions,), mask_shape
                        )
                    )
                self.has_action_mask = True
            if ENV_STATE in space_keys:
                self.env_global_state_shape = _get_size(
                    agent_obs_space.spaces[ENV_STATE]
                )
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
            default_model=RNNModel,
        ).to(self.device)

        self.target_model = ModelCatalog.get_model_v2(
            agent_obs_space, action_space.spaces[0], self.n_actions,
            config["model"], framework="torch", name="target_model",
            default_model=RNNModel,
        ).to(self.device)

        self.exploration = self._create_exploration()

        self.mixer = DuelMixerV2(
            self.args, self.n_agents, self.n_actions, self.env_global_state_shape,
            config['mixing_embed_dim'], self.args.ffn_hidden_dim, self.args.num_kernel,
        ).to(self.device)
        self.target_mixer = DuelMixerV2(
            self.args, self.n_agents, self.n_actions, self.env_global_state_shape,
            config['mixing_embed_dim'], self.args.ffn_hidden_dim, self.args.num_kernel,
        ).to(self.device)
        assert config['mixer'] == 'qplex_distill'

        self.cur_epsilon = 1.0
        self.update_target()

        self.params = list(self.model.parameters())
        if self.mixer:
            self.params += list(self.mixer.parameters())

        self.greedy_bc_coeff = config.get('greedy_bc_coeff', 0.1)
        self.greedy_bc_temperature = config.get('greedy_bc_temperature', 1.0)

        self.loss = QPLEXDistillLoss(
            self.model, self.target_model,
            self.mixer, self.target_mixer,
            self.n_agents, self.n_actions,
            config["double_q"], config["gamma"],
            greedy_bc_coeff=self.greedy_bc_coeff,
            greedy_bc_temperature=self.greedy_bc_temperature,
        )

        from torch.optim import RMSprop
        self.optimiser = RMSprop(
            params=self.params,
            lr=config["lr"],
            alpha=config["optim_alpha"],
            eps=config["optim_eps"],
        )

    # ------------------------------------------------------------------
    # Helpers (same as QPLEXTorchPolicy)
    # ------------------------------------------------------------------

    def _unpack_observation(self, obs_batch):
        """Unpack obs, action mask, and global state from grouped agent obs."""
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

    def _get_group_rewards(self, info_batch):
        group_rewards = np.array(
            [
                info.get(GROUP_REWARDS, np.zeros(self.n_agents, dtype=np.float32))
                for info in info_batch
            ]
        )
        return group_rewards

    def update_target(self):
        self.target_model.load_state_dict(self.model.state_dict())
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

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
            q_values, hiddens = _mac(
                self.model,
                torch.as_tensor(obs_batch, dtype=torch.float, device=self.device),
                [
                    torch.as_tensor(np.array(s), dtype=torch.float, device=self.device)
                    for s in state_batches
                ],
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
                timestep=timestep,
                explore=explore,
            )
            actions = (
                torch.reshape(actions, list(masked_q_values.shape)[:-1]).cpu().numpy()
            )
            hiddens = [s.cpu().numpy() for s in hiddens]

        return tuple(actions.transpose([1, 0])), hiddens, {}

    @override(Policy)
    def compute_log_likelihoods(
        self,
        actions,
        obs_batch,
        state_batches=None,
        prev_action_batch=None,
        prev_reward_batch=None,
    ):
        obs_batch, action_mask, _ = self._unpack_observation(obs_batch)
        return np.zeros(obs_batch.size()[0])

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    @override(Policy)
    def learn_on_batch(self, samples):
        obs_batch, action_mask, env_global_state = self._unpack_observation(
            samples[SampleBatch.CUR_OBS]
        )
        (
            next_obs_batch,
            next_action_mask,
            next_env_global_state,
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
            (
                rew, action_mask, next_action_mask, act, dones,
                obs, next_obs, env_global_state, next_env_global_state,
            ) = output_list
        else:
            (rew, action_mask, next_action_mask, act, dones, obs, next_obs) = output_list

        B, T = len(seq_lens), max(seq_lens)

        def to_batches(arr, dtype):
            new_shape = [B, T] + list(arr.shape[1:])
            return torch.as_tensor(
                np.reshape(arr, new_shape), dtype=dtype, device=self.device
            )

        rewards = to_batches(rew, torch.float)
        actions = to_batches(act, torch.long)
        obs = to_batches(obs, torch.float).reshape([B, T, self.n_agents, self.obs_size])
        action_mask = to_batches(action_mask, torch.float)
        next_obs = to_batches(next_obs, torch.float).reshape(
            [B, T, self.n_agents, self.obs_size]
        )
        next_action_mask = to_batches(next_action_mask, torch.float)
        if self.has_env_global_state:
            env_global_state = to_batches(env_global_state, torch.float)
            next_env_global_state = to_batches(next_env_global_state, torch.float)

        terminated = (
            to_batches(dones, torch.float).unsqueeze(2).expand(B, T, self.n_agents)
        )

        filled = np.reshape(
            np.tile(np.arange(T, dtype=np.float32), B), [B, T]
        ) < np.expand_dims(seq_lens, 1)
        mask = (
            torch.as_tensor(filled, dtype=torch.float, device=self.device)
            .unsqueeze(2)
            .expand(B, T, self.n_agents)
        )

        # Compute memory-augmented greedy actions across the sequence
        greedy_actions = None
        if self.greedy_bc_coeff > 0.0:
            greedy_actions = compute_greedy_actions(
                obs, self.n_actions, self.device,
                memory_period=self.config.get('greedy_memory_period', 25),
            )

        total_loss, loss_stats, mask, masked_td_error, chosen_action_qvals, targets = self.loss(
            rewards, actions, terminated, mask,
            obs, next_obs, action_mask, next_action_mask,
            env_global_state if self.has_env_global_state else None,
            next_env_global_state if self.has_env_global_state else None,
            greedy_actions=greedy_actions,
        )

        self.optimiser.zero_grad()
        total_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.params, self.config["grad_norm_clipping"]
        )
        self.optimiser.step()

        mask_elems = mask.sum().item()
        stats = {
            **loss_stats,
            "loss": total_loss.item(),
            "grad_norm": grad_norm if isinstance(grad_norm, float) else grad_norm.item(),
            "td_error_abs": masked_td_error.abs().sum().item() / mask_elems,
            "q_taken_mean": (chosen_action_qvals * mask).sum().item() / mask_elems,
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
            "mixer": self._cpu_dict(self.mixer.state_dict()) if self.mixer else None,
            "target_mixer": self._cpu_dict(self.target_mixer.state_dict())
            if self.mixer else None,
        }

    @override(Policy)
    def set_weights(self, weights):
        self.model.load_state_dict(self._device_dict(weights["model"]))
        self.target_model.load_state_dict(self._device_dict(weights["target_model"]))
        if self.mixer and weights.get("mixer") is not None:
            self.mixer.load_state_dict(self._device_dict(weights["mixer"]))
        if self.mixer and weights.get("target_mixer") is not None:
            self.target_mixer.load_state_dict(self._device_dict(weights["target_mixer"]))

    @override(Policy)
    def get_state(self):
        state = self.get_weights()
        state["cur_epsilon"] = self.cur_epsilon
        return state

    @override(Policy)
    def set_state(self, state):
        self.set_weights(state)
        if "cur_epsilon" in state:
            self.cur_epsilon = state["cur_epsilon"]

    def _cpu_dict(self, state_dict):
        return {k: v.cpu() for k, v in state_dict.items()}

    def _device_dict(self, state_dict):
        return {k: v.to(self.device) for k, v in state_dict.items()}
