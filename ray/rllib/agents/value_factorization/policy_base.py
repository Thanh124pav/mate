"""RLlib integration shared by the paper-specific grouped CTDE policies."""

import copy
import logging
from dataclasses import dataclass
from typing import Dict as TypingDict, Optional

import numpy as np
import tree
import torch
import torch.nn.functional as F
from gym.spaces import Dict, Discrete, Tuple

from ray.rllib.agents.qplex.model import _get_size
from ray.rllib.env.multi_agent_env import ENV_STATE
from ray.rllib.env.wrappers.group_agents_wrapper import GROUP_REWARDS
from ray.rllib.models.modelv2 import _unpack_obs
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.rnn_sequencing import chop_into_sequences
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.annotations import override
from ray.rllib.utils.metrics.learner_info import LEARNER_STATS_KEY

from ray.rllib.agents.focus_common.responsibility_engine import (
    FocusResponsibilityEngine,
)
from ray.rllib.agents.focus_official import (
    GradientFloor,
    action_credit_loss,
    carrier_credit_loss,
)
from ray.rllib.agents.qplex_focus.qplex_policy import (
    build_occupancy_model,
    resolve_focus_config,
)


logger = logging.getLogger(__name__)


@dataclass
class GroupedSequenceBatch:
    rewards: torch.Tensor
    actions: torch.Tensor
    terminated: torch.Tensor
    mask: torch.Tensor
    obs: torch.Tensor
    next_obs: torch.Tensor
    action_mask: torch.Tensor
    next_action_mask: torch.Tensor
    state: torch.Tensor
    next_state: torch.Tensor
    seq_lens: np.ndarray

    @property
    def agent_mask(self):
        return self.mask.expand(-1, -1, self.actions.size(-1))


def validate_grouped_spaces(obs_space, action_space, algorithm="algorithm"):
    if not hasattr(obs_space, "original_space") or not isinstance(
        obs_space.original_space, Tuple
    ):
        raise ValueError(
            f"{algorithm} requires a grouped Tuple observation space; call "
            "MultiAgentEnv.with_agent_groups()."
        )
    if not isinstance(action_space, Tuple):
        raise ValueError(f"{algorithm} requires a grouped Tuple action space")
    if not all(isinstance(space, Discrete) for space in action_space.spaces):
        raise ValueError(f"{algorithm} supports homogeneous discrete actions only")
    if len({str(space) for space in obs_space.original_space.spaces}) != 1:
        raise ValueError(f"{algorithm} requires homogeneous agent observations")
    if len({str(space) for space in action_space.spaces}) != 1:
        raise ValueError(f"{algorithm} requires homogeneous agent actions")


class GroupedCTDEPolicy(Policy):
    """Base policy for grouped homogeneous recurrent CTDE algorithms.

    Subclasses create ``self.model`` and all optimizer/modules after invoking
    this constructor, then implement ``_scores_from_output`` and
    ``_learn_sequence_batch``.
    """

    algorithm_name = "GroupedCTDE"

    def __init__(self, obs_space, action_space, config):
        validate_grouped_spaces(obs_space, action_space, self.algorithm_name)
        self.framework = "torch"
        super().__init__(obs_space, action_space, config)
        self.n_agents = len(obs_space.original_space.spaces)
        self.n_actions = int(action_space.spaces[0].n)
        self.has_action_mask = False
        self.has_env_global_state = False
        agent_space = obs_space.original_space.spaces[0]
        if isinstance(agent_space, Dict):
            keys = set(agent_space.spaces)
            if "obs" not in keys:
                raise ValueError("agent Dict observation must contain `obs`")
            self.obs_dim = _get_size(agent_space.spaces["obs"])
            if "action_mask" in keys:
                if tuple(agent_space.spaces["action_mask"].shape) != (self.n_actions,):
                    raise ValueError("action_mask has an incompatible shape")
                self.has_action_mask = True
            if ENV_STATE in keys:
                self.state_dim = _get_size(agent_space.spaces[ENV_STATE])
                self.has_env_global_state = True
            else:
                self.state_dim = self.n_agents * self.obs_dim
        else:
            self.obs_dim = _get_size(agent_space)
            self.state_dim = self.n_agents * self.obs_dim

        use_gpu = bool(config.get("num_gpus", 0)) and torch.cuda.is_available()
        self.device = torch.device("cuda" if use_gpu else "cpu")
        self.exploration = self._create_exploration()
        self.cur_epsilon = float(
            config.get("exploration_config", {}).get("initial_epsilon", 1.0)
        )
        self.focus_config = resolve_focus_config(config)
        self.focus_engine = None
        self._named_modules_for_weights = {}
        self._optimizers = {}
        self._learner_updates = 0
        self._last_target_update = 0
        self.focuss_gradient_floor = GradientFloor(
            ratio=float(
                self.focus_config.get(
                    "action_grad_floor",
                    self.focus_config.get("action_gradient_floor", 0.10),
                )
            ),
            ema_decay=float(
                self.focus_config.get(
                    "action_grad_ema",
                    self.focus_config.get("action_gradient_ema_decay", 0.9),
                )
            ),
            eps=float(self.focus_config.get("eps", 1e-8)),
        ).to(self.device)
        self._register_module("focuss_gradient_floor", self.focuss_gradient_floor)

    def _initialize_focus(self):
        if not self.focus_config.get("enabled", False):
            return
        mode = self.focus_config.get("belief_mode", "learned")
        occupancy = None
        if mode == "learned":
            occupancy = build_occupancy_model(
                self.state_dim,
                self.n_agents,
                int(self.focus_config.get("n_targets", 8)),
                self.focus_config,
            ).to(self.device)
        self.focus_engine = FocusResponsibilityEngine(
            occupancy,
            n_agents=self.n_agents,
            n_actions=self.n_actions,
            focus_config=self.focus_config,
        ).to(self.device)
        self._register_module("focus_engine", self.focus_engine)

    def _focus_output(self, batch: GroupedSequenceBatch):
        if self.focus_engine is None:
            return None
        return self.focus_engine(
            batch.state,
            batch.actions,
            valid_mask=batch.agent_mask,
            next_global_state=batch.next_state,
            learner_update=self._learner_updates,
        )

    def _focuss_action_objective(
        self,
        action_scores,
        focus,
        reference_loss,
        parameters,
    ):
        """Return official action CE scaled to at least 10% backbone gradient."""

        zero = reference_loss.new_zeros(())
        credit_level = str(self.focus_config.get("credit_level", "both")).lower()
        action_enabled = bool(
            self.focus_config.get("action_credit_enabled", True)
        ) and credit_level in ("action", "both")
        if not action_enabled:
            return zero, {
                "focuss_action_loss": zero,
                "focuss_action_gradient_scale": 0.0,
                "focuss_action_gradient_floor_active": 0.0,
            }
        if focus is None or focus.pair_gains is None:
            return zero, {
                "focuss_action_loss": zero,
                "focuss_action_gradient_scale": 0.0,
                "focuss_action_gradient_floor_active": 0.0,
            }
        valid = focus.valid
        signal = focus.total_gain.detach()
        if valid.any():
            signal = signal / signal[valid].mean().clamp_min(1e-8)
        else:
            signal = torch.ones_like(signal)
        signal = signal.clamp(
            min=float(self.focus_config.get("signal_weight_min", 0.1)),
            max=float(self.focus_config.get("signal_weight_max", 3.0)),
        )
        action_loss, action_gains = action_credit_loss(
            action_scores,
            focus.pair_gains,
            valid,
            confidence=focus.confidence,
            signal_weights=signal,
            temperature=float(
                self.focus_config.get(
                    "action_temperature",
                    self.focus_config.get("action_credit_temperature", 1.0),
                )
            ),
            q_std_floor=float(self.focus_config.get("action_q_std_floor", 0.01)),
            eps=float(self.focus_config.get("eps", 1e-8)),
        )
        coefficient = float(
            self.focus_config.get(
                "action_alpha", self.focus_config.get("action_credit_coeff", 1.0)
            )
        )
        scaled, floor = self.focuss_gradient_floor.scale(
            reference_loss,
            coefficient * action_loss,
            parameters,
        )
        return scaled, {
            "focuss_action_loss": action_loss.detach(),
            "focuss_action_gain_mean": action_gains.mean().detach(),
            "focuss_action_gradient_scale": floor.scale,
            "focuss_action_gradient_norm": floor.action_norm,
            "focuss_backbone_gradient_norm": floor.td_norm,
            "focuss_backbone_gradient_norm_ema": floor.td_norm_ema,
            "focuss_action_gradient_floor_active": float(floor.active),
        }

    def _focuss_gradient_carrier(self, joint_value, agent_utility):
        """Return the official gradient carrier |dQ_tot/du_i| normalized over agents.

        ``create_graph=True`` is intentional: the carrier CE must update the
        mixer's parameters, while the environmental target remains detached.
        Extra axes (for example RiskQ quantiles) are averaged into one carrier
        probability per agent and transition.
        """

        (gradient,) = torch.autograd.grad(
            joint_value.sum(),
            agent_utility,
            create_graph=True,
            retain_graph=True,
            allow_unused=False,
        )
        gradient = gradient.abs()
        while gradient.dim() > 3:
            gradient = gradient.mean(dim=-1)
        eps = float(self.focus_config.get("eps", 1e-8))
        return gradient / gradient.sum(dim=-1, keepdim=True).clamp_min(eps)

    def _focuss_agent_objective(self, joint_value, agent_utility, focus):
        """Official agent-level CE(rho, gradient carrier) for value backbones."""

        zero = joint_value.new_zeros(())
        stats = {
            "focuss_credit_loss": zero,
            "focuss_alpha_credit": 0.0,
        }
        if focus is None:
            return zero, stats
        credit_level = str(self.focus_config.get("credit_level", "both")).lower()
        mechanism = str(self.focus_config.get("credit_mechanism", "align")).lower()
        if credit_level not in ("agent", "both") or mechanism == "route":
            return zero, stats
        if mechanism not in ("align", "both"):
            raise ValueError(
                "focus.credit_mechanism must be one of align, route, or both"
            )
        carrier = self._focuss_gradient_carrier(joint_value, agent_utility)
        signal = focus.total_gain.detach()
        if focus.valid.any():
            signal = signal / signal[focus.valid].mean().clamp_min(1e-8)
        else:
            signal = torch.ones_like(signal)
        signal = signal.clamp(
            min=float(self.focus_config.get("signal_weight_min", 0.1)),
            max=float(self.focus_config.get("signal_weight_max", 3.0)),
        )
        credit_loss, carrier = carrier_credit_loss(
            focus.rho,
            carrier,
            focus.valid,
            confidence=focus.confidence,
            signal_weights=signal,
            eps=float(self.focus_config.get("eps", 1e-8)),
        )
        alpha = float(self.focus_config.get("alpha_credit", 0.1))
        valid = focus.valid
        entropy = -(carrier.detach() * carrier.detach().clamp_min(1e-8).log()).sum(-1)
        stats = {
            "focuss_credit_loss": credit_loss.detach(),
            "focuss_alpha_credit": alpha,
            "focuss_carrier_entropy": (
                entropy[valid].mean().detach() if valid.any() else entropy.new_zeros(())
            ),
        }
        return alpha * credit_loss, stats

    def _focuss_route_weights(self, focus):
        """Return the optional route-ablation weights; official align returns None."""

        if focus is None:
            return None
        credit_level = str(self.focus_config.get("credit_level", "both")).lower()
        mechanism = str(self.focus_config.get("credit_mechanism", "align")).lower()
        if credit_level not in ("agent", "both") or mechanism not in ("route", "both"):
            return None
        from ray.rllib.agents.value_factorization.components import focus_agent_weights

        return focus_agent_weights(
            focus.rho,
            self.n_agents,
            float(self.focus_config.get("route_strength", self.focus_config.get("eta", 1.0))),
            focus.confidence * focus.valid.to(focus.confidence.dtype),
        )

    def _register_module(self, name, module):
        self._named_modules_for_weights[name] = module

    def _register_optimizer(self, name, optimizer):
        self._optimizers[name] = optimizer

    def _scores_from_output(self, output):
        return output

    def _select_actions(self, scores, action_mask, explore, timestep):
        """Select flattened decentralized actions from local policy scores."""

        flat = scores.reshape(-1, self.n_actions)
        # Ray 1.12 creates EpsilonGreedy's Bernoulli selector on
        # ``exploration.device``.  That device remains CPU for a local policy,
        # even when its model and logits are on CUDA, so torch.where fails with
        # a mixed-device error.  Keep the same epsilon-greedy semantics while
        # creating every sampling tensor next to the policy scores.
        self.exploration.last_timestep = timestep
        epsilon = float(np.asarray(self.exploration.epsilon_schedule(timestep)))
        self.cur_epsilon = epsilon
        actions = flat.argmax(dim=-1)
        if bool(explore):
            valid = action_mask.reshape(-1, self.n_actions).to(
                device=flat.device, dtype=torch.float32
            )
            random_actions = torch.multinomial(valid, 1).squeeze(1)
            choose_random = torch.rand(flat.size(0), device=flat.device) < epsilon
            actions = torch.where(choose_random, random_actions, actions)
        return actions.reshape(scores.shape[:-1])

    def _reshape_model_output(self, output):
        return output

    def _model_step(self, obs, hidden, previous_action_onehot):
        output, hidden = self.model(obs, hidden, previous_action_onehot)
        return self._reshape_model_output(output), hidden

    def _previous_action_onehot(self, prev_action_batch, batch_size):
        if prev_action_batch is None:
            return torch.zeros(
                batch_size,
                self.n_agents,
                self.n_actions,
                dtype=torch.float32,
                device=self.device,
            )
        values = np.asarray(prev_action_batch)
        if values.dtype == object:
            values = np.stack([np.asarray(item) for item in prev_action_batch])
        values = values.reshape(batch_size, self.n_agents).astype(np.int64)
        return F.one_hot(
            torch.as_tensor(values, device=self.device), self.n_actions
        ).float()

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
        del prev_reward_batch, info_batch, episodes, kwargs
        explore = self.config.get("explore", True) if explore is None else explore
        observations, action_mask, _ = self._unpack_observation(obs_batch)
        obs = torch.as_tensor(observations, dtype=torch.float32, device=self.device)
        batch_size = obs.size(0)
        if state_batches:
            hidden = torch.as_tensor(
                np.asarray(state_batches[0]), dtype=torch.float32, device=self.device
            ).reshape(batch_size, self.n_agents, -1)
        else:
            hidden = self.model.initial_state(batch_size, device=self.device)
        previous = self._previous_action_onehot(prev_action_batch, batch_size)
        with torch.no_grad():
            output, next_hidden = self._model_step(obs, hidden, previous)
            scores = self._scores_from_output(output)
            available = torch.as_tensor(
                action_mask, dtype=torch.float32, device=self.device
            )
            invalid_rows = available.sum(dim=-1) <= 0
            if invalid_rows.any():
                raise RuntimeError(
                    f"{self.algorithm_name} received a live observation with no valid action"
                )
            scores = scores.masked_fill(available <= 0.0, -torch.inf)
            if timestep is None:
                timestep = int(1e9)
            actions = self._select_actions(
                scores, available, bool(explore), timestep
            ).reshape(batch_size, self.n_agents).cpu().numpy()
            next_hidden = next_hidden.cpu().numpy()
        return tuple(actions.transpose(1, 0)), [next_hidden], {}

    @override(Policy)
    def compute_log_likelihoods(
        self,
        actions,
        obs_batch,
        state_batches=None,
        prev_action_batch=None,
        prev_reward_batch=None,
    ):
        del actions, state_batches, prev_action_batch, prev_reward_batch
        return np.zeros(len(obs_batch), dtype=np.float32)

    @override(Policy)
    def learn_on_batch(self, samples):
        batch = self._sequence_batch(samples)
        stats = self._learn_sequence_batch(batch)
        self._learner_updates += 1
        target_interval = int(self.config.get("target_network_update_freq", 0))
        target_updated = False
        if (
            target_interval > 0
            and self._learner_updates - self._last_target_update >= target_interval
        ):
            self.update_target()
            self._last_target_update = self._learner_updates
            target_updated = True
        stats["learner_updates"] = self._learner_updates
        stats["target_updated"] = float(target_updated)
        clean = {}
        for key, value in stats.items():
            if torch.is_tensor(value):
                value = value.detach().cpu().item()
            clean[key] = value
        return {LEARNER_STATS_KEY: clean}

    def _learn_sequence_batch(self, batch: GroupedSequenceBatch):
        raise NotImplementedError

    @override(Policy)
    def get_initial_state(self):
        return [
            self.model.initial_state(1, device=self.device)[0].detach().cpu().numpy()
        ]

    @override(Policy)
    def get_weights(self):
        return {
            name: self._cpu_state_dict(module.state_dict())
            for name, module in self._named_modules_for_weights.items()
        }

    @override(Policy)
    def set_weights(self, weights):
        for name, module in self._named_modules_for_weights.items():
            if name in weights:
                module.load_state_dict(self._device_state_dict(weights[name]))

    @override(Policy)
    def get_state(self):
        state = self.get_weights()
        state["cur_epsilon"] = self.cur_epsilon
        state["learner_updates"] = self._learner_updates
        state["last_target_update"] = self._last_target_update
        if self.focus_engine is not None:
            state["focus_runtime_state"] = self.focus_engine.get_runtime_state()
        state["optimizer_states"] = {
            name: copy.deepcopy(optimizer.state_dict())
            for name, optimizer in self._optimizers.items()
        }
        return state

    @override(Policy)
    def set_state(self, state):
        self.set_weights(state)
        self.set_epsilon(state.get("cur_epsilon", self.cur_epsilon))
        self._learner_updates = int(state.get("learner_updates", self._learner_updates))
        self._last_target_update = int(
            state.get("last_target_update", self._last_target_update)
        )
        if self.focus_engine is not None:
            self.focus_engine.set_runtime_state(state.get("focus_runtime_state", {}))
        for name, optimizer_state in state.get("optimizer_states", {}).items():
            if name in self._optimizers:
                self._optimizers[name].load_state_dict(optimizer_state)
                for item in self._optimizers[name].state.values():
                    for key, value in item.items():
                        if torch.is_tensor(value):
                            item[key] = value.to(self.device)

    def set_epsilon(self, epsilon):
        self.cur_epsilon = float(epsilon)

    def update_target(self):
        raise NotImplementedError

    def _sequence_batch(self, samples) -> GroupedSequenceBatch:
        obs, action_mask, state = self._unpack_observation(samples[SampleBatch.CUR_OBS])
        next_obs, next_action_mask, next_state = self._unpack_observation(
            samples[SampleBatch.NEXT_OBS]
        )
        reward = self._group_reward(samples[SampleBatch.INFOS])
        columns = [
            reward,
            action_mask,
            next_action_mask,
            samples[SampleBatch.ACTIONS],
            samples[SampleBatch.DONES],
            obs,
            next_obs,
        ]
        if self.has_env_global_state:
            columns.extend([state, next_state])
        output, _, seq_lens = chop_into_sequences(
            episode_ids=samples[SampleBatch.EPS_ID],
            unroll_ids=samples[SampleBatch.UNROLL_ID],
            agent_indices=samples[SampleBatch.AGENT_INDEX],
            feature_columns=columns,
            state_columns=[],
            max_seq_len=self.config["model"]["max_seq_len"],
            dynamic_max=True,
        )
        if self.has_env_global_state:
            rew, am, nam, act, done, ob, nob, state, next_state = output
        else:
            rew, am, nam, act, done, ob, nob = output
        batch_size, time = len(seq_lens), max(seq_lens)

        def tensor(value, dtype):
            shape = [batch_size, time] + list(value.shape[1:])
            return torch.as_tensor(
                np.reshape(value, shape), dtype=dtype, device=self.device
            )

        obs_t = tensor(ob, torch.float32).reshape(
            batch_size, time, self.n_agents, self.obs_dim
        )
        next_obs_t = tensor(nob, torch.float32).reshape(
            batch_size, time, self.n_agents, self.obs_dim
        )
        if self.has_env_global_state:
            state_t = tensor(state, torch.float32).reshape(batch_size, time, self.state_dim)
            next_state_t = tensor(next_state, torch.float32).reshape(
                batch_size, time, self.state_dim
            )
        else:
            state_t = obs_t.reshape(batch_size, time, -1)
            next_state_t = next_obs_t.reshape(batch_size, time, -1)
        valid = np.arange(time)[None, :] < np.asarray(seq_lens)[:, None]
        mask = torch.as_tensor(valid, dtype=torch.float32, device=self.device).unsqueeze(-1)
        return GroupedSequenceBatch(
            rewards=tensor(rew, torch.float32).reshape(batch_size, time, 1),
            actions=tensor(act, torch.long).reshape(batch_size, time, self.n_agents),
            terminated=tensor(done, torch.float32).reshape(batch_size, time, 1),
            mask=mask,
            obs=obs_t,
            next_obs=next_obs_t,
            action_mask=tensor(am, torch.float32).reshape(
                batch_size, time, self.n_agents, self.n_actions
            ),
            next_action_mask=tensor(nam, torch.float32).reshape(
                batch_size, time, self.n_agents, self.n_actions
            ),
            state=state_t,
            next_state=next_state_t,
            seq_lens=np.asarray(seq_lens),
        )

    def _group_reward(self, info_batch):
        reward = np.asarray(
            [
                info.get(GROUP_REWARDS, [0.0] * self.n_agents)
                if isinstance(info, dict)
                else [0.0] * self.n_agents
                for info in info_batch
            ],
            dtype=np.float32,
        )
        if reward.ndim == 1:
            reward = reward[:, None]
        if reward.shape[-1] != self.n_agents:
            raise ValueError(
                f"expected {self.n_agents} grouped rewards, got {reward.shape}"
            )
        spread = np.max(np.abs(reward - reward.mean(axis=-1, keepdims=True)), axis=-1)
        if self.config.get("strict_shared_reward", True) and np.any(spread > 1e-5):
            raise ValueError(
                f"{self.algorithm_name} requires a cooperative shared reward; "
                f"maximum within-team difference is {spread.max():.6g}"
            )
        return reward.mean(axis=-1, keepdims=True)

    def _unpack_observation(self, obs_batch):
        unpacked = _unpack_obs(
            np.asarray(obs_batch, dtype=np.float32),
            self.observation_space.original_space,
            tensorlib=np,
        )
        if isinstance(unpacked[0], dict):
            local = [np.concatenate(tree.flatten(item["obs"]), axis=1) for item in unpacked]
        else:
            local = unpacked
        obs = np.concatenate(local, axis=1).reshape(
            len(obs_batch), self.n_agents, self.obs_dim
        )
        if self.has_action_mask:
            action_mask = np.concatenate(
                [item["action_mask"] for item in unpacked], axis=1
            ).reshape(len(obs_batch), self.n_agents, self.n_actions)
        else:
            action_mask = np.ones(
                (len(obs_batch), self.n_agents, self.n_actions), dtype=np.float32
            )
        state = None
        if self.has_env_global_state:
            state = np.concatenate(tree.flatten(unpacked[0][ENV_STATE]), axis=1)
        return obs, action_mask, state

    def _device_state_dict(self, state):
        # Ray's object store exposes received NumPy buffers as read-only.  Even
        # though ``load_state_dict`` copies them immediately, ``as_tensor``
        # warns that writes through the resulting tensor would be undefined.
        # Own the memory before moving it to the learner/worker device.
        return {
            key: torch.as_tensor(np.array(value, copy=True), device=self.device)
            for key, value in state.items()
        }

    @staticmethod
    def _cpu_state_dict(state):
        return {key: value.detach().cpu().numpy() for key, value in state.items()}


def safe_action_values(values, action_mask, active):
    """Mask unavailable actions without introducing terminal/padding infinities."""

    live_agents = active.expand(*action_mask.shape[:-1]).bool()
    invalid_live = live_agents & (action_mask.sum(dim=-1) <= 0)
    if invalid_live.any():
        raise RuntimeError("a live transition has no available action")
    safe_mask = action_mask.clone()
    safe_mask[~live_agents] = 1.0
    return values.masked_fill(safe_mask <= 0.0, -torch.inf)


def basic_stats(loss, grad_norm, td_error, q_taken, target, mask):
    denominator = mask.sum().clamp_min(1.0)
    return {
        "loss": loss.detach(),
        "grad_norm": grad_norm if torch.is_tensor(grad_norm) else float(grad_norm),
        "td_error_abs": (td_error.detach().abs() * mask).sum() / denominator,
        "q_taken_mean": (q_taken.detach() * mask).sum() / denominator,
        "target_mean": (target.detach() * mask).sum() / denominator,
    }
