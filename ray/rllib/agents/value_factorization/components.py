"""Neural and mathematical components shared by the new CTDE algorithms."""

import math
from typing import Callable, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _prod(shape) -> int:
    return int(np.prod(shape))


def forward_preserving_scale(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Return ``x`` in the forward pass while scaling its backward gradient.

    ``weight`` must be detached by the caller (or be independent of trainable
    parameters).  FOCUS uses this operation to redistribute agent credit while
    preserving the backbone's exact scalar objective and CTDE semantics.
    """

    while weight.dim() < x.dim():
        weight = weight.unsqueeze(-1)
    return x.detach() + weight * (x - x.detach())


def focus_agent_weights(
    rho: torch.Tensor,
    n_agents: int,
    eta: float,
    confidence: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Convert responsibilities to positive, mean-one agent gradient weights."""

    rho = rho.detach()
    if confidence is None:
        eta_t = rho.new_tensor(float(eta))
    else:
        eta_t = float(eta) * confidence.detach().unsqueeze(-1)
    return 1.0 - eta_t + eta_t * int(n_agents) * rho


def masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.to(dtype=value.dtype)
    while mask.dim() < value.dim():
        mask = mask.unsqueeze(-1)
    mask = mask.expand_as(value)
    return (value * mask).sum() / mask.sum().clamp_min(1.0)


def td_lambda_targets(
    rewards: torch.Tensor,
    terminated: torch.Tensor,
    mask: torch.Tensor,
    target_next: torch.Tensor,
    gamma: float,
    td_lambda: float,
) -> torch.Tensor:
    """Build finite-horizon TD(lambda) targets for transition-aligned tensors.

    All inputs have time dimension ``T`` and the bootstrap value at transition
    ``t`` is ``target_next[:, t] == Q_target(s_{t+1})``.  Padded rows are
    ignored and terminals never bootstrap.
    """

    if rewards.shape != terminated.shape or rewards.shape != target_next.shape:
        raise ValueError(
            "rewards, terminated, and target_next must have identical shapes; "
            f"got {rewards.shape}, {terminated.shape}, {target_next.shape}"
        )
    if mask.shape != rewards.shape:
        mask = mask.expand_as(rewards)

    result = torch.zeros_like(target_next)
    running = target_next[:, -1]
    for t in range(rewards.size(1) - 1, -1, -1):
        one_step = rewards[:, t] + float(gamma) * (
            1.0 - terminated[:, t]
        ) * target_next[:, t]
        if t == rewards.size(1) - 1:
            running = one_step
        else:
            recursive = rewards[:, t] + float(gamma) * (
                1.0 - terminated[:, t]
            ) * running
            running = (1.0 - float(td_lambda)) * one_step + float(td_lambda) * recursive
        running = running * mask[:, t]
        result[:, t] = running
    return result


class RecurrentAgent(nn.Module):
    """Shared-parameter decentralized recurrent agent.

    The input can include the previous one-hot action and agent identity, as in
    the official PyMARL configurations.  Neither global state nor future data
    is accepted by this module.
    """

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        n_agents: int,
        hidden_dim: int = 64,
        output_dim: Optional[int] = None,
        include_last_action: bool = True,
        include_agent_id: bool = True,
    ):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.n_actions = int(n_actions)
        self.n_agents = int(n_agents)
        self.hidden_dim = int(hidden_dim)
        self.include_last_action = bool(include_last_action)
        self.include_agent_id = bool(include_agent_id)
        input_dim = self.obs_dim
        if self.include_last_action:
            input_dim += self.n_actions
        if self.include_agent_id:
            input_dim += self.n_agents
        self.fc1 = nn.Linear(input_dim, self.hidden_dim)
        self.rnn = nn.GRUCell(self.hidden_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, int(output_dim or self.n_actions))

    def initial_state(self, batch_size: int, device=None, dtype=None) -> torch.Tensor:
        ref = self.fc1.weight
        return torch.zeros(
            int(batch_size),
            self.n_agents,
            self.hidden_dim,
            device=device or ref.device,
            dtype=dtype or ref.dtype,
        )

    def forward(
        self,
        obs: torch.Tensor,
        hidden: torch.Tensor,
        previous_action: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if obs.shape[-2:] != (self.n_agents, self.obs_dim):
            raise ValueError(
                f"expected obs [...,{self.n_agents},{self.obs_dim}], got {obs.shape}"
            )
        pieces = [obs]
        if self.include_last_action:
            if previous_action is None:
                previous_action = obs.new_zeros(*obs.shape[:-1], self.n_actions)
            pieces.append(previous_action.to(dtype=obs.dtype))
        if self.include_agent_id:
            ids = torch.eye(
                self.n_agents, dtype=obs.dtype, device=obs.device
            ).view(*([1] * (obs.dim() - 2)), self.n_agents, self.n_agents)
            ids = ids.expand(*obs.shape[:-2], self.n_agents, self.n_agents)
            pieces.append(ids)
        x = torch.cat(pieces, dim=-1)
        leading = x.shape[:-1]
        x = F.relu(self.fc1(x.reshape(-1, x.size(-1))))
        h = self.rnn(x, hidden.reshape(-1, self.hidden_dim))
        out = self.fc2(h).reshape(*leading, -1)
        return out, h.reshape(*leading, self.hidden_dim)


def unroll_current_and_next(
    model: RecurrentAgent,
    obs: torch.Tensor,
    next_obs: torch.Tensor,
    actions: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Unroll current histories and evaluate each corresponding next history.

    The next observation at ``t`` is evaluated from the hidden state *after*
    consuming current observation ``t`` and with current action ``a_t`` as its
    previous action.  The branch is not fed back into the current trajectory,
    avoiding both a reset-at-every-transition bug and future-state leakage.
    """

    batch, time, n_agents = obs.shape[:3]
    if n_agents != model.n_agents:
        raise ValueError("agent dimension does not match model")
    hidden = model.initial_state(batch, device=obs.device, dtype=obs.dtype)
    previous = obs.new_zeros(batch, n_agents, model.n_actions)
    current_outputs, next_outputs, hidden_outputs = [], [], []
    for t in range(time):
        current, hidden = model(obs[:, t], hidden, previous)
        action_onehot = F.one_hot(actions[:, t].long(), model.n_actions).to(obs.dtype)
        nxt, _ = model(next_obs[:, t], hidden, action_onehot)
        current_outputs.append(current)
        next_outputs.append(nxt)
        hidden_outputs.append(hidden)
        previous = action_onehot
    return (
        torch.stack(current_outputs, dim=1),
        torch.stack(next_outputs, dim=1),
        torch.stack(hidden_outputs, dim=1),
    )


class MonotonicMixer(nn.Module):
    """QMIX monotonic hypernetwork mixer used by FACMAC and ResQ."""

    def __init__(
        self,
        n_agents: int,
        state_dim: int,
        embed_dim: int = 32,
        hypernet_layers: int = 2,
        hypernet_embed: int = 64,
        skip_connections: bool = False,
        gated: bool = False,
    ):
        super().__init__()
        self.n_agents = int(n_agents)
        self.state_dim = int(state_dim)
        self.embed_dim = int(embed_dim)
        if hypernet_layers == 1:
            self.hyper_w_1 = nn.Linear(self.state_dim, self.n_agents * self.embed_dim)
            self.hyper_w_final = nn.Linear(self.state_dim, self.embed_dim)
        elif hypernet_layers == 2:
            self.hyper_w_1 = nn.Sequential(
                nn.Linear(self.state_dim, int(hypernet_embed)),
                nn.ReLU(),
                nn.Linear(int(hypernet_embed), self.n_agents * self.embed_dim),
            )
            self.hyper_w_final = nn.Sequential(
                nn.Linear(self.state_dim, int(hypernet_embed)),
                nn.ReLU(),
                nn.Linear(int(hypernet_embed), self.embed_dim),
            )
        else:
            raise ValueError("QMIX supports one or two hypernetwork layers")
        self.hyper_b_1 = nn.Linear(self.state_dim, self.embed_dim)
        self.v = nn.Sequential(
            nn.Linear(self.state_dim, self.embed_dim), nn.ReLU(), nn.Linear(self.embed_dim, 1)
        )
        self.skip_connections = bool(skip_connections)
        self.gated = bool(gated)
        if self.gated:
            self.gate = nn.Parameter(torch.full((1,), 0.5))

    def forward(self, agent_qs: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
        prefix = agent_qs.shape[:-1]
        if agent_qs.size(-1) != self.n_agents:
            raise ValueError(f"expected {self.n_agents} agent values, got {agent_qs.shape}")
        flat_q = agent_qs.reshape(-1, 1, self.n_agents)
        flat_s = states.reshape(-1, self.state_dim)
        w1 = self.hyper_w_1(flat_s).abs().view(-1, self.n_agents, self.embed_dim)
        b1 = self.hyper_b_1(flat_s).view(-1, 1, self.embed_dim)
        hidden = F.elu(torch.bmm(flat_q, w1) + b1)
        wf = self.hyper_w_final(flat_s).abs().view(-1, self.embed_dim, 1)
        mixed = torch.bmm(hidden, wf)
        if self.gated:
            mixed = mixed * self.gate
        mixed = mixed + self.v(flat_s).view(-1, 1, 1)
        if self.skip_connections:
            mixed = mixed + flat_q.sum(dim=-1, keepdim=True)
        return mixed.view(*prefix, 1)


class FACMACCritic(nn.Module):
    """Official discrete FACMAC per-agent action critic (two ReLU layers)."""

    def __init__(self, obs_dim: int, n_actions: int, hidden_dim: int = 64):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.n_actions = int(n_actions)
        self.fc1 = nn.Linear(self.obs_dim + self.n_actions, int(hidden_dim))
        self.fc2 = nn.Linear(int(hidden_dim), int(hidden_dim))
        self.fc3 = nn.Linear(int(hidden_dim), 1)

    def forward(self, obs: torch.Tensor, actions_onehot: torch.Tensor) -> torch.Tensor:
        x = torch.cat([obs, actions_onehot.to(obs.dtype)], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x).squeeze(-1)


class UnrestrictedMixer(nn.Module):
    """Unrestricted centralized MLP over agent utilities and state."""

    def __init__(
        self,
        n_agents: int,
        state_dim: int,
        hidden_dims: Sequence[int] = (64, 64),
        include_state_value: bool = False,
    ):
        super().__init__()
        self.n_agents = int(n_agents)
        self.state_dim = int(state_dim)
        layers = []
        width = self.n_agents + self.state_dim
        for hidden in hidden_dims:
            layers.extend([nn.Linear(width, int(hidden)), nn.ReLU()])
            width = int(hidden)
        layers.append(nn.Linear(width, 1))
        self.net = nn.Sequential(*layers)
        self.state_value = (
            nn.Sequential(
                nn.Linear(self.state_dim, int(hidden_dims[0])),
                nn.ReLU(),
                nn.Linear(int(hidden_dims[0]), 1),
            )
            if include_state_value
            else None
        )

    def forward(self, agent_values: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
        x = torch.cat(
            [agent_values.reshape(-1, self.n_agents), states.reshape(-1, self.state_dim)],
            dim=-1,
        )
        out = self.net(x)
        if self.state_value is not None:
            out = out + self.state_value(states.reshape(-1, self.state_dim))
        return out.view(*agent_values.shape[:-1], 1)


def _normal_cdf(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def _normal_icdf(x: torch.Tensor) -> torch.Tensor:
    return math.sqrt(2.0) * torch.erfinv(2.0 * x - 1.0)


def risk_distortion_weights(
    n_quantiles: int,
    risk_type: str = "wang",
    risk_param: float = 0.75,
    device=None,
    dtype=torch.float32,
    derivative_clip: float = 5.0,
) -> torch.Tensor:
    """RiskQ midpoint quadrature weights for neutral/CVaR/Wang/CPW risk."""

    n = int(n_quantiles)
    tau = (torch.arange(n, device=device, dtype=dtype) + 0.5) / n
    kind = str(risk_type).lower()
    alpha = float(risk_param)
    # The official implementation reflects tau for negative distortion
    # parameters and evaluates the positive-parameter derivative there.
    tau_eval = 1.0 - tau if alpha < 0.0 else tau
    alpha = abs(alpha)
    if kind in ("neutral", "mean", "expectation"):
        derivative = torch.ones_like(tau)
    elif kind == "wang":
        z = _normal_icdf(tau_eval.clamp(1e-6, 1.0 - 1e-6))
        derivative = torch.exp(-alpha * z - 0.5 * alpha * alpha)
    elif kind in ("cvar", "cvar_upper"):
        if not 0.0 < alpha <= 1.0:
            raise ValueError("CVaR alpha must be in (0, 1]")
        derivative = (tau_eval < alpha).to(dtype) / alpha
    elif kind in ("cpw", "cumulative_prospect"):
        if alpha <= 0.0:
            raise ValueError("CPW alpha must be positive")
        g = tau_eval.pow(alpha)
        base = tau_eval.pow(alpha) + (1.0 - tau_eval).pow(alpha)
        h = base.pow(1.0 / alpha)
        g_prime = alpha * tau_eval.pow(alpha - 1.0)
        h_prime = base.pow(1.0 / alpha - 1.0) * (
            tau_eval.pow(alpha - 1.0) - (1.0 - tau_eval).pow(alpha - 1.0)
        )
        derivative = (g_prime * h - g * h_prime) / h.square().clamp_min(1e-12)
    elif kind == "var":
        index = max(0, min(n - 1, int(math.floor((alpha - 0.5 / n) * n))))
        derivative = torch.zeros_like(tau)
        derivative[index] = float(n)
    else:
        raise ValueError(f"unknown RiskQ distortion: {risk_type}")
    derivative = derivative.clamp(min=0.0, max=float(derivative_clip))
    # RiskQ multiplies g'(tau_hat) by the uniform quantile mass 1/N.  Do not
    # renormalize the finite midpoint quadrature: that would define a
    # different distorted action value, especially for CVaR.
    return derivative / float(n)


def distorted_expectation(
    quantiles: torch.Tensor, risk_type: str, risk_param: float
) -> torch.Tensor:
    weights = risk_distortion_weights(
        quantiles.size(-1), risk_type, risk_param, quantiles.device, quantiles.dtype
    )
    return (quantiles * weights).sum(dim=-1)


def quantile_huber_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    kappa: float = 1.0,
) -> torch.Tensor:
    """Pairwise QR-Huber loss with explicit current/target atom axes."""

    if predicted.size(-1) <= 0 or target.size(-1) <= 0:
        raise ValueError("quantile tensors must have a non-empty final axis")
    delta = target.unsqueeze(-2) - predicted.unsqueeze(-1)
    absolute = delta.abs()
    k = float(kappa)
    huber = torch.where(absolute <= k, 0.5 * delta.square(), k * (absolute - 0.5 * k))
    n = predicted.size(-1)
    tau = (
        (torch.arange(n, device=predicted.device, dtype=predicted.dtype) + 0.5) / n
    ).view(*([1] * (delta.dim() - 2)), n, 1)
    weight = (tau - (delta.detach() < 0.0).to(predicted.dtype)).abs()
    per_item = (weight * huber / k).mean(dim=-1).sum(dim=-1)
    if mask is None:
        return per_item.mean()
    return masked_mean(per_item, mask)


class RiskQMixer(nn.Module):
    """RiskQ state-attention distributional mixer with nonnegative weights."""

    def __init__(
        self,
        n_agents: int,
        state_dim: int,
        agent_key_dim: int,
        n_heads: int = 1,
        query_hidden: int = 32,
        attention_dim: int = 16,
        key_embedding_dim: Optional[int] = None,
        head_hidden: int = 32,
        constant_hidden: int = 16,
    ):
        super().__init__()
        self.n_agents = int(n_agents)
        self.state_dim = int(state_dim)
        self.agent_key_dim = int(agent_key_dim)
        self.n_heads = int(n_heads)
        self.attention_dim = int(attention_dim)
        self.key_embedding_dim = int(key_embedding_dim or attention_dim)
        if self.key_embedding_dim != self.attention_dim:
            raise ValueError(
                "RiskQ query and key embedding dimensions must match for "
                "scaled dot-product attention"
            )
        self.queries = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(self.state_dim, int(query_hidden)),
                    nn.ReLU(),
                    nn.Linear(int(query_hidden), self.attention_dim),
                )
                for _ in range(self.n_heads)
            ]
        )
        self.keys = nn.ModuleList(
            [
                nn.Linear(self.agent_key_dim, self.key_embedding_dim)
                for _ in range(self.n_heads)
            ]
        )
        self.head_weight = nn.Sequential(
            nn.Linear(self.state_dim, int(head_hidden)),
            nn.ReLU(),
            nn.Linear(int(head_hidden), self.n_heads),
        )
        self.constant = nn.Sequential(
            nn.Linear(self.state_dim, int(constant_hidden)),
            nn.ReLU(),
            nn.Linear(int(constant_hidden), 1),
        )

    def coefficients(
        self, states: torch.Tensor, agent_keys: torch.Tensor
    ) -> torch.Tensor:
        flat_s = states.reshape(-1, self.state_dim)
        flat_k = agent_keys.reshape(-1, self.n_agents, self.agent_key_dim)
        coefficients = flat_s.new_zeros(flat_s.size(0), self.n_agents)
        head_weights = self.head_weight(flat_s).abs()
        for head, (query_net, key_net) in enumerate(zip(self.queries, self.keys)):
            query = query_net(flat_s).unsqueeze(1)
            keys = key_net(flat_k)
            logits = (query * keys).sum(dim=-1) / math.sqrt(self.attention_dim)
            attention = F.softmax(logits, dim=-1)
            coefficients = coefficients + attention * head_weights[:, head : head + 1]
        return coefficients.view(*states.shape[:-1], self.n_agents)

    def forward(
        self,
        agent_quantiles: torch.Tensor,
        states: torch.Tensor,
        agent_keys: torch.Tensor,
        coefficient_scale: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        coefficients = self.coefficients(states, agent_keys)
        if coefficient_scale is not None:
            coefficients = coefficients * coefficient_scale
        mixed = (agent_quantiles * coefficients.unsqueeze(-1)).sum(dim=-2)
        constant = self.constant(states.reshape(-1, self.state_dim)).view(
            *states.shape[:-1], 1
        )
        return mixed + constant


class _PositiveHyperLayer(nn.Module):
    def __init__(self, state_dim: int, in_features: int, out_features: int, hidden: int):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        # ConcaveQ specifies a single linear hypernetwork followed by an
        # absolute activation for each state-conditioned weight matrix.
        self.weight = nn.Linear(
            int(state_dim), self.in_features * self.out_features
        )
        self.bias = nn.Linear(int(state_dim), self.out_features)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize a deep state-conditioned ICNN without changing its form.

        A default linear initialization gives every generated matrix entry
        O(1).  Repeated positive matrix products then grow with the ICNN width
        (64, 64, and 192 in ConcaveQ) before the first optimizer step.  Scaling
        only the *initial* hypernetwork parameters by the generated matrix's
        fan-in keeps row sums O(1); the learned matrices remain unrestricted
        apart from the paper's non-negativity constraint.
        """

        nn.init.xavier_uniform_(
            self.weight.weight, gain=1.0 / float(self.in_features)
        )
        nn.init.zeros_(self.weight.bias)
        nn.init.xavier_uniform_(self.bias.weight, gain=0.1)
        nn.init.zeros_(self.bias.bias)

    def forward(self, x: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        flat_x = x.reshape(-1, self.in_features)
        flat_s = state.reshape(flat_x.size(0), -1)
        # This is a pure reparameterization of a non-negative matrix: the
        # linear hypernetwork can absorb the constant factor, so the set of
        # representable W(s) is unchanged while updates no longer grow with
        # the dense ICNN fan-in.
        weights = (
            self.weight(flat_s).abs() / float(self.in_features)
        ).view(-1, self.in_features, self.out_features)
        bias = self.bias(flat_s).view(-1, 1, self.out_features)
        return (torch.bmm(flat_x.unsqueeze(1), weights) + bias).squeeze(1).view(
            *x.shape[:-1], self.out_features
        )

    def negative_weight_forward(self, x: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """Compute ``-W(s)x + b(s)`` for the final concave layer."""

        flat_x = x.reshape(-1, self.in_features)
        flat_s = state.reshape(flat_x.size(0), -1)
        weights = (
            self.weight(flat_s).abs() / float(self.in_features)
        ).view(-1, self.in_features, self.out_features)
        bias = self.bias(flat_s).view(-1, 1, self.out_features)
        return (-torch.bmm(flat_x.unsqueeze(1), weights) + bias).squeeze(1).view(
            *x.shape[:-1], self.out_features
        )


class ConcaveMixer(nn.Module):
    """Four-layer input-concave state-conditioned ConcaveQ mixer.

    Dense skip concatenation follows the paper: every hidden layer consumes all
    earlier activations.  Positive hypernetwork weights and a final negation
    make the output concave in the local utility vector.
    """

    def __init__(
        self,
        n_agents: int,
        state_dim: int,
        hidden_dims: Sequence[int] = (64, 64, 64),
        hypernet_hidden: int = 64,
    ):
        super().__init__()
        if len(hidden_dims) != 3:
            raise ValueError("ConcaveQ uses exactly three hidden layers before output")
        self.n_agents = int(n_agents)
        self.state_dim = int(state_dim)
        h1, h2, h3 = map(int, hidden_dims)
        self.layer1 = _PositiveHyperLayer(state_dim, self.n_agents, h1, hypernet_hidden)
        self.layer2 = _PositiveHyperLayer(state_dim, h1, h2, hypernet_hidden)
        self.layer3 = _PositiveHyperLayer(state_dim, h1 + h2, h3, hypernet_hidden)
        self.output_layer = _PositiveHyperLayer(
            state_dim, h1 + h2 + h3, 1, hypernet_hidden
        )

    def forward(self, utilities: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
        z1 = self.layer1(utilities, states)
        z2 = F.relu(self.layer2(z1, states))
        z3 = F.relu(self.layer3(torch.cat([z1, z2], dim=-1), states))
        return self.output_layer.negative_weight_forward(
            torch.cat([z1, z2, z3], dim=-1), states
        )


def coordinate_ascent_actions(
    action_values: torch.Tensor,
    action_mask: torch.Tensor,
    evaluate_joint: Callable[[torch.Tensor], torch.Tensor],
    initial_actions: Optional[torch.Tensor] = None,
    candidate_chunk_size: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """One exact asynchronous coordinate sweep from ConcaveQ Algorithm 1.

    ``evaluate_joint`` receives ``[..., n_agents]`` integer joint actions and
    returns ``[..., 1]`` values.  Each agent's coordinate is exhaustively
    evaluated and the current joint action is immediately updated.
    """

    n_actions = action_values.size(-1)
    n_agents = action_values.size(-2)
    safe = action_values.masked_fill(action_mask <= 0.0, -torch.inf)
    if initial_actions is None:
        actions = safe.argmax(dim=-1)
    else:
        actions = initial_actions.long().clone()
    best = evaluate_joint(actions).squeeze(-1)
    chunk_size = n_actions if candidate_chunk_size is None else int(candidate_chunk_size)
    if chunk_size <= 0:
        raise ValueError("candidate_chunk_size must be positive")
    for agent in range(n_agents):
        chosen_value = torch.full_like(best, -torch.inf)
        chosen_action = torch.zeros_like(actions[..., agent])
        # Exhaustive evaluation is unchanged; chunks only bound the temporary
        # [..., n_actions, n_agents] tensor used by large HRL action spaces.
        for start in range(0, n_actions, chunk_size):
            stop = min(start + chunk_size, n_actions)
            count = stop - start
            candidates = actions.unsqueeze(-2).expand(
                *actions.shape[:-1], count, n_agents
            ).clone()
            values = torch.arange(start, stop, device=actions.device)
            values = values.view(*([1] * (candidates.dim() - 2)), count)
            candidates[..., agent] = values
            candidate_values = evaluate_joint(candidates).squeeze(-1)
            candidate_values = candidate_values.masked_fill(
                action_mask[..., agent, start:stop] <= 0.0, -torch.inf
            )
            chunk_value, chunk_index = candidate_values.max(dim=-1)
            improve_chunk = chunk_value > chosen_value
            chosen_value = torch.where(improve_chunk, chunk_value, chosen_value)
            chosen_action = torch.where(
                improve_chunk, chunk_index + start, chosen_action
            )
        accept = chosen_value > best
        actions[..., agent] = torch.where(accept, chosen_action, actions[..., agent])
        best = torch.where(accept, chosen_value, best)
    return actions, best.unsqueeze(-1)
