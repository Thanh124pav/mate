"""Shared primitives for the official FOCUS training objective.

This module deliberately contains no MATE geometry.  The responsibility
estimator produces ``rho`` and camera-target gains, while a host algorithm
supplies its local action values and mixer carrier.  Keeping the objective
here makes the QPLEX, DuelMIX, and QMIX adapters use exactly the same
normalisation and gradient-strength rule.
"""

from dataclasses import dataclass

from ray.rllib.utils.framework import try_import_torch


torch, nn = try_import_torch(error=True)
F = torch.nn.functional


class RunningFeatureNormalizer(nn.Module):
    """Per-feature running mean/std with a numerically stable parallel update."""

    def __init__(self, size, clip=5.0, eps=1e-5):
        super().__init__()
        self.size = int(size)
        self.clip = float(clip)
        self.eps = float(eps)
        self.register_buffer("mean", torch.zeros(self.size))
        self.register_buffer("variance", torch.ones(self.size))
        self.register_buffer("count", torch.zeros(()))

    @torch.no_grad()
    def update(self, value):
        flat = value.detach().reshape(-1, self.size)
        if flat.numel() == 0:
            return
        batch_count = flat.new_tensor(float(flat.size(0)))
        batch_mean = flat.mean(dim=0)
        batch_variance = flat.var(dim=0, unbiased=False)
        if self.count.item() == 0:
            self.mean.copy_(batch_mean)
            self.variance.copy_(batch_variance.clamp_min(self.eps))
            self.count.copy_(batch_count)
            return
        delta = batch_mean - self.mean
        total = self.count + batch_count
        new_mean = self.mean + delta * (batch_count / total)
        old_m2 = self.variance * self.count
        batch_m2 = batch_variance * batch_count
        correction = delta.square() * self.count * batch_count / total
        self.mean.copy_(new_mean)
        self.variance.copy_(((old_m2 + batch_m2 + correction) / total).clamp_min(self.eps))
        self.count.copy_(total)

    def forward(self, value, update=None):
        if update is None:
            update = self.training
        if update:
            self.update(value)
        normalized = (value - self.mean) / torch.sqrt(self.variance + self.eps)
        return normalized.clamp(-self.clip, self.clip)


def subset_action_matrix(n_actions, n_targets, *, device=None, dtype=None):
    """Decode integer actions using MATE's most-significant-target-first order."""

    n_actions, n_targets = int(n_actions), int(n_targets)
    if n_actions != 2 ** n_targets:
        raise ValueError(
            "FOCUS action credit requires the target-subset action space: "
            f"n_actions={n_actions}, expected 2**{n_targets}={2 ** n_targets}."
        )
    actions = torch.arange(n_actions, device=device, dtype=torch.long)
    shifts = torch.arange(n_targets - 1, -1, -1, device=device)
    return ((actions.unsqueeze(-1) >> shifts) & 1).to(dtype=dtype or torch.float32)


def _standardize(value, std_floor):
    centered = value - value.mean(dim=-1, keepdim=True)
    std = centered.std(dim=-1, keepdim=True, unbiased=False).clamp_min(float(std_floor))
    return centered / std


def action_credit_loss(
    q_values,
    pair_gains,
    valid,
    *,
    confidence=None,
    signal_weights=None,
    temperature=1.0,
    q_std_floor=0.01,
    eps=1e-8,
):
    """CE(softmax(z(G_i(a))/T), softmax(z(Q_i(a))/T))."""

    if q_values.dim() != 4 or pair_gains.dim() != 4:
        raise ValueError("q_values and pair_gains must have shapes [B,T,C,A/J]")
    if q_values.shape[:3] != pair_gains.shape[:3]:
        raise ValueError("q_values and pair_gains must share [B,T,C]")
    action_bits = subset_action_matrix(
        q_values.size(-1), pair_gains.size(-1), device=q_values.device, dtype=q_values.dtype
    )
    action_gains = torch.einsum("btcj,aj->btca", pair_gains.detach(), action_bits)
    target = torch.softmax(_standardize(action_gains, eps) / float(temperature), dim=-1)
    prediction = torch.log_softmax(
        _standardize(q_values, q_std_floor) / float(temperature), dim=-1
    )
    per_agent = -(target * prediction).sum(dim=-1)
    weights = valid.to(dtype=q_values.dtype)
    if confidence is not None:
        weights = weights * confidence.detach().to(dtype=weights.dtype)
    if signal_weights is not None:
        weights = weights * signal_weights.detach().to(dtype=weights.dtype)
    weights = weights.unsqueeze(-1).expand_as(per_agent)
    denominator = weights.sum()
    if denominator.detach().item() <= eps:
        return q_values.new_zeros(()), action_gains
    return (per_agent * weights).sum() / (denominator + eps), action_gains


def carrier_credit_loss(
    rho,
    carrier,
    valid,
    *,
    confidence=None,
    signal_weights=None,
    eps=1e-8,
):
    """Cross entropy from environmental responsibility to mixer credit."""

    carrier_prob = carrier.clamp_min(eps)
    carrier_prob = carrier_prob / carrier_prob.sum(dim=-1, keepdim=True).clamp_min(eps)
    per_step = -(rho.detach() * torch.log(carrier_prob)).sum(dim=-1)
    weights = valid.to(dtype=per_step.dtype)
    if confidence is not None:
        weights = weights * confidence.detach().to(dtype=weights.dtype)
    if signal_weights is not None:
        weights = weights * signal_weights.detach().to(dtype=weights.dtype)
    denominator = weights.sum()
    if denominator.detach().item() <= eps:
        return per_step.new_zeros(()), carrier_prob
    return (per_step * weights).sum() / (denominator + eps), carrier_prob


def _gradient_norm(loss, parameters):
    if not loss.requires_grad:
        return loss.detach().new_zeros(())
    parameters = [parameter for parameter in parameters if parameter.requires_grad]
    gradients = torch.autograd.grad(
        loss, parameters, retain_graph=True, allow_unused=True, create_graph=False
    )
    squared = [gradient.detach().square().sum() for gradient in gradients if gradient is not None]
    if not squared:
        return loss.detach().new_zeros(())
    return torch.sqrt(torch.stack(squared).sum())


@dataclass
class GradientFloorStats:
    scale: float
    td_norm: float
    action_norm: float
    td_norm_ema: float
    active: bool


class GradientFloor(nn.Module):
    """Raise action-credit gradient to a fraction of the TD gradient EMA."""

    def __init__(self, ratio=0.10, ema_decay=0.9, eps=1e-12):
        super().__init__()
        self.ratio = float(ratio)
        self.ema_decay = float(ema_decay)
        self.eps = float(eps)
        self.register_buffer("td_norm_ema", torch.zeros(()))
        self.register_buffer("num_updates", torch.zeros((), dtype=torch.long))
        self.register_buffer("num_activations", torch.zeros((), dtype=torch.long))

    def scale(self, td_loss, action_loss, parameters):
        parameters = list(parameters)
        td_norm = _gradient_norm(td_loss, parameters)
        action_norm = _gradient_norm(action_loss, parameters)
        with torch.no_grad():
            if self.num_updates.item() == 0:
                self.td_norm_ema.copy_(td_norm)
            else:
                self.td_norm_ema.mul_(self.ema_decay).add_(
                    td_norm, alpha=1.0 - self.ema_decay
                )
            required = self.ratio * self.td_norm_ema
            scale = torch.maximum(
                action_norm.new_ones(()), required / action_norm.clamp_min(self.eps)
            )
            active = bool(scale.item() > 1.0 + 1e-7)
            self.num_updates.add_(1)
            if active:
                self.num_activations.add_(1)
        stats = GradientFloorStats(
            scale=float(scale.item()),
            td_norm=float(td_norm.item()),
            action_norm=float(action_norm.item()),
            td_norm_ema=float(self.td_norm_ema.item()),
            active=active,
        )
        return action_loss * scale.detach(), stats
