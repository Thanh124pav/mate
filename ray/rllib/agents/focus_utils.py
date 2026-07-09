import math

from ray.rllib.utils.framework import try_import_torch


torch, _ = try_import_torch(error=True)


FOCUS_CONFIDENCE_DEFAULTS = {
    "confidence_gate_enabled": True,
    "confidence_gate_mode": "auto",
    "confidence_entropy_kappa": 2.0,
    "confidence_loss_threshold": None,
    "confidence_loss_temperature": 1.0,
}


def add_confidence_defaults(focus_config):
    for key, value in FOCUS_CONFIDENCE_DEFAULTS.items():
        focus_config.setdefault(key, value)
    return focus_config


def confidence_mode(focus_config, default_mode):
    if not focus_config.get("confidence_gate_enabled", True):
        return "off"
    mode = str(focus_config.get("confidence_gate_mode", "auto")).lower()
    if mode == "auto":
        return default_mode
    return mode


def entropy_confidence(probs, focus_config):
    eps = focus_config.get("eps", 1e-8)
    num_classes = max(int(probs.size(-1)), 2)
    entropy = -(probs * torch.log(probs + eps)).sum(dim=-1) / math.log(num_classes)
    while entropy.dim() > 2:
        entropy = entropy.mean(dim=-1)
    kappa = float(focus_config.get("confidence_entropy_kappa", 2.0))
    return torch.exp(-kappa * entropy).detach().clamp(0.0, 1.0)


def loss_confidence(per_step_loss, valid, focus_config):
    eps = focus_config.get("eps", 1e-8)
    loss = per_step_loss.detach()
    threshold = focus_config.get("confidence_loss_threshold")
    if threshold is None:
        if valid is not None and valid.any():
            threshold = loss[valid].mean().detach()
        else:
            threshold = loss.mean().detach()
    else:
        threshold = torch.as_tensor(threshold, dtype=loss.dtype, device=loss.device)
    temperature = max(float(focus_config.get("confidence_loss_temperature", 1.0)), eps)
    return torch.sigmoid((threshold - loss) / temperature).detach()


def resolve_confidence(
    focus_config,
    valid,
    reference,
    default_mode,
    probs=None,
    per_step_loss=None,
):
    mode = confidence_mode(focus_config, default_mode)
    if mode == "off":
        confidence = torch.ones_like(reference, dtype=reference.dtype, device=reference.device)
    elif mode == "entropy" and probs is not None:
        confidence = entropy_confidence(probs, focus_config)
    elif mode == "loss" and per_step_loss is not None:
        confidence = loss_confidence(per_step_loss, valid, focus_config)
    else:
        confidence = torch.ones_like(reference, dtype=reference.dtype, device=reference.device)
        mode = "off"
    return confidence.to(dtype=reference.dtype, device=reference.device), mode


def gated_focus_loss(per_step_loss, valid, confidence, reference_loss, eps=1e-8, weights=None):
    if not valid.any():
        return torch.zeros_like(reference_loss)
    if weights is None:
        weights = torch.ones_like(per_step_loss)
    gated_weights = weights * confidence.detach()
    valid_weights = gated_weights[valid]
    denom = valid_weights.sum()
    if denom.detach().item() <= eps:
        return torch.zeros_like(reference_loss)
    return (per_step_loss[valid] * valid_weights).sum() / (denom + eps)


def confidence_stats(confidence, valid, mode):
    values = confidence[valid] if valid is not None and valid.any() else confidence.reshape(-1)
    return {
        "focus_confidence_mean": values.mean().detach().item(),
        "focus_confidence_min": values.min().detach().item(),
        "focus_confidence_max": values.max().detach().item(),
        "focus_confidence_mode": mode,
    }
