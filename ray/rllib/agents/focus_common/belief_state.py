"""Reusable local-to-global belief models for CTDE policy backbones.

The model is deliberately agnostic to the MARL optimizer.  It consumes only an
agent's available feature vector (or a flattened local history) and predicts a
normalized global-state representation.  During training the prediction can be
supervised with the privileged environment state; during execution the
prediction is the only state-like context exposed to the actor.
"""

from __future__ import annotations

from typing import Optional, Tuple

from ray.rllib.utils.framework import try_import_torch


torch, nn = try_import_torch(error=True)


class GlobalStateBelief(nn.Module):
    """MLP/GRU belief estimator with a stable batched API.

    ``features`` may have shape ``[B, D]`` or ``[B, T, D]``.  The latter is
    useful for recurrent policy minibatches.  The returned state prediction has
    the same leading dimensions and is bounded to ``[-1, 1]`` by default,
    matching MATE's normalized state convention.
    """

    def __init__(
        self,
        input_dim: int,
        state_dim: int,
        hidden_dim: int = 128,
        *,
        recurrent: bool = False,
        output_activation: str = "tanh",
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.state_dim = int(state_dim)
        self.hidden_dim = int(hidden_dim)
        self.recurrent = bool(recurrent)
        self.output_activation = str(output_activation).lower()
        if self.input_dim <= 0 or self.state_dim <= 0 or self.hidden_dim <= 0:
            raise ValueError("input_dim, state_dim and hidden_dim must be positive")

        if self.recurrent:
            self.encoder = nn.GRU(
                input_size=self.input_dim,
                hidden_size=self.hidden_dim,
                batch_first=True,
            )
        else:
            self.encoder = nn.Sequential(
                nn.Linear(self.input_dim, self.hidden_dim),
                nn.LayerNorm(self.hidden_dim),
                nn.Tanh(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.Tanh(),
            )
        self.head = nn.Linear(self.hidden_dim, self.state_dim)

    def forward(
        self,
        features: torch.Tensor,
        hidden: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if features.dim() not in (2, 3):
            raise ValueError(
                f"features must have shape [B,D] or [B,T,D], got {tuple(features.shape)}"
            )
        if features.size(-1) != self.input_dim:
            raise ValueError(
                f"expected feature dimension {self.input_dim}, got {features.size(-1)}"
            )

        if self.recurrent:
            sequence = features if features.dim() == 3 else features.unsqueeze(1)
            encoded, next_hidden = self.encoder(sequence, hidden)
            logits = self.head(encoded)
            if features.dim() == 2:
                logits = logits[:, 0]
            return self._activate(logits), next_hidden

        flat = features.reshape(-1, self.input_dim)
        logits = self.head(self.encoder(flat)).reshape(*features.shape[:-1], self.state_dim)
        return self._activate(logits), None

    def _activate(self, value: torch.Tensor) -> torch.Tensor:
        if self.output_activation == "tanh":
            return torch.tanh(value)
        if self.output_activation in ("identity", "none", "linear"):
            return value
        raise ValueError(f"unknown output_activation: {self.output_activation}")

    def loss(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        *,
        mode: str = "smooth_l1",
    ) -> torch.Tensor:
        if prediction.shape != target.shape:
            raise ValueError(
                f"prediction and target shapes must match, got {tuple(prediction.shape)} and {tuple(target.shape)}"
            )
        mode = str(mode).lower()
        if mode in ("smooth_l1", "huber"):
            per_element = nn.functional.smooth_l1_loss(prediction, target, reduction="none")
        elif mode in ("mse", "l2"):
            per_element = (prediction - target).square()
        else:
            raise ValueError(f"unknown belief loss mode: {mode}")
        per_sample = per_element.mean(dim=-1)
        if mask is None:
            return per_sample.mean()
        mask = mask.to(device=per_sample.device, dtype=per_sample.dtype)
        while mask.dim() < per_sample.dim():
            mask = mask.unsqueeze(-1)
        mask = mask.expand_as(per_sample)
        denominator = mask.sum().clamp_min(torch.finfo(per_sample.dtype).eps)
        return (per_sample * mask).sum() / denominator


__all__ = ["GlobalStateBelief"]
