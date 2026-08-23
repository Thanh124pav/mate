"""Local-to-global belief model used by the standalone HMVFE trainer."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class GlobalStateBelief(nn.Module):
    """Predict normalized environment state from HMVFE's local representation."""

    def __init__(self, input_dim: int, state_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.state_dim = int(state_dim)
        self.net = nn.Sequential(
            nn.Linear(self.input_dim, int(hidden_dim)),
            nn.LayerNorm(int(hidden_dim)),
            nn.Tanh(),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.Tanh(),
            nn.Linear(int(hidden_dim), self.state_dim),
            nn.Tanh(),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.size(-1) != self.input_dim:
            raise ValueError(
                f"expected feature dimension {self.input_dim}, got {features.size(-1)}"
            )
        return self.net(features)

    @staticmethod
    def loss(prediction: torch.Tensor, target: torch.Tensor, mode: str = 'smooth_l1') -> torch.Tensor:
        mode = str(mode).lower()
        if mode in ('smooth_l1', 'huber'):
            return nn.functional.smooth_l1_loss(prediction, target)
        if mode in ('mse', 'l2'):
            return (prediction - target).square().mean()
        raise ValueError(f'unknown belief loss mode: {mode}')


__all__ = ['GlobalStateBelief']
