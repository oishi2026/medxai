"""Two losses for the imbalanced multi-label chest task, toggled via config.

- WeightedBCE: BCE with per-class pos_weight = neg/pos (clamped). Simple, standard.
- FocalLoss: down-weights easy examples; better for the rare pathologies.
Pick one via --loss {bce,focal}.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedBCE(nn.Module):
    def __init__(self, pos_weight: torch.Tensor | None = None):
        super().__init__()
        if pos_weight is not None:
            self.register_buffer("pos_weight", pos_weight)
        else:
            self.pos_weight = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weight
        )


class FocalLoss(nn.Module):
    """Multi-label sigmoid focal loss (Lin et al. 2017), per-class."""

    def __init__(self, gamma: float = 2.0, alpha: float = 0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p = torch.sigmoid(logits)
        p_t = p * targets + (1 - p) * (1 - targets)
        focal = (1 - p_t) ** self.gamma * bce
        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            focal = alpha_t * focal
        return focal.mean()


def build_loss(name: str, pos_weight: torch.Tensor | None = None) -> nn.Module:
    name = name.lower()
    if name == "bce":
        return WeightedBCE(pos_weight=pos_weight)
    if name == "focal":
        return FocalLoss()
    raise ValueError(f"Unknown loss {name!r}; use 'bce' or 'focal'.")
