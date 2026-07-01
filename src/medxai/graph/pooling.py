"""Pool a CNN feature map onto superpixel regions -> node features.

The backbone's layer3 map (1024 x 20x20 at 320 input) is bilinearly upsampled to
a mid-resolution grid, the label map is nearest-downsampled to match, and features
are mean-pooled per region. Mid-resolution (default 80x80) guarantees every ~185
regions gets enough cells, avoiding empty nodes from tiny superpixels.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def pool_features_to_regions(
    feat: torch.Tensor, labels: np.ndarray, k: int, pool_size: int = 80
) -> torch.Tensor:
    """feat: (C, h, w) tensor. labels: (H, W) int array with ids 0..k-1.
    Returns (k, C) mean-pooled node features."""
    c = feat.shape[0]
    feat_up = F.interpolate(
        feat.unsqueeze(0), size=(pool_size, pool_size),
        mode="bilinear", align_corners=False,
    ).squeeze(0)                                   # (C, P, P)
    feat_flat = feat_up.reshape(c, -1).t()         # (P*P, C)

    lab = torch.from_numpy(labels.astype(np.int64)).float().view(1, 1, *labels.shape)
    lab_small = (
        F.interpolate(lab, size=(pool_size, pool_size), mode="nearest")
        .long().view(-1).to(feat_flat.device)
    )                                              # (P*P,)

    node = torch.zeros(k, c, dtype=feat_flat.dtype, device=feat_flat.device)
    node.index_add_(0, lab_small, feat_flat)
    counts = torch.bincount(lab_small, minlength=k)
    node = node / counts.clamp(min=1).unsqueeze(1).to(feat_flat.dtype)
    empty = counts == 0
    if empty.any():                                # rare: tiny region lost in downsample
        node[empty] = feat_flat.mean(0)
    return node
