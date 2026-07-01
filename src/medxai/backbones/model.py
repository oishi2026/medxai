"""ResNet-50 multi-label backbone with a layer3 feature-export path.

The manual forward lets us grab the layer3 feature map (the GNN arm pools region
features over it). At 320x320 input, layer3 is 20x20x1024 — a finer spatial grid
than layer4's 10x10, which is why we pool nodes from layer3 (better superpixel
alignment), exactly as recorded in conf/frozen.yaml.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models

# ResNet stage names in forward order, for freezing.
STAGE_ORDER = ["conv1", "bn1", "layer1", "layer2", "layer3", "layer4"]


class ResNet50MultiLabel(nn.Module):
    def __init__(self, num_classes: int = 10, pretrained: bool = True,
                 dropout: float = 0.0):
        super().__init__()
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        self.backbone = models.resnet50(weights=weights)
        self.feature_channels = 1024  # layer3 output channels
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(2048, num_classes),
        )

    def forward(self, x: torch.Tensor, return_features: bool = False):
        b = self.backbone
        x = b.conv1(x)
        x = b.bn1(x)
        x = b.relu(x)
        x = b.maxpool(x)
        x = b.layer1(x)
        x = b.layer2(x)
        f3 = b.layer3(x)            # (B, 1024, H/16, W/16)  -> 20x20 at 320
        x = b.layer4(f3)
        x = b.avgpool(x)
        x = torch.flatten(x, 1)
        logits = b.fc(x)
        if return_features:
            return logits, f3
        return logits


def freeze_until(model: ResNet50MultiLabel, stage: str | None) -> None:
    """Freeze all stages up to and including `stage` (e.g. 'layer2' freezes
    conv1/bn1/layer1/layer2, leaving layer3/layer4/fc trainable). 'none' = no-op.
    layer3 is intentionally left trainable so its features adapt for the GNN arm."""
    if not stage or stage == "none":
        return
    b = model.backbone
    idx = STAGE_ORDER.index(stage)
    for name in STAGE_ORDER[: idx + 1]:
        for p in getattr(b, name).parameters():
            p.requires_grad = False


def set_frozen_bn_eval(model: ResNet50MultiLabel, stage: str | None) -> None:
    """Keep BatchNorm in frozen stages in eval mode so their running stats don't
    drift during training (call after model.train() each epoch)."""
    if not stage or stage == "none":
        return
    b = model.backbone
    idx = STAGE_ORDER.index(stage)
    for name in STAGE_ORDER[: idx + 1]:
        getattr(b, name).eval()
