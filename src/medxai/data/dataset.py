"""CheXpert chest dataset + transforms, driven by the split manifests.

Reads chest_train.csv / chest_val.csv (produced by medxai.data.build_splits),
resolves image paths under an image root, returns (image_tensor, label_vector).
Images are grayscale X-rays converted to 3-channel for ImageNet-pretrained nets.
"""
from __future__ import annotations

import os
from typing import Sequence

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_transforms(resolution: int, train: bool, hflip: bool = False,
                     strong_aug: bool = False):
    """Mild augmentation by default. NOTE: horizontal flip is OFF for chest —
    cardiac silhouette / situs is left-right asymmetric, so flipping can teach
    the wrong anatomy. strong_aug adds translation + mild contrast jitter to
    regularize when overfitting (anatomy-safe magnitudes)."""
    if train:
        ops = [transforms.Resize((resolution, resolution))]
        if strong_aug:
            ops += [
                transforms.RandomResizedCrop(
                    resolution, scale=(0.85, 1.0), ratio=(0.9, 1.1)
                ),
                transforms.RandomAffine(degrees=7, translate=(0.05, 0.05)),
                transforms.ColorJitter(brightness=0.1, contrast=0.1),
            ]
        else:
            ops += [
                transforms.RandomResizedCrop(
                    resolution, scale=(0.9, 1.0), ratio=(0.95, 1.05)
                ),
                transforms.RandomRotation(7),
            ]
        if hflip:
            ops.append(transforms.RandomHorizontalFlip())
    else:
        ops = [transforms.Resize((resolution, resolution))]
    ops += [
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
    return transforms.Compose(ops)


class ChestDataset(Dataset):
    def __init__(
        self,
        manifest_csv: str,
        image_root: str,
        label_cols: Sequence[str],
        resolution: int,
        train: bool,
        path_col: str = "Path",
        hflip: bool = False,
        verify: bool = True,
        strip_prefix: str = "",
        strong_aug: bool = False,
    ):
        self.df = pd.read_csv(manifest_csv).reset_index(drop=True)
        self.image_root = image_root
        self.label_cols = list(label_cols)
        self.path_col = path_col if path_col in self.df.columns else self.df.columns[0]
        # Some CheXpert Kaggle mirrors omit the 'CheXpert-v1.0-small/' folder that
        # the manifest paths carry. strip_prefix removes that leading segment so
        # image_root + (Path - prefix) resolves on disk. No CSV edit needed.
        self.strip_prefix = strip_prefix
        self.tf = build_transforms(resolution, train, hflip, strong_aug)

        if verify:  # fail fast with a helpful message if the root is wrong
            sample = self._full_path(self.df.iloc[0][self.path_col])
            if not os.path.exists(sample):
                raise FileNotFoundError(
                    f"Image not found:\n  {sample}\n"
                    f"Check --image_root / --strip_prefix. Manifest path is "
                    f"'{self.df.iloc[0][self.path_col]}'. Run e.g. "
                    f"`find {image_root} -name '*.jpg' | head` to see the real layout."
                )

    def _full_path(self, rel: str) -> str:
        rel = str(rel)
        if self.strip_prefix and rel.startswith(self.strip_prefix):
            rel = rel[len(self.strip_prefix):].lstrip("/")
        return os.path.join(self.image_root, rel)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img = Image.open(self._full_path(row[self.path_col])).convert("RGB")
        x = self.tf(img)
        y = torch.tensor(
            [float(row[c]) for c in self.label_cols], dtype=torch.float32
        )
        return x, y


def compute_pos_weight(
    manifest_csv: str, label_cols: Sequence[str], clamp_max: float = 10.0
) -> torch.Tensor:
    """pos_weight_c = negatives/positives, clamped so ultra-rare classes don't
    destabilize training. Used by weighted BCE."""
    df = pd.read_csv(manifest_csv)
    pos = df[list(label_cols)].sum().clip(lower=1)
    neg = len(df) - pos
    w = (neg / pos).clip(upper=clamp_max)
    return torch.tensor(w.values, dtype=torch.float32)
