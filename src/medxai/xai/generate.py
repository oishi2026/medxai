"""Generate and visualize post-hoc explanations for the chest backbone.

Quick visual check (the milestone): load best.ckpt, explain a few validation
images with Grad-CAM++ and Integrated Gradients, save side-by-side overlays you
can open, plus the raw maps as .npz for the later faithfulness stage.

Example (Kaggle):
    python -m medxai.xai.generate \
        --ckpt        /kaggle/input/medxai-chest-resnet50-v1/best.ckpt \
        --chest_val   /kaggle/input/datasets/eemiir/medxai-splits-v1/chest_val.csv \
        --image_root  /kaggle/input/datasets/ashery/chexpert \
        --strip_prefix "CheXpert-v1.0-small/" \
        --out_dir     /kaggle/working/outputs/xai \
        --num_samples 8
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import torch
import yaml

from medxai.data.dataset import IMAGENET_MEAN, IMAGENET_STD, ChestDataset
from medxai.data.splits import CHEXLOCALIZE_PATHOLOGIES
from medxai.utils.determinism import set_determinism
from medxai.xai.explainers import (
    choose_target_classes,
    gradcampp_maps,
    integrated_gradients_maps,
    load_backbone,
)


def _denorm(t: torch.Tensor) -> np.ndarray:
    """Undo ImageNet normalization for display -> HxWx3 in [0,1]."""
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    img = (t.cpu() * std + mean).clamp(0, 1)
    return img.permute(1, 2, 0).numpy()


def _overlay(ax, base_img, heat, title):
    ax.imshow(base_img)
    ax.imshow(heat, cmap="jet", alpha=0.45)
    ax.set_title(title, fontsize=9)
    ax.axis("off")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--chest_val", required=True)
    ap.add_argument("--image_root", required=True)
    ap.add_argument("--strip_prefix", default="")
    ap.add_argument("--out_dir", default="/kaggle/working/outputs/xai")
    ap.add_argument("--frozen", default="conf/frozen.yaml")
    ap.add_argument("--num_samples", type=int, default=8)
    ap.add_argument("--ig_steps", type=int, default=32)
    args = ap.parse_args()

    with open(args.frozen) as f:
        cfg = yaml.safe_load(f)
    seed = cfg["seeds"][0]
    set_determinism(seed)
    resolution = cfg["input_resolution"]["chest"]
    label_cols = CHEXLOCALIZE_PATHOLOGIES
    device = "cuda" if torch.cuda.is_available() else "cpu"
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    ds = ChestDataset(args.chest_val, args.image_root, label_cols, resolution,
                      train=False, strip_prefix=args.strip_prefix)
    n = min(args.num_samples, len(ds))
    imgs = torch.stack([ds[i][0] for i in range(n)]).to(device)
    labels = np.stack([ds[i][1].numpy() for i in range(n)])

    model = load_backbone(args.ckpt, num_classes=len(label_cols), device=device)
    with torch.no_grad():
        probs = torch.sigmoid(model(imgs)).cpu().numpy()
    targets = choose_target_classes(probs, labels)
    print("explained classes:",
          [label_cols[c] for c in targets])

    cam = gradcampp_maps(model, imgs, targets)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    ig = integrated_gradients_maps(model, imgs, targets, n_steps=args.ig_steps)

    # save raw maps for the faithfulness stage
    np.savez_compressed(
        os.path.join(args.out_dir, "maps.npz"),
        gradcampp=cam, integrated_gradients=ig,
        targets=targets, probs=probs, labels=labels,
    )

    # save a visual grid
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(n, 3, figsize=(7, 2.4 * n))
    if n == 1:
        axes = axes[None, :]
    for i in range(n):
        base = _denorm(imgs[i])
        cls = label_cols[targets[i]]
        p = probs[i, targets[i]]
        axes[i, 0].imshow(base); axes[i, 0].axis("off")
        axes[i, 0].set_title(f"{cls}  p={p:.2f}", fontsize=9)
        _overlay(axes[i, 1], base, cam[i], "Grad-CAM++")
        _overlay(axes[i, 2], base, ig[i], "Integrated Gradients")
    plt.tight_layout()
    out_png = os.path.join(args.out_dir, "explanations.png")
    plt.savefig(out_png, dpi=110, bbox_inches="tight")
    print("saved:", out_png)
    print("saved raw maps:", os.path.join(args.out_dir, "maps.npz"))


if __name__ == "__main__":
    main()
