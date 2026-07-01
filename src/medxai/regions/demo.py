"""Visualize the shared region layer on real chest X-rays.

Shows, per image: SLIC boundaries over the X-ray, and — if you pass the maps.npz
from the XAI step — the Grad-CAM++ saliency both raw and aggregated onto the
regions. The raw-vs-regionized comparison is the visual proof of the protocol:
the same saliency, now expressed on the shared regions the GNN also uses.

Example (Kaggle):
    python -m medxai.regions.demo \
        --chest_val   /kaggle/input/datasets/eemiir/medxai-splits-v1/chest_val.csv \
        --image_root  /kaggle/input/datasets/ashery/chexpert \
        --strip_prefix "CheXpert-v1.0-small/" \
        --maps        /kaggle/working/outputs/xai/maps.npz \
        --out_dir     /kaggle/working/outputs/regions \
        --num_samples 6
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import yaml

from medxai.data.dataset import IMAGENET_MEAN, IMAGENET_STD, ChestDataset
from medxai.data.splits import CHEXLOCALIZE_PATHOLOGIES
from medxai.regions import (
    aggregate_map_to_regions,
    build_rag,
    compute_superpixels,
    n_regions,
    regions_to_map,
)


def _denorm(t) -> np.ndarray:
    import torch
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return (t.cpu() * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chest_val", required=True)
    ap.add_argument("--image_root", required=True)
    ap.add_argument("--strip_prefix", default="")
    ap.add_argument("--maps", default=None, help="optional maps.npz from XAI step")
    ap.add_argument("--out_dir", default="/kaggle/working/outputs/regions")
    ap.add_argument("--frozen", default="conf/frozen.yaml")
    ap.add_argument("--num_samples", type=int, default=6)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.frozen))
    resolution = cfg["input_resolution"]["chest"]
    sp = cfg["superpixels"]
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    ds = ChestDataset(args.chest_val, args.image_root, CHEXLOCALIZE_PATHOLOGIES,
                      resolution, train=False, strip_prefix=args.strip_prefix)
    n = min(args.num_samples, len(ds))

    maps = np.load(args.maps) if args.maps else None
    cam = maps["gradcampp"] if maps is not None else None

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from skimage.segmentation import mark_boundaries

    ncols = 3 if cam is not None else 2
    fig, axes = plt.subplots(n, ncols, figsize=(2.6 * ncols, 2.6 * n))
    if n == 1:
        axes = axes[None, :]

    region_counts = []
    for i in range(n):
        img = _denorm(ds[i][0])
        labels = compute_superpixels(img, sp["n_segments"], sp["compactness"])
        edges, _, _ = build_rag(labels)
        region_counts.append((n_regions(labels), len(edges)))

        axes[i, 0].imshow(img); axes[i, 0].axis("off")
        axes[i, 0].set_title("X-ray", fontsize=9)
        axes[i, 1].imshow(mark_boundaries(img, labels, color=(1, 1, 0)))
        axes[i, 1].axis("off")
        axes[i, 1].set_title(f"{n_regions(labels)} regions", fontsize=9)

        if cam is not None:
            region_vals = aggregate_map_to_regions(cam[i], labels)
            axes[i, 2].imshow(img)
            axes[i, 2].imshow(regions_to_map(region_vals, labels),
                              cmap="jet", alpha=0.5)
            axes[i, 2].axis("off")
            axes[i, 2].set_title("Grad-CAM++ on regions", fontsize=9)

    plt.tight_layout()
    out_png = os.path.join(args.out_dir, "regions.png")
    plt.savefig(out_png, dpi=110, bbox_inches="tight")
    print("regions/edges per image:", region_counts)
    print("saved:", out_png)


if __name__ == "__main__":
    main()
