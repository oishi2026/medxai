"""Run the faithfulness comparison and produce the paper's core result table.

For an evaluation subset of chest val images, scores Grad-CAM++, GNN attention,
and a random control (Integrated Gradients optional via --methods) with region-
level insertion/deletion, and reports per-image scores + bootstrap CIs.

Default methods keep the FIRST result fast (Grad-CAM++ vs GNN attention vs random,
the headline three-way). Add integrated_gradients when you want the full table.

Example (Kaggle):
    python -m medxai.eval.run_faithfulness \
        --cnn_ckpt   /kaggle/input/datasets/eemiir/medxai-chest-resnet50-v1/best.ckpt \
        --gnn_ckpt   /kaggle/input/datasets/eemiir/medxai-chest-gnn-gat-v1/best.ckpt \
        --chest_val  /kaggle/input/datasets/eemiir/medxai-splits-v1/chest_val.csv \
        --graph_dir  /kaggle/input/datasets/eemiir/medxai-chest-graphs-v1/val \
        --image_root /kaggle/input/datasets/ashery/chexpert \
        --strip_prefix "CheXpert-v1.0-small/" \
        --out_dir /kaggle/working/outputs/faithfulness --num_samples 100
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from medxai.data.dataset import ChestDataset
from medxai.data.splits import CHEXLOCALIZE_PATHOLOGIES
from medxai.eval.faithfulness import (
    bootstrap_ci,
    cnn_curves,
    curve_auc,
    gnn_curves,
)
from medxai.graph.gnn import RegionGNN
from medxai.regions import aggregate_map_to_regions, compute_superpixels, n_regions
from medxai.utils.determinism import set_determinism
from medxai.xai.explainers import (
    choose_target_classes,
    gradcampp_maps,
    integrated_gradients_maps,
    load_backbone,
)


def _load_gnn(ckpt_path, in_dim, num_classes, device):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    c = ck.get("cfg", {})
    gnn = RegionGNN(
        in_dim=in_dim, hidden=c.get("hidden", 256), num_classes=num_classes,
        num_layers=c.get("num_layers", 3), conv=c.get("conv", "gat"),
        heads=c.get("heads", 4), dropout=c.get("dropout", 0.3),
    )
    gnn.load_state_dict(ck["model"])
    return gnn.to(device).eval()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cnn_ckpt", required=True)
    ap.add_argument("--gnn_ckpt", required=True)
    ap.add_argument("--chest_val", required=True)
    ap.add_argument("--graph_dir", required=True, help="val graph cache dir")
    ap.add_argument("--image_root", required=True)
    ap.add_argument("--strip_prefix", default="")
    ap.add_argument("--out_dir", default="/kaggle/working/outputs/faithfulness")
    ap.add_argument("--frozen", default="conf/frozen.yaml")
    ap.add_argument("--num_samples", type=int, default=100)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--ig_steps", type=int, default=16)
    ap.add_argument("--baseline", choices=["zero", "blur"], default="zero")
    ap.add_argument("--methods", default="gradcampp,gnn_attention,random",
                    help="comma list; add integrated_gradients for full table")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.frozen))
    seed = cfg["seeds"][0]
    set_determinism(seed)
    resolution = cfg["input_resolution"]["chest"]
    sp = cfg["superpixels"]
    label_cols = CHEXLOCALIZE_PATHOLOGIES
    device = "cuda" if torch.cuda.is_available() else "cpu"
    methods = [m.strip() for m in args.methods.split(",")]
    rng = np.random.default_rng(seed)
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    ds = ChestDataset(args.chest_val, args.image_root, label_cols, resolution,
                      train=False, strip_prefix=args.strip_prefix)
    n = min(args.num_samples, len(ds))
    cnn = load_backbone(args.cnn_ckpt, num_classes=len(label_cols), device=device)

    # peek graph feature dim to build the GNN
    g0 = torch.load(os.path.join(args.graph_dir, "graph_000000.pt"),
                    weights_only=False)
    gnn = _load_gnn(args.gnn_ckpt, g0.x.shape[1], len(label_cols), device)

    rows = []
    curves = {m: {"del": [], "ins": []} for m in methods}
    for i in range(n):
        img = ds[i][0]
        labels_np = compute_superpixels(
            _denorm(img), sp["n_segments"], sp["compactness"])
        k = n_regions(labels_np)
        with torch.no_grad():
            prob = torch.sigmoid(cnn(img.unsqueeze(0).to(device))).cpu().numpy()
        target = int(choose_target_classes(prob, ds[i][1].numpy()[None])[0])

        # region importances per method
        imp = {}
        if "gradcampp" in methods:
            m = gradcampp_maps(cnn, img.unsqueeze(0).to(device), [target])[0]
            imp["gradcampp"] = aggregate_map_to_regions(m, labels_np)
        if "integrated_gradients" in methods:
            m = integrated_gradients_maps(cnn, img.unsqueeze(0).to(device), [target],
                                          n_steps=args.ig_steps)[0]
            imp["integrated_gradients"] = aggregate_map_to_regions(m, labels_np)
        graph = None
        if "gnn_attention" in methods:
            graph = torch.load(os.path.join(args.graph_dir, f"graph_{i:06d}.pt"),
                               weights_only=False)
            with torch.no_grad():
                _, alpha = gnn(graph.to(device), return_attention=True)
            imp["gnn_attention"] = alpha.float().cpu().numpy()
        if "random" in methods:
            imp["random"] = rng.random(k)

        for method, importance in imp.items():
            if method == "gnn_attention":
                dcurve, icurve = gnn_curves(gnn, graph, importance, target, device,
                                            steps=args.steps)
            else:
                dcurve, icurve = cnn_curves(cnn, img, labels_np, importance, target,
                                            device, steps=args.steps,
                                            baseline=args.baseline)
            rows.append({"image": i, "target": label_cols[target], "method": method,
                         "deletion_auc": curve_auc(dcurve),
                         "insertion_auc": curve_auc(icurve)})
            curves[method]["del"].append(dcurve)
            curves[method]["ins"].append(icurve)
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{n} images")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.out_dir, "per_image_scores.csv"), index=False)

    # aggregate table with bootstrap CIs
    print("\n=== FAITHFULNESS (region-level insertion/deletion) ===")
    print(f"{'method':<22} {'deletion AUC (lower=better)':<30} insertion AUC (higher=better)")
    summary = {}
    for method in methods:
        sub = df[df["method"] == method]
        dm, dlo, dhi = bootstrap_ci(sub["deletion_auc"].values)
        im, ilo, ihi = bootstrap_ci(sub["insertion_auc"].values)
        summary[method] = {"deletion": [dm, dlo, dhi], "insertion": [im, ilo, ihi]}
        print(f"{method:<22} {dm:.4f} [{dlo:.4f},{dhi:.4f}]        "
              f"{im:.4f} [{ilo:.4f},{ihi:.4f}]")
    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # mean curves figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    x = np.linspace(0, 1, args.steps + 1)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    for method in methods:
        a1.plot(x, np.mean(curves[method]["del"], axis=0), label=method)
        a2.plot(x, np.mean(curves[method]["ins"], axis=0), label=method)
    a1.set_title("Deletion (lower curve = more faithful)")
    a1.set_xlabel("fraction of regions removed"); a1.set_ylabel("target prob")
    a2.set_title("Insertion (higher curve = more faithful)")
    a2.set_xlabel("fraction of regions added"); a2.set_ylabel("target prob")
    a1.legend(); a2.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "curves.png"), dpi=110, bbox_inches="tight")
    print("\nsaved:", os.path.join(args.out_dir, "curves.png"),
          "and per_image_scores.csv")


def _denorm(t):
    from medxai.data.dataset import IMAGENET_MEAN, IMAGENET_STD
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return (t.cpu() * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()


if __name__ == "__main__":
    main()
