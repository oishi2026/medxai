"""Precompute region graphs and cache them to disk (one .pt per image).

Per image: frozen backbone -> layer3 features; SLIC -> regions (nodes); RAG ->
edges; pool features per region -> node features (+ normalized centroid coords);
centroid-distance edge features. Saved as a PyTorch Geometric Data object so GNN
training just loads tensors instead of recomputing SLIC every epoch.

Resumable: skips images whose graph already exists. Run val first (small) to
verify, then train.

Example (Kaggle):
    python -m medxai.graph.build_graphs \
        --ckpt        /kaggle/input/datasets/eemiir/medxai-chest-resnet50-v1/best.ckpt \
        --manifest    /kaggle/input/datasets/eemiir/medxai-splits-v1/chest_val.csv \
        --image_root  /kaggle/input/datasets/ashery/chexpert \
        --strip_prefix "CheXpert-v1.0-small/" \
        --out_dir     /kaggle/working/graphs --split_name val
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from medxai.data.dataset import IMAGENET_MEAN, IMAGENET_STD, ChestDataset
from medxai.data.splits import CHEXLOCALIZE_PATHOLOGIES
from medxai.graph.pooling import pool_features_to_regions
from medxai.regions import build_rag, compute_superpixels, edge_features, n_regions
from medxai.utils.determinism import set_determinism
from medxai.xai.explainers import load_backbone


def _denorm_np(t: torch.Tensor) -> np.ndarray:
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return (t.cpu() * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()


def _to_graph(node_feats, centroids, edges, edge_attr, y, labels_shape):
    from torch_geometric.data import Data

    k = node_feats.shape[0]
    # append normalized centroid coords (row/H, col/W) to node features
    cen = torch.tensor(centroids, dtype=torch.float32)
    cen[:, 0] /= labels_shape[0]
    cen[:, 1] /= labels_shape[1]
    x = torch.cat([node_feats.float(), cen], dim=1).half()  # (k, C+2) fp16

    if len(edges):
        ei = torch.tensor(edges.T, dtype=torch.long)
        edge_index = torch.cat([ei, ei.flip(0)], dim=1)       # undirected -> both dirs
        ea = torch.tensor(edge_attr, dtype=torch.float32).view(-1, 1)
        edge_attr_t = torch.cat([ea, ea], dim=0).half()
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr_t = torch.zeros((0, 1), dtype=torch.float16)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr_t,
                y=torch.tensor(y, dtype=torch.float32).view(1, -1), num_nodes=k)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--image_root", required=True)
    ap.add_argument("--strip_prefix", default="")
    ap.add_argument("--out_dir", default="/kaggle/working/graphs")
    ap.add_argument("--split_name", default="val")
    ap.add_argument("--frozen", default="conf/frozen.yaml")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--pool_size", type=int, default=80)
    ap.add_argument("--limit", type=int, default=0, help="0 = all; else first N")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.frozen))
    set_determinism(cfg["seeds"][0])
    resolution = cfg["input_resolution"]["chest"]
    sp = cfg["superpixels"]
    label_cols = CHEXLOCALIZE_PATHOLOGIES
    device = "cuda" if torch.cuda.is_available() else "cpu"

    out = Path(args.out_dir) / args.split_name
    out.mkdir(parents=True, exist_ok=True)

    ds = ChestDataset(args.manifest, args.image_root, label_cols, resolution,
                      train=False, strip_prefix=args.strip_prefix)  # deterministic
    total = len(ds) if args.limit == 0 else min(args.limit, len(ds))
    model = load_backbone(args.ckpt, num_classes=len(label_cols), device=device)

    loader = DataLoader(range(total), batch_size=args.batch_size, shuffle=False)
    done, node_counts, edge_counts = 0, [], []
    for batch_idx in loader:
        idxs = [int(i) for i in batch_idx]
        pending = [i for i in idxs if not (out / f"graph_{i:06d}.pt").exists()]
        if pending:
            imgs = torch.stack([ds[i][0] for i in pending]).to(device)
            with torch.no_grad():
                _, feats = model(imgs, return_features=True)   # (B,1024,20,20)
            feats = feats.float().cpu()
            for j, i in enumerate(pending):
                img_np = _denorm_np(ds[i][0])
                labels = compute_superpixels(img_np, sp["n_segments"], sp["compactness"])
                k = n_regions(labels)
                edges, centroids, _ = build_rag(labels)
                node_feats = pool_features_to_regions(feats[j], labels, k, args.pool_size)
                ea = edge_features(edges, centroids, labels)
                y = ds[i][1].numpy()
                g = _to_graph(node_feats, centroids, edges, ea, y, labels.shape)
                torch.save(g, out / f"graph_{i:06d}.pt")
                node_counts.append(k); edge_counts.append(len(edges))
        done += len(idxs)
        if done % (args.batch_size * 10) < args.batch_size:
            print(f"  {done}/{total} graphs")

    meta = {
        "split": args.split_name, "n_graphs": total,
        "feature_dim": 1024 + 2, "pool_size": args.pool_size,
        "avg_nodes": float(np.mean(node_counts)) if node_counts else None,
        "avg_edges": float(np.mean(edge_counts)) if edge_counts else None,
        "superpixels": sp,
    }
    with open(Path(args.out_dir) / f"{args.split_name}_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Done: {total} graphs in {out}")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
