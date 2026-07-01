"""Region-level insertion/deletion faithfulness — the core scientific measurement.

Both paradigms are scored through the SAME region-level perturbation, each against
its own model:
  * CNN saliency: rank regions -> mask those regions' PIXELS -> re-run CNN.
  * GNN attention: rank regions -> mask those NODES' features -> re-run GNN.
Identical region partition and identical "remove top-k regions" schedule make the
comparison fair; scoring each against its own model keeps 'faithfulness' correct.

Deletion AUC: lower = better (removing important regions collapses the prediction).
Insertion AUC: higher = better (adding important regions restores it).
"""
from __future__ import annotations

import numpy as np
import torch


def curve_auc(y: np.ndarray) -> float:
    """Trapezoidal area under a curve sampled at equal x-steps on [0,1].
    Version-agnostic (no np.trapz/np.trapezoid dependency)."""
    y = np.asarray(y, dtype=np.float64)
    x = np.linspace(0.0, 1.0, len(y))
    return float(np.sum((y[:-1] + y[1:]) / 2.0 * np.diff(x)))


def _rank_of_region(importance: np.ndarray) -> np.ndarray:
    """rank[r] = 0 for the most important region, ... ascending."""
    order = np.argsort(-np.asarray(importance))
    rank = np.empty(len(order), dtype=np.int64)
    rank[order] = np.arange(len(order))
    return rank


def _blur_baseline(img: torch.Tensor) -> torch.Tensor:
    from torchvision.transforms.functional import gaussian_blur
    h = img.shape[-1]
    k = max(3, (h // 10) | 1)  # odd kernel ~10% of image
    return gaussian_blur(img.unsqueeze(0), kernel_size=[k, k]).squeeze(0)


@torch.no_grad()
def cnn_curves(model, img, labels, importance, target, device,
               steps=20, baseline="zero"):
    """Region-level insertion & deletion for a pixel-space CNN saliency ranking."""
    img = img.to(device)
    base = torch.zeros_like(img) if baseline == "zero" else _blur_baseline(img).to(device)
    rank = _rank_of_region(importance)
    k = len(rank)
    rankmap = torch.from_numpy(rank[labels]).to(device)     # (H,W) pixel ranks
    ns = (np.linspace(0, 1, steps + 1) * k).astype(int)

    del_imgs, ins_imgs = [], []
    for n in ns:
        keep_del = (rankmap >= n).float()                   # keep not-yet-removed
        del_imgs.append(img * keep_del + base * (1 - keep_del))
        keep_ins = (rankmap < n).float()                    # only top-n present
        ins_imgs.append(img * keep_ins + base * (1 - keep_ins))

    del_p = torch.sigmoid(model(torch.stack(del_imgs)))[:, target].float().cpu().numpy()
    ins_p = torch.sigmoid(model(torch.stack(ins_imgs)))[:, target].float().cpu().numpy()
    return del_p, ins_p


@torch.no_grad()
def gnn_curves(gnn, graph, importance, target, device, steps=20, img_feat_dim=1024):
    """Region-level insertion & deletion for a GNN attention ranking. Masking a
    region zeros that node's image-feature dims (centroid coords kept, since
    position is graph structure, not ablated image content)."""
    from torch_geometric.data import Data

    g = graph.to(device)
    x0 = g.x.float()
    k = x0.shape[0]
    rank = torch.from_numpy(_rank_of_region(importance)).to(device)
    batch = torch.zeros(k, dtype=torch.long, device=device)
    ns = (np.linspace(0, 1, steps + 1) * k).astype(int)

    def run(x):
        d = Data(x=x, edge_index=g.edge_index, edge_attr=g.edge_attr.float(),
                 batch=batch)
        return torch.sigmoid(gnn(d))[0, target].item()

    del_p, ins_p = [], []
    for n in ns:
        m = rank < n
        xdel = x0.clone(); xdel[m, :img_feat_dim] = 0.0         # remove top-n
        del_p.append(run(xdel))
        xins = x0.clone(); xins[~m, :img_feat_dim] = 0.0        # keep only top-n
        ins_p.append(run(xins))
    return np.array(del_p), np.array(ins_p)


def bootstrap_ci(values, n_boot=2000, alpha=0.05, seed=0):
    """Mean and (lo, hi) percentile bootstrap CI."""
    v = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    boots = [rng.choice(v, size=len(v), replace=True).mean() for _ in range(n_boot)]
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(v.mean()), float(lo), float(hi)
