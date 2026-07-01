"""Load cached region graphs for GNN training.

A thin Dataset over the .pt files written by build_graphs, plus a loader helper.
Uses torch_geometric's DataLoader for correct graph-batch collation.
"""
from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import Dataset


class CachedGraphDataset(Dataset):
    def __init__(self, graph_dir: str):
        self.files = sorted(Path(graph_dir).glob("graph_*.pt"))
        if not self.files:
            raise FileNotFoundError(f"No graph_*.pt files in {graph_dir}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx):
        g = torch.load(self.files[idx], weights_only=False)
        g.x = g.x.float()                # fp16 on disk -> fp32 for compute
        g.edge_attr = g.edge_attr.float()
        return g


def make_graph_loader(graph_dir: str, batch_size: int = 32, shuffle: bool = True):
    from torch_geometric.loader import DataLoader as GeoLoader

    ds = CachedGraphDataset(graph_dir)
    return ds, GeoLoader(ds, batch_size=batch_size, shuffle=shuffle)
