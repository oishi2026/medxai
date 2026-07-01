"""Region-graph GNN — the second explanation paradigm.

Message-passing (GAT or GraphSAGE) over region graphs, then a GLOBAL ATTENTION
POOLING readout whose per-node weight IS the explanation: one importance value
per region, directly comparable to CNN saliency aggregated onto the same regions.

The attention readout is deliberately the primary explanation signal (cleaner
than reading internal GAT head attention), and node importance is exposed via
`forward(..., return_attention=True)` for the faithfulness stage.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, SAGEConv
from torch_geometric.utils import softmax


class AttentionReadout(nn.Module):
    """Global attention pooling: scores each node, softmax-normalizes within its
    graph, returns the attention-weighted sum plus the per-node weights."""

    def __init__(self, in_dim: int):
        super().__init__()
        self.gate = nn.Linear(in_dim, 1)

    def forward(self, x, batch):
        scores = self.gate(x).squeeze(-1)          # (N,)
        alpha = softmax(scores, batch)             # normalized within each graph
        weighted = x * alpha.unsqueeze(-1)
        # sum per graph
        out = torch.zeros(int(batch.max()) + 1, x.size(1), device=x.device, dtype=x.dtype)
        out.index_add_(0, batch, weighted)
        return out, alpha


class RegionGNN(nn.Module):
    def __init__(
        self,
        in_dim: int = 1026,
        hidden: int = 256,
        num_classes: int = 10,
        num_layers: int = 3,
        conv: str = "gat",
        heads: int = 4,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.dropout = dropout
        self.convs = nn.ModuleList()
        d = in_dim
        for _ in range(num_layers):
            if conv == "gat":
                # concat heads -> hidden; set per-head dim so output is `hidden`
                self.convs.append(GATv2Conv(d, hidden // heads, heads=heads,
                                            edge_dim=1, dropout=dropout))
            elif conv == "sage":
                self.convs.append(SAGEConv(d, hidden))
            else:
                raise ValueError(f"conv must be 'gat' or 'sage', got {conv!r}")
            d = hidden
        self.readout = AttentionReadout(hidden)
        self.head = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(hidden, num_classes)
        )
        self.conv_type = conv

    def forward(self, data, return_attention: bool = False):
        x, ei, batch = data.x, data.edge_index, data.batch
        ea = getattr(data, "edge_attr", None)
        for conv in self.convs:
            if self.conv_type == "gat":
                x = conv(x, ei, edge_attr=ea)
            else:
                x = conv(x, ei)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        pooled, alpha = self.readout(x, batch)
        logits = self.head(pooled)
        if return_attention:
            return logits, alpha
        return logits
