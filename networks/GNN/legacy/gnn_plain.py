"""
Plain (non-invariant) message-passing GNN baseline for QM9.

Uses the raw 4D edge features straight from the loader: [bond_type, theta,
phi, distance]. theta/phi are lab-frame so this model is NOT rotation
invariant. Serves as the lower-bound baseline for comparison against
`gnn_invariant.py` and `gnn_qfim.py`.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing, global_max_pool


class MP(MessagePassing):
    def __init__(self, node_dim: int, edge_dim: int):
        super().__init__(aggr="add")
        self.mlp = nn.Sequential(
            nn.Linear(2 * node_dim + edge_dim, node_dim),
            nn.LayerNorm(node_dim),
            nn.ReLU(),
        )

    def forward(self, x, edge_index, edge_attr):
        return x + self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr):
        return self.mlp(torch.cat([x_i, x_j, edge_attr], dim=-1))


class GNN(nn.Module):
    def __init__(
        self,
        node_dim: int = 9,
        edge_dim: int = 4,
        hidden_dim: int = 64,
        num_layers: int = 6,
        out_dim: int = 1,
    ):
        super().__init__()
        self.node_embed = nn.Linear(node_dim, hidden_dim)
        self.edge_embed = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.layers = nn.ModuleList([MP(hidden_dim, hidden_dim) for _ in range(num_layers)])
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, out_dim),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        h = self.node_embed(x)
        e = self.edge_embed(edge_attr)
        for layer in self.layers:
            h = layer(h, edge_index, e)
        g = global_max_pool(h, batch) if batch is not None else h.max(0, keepdim=True)[0]
        return self.readout(g).squeeze(-1)
