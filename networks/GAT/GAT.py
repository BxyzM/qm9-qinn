"""
Graph Attention Network (GAT) for QM9 HOMO-LUMO gap prediction.

A GAT learns which bonds are important via attention weights α_ij for each edge.
Multi-head attention allows the model to attend to different types of bonds in parallel.

References:
    Veličković et al. "Graph Attention Networks" (ICLR 2018)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.data import Data
import numpy as np


# ---------------------------------------------------------------------------
# GAT Model
# ---------------------------------------------------------------------------

class GAT(nn.Module):
    """
    Graph Attention Network for molecular property regression.

    Architecture:
        Input node features: (n_atoms, 7)
        Input edge features: (n_edges, 4)

        GATConv(7 → 64, heads=4) → (n_atoms, 256) + ELU
        GATConv(256 → 64, heads=4) → (n_atoms, 256) + ELU
        GATConv(256 → 64, heads=1) → (n_atoms, 64) + ELU

        Global mean pool → (64,)
        MLP: Linear(64→32) + ReLU + Linear(32→1)
    """

    def __init__(
        self,
        in_channels: int = 7,
        hidden_channels: int = 64,
        num_heads: int = 4,
        edge_dim: int = 4,
        dropout: float = 0.2,
    ):
        super().__init__()

        # Layer 1: in_channels → hidden_channels (multi-head)
        self.gat1 = GATConv(
            in_channels,
            hidden_channels,
            heads=num_heads,
            edge_dim=edge_dim,
            dropout=dropout,
            concat=True,
        )

        # Layer 2: hidden_channels*num_heads → hidden_channels (multi-head)
        self.gat2 = GATConv(
            hidden_channels * num_heads,
            hidden_channels,
            heads=num_heads,
            edge_dim=edge_dim,
            dropout=dropout,
            concat=True,
        )

        # Layer 3: hidden_channels*num_heads → hidden_channels (single head)
        self.gat3 = GATConv(
            hidden_channels * num_heads,
            hidden_channels,
            heads=1,
            edge_dim=edge_dim,
            dropout=dropout,
            concat=False,
        )

        # MLP head
        self.mlp = nn.Sequential(
            nn.Linear(hidden_channels, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

        self.elu = nn.ELU()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        """Forward pass: x (n_atoms, 7), edge_index (2, n_edges), edge_attr (n_edges, 4)"""
        x = self.gat1(x, edge_index, edge_attr)
        x = self.elu(x)
        x = F.dropout(x, p=0.2, training=self.training)

        x = self.gat2(x, edge_index, edge_attr)
        x = self.elu(x)
        x = F.dropout(x, p=0.2, training=self.training)

        x = self.gat3(x, edge_index, edge_attr)
        x = self.elu(x)

        x = global_mean_pool(x, batch=None)
        gap = self.mlp(x)
        return gap.squeeze(-1)


# ---------------------------------------------------------------------------
# Data utilities
# ---------------------------------------------------------------------------

def build_graph_from_features(
    node_feat: np.ndarray,
    edge_feat: np.ndarray,
    n_heavy: int,
) -> Data:
    """Convert node/edge features to PyG Data object."""
    x = torch.tensor(node_feat[:n_heavy], dtype=torch.float32)

    src, dst, edge_attrs = [], [], []

    # Bond edges
    for i in range(n_heavy):
        for j in range(n_heavy):
            bond_type = edge_feat[i, j, 0]
            if bond_type > 0:
                src.append(i)
                dst.append(j)
                edge_attrs.append(edge_feat[i, j])

    # Self-loops
    for i in range(n_heavy):
        src.append(i)
        dst.append(i)
        edge_attrs.append([0.0, 0.0, 0.0, 0.0])

    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_attr = torch.tensor(edge_attrs, dtype=torch.float32)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


class TargetNormaliser:
    """Normalize target to zero mean, unit std."""

    def __init__(self, mean: float = 0.0, std: float = 1.0):
        self.mean = mean
        self.std = std

    @classmethod
    def from_data(cls, targets: np.ndarray):
        return cls(float(np.mean(targets)), float(np.std(targets)))

    def normalise(self, target: torch.Tensor) -> torch.Tensor:
        return (target - self.mean) / (self.std + 1e-8)

    def denormalise(self, target_norm: torch.Tensor) -> torch.Tensor:
        return target_norm * self.std + self.mean
