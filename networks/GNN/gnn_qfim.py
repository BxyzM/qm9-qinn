"""
QFIM-enhanced GNN where the QFIM enters ONLY as per-edge features.

Motivation: the off-diagonal rot-gate block of the QFIM is the parameter
coupling between different qubits -- a genuine edge-level quantity. Feeding
only the diagonal to nodes throws that away. Here we instead gather the
(per_qubit_dim, per_qubit_dim) sub-block for each bonded (atom_i, atom_j)
pair and flatten it into the edge attribute.

Design:
- Pure EdgeConv-style message passing.
- Node input: classical 9D features, same embedding as gnn_invariant.
- Edge input: invariant geometric features (as in gnn_invariant) concatenated
  with the (pd*pd,)-flattened QFIM sub-block for (qubit_i, qubit_j).
- No qfim_embed, no fuse. QFIM's only entry point is through edges, once per
  message-passing layer (it is part of edge_attr, re-consumed each layer).

Atoms beyond the qubit budget (n_heavy > n_qubits) get zero QFIM blocks.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing, global_max_pool

from .gnn_invariant import build_invariant_edge_attr


class QFIMEdgeConv(MessagePassing):
    """EdgeConv with edge features: msg = MLP([h_i, h_j - h_i, edge_attr])."""

    def __init__(self, node_dim: int, edge_dim: int):
        super().__init__(aggr="max")
        self.mlp = nn.Sequential(
            nn.Linear(2 * node_dim + edge_dim, node_dim),
            nn.LayerNorm(node_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        return x + self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i: torch.Tensor, x_j: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        return self.mlp(torch.cat([x_i, x_j - x_i, edge_attr], dim=-1))


def _gather_edge_qfim(
    qfim_block: torch.Tensor,      # (B, nq, nq, pd, pd)
    qfim_nq: int,                  # qubits per molecule
    edge_index: torch.Tensor,      # (2, E) global node indices
    batch: torch.Tensor,           # (N,) graph id per node
) -> torch.Tensor:
    """
    For each edge, fetch the (pd, pd) QFIM sub-block linking the rot-gate
    params of the source atom and the destination atom, flatten to (pd*pd,).

    Atoms with local index >= n_qubits (i.e. beyond the qubit budget) yield
    a zero block. Edges where either endpoint is out of the qubit budget
    also yield zero.
    """
    src, dst = edge_index[0], edge_index[1]
    if src.numel() == 0:
        B, _, _, pd, _ = qfim_block.shape
        return qfim_block.new_zeros((0, pd * pd))

    # Local atom index within its graph: running counter reset per graph.
    # batch is sorted by graph id (PyG invariant); compute starts per graph.
    n_nodes = batch.numel()
    # First occurrence of each graph id:
    change = torch.ones_like(batch, dtype=torch.bool)
    change[1:] = batch[1:] != batch[:-1]
    graph_starts = torch.nonzero(change, as_tuple=False).view(-1)        # (B,)
    # Map global node -> local idx within its graph:
    local_idx = torch.arange(n_nodes, device=batch.device) - graph_starts[batch]

    mol_id = batch[src]                           # (E,)
    li = local_idx[src]                           # (E,)
    lj = local_idx[dst]                           # (E,)

    in_budget = (li < qfim_nq) & (lj < qfim_nq)
    li_clamped = li.clamp(max=qfim_nq - 1)
    lj_clamped = lj.clamp(max=qfim_nq - 1)

    sub = qfim_block[mol_id, li_clamped, lj_clamped]                     # (E, pd, pd)
    sub = sub * in_budget.view(-1, 1, 1).to(sub.dtype)
    E = sub.shape[0]
    return sub.reshape(E, -1)                                            # (E, pd*pd)


class QFIMGNN(nn.Module):
    """
    GNN where QFIM enters only as per-edge features.
    """

    def __init__(
        self,
        node_dim: int = 9,
        qfim_per_qubit_dim: int = 6,       # pd -> per-edge qfim dim = pd*pd
        hidden_dim: int = 32,
        num_layers: int = 6,
        include_dihedral: bool = True,
        coord_cols: slice = slice(4, 7),
        out_dim: int = 1,
    ):
        super().__init__()
        self.coord_cols = coord_cols
        self.include_dihedral = include_dihedral
        self.geom_edge_dim = 4 if include_dihedral else 3
        self.qfim_edge_dim = qfim_per_qubit_dim * qfim_per_qubit_dim
        self.edge_in_dim = self.geom_edge_dim + self.qfim_edge_dim

        self.node_embed = nn.Linear(node_dim, hidden_dim)
        self.edge_embed = nn.Sequential(
            nn.Linear(self.edge_in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.layers = nn.ModuleList(
            [QFIMEdgeConv(hidden_dim, hidden_dim) for _ in range(num_layers)]
        )
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
        qfim_block: torch.Tensor,           # (B, nq, nq, pd, pd)
        qfim_nq: int,
        batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        coords = x[:, self.coord_cols]
        geom_attr = build_invariant_edge_attr(
            edge_attr, coords, edge_index, self.include_dihedral
        )
        if batch is None:
            batch = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)

        qfim_edge = _gather_edge_qfim(qfim_block, qfim_nq, edge_index, batch)
        e_in = torch.cat([geom_attr, qfim_edge], dim=-1)

        h = self.node_embed(x)
        e = self.edge_embed(e_in)
        for layer in self.layers:
            h = layer(h, edge_index, e)
        g = global_max_pool(h, batch) if batch is not None else h.max(0, keepdim=True)[0]
        return self.readout(g).squeeze(-1)
