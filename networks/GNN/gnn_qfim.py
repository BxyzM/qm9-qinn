"""
QFIMGNN: baseline GNN + QFIM edge features via a swappable embedding head.

Architecture
------------
Inherits everything from GNN (node path, geometric edge path, MP stack,
readout, target standardization) and extends the edge features with a
4-dim QFIM summary per edge:

    e_full = concat(geom_edge_3dim, qfim_edge_4dim)  -> 7 dims per edge

The QFIM summary comes from one of four swappable heads selected by
config.qfim.embed_op:

    mlp    : expand-compress MLP  36 -> 64 -> 32 -> 16 -> 8 -> 4
    conv1d : Conv1d(6, 16, k=3) over one qubit-param axis, sym-averaged,
             pooled along the param axis, projected to 4
    conv2d : Conv2d(1, 16, k=3) over the (6, 6) block, pooled, projected to 4
    gated  : compute C_ij = ||Q[i,j]||_F (scalar), tiny MLP to 4 dims

All heads take the same input (the flattened (pd*pd)-dim QFIM sub-block per
edge) and produce the same output shape (E, 4). This lets the rest of the
model be identical across heads and lets ablations be a one-line config
change.

The QFIM diagonal sub-blocks Q[i, i] are not used here. PyG bond edges never
have i == j, so the gather returns only off-diagonal sub-blocks by
construction. If per-atom self-coupling is wanted later, it belongs in a
node-level extension (see legacy/gnn_qfim_node.py for the earlier attempt).
"""

from __future__ import annotations

from typing import Iterable, Optional, Tuple

import torch
import torch.nn as nn

from .gnn import (
    GNN,
    InvariantMP,
    _POOLINGS,
    _build_mlp,
    _COORD_COLS,
    MAX_NEIGHBORS,
    MAX_CHAINS,
)


# ---------------------------------------------------------------------------
# QFIM edge gather (moved here from legacy; kept signature-compatible)
# ---------------------------------------------------------------------------

def _gather_edge_qfim(
    qfim_block: torch.Tensor,      # (B, nq, nq, pd, pd)
    qfim_nq: int,
    edge_index: torch.Tensor,      # (2, E) global node indices
    batch: torch.Tensor,           # (N,) graph id per node
) -> torch.Tensor:
    """
    For each edge (i, j), fetch Q[mol(i), local(i), local(j)] and flatten
    to (pd*pd,). Atoms whose local index exceeds qfim_nq yield zeros.
    """
    src, dst = edge_index[0], edge_index[1]
    if src.numel() == 0:
        _, _, _, pd, _ = qfim_block.shape
        return qfim_block.new_zeros((0, pd * pd))

    n_nodes = batch.numel()
    change = torch.ones_like(batch, dtype=torch.bool)
    change[1:] = batch[1:] != batch[:-1]
    graph_starts = torch.nonzero(change, as_tuple=False).view(-1)
    local_idx = torch.arange(n_nodes, device=batch.device) - graph_starts[batch]

    mol_id = batch[src]
    li = local_idx[src]
    lj = local_idx[dst]

    in_budget = (li < qfim_nq) & (lj < qfim_nq)
    li_clamped = li.clamp(max=qfim_nq - 1)
    lj_clamped = lj.clamp(max=qfim_nq - 1)

    sub = qfim_block[mol_id, li_clamped, lj_clamped]              # (E, pd, pd)
    sub = sub * in_budget.view(-1, 1, 1).to(sub.dtype)
    return sub.reshape(sub.shape[0], -1)                          # (E, pd*pd)


# ---------------------------------------------------------------------------
# QFIM embedding heads
# ---------------------------------------------------------------------------

class _QFIMHeadMLP(nn.Module):
    """Expand-compress MLP: pd*pd -> 64 -> 32 -> 16 -> 8 -> out_dim."""

    def __init__(self, pd: int, out_dim: int = 4):
        super().__init__()
        self.mlp = _build_mlp((pd * pd, 64, 32, 16, 8, out_dim))

    def forward(self, qfim_edge: torch.Tensor) -> torch.Tensor:
        return self.mlp(qfim_edge)


class _QFIMHeadConv1d(nn.Module):
    """
    Reshape (E, pd*pd) -> (E, pd, pd) treating rows as channels and cols as
    length; Conv1d with kernel 3 padding 1, LayerNorm, ReLU; symmetrize by
    averaging the stack applied to Q and Q.T; AdaptiveAvgPool1d(1) over
    the length axis; Linear to out_dim; LayerNorm(out_dim).
    """

    def __init__(self, pd: int, out_dim: int = 4,
                 conv_channels: int = 16, kernel_size: int = 3):
        super().__init__()
        self.pd = pd
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(pd, conv_channels, kernel_size=kernel_size, padding=padding)
        self.norm1 = nn.LayerNorm([conv_channels, pd])
        self.conv2 = nn.Conv1d(conv_channels, conv_channels, kernel_size=kernel_size, padding=padding)
        self.norm2 = nn.LayerNorm([conv_channels, pd])
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.project = nn.Linear(conv_channels, out_dim)
        self.out_norm = nn.LayerNorm(out_dim)

    def _forward_one(self, Q: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.norm1(self.conv1(Q)))
        h = torch.relu(self.norm2(self.conv2(h)))
        return self.pool(h).squeeze(-1)                           # (E, conv_channels)

    def forward(self, qfim_edge: torch.Tensor) -> torch.Tensor:
        E = qfim_edge.shape[0]
        Q = qfim_edge.view(E, self.pd, self.pd)
        h = 0.5 * (self._forward_one(Q) + self._forward_one(Q.transpose(1, 2)))
        return self.out_norm(self.project(h))


class _QFIMHeadConv2d(nn.Module):
    """
    Reshape (E, pd*pd) -> (E, 1, pd, pd); shallow Conv2d stack; pool; project.
    """

    def __init__(self, pd: int, out_dim: int = 4,
                 conv_channels: int = 16, kernel_size: int = 3):
        super().__init__()
        self.pd = pd
        padding = kernel_size // 2
        self.conv1 = nn.Conv2d(1, conv_channels, kernel_size=kernel_size, padding=padding)
        self.norm1 = nn.LayerNorm([conv_channels, pd, pd])
        self.conv2 = nn.Conv2d(conv_channels, conv_channels, kernel_size=kernel_size, padding=padding)
        self.norm2 = nn.LayerNorm([conv_channels, pd, pd])
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.project = nn.Linear(conv_channels, out_dim)
        self.out_norm = nn.LayerNorm(out_dim)

    def forward(self, qfim_edge: torch.Tensor) -> torch.Tensor:
        E = qfim_edge.shape[0]
        Q = qfim_edge.view(E, 1, self.pd, self.pd)
        h = torch.relu(self.norm1(self.conv1(Q)))
        h = torch.relu(self.norm2(self.conv2(h)))
        h = self.pool(h).view(E, -1)                              # (E, conv_channels)
        return self.out_norm(self.project(h))


class _QFIMHeadGated(nn.Module):
    """C_ij = ||Q[i,j]||_F (scalar), tiny MLP to out_dim."""

    def __init__(self, pd: int, out_dim: int = 4):
        super().__init__()
        self.pd = pd
        self.mlp = nn.Sequential(
            nn.Linear(1, 8),
            nn.ReLU(),
            nn.Linear(8, out_dim),
        )

    def forward(self, qfim_edge: torch.Tensor) -> torch.Tensor:
        E = qfim_edge.shape[0]
        Q = qfim_edge.view(E, self.pd, self.pd)
        cij = torch.linalg.matrix_norm(Q, ord="fro").unsqueeze(-1)  # (E, 1)
        return self.mlp(cij)


_QFIM_HEADS = {
    "mlp": _QFIMHeadMLP,
    "conv1d": _QFIMHeadConv1d,
    "conv2d": _QFIMHeadConv2d,
    "gated": _QFIMHeadGated,
}


# ---------------------------------------------------------------------------
# QFIMGNN
# ---------------------------------------------------------------------------

class QFIMGNN(GNN):
    """
    Baseline GNN extended with a QFIM edge feature. The QFIM head is chosen
    by name and produces (E, qfim_out_dim) per edge; this is concatenated
    onto the 3-dim geometric edge feature so MP sees 7-dim edges.
    """

    def __init__(
        self,
        num_mp_layers: int = 6,
        node_mlp_dims: Tuple[int, ...] = (4, 8, 16, 8, 4),
        edge_mlp_dims: Tuple[int, ...] = (13, 6, 8, 16, 8, 3),
        max_neighbors: int = MAX_NEIGHBORS,
        max_chains: int = MAX_CHAINS,
        pooling: str = "mean",
        qfim_per_qubit_dim: int = 6,
        qfim_embed_op: str = "mlp",
        qfim_out_dim: int = 4,
    ):
        super().__init__(
            num_mp_layers=num_mp_layers,
            node_mlp_dims=node_mlp_dims,
            edge_mlp_dims=edge_mlp_dims,
            max_neighbors=max_neighbors,
            max_chains=max_chains,
            pooling=pooling,
        )
        if qfim_embed_op not in _QFIM_HEADS:
            raise ValueError(
                f"qfim_embed_op must be one of {list(_QFIM_HEADS)}; got {qfim_embed_op!r}"
            )
        self.qfim_pd = qfim_per_qubit_dim
        self.qfim_out_dim = qfim_out_dim
        self.qfim_embed_op = qfim_embed_op
        self.qfim_head = _QFIM_HEADS[qfim_embed_op](
            pd=qfim_per_qubit_dim, out_dim=qfim_out_dim,
        )

        # Rebuild MP stack with the new edge dim = geom 3 + qfim out dim.
        new_edge_dim = self.edge_dim + qfim_out_dim
        self.mp_layers = nn.ModuleList(
            [InvariantMP(self.node_dim, new_edge_dim) for _ in range(num_mp_layers)]
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
        if not bool(self._stats_fitted.item()):
            raise RuntimeError(
                "QFIMGNN.forward called before fit_target_stats. "
                "Call model.fit_target_stats(train_loader) once before training."
            )
        if batch is None:
            batch = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)

        node_in = self.build_node_feat(x)                             # (N, 4)
        h = self.node_mlp(node_in)                                    # (N, 4)
        e_geom = self.build_edge_feat(x, edge_index, edge_attr)       # (E, 3)

        qfim_edge = _gather_edge_qfim(qfim_block, qfim_nq, edge_index, batch)   # (E, pd*pd)
        qfim_feat = self.qfim_head(qfim_edge)                         # (E, 4)
        e = torch.cat([e_geom, qfim_feat], dim=-1)                    # (E, 7)

        for layer in self.mp_layers:
            h = layer(h, edge_index, e)

        g = self._pool(h, batch)                                      # (B, 4)
        z = self.readout(g).squeeze(-1)                               # (B,)
        return z * self.target_std + self.target_mean
