"""
QFIMGNN variant with a Conv1d-based edge-feature compressor.

Motivation
----------
The per-edge QFIM cross-block has shape (pd, pd) = (6, 6). The 6-dim param
axis is ordered as (layer, op) with 2 layers x 3 ops per layer, so "nearby
positions along the axis" correspond to "adjacent operations within a
layer." A Conv1d with small kernel can exploit this locality, while the
current flat-36 Linear cannot: it has no inductive bias for which entries
are neighbors.

Interpretation (axis A)
-----------------------
Treat each qubit's 6-dim param axis as the "length" axis for Conv1d. The
other qubit's 6 params are the input "channels" (Conv1d runs them in
parallel). Kernel size 3 exactly spans one layer's 3 ops; stacking two
convs with padding=1 gives a receptive field of 5 so cross-layer locality
is captured too.

Symmetrization
--------------
The raw QFIM block is symmetric: Q[i,j] = Q[j,i].T. Treating one axis as
channels and the other as length breaks that symmetry at the conv level.
We restore it by running the conv on both Q and Q.T, then averaging the
outputs. Costs 2x the conv compute (trivial on small matrices) and keeps
the feature invariant under swapping qubit i with qubit j at the block
level, mirroring the underlying physics.

Architecture
------------
    per edge:
        Q (E, 6, 6)
            -> Conv1d(6, 16, k=3, p=1) + LN + ReLU + Conv1d(16, 16, k=3, p=1) + LN + ReLU
            -> avg-pool along length -> (E, 16)
        applied twice (Q and Q.T), averaged -> (E, 16)
        -> Linear(16, 8) + LN  -> qfim_small (E, 8)

    concat with geom (E, 4)  -> (E, 12)
        -> edge_embed: Linear(12, H) + ReLU + Linear(H, H)  -> e (E, H)

This keeps the raw edge-embed input at 12 dims (vs 40 in QFIMGNN), aligned
with standard GNN practice of compressing edge features before the MP path.
"""

from __future__ import annotations

from typing import Iterable, Optional, Tuple

import torch
import torch.nn as nn
from torch_geometric.nn import global_add_pool, global_max_pool, global_mean_pool

from .gnn_invariant import InvariantMP, build_invariant_edge_attr
from .gnn_qfim import _gather_edge_qfim


_POOLINGS = {
    "add": global_add_pool,
    "mean": global_mean_pool,
    "max": global_max_pool,
}


class _QFIMConvCompress(nn.Module):
    """
    Two-stage Conv1d compressor applied symmetrically to Q and Q.T, then
    average-pooled along the length axis and projected to `out_dim`.
    """

    def __init__(
        self,
        pd: int,
        conv_channels: int = 16,
        kernel_size: int = 3,
        out_dim: int = 8,
    ):
        super().__init__()
        self.pd = pd
        padding = kernel_size // 2
        # Conv1d expects (N, C_in, L). We use C_in = pd (one channel per
        # "other qubit's param") and L = pd (positions along this qubit's
        # param axis).
        self.conv1 = nn.Conv1d(pd, conv_channels, kernel_size=kernel_size, padding=padding)
        self.norm1 = nn.LayerNorm([conv_channels, pd])
        self.conv2 = nn.Conv1d(conv_channels, conv_channels, kernel_size=kernel_size, padding=padding)
        self.norm2 = nn.LayerNorm([conv_channels, pd])
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.project = nn.Linear(conv_channels, out_dim)
        self.out_norm = nn.LayerNorm(out_dim)

    def _forward_one(self, Q: torch.Tensor) -> torch.Tensor:
        """Q has shape (E, pd, pd). Returns (E, out_dim_pre_project=conv_channels)."""
        h = self.conv1(Q)                      # (E, conv_channels, pd)
        h = self.norm1(h)
        h = torch.relu(h)
        h = self.conv2(h)                      # (E, conv_channels, pd)
        h = self.norm2(h)
        h = torch.relu(h)
        h = self.pool(h).squeeze(-1)           # (E, conv_channels)
        return h

    def forward(self, qfim_edge: torch.Tensor) -> torch.Tensor:
        """qfim_edge has shape (E, pd*pd). Returns (E, out_dim)."""
        E = qfim_edge.shape[0]
        Q = qfim_edge.view(E, self.pd, self.pd)
        h_a = self._forward_one(Q)             # qubit-j axis as length
        h_b = self._forward_one(Q.transpose(1, 2))  # qubit-i axis as length
        h = 0.5 * (h_a + h_b)                  # symmetric under i<->j swap
        out = self.project(h)
        out = self.out_norm(out)
        return out


class QFIMGNNConv(nn.Module):
    """
    InvariantGNN with a Conv1d-based QFIM edge compressor.

    Edge feature (per edge): concat(geom_attr (4), qfim_small (8)) = 12 dims.
    All other components mirror QFIMGNN / InvariantGNN (InvariantMP stack,
    mean pool, target standardization in eV).
    """

    DEFAULT_TARGET_INDEX = 4

    def __init__(
        self,
        node_dim: int = 9,
        qfim_per_qubit_dim: int = 6,
        qfim_conv_channels: int = 16,
        qfim_kernel_size: int = 3,
        qfim_out_dim: int = 8,
        hidden_dim: int = 64,
        num_layers: int = 6,
        include_dihedral: bool = True,
        coord_cols: slice = slice(4, 7),
        out_dim: int = 1,
        pooling: str = "mean",
    ):
        super().__init__()
        if pooling not in _POOLINGS:
            raise ValueError(f"pooling must be one of {list(_POOLINGS)}; got {pooling!r}")
        self.coord_cols = coord_cols
        self.include_dihedral = include_dihedral
        self.geom_edge_dim = 4 if include_dihedral else 3
        self.qfim_out_dim = qfim_out_dim
        self.edge_in_dim = self.geom_edge_dim + self.qfim_out_dim
        self._pool = _POOLINGS[pooling]

        self.qfim_compress = _QFIMConvCompress(
            pd=qfim_per_qubit_dim,
            conv_channels=qfim_conv_channels,
            kernel_size=qfim_kernel_size,
            out_dim=qfim_out_dim,
        )

        self.node_embed = nn.Linear(node_dim, hidden_dim)
        self.edge_embed = nn.Sequential(
            nn.Linear(self.edge_in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.layers = nn.ModuleList(
            [InvariantMP(hidden_dim, hidden_dim) for _ in range(num_layers)]
        )
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, out_dim),
        )

        self.register_buffer("target_mean", torch.zeros(1))
        self.register_buffer("target_std", torch.ones(1))
        self.register_buffer("_stats_fitted", torch.tensor(False))

    @torch.no_grad()
    def fit_target_stats(
        self,
        loader: Iterable,
        target_index: Optional[int] = None,
    ) -> Tuple[float, float]:
        if target_index is None:
            target_index = self.DEFAULT_TARGET_INDEX

        count = 0
        mean = 0.0
        m2 = 0.0

        for batch in loader:
            y = batch.y
            if y.dim() > 1:
                y = y[:, target_index]
            y = y.flatten().double()
            n_b = y.numel()
            if n_b == 0:
                continue
            mean_b = y.mean().item()
            m2_b = ((y - mean_b) ** 2).sum().item()
            delta = mean_b - mean
            new_count = count + n_b
            mean = mean + delta * n_b / new_count
            m2 = m2 + m2_b + delta ** 2 * count * n_b / new_count
            count = new_count

        if count < 2:
            raise RuntimeError(
                f"fit_target_stats needs at least 2 samples; got {count}."
            )

        std = (m2 / (count - 1)) ** 0.5
        if std < 1e-8:
            raise RuntimeError(
                f"Target std is ~0 ({std:.2e}); check target_index={target_index}."
            )

        device = self.target_mean.device
        self.target_mean.copy_(torch.tensor([mean], device=device))
        self.target_std.copy_(torch.tensor([std], device=device))
        self._stats_fitted.copy_(torch.tensor(True))
        return float(mean), float(std)

    @property
    def stats_fitted(self) -> bool:
        return bool(self._stats_fitted.item())

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        qfim_block: torch.Tensor,
        qfim_nq: int,
        batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if not bool(self._stats_fitted.item()):
            raise RuntimeError(
                "QFIMGNNConv.forward called before fit_target_stats. "
                "Call model.fit_target_stats(train_loader) once before "
                "training / evaluation, or load a checkpoint saved after it."
            )

        coords = x[:, self.coord_cols]
        geom_attr = build_invariant_edge_attr(
            edge_attr, coords, edge_index,
            include_dihedral=self.include_dihedral,
        )
        if batch is None:
            batch = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)

        qfim_edge = _gather_edge_qfim(qfim_block, qfim_nq, edge_index, batch)
        qfim_small = self.qfim_compress(qfim_edge)
        e_in = torch.cat([geom_attr, qfim_small], dim=-1)

        h = self.node_embed(x)
        e = self.edge_embed(e_in)
        for layer in self.layers:
            h = layer(h, edge_index, e)
        g = self._pool(h, batch)
        z = self.readout(g).squeeze(-1)

        return z * self.target_std + self.target_mean
