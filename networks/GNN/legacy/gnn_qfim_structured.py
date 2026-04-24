"""
QFIMGNN variant with per-layer 3x3 sub-block summaries.

Background
----------
The QFIM cross-block between qubit i and qubit j has shape (pd, pd) = (6, 6)
with pd = num_layers * ops_per_layer = 2 * 3. Stacking the per-qubit params
in (layer, op) order, the 6x6 block naturally decomposes into four 3x3
sub-blocks that correspond to (layer_of_i, layer_of_j) in {(0,0), (0,1),
(1,0), (1,1)}. Visualizing the QFIM heatmap shows these 3x3 sub-blocks have
internally coherent structure (same ansatz gate, same circuit moment).

Hypothesis
----------
The model can extract physical signal from (Frobenius norm, trace, max
singular value) of each of the 4 sub-blocks more easily than from the raw
36-dim flatten alone. Frobenius = overall coupling magnitude; trace = signed
sum of eigenvalues; max-SV = dominant mode. Three rotation-invariant
scalars per sub-block, across 4 sub-blocks = 12 extra scalar features per
edge. Concatenated alongside the raw flatten to keep information strictly
additive (no loss vs the default QFIMGNN).

Only the edge feature preparation differs from QFIMGNN; everything else
(node embed, InvariantMP stack, readout, target standardization) is
identical.
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


def _structured_qfim_summaries(
    qfim_edge: torch.Tensor,            # (E, pd*pd)
    num_layers: int,
    ops_per_layer: int,
) -> torch.Tensor:
    """
    Decompose each edge's (pd, pd) QFIM block into num_layers**2 sub-blocks
    of shape (ops, ops), and return (Frobenius, trace, max singular value)
    for each sub-block.

    Per-edge output dim = num_layers**2 * 3.

    The 6-dim parameter axis stacks params as (layer, op) in C order, so
    reshaping the (pd, pd) block to (num_layers, ops, num_layers, ops)
    groups rows by layer_i/op_i and cols by layer_j/op_j. This matches the
    way bioQINN stores the rot-gate weights: weights.shape = (num_layers,
    ops, n_qubits), flattened C-order with qubit fastest (handled upstream
    by the loader; here each per-qubit 6-dim axis is already (layer, op)).
    """
    pd = num_layers * ops_per_layer
    E = qfim_edge.shape[0]
    Q = qfim_edge.view(E, pd, pd)
    # Split axes: (E, num_layers, ops, num_layers, ops) -> transpose to
    # (E, num_layers, num_layers, ops, ops) so axis 1=layer_i, 2=layer_j.
    Qb = Q.view(E, num_layers, ops_per_layer, num_layers, ops_per_layer)
    Qb = Qb.permute(0, 1, 3, 2, 4).contiguous()
    # Qb now has shape (E, num_layers, num_layers, ops, ops).
    # Reshape so the 4 sub-blocks are batched: (E * L * L, ops, ops).
    L = num_layers
    sub = Qb.view(E * L * L, ops_per_layer, ops_per_layer)

    frob = torch.linalg.matrix_norm(sub, ord="fro")                        # (E*L*L,)
    trace = sub.diagonal(dim1=-2, dim2=-1).sum(-1)                         # (E*L*L,)
    # torch.linalg.svdvals is stable for small matrices; take the largest.
    svals = torch.linalg.svdvals(sub)                                      # (E*L*L, ops)
    max_sv = svals[:, 0]                                                   # (E*L*L,)

    summ = torch.stack([frob, trace, max_sv], dim=-1)                      # (E*L*L, 3)
    return summ.view(E, L * L * 3)                                         # (E, 12)


class QFIMGNNStructured(nn.Module):
    """
    InvariantGNN + edge QFIM (36 dims) + per-sub-block structured summaries
    (num_layers**2 * 3 extra scalars per edge). Strict superset of
    QFIMGNN's edge features.
    """

    DEFAULT_TARGET_INDEX = 4

    def __init__(
        self,
        node_dim: int = 9,
        qfim_per_qubit_dim: int = 6,       # pd = num_layers * ops_per_layer
        qfim_num_layers: int = 2,
        qfim_ops_per_layer: int = 3,
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
        if qfim_num_layers * qfim_ops_per_layer != qfim_per_qubit_dim:
            raise ValueError(
                f"qfim_num_layers ({qfim_num_layers}) * qfim_ops_per_layer "
                f"({qfim_ops_per_layer}) must equal qfim_per_qubit_dim "
                f"({qfim_per_qubit_dim})"
            )
        self.coord_cols = coord_cols
        self.include_dihedral = include_dihedral
        self.qfim_num_layers = qfim_num_layers
        self.qfim_ops_per_layer = qfim_ops_per_layer
        self.geom_edge_dim = 4 if include_dihedral else 3
        self.qfim_edge_dim = qfim_per_qubit_dim * qfim_per_qubit_dim
        self.qfim_summary_dim = (qfim_num_layers * qfim_num_layers) * 3
        self.edge_in_dim = (
            self.geom_edge_dim + self.qfim_edge_dim + self.qfim_summary_dim
        )
        self._pool = _POOLINGS[pooling]

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
        qfim_block: torch.Tensor,           # (B, nq, nq, pd, pd)
        qfim_nq: int,
        batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if not bool(self._stats_fitted.item()):
            raise RuntimeError(
                "QFIMGNNStructured.forward called before fit_target_stats. "
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
        qfim_summ = _structured_qfim_summaries(
            qfim_edge, self.qfim_num_layers, self.qfim_ops_per_layer,
        )
        e_in = torch.cat([geom_attr, qfim_edge, qfim_summ], dim=-1)

        h = self.node_embed(x)
        e = self.edge_embed(e_in)
        for layer in self.layers:
            h = layer(h, edge_index, e)
        g = self._pool(h, batch)
        z = self.readout(g).squeeze(-1)

        return z * self.target_std + self.target_mean
