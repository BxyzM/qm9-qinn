"""
QFIMGNN variant using the paper's C_ij reduction.

Motivation
----------
The jet-tomography QINN paper (Binder/Bal et al.) reduces the full per-pair
QFIM sub-block Q[i,j] of shape (pd, pd) to a single scalar via the
Frobenius norm:

    C_ij = ||Q[i,j]||_F

The resulting NxN correlation matrix captures pairwise quantum coupling
magnitude while discarding within-block detail that is likely redundant
with geometry. The paper reports that feeding C_ij as an edge feature
into a classical GNN improves classification AUC over a geometry-only
baseline.

This model tests the paper's method directly on QM9 HOMO-LUMO gap
regression. It is strictly the paper's minimal injection -- one scalar
per edge -- with no other QFIM information.

Edge feature layout per edge:
    [ bond_type, distance, bond_angle, dihedral, C_ij ]  -> 5 dims

Everything else (node embed, InvariantMP stack, mean pool, readout,
target standardization) mirrors InvariantGNN / QFIMGNN.
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


def _cij_from_edge_qfim(qfim_edge: torch.Tensor, pd: int) -> torch.Tensor:
    """
    Reduce each edge's flattened (pd, pd) QFIM sub-block to a single scalar
    via the Frobenius norm:   C_ij = ||Q[i,j]||_F

    Numerically equivalent to sqrt(sum of squared entries) per edge. Kept
    as a function so it is easy to swap in alternative reductions (trace,
    max singular value, etc.) for ablations without touching the model.
    """
    E = qfim_edge.shape[0]
    Q = qfim_edge.view(E, pd, pd)
    # matrix_norm with ord='fro' is equivalent to sqrt(sum(Q*Q)) but uses
    # the numerically-stable path. Output shape: (E,).
    return torch.linalg.matrix_norm(Q, ord="fro")


class QFIMGNNCij(nn.Module):
    """
    InvariantGNN + single scalar C_ij = ||Q[i,j]||_F per edge.

    Edge input: geom_attr (4 dims) + C_ij (1 dim) = 5 dims.
    All other components mirror QFIMGNN.
    """

    DEFAULT_TARGET_INDEX = 4

    def __init__(
        self,
        node_dim: int = 9,
        qfim_per_qubit_dim: int = 6,       # pd; used only to reshape Q[i,j]
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
        self.qfim_per_qubit_dim = qfim_per_qubit_dim
        self.geom_edge_dim = 4 if include_dihedral else 3
        self.edge_in_dim = self.geom_edge_dim + 1        # +1 for C_ij
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
                "QFIMGNNCij.forward called before fit_target_stats. "
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
        cij = _cij_from_edge_qfim(qfim_edge, pd=self.qfim_per_qubit_dim)     # (E,)
        e_in = torch.cat([geom_attr, cij.unsqueeze(-1)], dim=-1)

        h = self.node_embed(x)
        e = self.edge_embed(e_in)
        for layer in self.layers:
            h = layer(h, edge_index, e)
        g = self._pool(h, batch)
        z = self.readout(g).squeeze(-1)

        return z * self.target_std + self.target_mean
