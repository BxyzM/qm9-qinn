"""
Invariant GNN + QFIM edge features.

Identical to `gnn_invariant.InvariantGNN` except that each edge also carries
the flattened (pd, pd) QFIM sub-block linking the rot-gate parameters of the
two endpoint atoms. This isolates the QFIM contribution as the single
architectural change, so any delta vs. the invariant baseline is attributable
to the QFIM features and nothing else.

Edge feature layout per edge:
    [ geometry (3 or 4 dims) || flatten(QFIM sub-block) (pd*pd dims) ]

Everything else -- node embedding, message form (`MLP([x_i, x_j, e])`, sum
aggregation via InvariantMP), hidden dim, pooling, target standardization --
mirrors InvariantGNN.

Atoms beyond the qubit budget (n_heavy > n_qubits) and edges where either
endpoint is out of budget yield a zero QFIM block.
"""

from __future__ import annotations

from typing import Iterable, Optional, Tuple

import torch
import torch.nn as nn
from torch_geometric.nn import global_add_pool, global_max_pool, global_mean_pool

from .gnn_invariant import InvariantMP, build_invariant_edge_attr


_POOLINGS = {
    "add": global_add_pool,
    "mean": global_mean_pool,
    "max": global_max_pool,
}


def _gather_edge_qfim(
    qfim_block: torch.Tensor,      # (B, nq, nq, pd, pd)
    qfim_nq: int,                  # qubits per molecule
    edge_index: torch.Tensor,      # (2, E) global node indices
    batch: torch.Tensor,           # (N,) graph id per node
) -> torch.Tensor:
    """
    For each edge, fetch the (pd, pd) QFIM sub-block linking the rot-gate
    params of the source and destination atoms, flatten to (pd*pd,).

    Atoms with local index >= n_qubits yield a zero block; edges where either
    endpoint is out of budget also yield zero.
    """
    src, dst = edge_index[0], edge_index[1]
    if src.numel() == 0:
        _, _, _, pd, _ = qfim_block.shape
        return qfim_block.new_zeros((0, pd * pd))

    # Local atom index within its graph: batch is sorted by graph id (PyG
    # invariant), so graph starts are at the first occurrence of each id.
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

    sub = qfim_block[mol_id, li_clamped, lj_clamped]           # (E, pd, pd)
    sub = sub * in_budget.view(-1, 1, 1).to(sub.dtype)
    return sub.reshape(sub.shape[0], -1)                       # (E, pd*pd)


class QFIMGNN(nn.Module):
    """
    InvariantGNN with QFIM sub-blocks concatenated onto each edge's invariant
    geometric features. Mirrors InvariantGNN in every other respect, including
    target standardization in physical units (eV).
    """

    # PyG QM9 target layout: 4 = HOMO-LUMO gap (eV).
    DEFAULT_TARGET_INDEX = 4

    def __init__(
        self,
        node_dim: int = 9,
        qfim_per_qubit_dim: int = 6,       # pd; per-edge QFIM dim is pd*pd
        hidden_dim: int = 64,
        num_layers: int = 6,
        include_dihedral: bool = True,
        coord_cols: slice = slice(4, 7),
        out_dim: int = 1,
        pooling: str = "mean",             # intensive target -> mean
    ):
        super().__init__()
        if pooling not in _POOLINGS:
            raise ValueError(f"pooling must be one of {list(_POOLINGS)}; got {pooling!r}")
        self.coord_cols = coord_cols
        self.include_dihedral = include_dihedral
        self.geom_edge_dim = 4 if include_dihedral else 3
        self.qfim_edge_dim = qfim_per_qubit_dim * qfim_per_qubit_dim
        self.edge_in_dim = self.geom_edge_dim + self.qfim_edge_dim
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
        """
        Streaming (Welford) mean/std of the target over `loader`. MUST be
        called on the training split only. Stats save with state_dict.
        """
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
        coords = x[:, self.coord_cols]
        geom_attr = build_invariant_edge_attr(
            edge_attr, coords, edge_index,
            include_dihedral=self.include_dihedral,
        )
        if batch is None:
            batch = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)

        qfim_edge = _gather_edge_qfim(qfim_block, qfim_nq, edge_index, batch)
        e_in = torch.cat([geom_attr, qfim_edge], dim=-1)

        h = self.node_embed(x)
        e = self.edge_embed(e_in)
        for layer in self.layers:
            h = layer(h, edge_index, e)
        g = self._pool(h, batch)
        z = self.readout(g).squeeze(-1)

        return z * self.target_std + self.target_mean
