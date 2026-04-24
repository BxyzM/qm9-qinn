"""
QFIMGNN variant with node-level QFIM diagonal injection.

Motivation
----------
Prior QFIM experiments (QFIMGNN, QFIMGNNStructured, QFIMGNNConv) all used
only the off-diagonal sub-blocks Q[i, j] for i != j -- the entanglement
between atoms -- attached as edge features. All three plateau at the
baseline InvariantGNN val_loss, suggesting that off-diagonal QFIM is
redundant with geometric edge features for HOMO-LUMO gap prediction.

The QFIM also contains diagonal sub-blocks Q[i, i] (one per qubit). These
are the **self-coupling** of qubit i's rot-gate parameters -- an atom-local
quantity that depends on that atom's encoding but not on its neighbors.
They were never consulted by the previous models because PyG bond edges
don't include self-loops (i != j always), so Q[i, i] was generated, stored
in the HDF5, and silently dropped.

This model exposes the diagonal by gathering Q[i, i] per atom and mixing
it into the node embedding via a dedicated branch. Off-diagonal edge
injection is kept unchanged, so QFIMGNNNode is a strict superset of
QFIMGNN in terms of information consumed.

Architecture
------------
    per atom:
        Q_diag = Q[mol, i, i]                       # (N, pd, pd)
        atom_qfim = flatten(Q_diag)                  # (N, pd*pd)
        h_q = MLP(atom_qfim)                         # (N, hidden_dim)

    per node:
        h = node_embed(x) + node_qfim_embed(atom_qfim)

    per edge (unchanged from QFIMGNN):
        e_in = concat(geom_attr, flatten(Q[mol, i, j]))     # (E, 4 + pd*pd)
        e = edge_embed(e_in)

    then 6 x InvariantMP + mean pool + readout as usual.

The additive combination in the node embedding is standard for
multi-channel inductive biases: each branch has its own weights and
cannot starve the other for capacity in a shared linear projection.
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


def _gather_node_qfim(
    qfim_block: torch.Tensor,    # (B, nq, nq, pd, pd)
    qfim_nq: int,
    batch: torch.Tensor,         # (N,)
) -> torch.Tensor:
    """
    For each atom i in the batched graph, fetch Q[mol(i), local(i), local(i)]
    -- its own diagonal QFIM sub-block -- and flatten to (pd*pd,).

    Atoms with local index >= n_qubits (i.e. past the qubit budget) yield
    a zero block, mirroring _gather_edge_qfim's treatment of out-of-budget
    edges.
    """
    if batch.numel() == 0:
        _, _, _, pd, _ = qfim_block.shape
        return qfim_block.new_zeros((0, pd * pd))

    # Local atom index within its graph: batch is sorted by graph id (PyG
    # invariant), so graph starts are at the first occurrence of each id.
    n_nodes = batch.numel()
    change = torch.ones_like(batch, dtype=torch.bool)
    change[1:] = batch[1:] != batch[:-1]
    graph_starts = torch.nonzero(change, as_tuple=False).view(-1)
    local_idx = torch.arange(n_nodes, device=batch.device) - graph_starts[batch]

    in_budget = local_idx < qfim_nq
    li_clamped = local_idx.clamp(max=qfim_nq - 1)

    sub = qfim_block[batch, li_clamped, li_clamped]        # (N, pd, pd)
    sub = sub * in_budget.view(-1, 1, 1).to(sub.dtype)
    return sub.reshape(sub.shape[0], -1)                    # (N, pd*pd)


class QFIMGNNNode(nn.Module):
    """
    InvariantGNN + QFIM on both node-level (diagonal self-coupling) and
    edge-level (off-diagonal pair coupling) channels.
    """

    DEFAULT_TARGET_INDEX = 4

    def __init__(
        self,
        node_dim: int = 9,
        qfim_per_qubit_dim: int = 6,       # pd
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
        self.qfim_block_dim = qfim_per_qubit_dim * qfim_per_qubit_dim    # pd*pd
        self.edge_in_dim = self.geom_edge_dim + self.qfim_block_dim
        self._pool = _POOLINGS[pooling]

        # Two-branch node embedding. Atomic features and the QFIM diagonal
        # each get their own MLP to the hidden space; the sum avoids one
        # channel starving the other in a shared linear projection.
        self.node_embed = nn.Linear(node_dim, hidden_dim)
        self.node_qfim_embed = nn.Sequential(
            nn.Linear(self.qfim_block_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

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
                "QFIMGNNNode.forward called before fit_target_stats. "
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

        # Node-level QFIM: diagonal sub-block Q[i, i] per atom.
        atom_qfim = _gather_node_qfim(qfim_block, qfim_nq, batch)          # (N, pd*pd)

        # Edge-level QFIM: off-diagonal sub-block Q[i, j] per bonded edge.
        # (Bond edges never have i == j, so the diagonal is not double-counted.)
        qfim_edge = _gather_edge_qfim(qfim_block, qfim_nq, edge_index, batch)
        e_in = torch.cat([geom_attr, qfim_edge], dim=-1)

        h = self.node_embed(x) + self.node_qfim_embed(atom_qfim)
        e = self.edge_embed(e_in)
        for layer in self.layers:
            h = layer(h, edge_index, e)
        g = self._pool(h, batch)
        z = self.readout(g).squeeze(-1)

        return z * self.target_std + self.target_mean
