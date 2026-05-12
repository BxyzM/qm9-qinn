"""
QFIMAttnGNN: QFIM coupling matrix defines both the graph topology and the
attention weights. Bond-graph edges are not used.

Architecture
------------
- Edges = every directed pair (i, j) with i != j and both atoms within the
  qubit budget. The "adjacency" comes from QFIM, not chemistry.
- Edge features = RBF expansion of the Euclidean distance between i and j.
  No bond angles or dihedrals (those concepts are bond-graph specific).
- Attention scores = beta * ||Q[i, j]||_F, with beta a learnable per-layer
  temperature scalar. Softmax is taken over j (neighbours of source i).
- Messages = msg_mlp([h_i, h_j, e_ij]); summed with attention weights.

The whole point of Option A is to test whether QFIM, used as both topology
and attention signal, carries useful coupling information beyond what the
geometric bond graph captures.

Notes
-----
- Atoms with local index >= qfim_nq are silently dropped from the QFIM
  graph (they have no QFIM block to source from). Their node features
  still go through the readout pool, but they receive no messages and
  send none. This is the same heavy-atom-budget rule QFIMGNN uses.
- The graph is dense: nq*(nq-1) edges per molecule. With nq=10 this is
  90 edges per molecule, comparable to the bond graph for small QM9
  molecules — runtime should not be drastically different.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax as edge_softmax

from .gnn import (
    GNN,
    MAX_NEIGHBORS,
    MAX_CHAINS,
    _COORD_COLS,
    _make_activation,
    gaussian_rbf,
)
from .gnn_qfim import _gather_edge_qfim


# ---------------------------------------------------------------------------
# QFIM-adjacency graph builder
# ---------------------------------------------------------------------------

def _build_qfim_graph(
    coords: torch.Tensor,           # (N, 3)
    batch: torch.Tensor,            # (N,) graph id per node
    qfim_block: torch.Tensor,       # (B, nq, nq, pd, pd)
    qfim_nq: int,
    rbf_centers: torch.Tensor,
    rbf_gamma: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build a dense QFIM-adjacency graph over the qubit-budgeted atoms of
    each molecule.

    Returns:
        edge_index: (2, E) global node indices, long.
        edge_attr:  (E, K) RBF expansion of pairwise distances.
        coupling:   (E,) scalar Frobenius norm of Q[i, j], used as
                    attention pre-score.

    All three are concatenated across molecules in batch order.
    """
    device = coords.device
    n_nodes = batch.numel()
    if n_nodes == 0:
        K = rbf_centers.numel()
        empty_edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
        empty_edge_attr = coords.new_zeros((0, K))
        empty_coupling = coords.new_zeros((0,))
        return empty_edge_index, empty_edge_attr, empty_coupling

    # Local atom indices: position of each node within its own molecule.
    change = torch.ones_like(batch, dtype=torch.bool)
    change[1:] = batch[1:] != batch[:-1]
    graph_starts = torch.nonzero(change, as_tuple=False).view(-1)
    local_idx = torch.arange(n_nodes, device=device) - graph_starts[batch]

    # Per-molecule sizes and atom counts within the qubit budget.
    B = int(qfim_block.shape[0])
    counts = torch.bincount(batch, minlength=B)                   # (B,) atoms per mol
    in_budget_count = torch.minimum(
        counts, torch.tensor(qfim_nq, device=device)
    )                                                              # (B,) <= nq

    # Build edges only between in-budget atoms. For each molecule with
    # k = in_budget_count[m] in-budget atoms, we add k*(k-1) directed edges
    # (i != j) over the local indices [0, k).
    edges_per_mol = in_budget_count * (in_budget_count - 1)        # (B,)
    total_E = int(edges_per_mol.sum().item())
    if total_E == 0:
        K = rbf_centers.numel()
        empty_edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
        empty_edge_attr = coords.new_zeros((0, K))
        empty_coupling = coords.new_zeros((0,))
        return empty_edge_index, empty_edge_attr, empty_coupling

    # For each molecule m, generate the (i, j) local-index pairs i != j.
    # Vectorised: for each m, repeat 0..k-1 once for src (k times each)
    # and tile 0..k-1 for dst, then drop diagonal.
    mol_repeat = torch.repeat_interleave(
        torch.arange(B, device=device), edges_per_mol
    )                                                              # (E,) molecule id per edge
    # within-molecule offset (0..edges_per_mol[m]-1) per edge
    mol_edge_offsets = (
        torch.arange(total_E, device=device)
        - torch.repeat_interleave(
            torch.cumsum(edges_per_mol, 0) - edges_per_mol, edges_per_mol
        )
    )
    k_per_edge = in_budget_count[mol_repeat]                       # (E,)
    # local i = offset // (k - 1), local j_dense = offset %  (k - 1)
    # then j = j_dense if j_dense < i else j_dense + 1   (skips diagonal)
    denom = (k_per_edge - 1).clamp(min=1)
    li = mol_edge_offsets // denom
    j_dense = mol_edge_offsets % denom
    lj = torch.where(j_dense < li, j_dense, j_dense + 1)

    # Convert local indices to global node indices.
    src = graph_starts[mol_repeat] + li
    dst = graph_starts[mol_repeat] + lj
    edge_index = torch.stack([src, dst], dim=0)                    # (2, E)

    # Edge feature: RBF of Euclidean distance between i and j.
    d = (coords[src] - coords[dst]).norm(dim=-1)                   # (E,)
    edge_attr = gaussian_rbf(d, rbf_centers, rbf_gamma)            # (E, K)

    # Coupling strength: Frobenius norm of Q[mol, li, lj].
    sub = qfim_block[mol_repeat, li, lj]                           # (E, pd, pd)
    coupling = torch.linalg.matrix_norm(sub, ord="fro")            # (E,)

    return edge_index, edge_attr, coupling


# ---------------------------------------------------------------------------
# QFIM-attention message-passing layer
# ---------------------------------------------------------------------------

class _QFIMAttnMP(MessagePassing):
    """
    Sum-aggregating MP block whose messages are scaled by a per-edge weight
    derived from QFIM coupling C_ij. The weighting form is selected by
    ``gate_mode``:

      - "softmax"  (default): alpha_ij = softmax_dst( beta * C_ij ).
                              Standard attention; weights into each node sum
                              to 1; magnitude information is lost.
      - "uniform"           : alpha_ij = 1 / in_degree(dst). Ablation: same
                              graph, attention discarded entirely.
      - "softplus_gate"     : g_ij = 1 + alpha * softplus(beta * C_ij - theta).
                              Learnable multiplicative gate, bounded below
                              by 1 ("amplify only"). Magnitude of total
                              incoming signal scales with both coupling and
                              count of neighbours. Falls back to baseline
                              gracefully (alpha -> 0).
      - "raw"               : g_ij = beta * C_ij. Simplest possible
                              multiplicative weight. Magnitude unbounded;
                              relies on per-batch scale of C_ij being
                              somewhat stable.

    Conventions:
      - softmax / uniform produce *probabilities* (sum-to-1 per dst).
      - softplus_gate / raw produce *gain factors* (no normalisation).
    """

    _VALID_GATE_MODES = ("softmax", "uniform", "softplus_gate", "raw")

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        activation: str = "silu",
        msg_layers: int = 2,
        beta_init: float = 1.0,
        attn_uniform: bool = False,
        gate_mode: str = "softmax",
        gate_alpha_init: float = 1.0,
        gate_theta_init: float = 0.0,
    ):
        super().__init__(aggr="add", node_dim=0)
        msg_in = 2 * node_dim + edge_dim
        if msg_layers == 1:
            self.msg_mlp = nn.Sequential(
                nn.Linear(msg_in, node_dim),
                nn.LayerNorm(node_dim),
                _make_activation(activation),
            )
        elif msg_layers == 2:
            hidden = max(node_dim, msg_in // 2)
            self.msg_mlp = nn.Sequential(
                nn.Linear(msg_in, hidden),
                nn.LayerNorm(hidden),
                _make_activation(activation),
                nn.Linear(hidden, node_dim),
                nn.LayerNorm(node_dim),
                _make_activation(activation),
            )
        else:
            raise ValueError(f"msg_layers must be 1 or 2; got {msg_layers}")

        # attn_uniform is the legacy flag that hard-selects "uniform" mode.
        # gate_mode is the new, more expressive selector. If attn_uniform is
        # True we override gate_mode to "uniform" for backward compatibility.
        if attn_uniform:
            gate_mode = "uniform"
        if gate_mode not in self._VALID_GATE_MODES:
            raise ValueError(
                f"gate_mode must be one of {self._VALID_GATE_MODES}; got {gate_mode!r}"
            )
        self.gate_mode = gate_mode

        # `beta` is the temperature/scale on coupling. Used by all modes
        # except "uniform" (where it's a buffer for state_dict symmetry).
        if gate_mode == "uniform":
            self.register_buffer("beta", torch.tensor(beta_init))
        else:
            self.beta = nn.Parameter(torch.tensor(beta_init))

        # alpha and theta only used by softplus_gate.
        if gate_mode == "softplus_gate":
            self.gate_alpha = nn.Parameter(torch.tensor(gate_alpha_init))
            self.gate_theta = nn.Parameter(torch.tensor(gate_theta_init))

    def _compute_weights(
        self, edge_index: torch.Tensor, coupling: torch.Tensor, x: torch.Tensor,
    ) -> torch.Tensor:
        if self.gate_mode == "uniform":
            dst = edge_index[1]
            ones = torch.ones_like(dst, dtype=x.dtype)
            in_degree = torch.zeros(x.shape[0], dtype=x.dtype, device=x.device)
            in_degree.scatter_add_(0, dst, ones)
            return 1.0 / in_degree[dst].clamp(min=1.0)
        if self.gate_mode == "softmax":
            score = self.beta * coupling
            return edge_softmax(score, edge_index[1])
        if self.gate_mode == "softplus_gate":
            # g_ij = 1 + alpha * softplus(beta * C - theta), per edge.
            return 1.0 + self.gate_alpha * torch.nn.functional.softplus(
                self.beta * coupling - self.gate_theta
            )
        if self.gate_mode == "raw":
            return self.beta * coupling
        raise RuntimeError(f"unhandled gate_mode={self.gate_mode!r}")

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        coupling: torch.Tensor,
    ) -> torch.Tensor:
        alpha = self._compute_weights(edge_index, coupling, x)
        out = self.propagate(
            edge_index, x=x, edge_attr=edge_attr, alpha=alpha,
        )
        return x + out

    def message(
        self,
        x_i: torch.Tensor,
        x_j: torch.Tensor,
        edge_attr: torch.Tensor,
        alpha: torch.Tensor,
    ) -> torch.Tensor:
        m = self.msg_mlp(torch.cat([x_i, x_j, edge_attr], dim=-1))
        return m * alpha.unsqueeze(-1)


# ---------------------------------------------------------------------------
# QFIM bond-graph residual gate layer
# ---------------------------------------------------------------------------

class _QFIMBondGateMP(MessagePassing):
    """
    v36-style bond-graph message passing with a residual multiplicative QFIM
    gate on each message:

        m_ij = msg_mlp([h_i, h_j, e_ij])
        g_ij = 1 + alpha * softplus(beta * C_ij - theta)
        out_i = sum_j g_ij * m_ij

    With alpha_init=0 this is exactly the baseline MP layer at
    initialisation. The model must learn to turn QFIM on.
    """

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        activation: str = "silu",
        msg_layers: int = 2,
        per_layer_edge_update: bool = True,
        beta_init: float = 1.0,
        gate_alpha_init: float = 0.0,
        gate_theta_init: float = 0.0,
    ):
        super().__init__(aggr="add")
        msg_in = 2 * node_dim + edge_dim
        if msg_layers == 1:
            self.msg_mlp = nn.Sequential(
                nn.Linear(msg_in, node_dim),
                nn.LayerNorm(node_dim),
                _make_activation(activation),
            )
        elif msg_layers == 2:
            hidden = max(node_dim, msg_in // 2)
            self.msg_mlp = nn.Sequential(
                nn.Linear(msg_in, hidden),
                nn.LayerNorm(hidden),
                _make_activation(activation),
                nn.Linear(hidden, node_dim),
                nn.LayerNorm(node_dim),
                _make_activation(activation),
            )
        else:
            raise ValueError(f"msg_layers must be 1 or 2; got {msg_layers}")

        self._edge_update_mlp = None
        if per_layer_edge_update:
            self._edge_update_mlp = nn.Sequential(
                nn.Linear(edge_dim, edge_dim),
                _make_activation(activation),
            )

        self.beta = nn.Parameter(torch.tensor(float(beta_init)))
        self.gate_alpha = nn.Parameter(torch.tensor(float(gate_alpha_init)))
        self.gate_theta = nn.Parameter(torch.tensor(float(gate_theta_init)))

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        coupling: torch.Tensor,
    ) -> torch.Tensor:
        if self._edge_update_mlp is not None:
            edge_attr = edge_attr + self._edge_update_mlp(edge_attr)
        gate = 1.0 + self.gate_alpha * torch.nn.functional.softplus(
            self.beta * coupling - self.gate_theta
        )
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, gate=gate)
        return x + out

    def message(
        self,
        x_i: torch.Tensor,
        x_j: torch.Tensor,
        edge_attr: torch.Tensor,
        gate: torch.Tensor,
    ) -> torch.Tensor:
        m = self.msg_mlp(torch.cat([x_i, x_j, edge_attr], dim=-1))
        return m * gate.unsqueeze(-1)


# ---------------------------------------------------------------------------
# QFIMAttnGNN
# ---------------------------------------------------------------------------

class QFIMAttnGNN(GNN):
    """
    QFIM-as-graph + QFIM-as-attention.

    Inherits node-feature path from GNN (Z embedding + xyz + node_mlp). The
    inherited edge_mlp/build_edge_feat path is *not used* here -- edges are
    built from the QFIM adjacency, not bonds. The MP stack uses
    _QFIMAttnMP layers that compute softmax-attention from a Frobenius-norm
    coupling score with a learnable per-layer temperature.
    """

    def __init__(
        self,
        num_mp_layers: int = 6,
        embed_z_dim: int = 16,
        node_mlp_dims: Tuple[int, ...] = (19, 32, 64, 64, 32),
        edge_mlp_dims: Tuple[int, ...] = (28, 32, 32, 16, 8),
        max_neighbors: int = MAX_NEIGHBORS,
        max_chains: int = MAX_CHAINS,
        rbf_num_centers: int = 16,
        rbf_range: Tuple[float, float] = (0.0, 5.0),
        rbf_gamma: float = 4.0,
        pooling: str = "mean",
        activation: str = "silu",
        mlp_residual: bool = False,
        msg_layers: int = 2,
        per_layer_edge_update: bool = False,           # unused here; accepted for API symmetry
        qfim_per_qubit_dim: int = 6,
        qfim_attn_beta_init: float = 1.0,
        qfim_edge_dim: int = 16,                        # MP edge-feature dim after projection
        attn_uniform: bool = False,
        gate_mode: str = "softmax",
        gate_alpha_init: float = 1.0,
        gate_theta_init: float = 0.0,
    ):
        super().__init__(
            num_mp_layers=num_mp_layers,
            embed_z_dim=embed_z_dim,
            node_mlp_dims=node_mlp_dims,
            edge_mlp_dims=edge_mlp_dims,
            max_neighbors=max_neighbors,
            max_chains=max_chains,
            rbf_num_centers=rbf_num_centers,
            rbf_range=rbf_range,
            rbf_gamma=rbf_gamma,
            pooling=pooling,
            activation=activation,
            mlp_residual=mlp_residual,
            msg_layers=msg_layers,
            per_layer_edge_update=per_layer_edge_update,
        )
        self.qfim_pd = qfim_per_qubit_dim

        # Project the K-dim RBF distance feature to MP edge_dim. We do NOT
        # reuse the parent's edge_mlp because that one was sized for the
        # 28-dim raw bond features.
        self.qfim_edge_proj = nn.Sequential(
            nn.Linear(rbf_num_centers, qfim_edge_dim),
            _make_activation(activation),
            nn.LayerNorm(qfim_edge_dim),
        )

        # Replace the parent's MP layers with QFIM-attention layers using
        # the projected edge dim. attn_uniform=True flips them to uniform
        # aggregation on the same dense graph (ablation: tests whether
        # QFIM coupling is what helps, vs. dense graph topology alone).
        self.mp_layers = nn.ModuleList([
            _QFIMAttnMP(
                self.node_dim, qfim_edge_dim,
                activation=activation, msg_layers=msg_layers,
                beta_init=qfim_attn_beta_init,
                attn_uniform=attn_uniform,
                gate_mode=gate_mode,
                gate_alpha_init=gate_alpha_init,
                gate_theta_init=gate_theta_init,
            )
            for _ in range(num_mp_layers)
        ])

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,                       # bond graph -- IGNORED here
        edge_attr: torch.Tensor,                        # bond edge attrs -- IGNORED here
        qfim_block: torch.Tensor,                       # (B, nq, nq, pd, pd)
        qfim_nq: int,
        batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if batch is None:
            batch = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)

        coords = x[:, _COORD_COLS]

        node_in = self.build_node_feat(x)                           # (N, 19)
        h = self.node_mlp(node_in)                                  # (N, node_dim)

        qfim_edge_index, qfim_edge_rbf, coupling = _build_qfim_graph(
            coords=coords,
            batch=batch,
            qfim_block=qfim_block,
            qfim_nq=int(qfim_nq),
            rbf_centers=self.rbf_centers,
            rbf_gamma=self.rbf_gamma,
        )
        e = self.qfim_edge_proj(qfim_edge_rbf)                      # (E, qfim_edge_dim)

        for layer in self.mp_layers:
            h = layer(h, qfim_edge_index, e, coupling)

        g = self._pool_nodes(h, batch)
        z = self.readout(g).squeeze(-1)
        return z


# ---------------------------------------------------------------------------
# Option D: QFIM-attention on the existing bond graph
# ---------------------------------------------------------------------------

class QFIMBondAttnGNN(GNN):
    """
    Option D: QFIM coupling acts as the attention signal on the *baseline*
    bond graph. Topology, edge features, node path -- all identical to v36.
    The only change vs baseline is that messages are weighted by softmax-
    attention scored from per-edge QFIM coupling magnitudes:

        alpha_ij = softmax_dst( beta * ||Q[i, j]||_F )
        m_{i<-j} = alpha_ij * msg_mlp([h_i, h_j, e_ij])

    Compared to QFIMAttnGNN (Option A) this isolates the attention
    mechanism: no topology change, no loss of bond-angle / dihedral edge
    features. If this beats baseline cleanly, the result is unambiguous.
    """

    def __init__(
        self,
        num_mp_layers: int = 6,
        embed_z_dim: int = 16,
        node_mlp_dims: Tuple[int, ...] = (19, 32, 64, 64, 32),
        edge_mlp_dims: Tuple[int, ...] = (28, 32, 32, 16, 8),
        max_neighbors: int = MAX_NEIGHBORS,
        max_chains: int = MAX_CHAINS,
        rbf_num_centers: int = 16,
        rbf_range: Tuple[float, float] = (0.0, 5.0),
        rbf_gamma: float = 4.0,
        pooling: str = "mean",
        activation: str = "silu",
        mlp_residual: bool = False,
        msg_layers: int = 2,
        per_layer_edge_update: bool = False,
        qfim_per_qubit_dim: int = 6,
        qfim_attn_beta_init: float = 1.0,
    ):
        super().__init__(
            num_mp_layers=num_mp_layers,
            embed_z_dim=embed_z_dim,
            node_mlp_dims=node_mlp_dims,
            edge_mlp_dims=edge_mlp_dims,
            max_neighbors=max_neighbors,
            max_chains=max_chains,
            rbf_num_centers=rbf_num_centers,
            rbf_range=rbf_range,
            rbf_gamma=rbf_gamma,
            pooling=pooling,
            activation=activation,
            mlp_residual=mlp_residual,
            msg_layers=msg_layers,
            per_layer_edge_update=per_layer_edge_update,
        )
        self.qfim_pd = qfim_per_qubit_dim

        # Replace MP stack with QFIM-attention layers, keeping the parent's
        # edge_dim (8) since we keep the baseline edge_mlp output unchanged.
        self.mp_layers = nn.ModuleList([
            _QFIMAttnMP(
                self.node_dim, self.edge_dim,
                activation=activation, msg_layers=msg_layers,
                beta_init=qfim_attn_beta_init,
            )
            for _ in range(num_mp_layers)
        ])

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,                  # bond graph -- USED
        edge_attr: torch.Tensor,                   # bond edge attrs -- USED
        qfim_block: torch.Tensor,                  # (B, nq, nq, pd, pd)
        qfim_nq: int,
        batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if batch is None:
            batch = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)

        node_in = self.build_node_feat(x)
        h = self.node_mlp(node_in)
        e = self.build_edge_feat(x, edge_index, edge_attr)            # (E, edge_dim) = baseline

        # Per-bond-edge QFIM coupling magnitude. Atoms beyond qfim_nq get
        # zero blocks via _gather_edge_qfim, so coupling = 0 for those edges
        # -> uniform attention contribution from them after softmax.
        qfim_edge_flat = _gather_edge_qfim(
            qfim_block, int(qfim_nq), edge_index, batch,
        )                                                              # (E, pd*pd)
        if qfim_edge_flat.numel() == 0:
            coupling = qfim_edge_flat.new_zeros((0,))
        else:
            Q = qfim_edge_flat.view(-1, self.qfim_pd, self.qfim_pd)
            coupling = torch.linalg.matrix_norm(Q, ord="fro")          # (E,)

        for layer in self.mp_layers:
            h = layer(h, edge_index, e, coupling)

        g = self._pool_nodes(h, batch)
        z = self.readout(g).squeeze(-1)
        return z


class QFIMBondGateGNN(GNN):
    """
    QFIM as a residual multiplicative message gate on the baseline v36 bond
    graph. This is the closest implementation of:

        message_ij = MLP([h_i, h_j, geom_ij])
        message_ij *= gate(QFIM_ij)

    The graph topology, node features, geometric edge features, pooling, and
    readout are inherited from GNN. The MP layers are replaced by v36-style
    layers that multiply each per-bond message by
    1 + alpha * softplus(beta * ||Q[i,j]||_F - theta). With alpha initialised
    to 0 this starts exactly as the baseline and learns whether QFIM helps.
    """

    def __init__(
        self,
        num_mp_layers: int = 6,
        embed_z_dim: int = 16,
        node_mlp_dims: Tuple[int, ...] = (19, 32, 64, 64, 32),
        edge_mlp_dims: Tuple[int, ...] = (28, 32, 32, 16, 8),
        max_neighbors: int = MAX_NEIGHBORS,
        max_chains: int = MAX_CHAINS,
        rbf_num_centers: int = 16,
        rbf_range: Tuple[float, float] = (0.0, 5.0),
        rbf_gamma: float = 4.0,
        pooling: str = "mean",
        activation: str = "silu",
        mlp_residual: bool = False,
        msg_layers: int = 2,
        per_layer_edge_update: bool = True,
        qfim_per_qubit_dim: int = 6,
        qfim_gate_beta_init: float = 1.0,
        qfim_gate_alpha_init: float = 0.0,
        qfim_gate_theta_init: float = 0.0,
    ):
        super().__init__(
            num_mp_layers=num_mp_layers,
            embed_z_dim=embed_z_dim,
            node_mlp_dims=node_mlp_dims,
            edge_mlp_dims=edge_mlp_dims,
            max_neighbors=max_neighbors,
            max_chains=max_chains,
            rbf_num_centers=rbf_num_centers,
            rbf_range=rbf_range,
            rbf_gamma=rbf_gamma,
            pooling=pooling,
            activation=activation,
            mlp_residual=mlp_residual,
            msg_layers=msg_layers,
            per_layer_edge_update=per_layer_edge_update,
        )
        self.qfim_pd = qfim_per_qubit_dim
        self.mp_layers = nn.ModuleList([
            _QFIMBondGateMP(
                self.node_dim,
                self.edge_dim,
                activation=activation,
                msg_layers=msg_layers,
                per_layer_edge_update=per_layer_edge_update,
                beta_init=qfim_gate_beta_init,
                gate_alpha_init=qfim_gate_alpha_init,
                gate_theta_init=qfim_gate_theta_init,
            )
            for _ in range(num_mp_layers)
        ])

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        qfim_block: torch.Tensor,
        qfim_nq: int,
        batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if batch is None:
            batch = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)

        node_in = self.build_node_feat(x)
        h = self.node_mlp(node_in)
        e = self.build_edge_feat(x, edge_index, edge_attr)

        qfim_edge_flat = _gather_edge_qfim(
            qfim_block, int(qfim_nq), edge_index, batch,
        )
        if qfim_edge_flat.numel() == 0:
            coupling = qfim_edge_flat.new_zeros((0,))
        else:
            Q = qfim_edge_flat.view(-1, self.qfim_pd, self.qfim_pd)
            coupling = torch.linalg.matrix_norm(Q, ord="fro")

        for layer in self.mp_layers:
            h = layer(h, edge_index, e, coupling)

        g = self._pool_nodes(h, batch)
        z = self.readout(g).squeeze(-1)
        return z
