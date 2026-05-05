"""
Residual QFIM branch for the QM9 GNN.

This model keeps the baseline bond-graph message passing unchanged and adds
an extra QFIM message branch behind a learnable residual gate:

    h = baseline_mp(h, bond_edges)
    h = h + alpha * qfim_mp(h, bond_edges, Q_ij)

With alpha initialised to 0, the model starts at the baseline path and learns
whether aligned QFIM blocks are useful.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing

from .gnn import (
    GNN,
    InvariantMP,
    _make_activation,
    MAX_CHAINS,
    MAX_NEIGHBORS,
)
from .gnn_qfim import _QFIM_HEADS, _gather_edge_qfim


class _QFIMHeadFrobenius(nn.Module):
    """Cheap scalar summary: Frobenius norm of each 6x6 block."""

    def __init__(self, pd: int, out_dim: int = 4):
        super().__init__()
        self.pd = pd
        self.proj = nn.Sequential(
            nn.Linear(1, out_dim),
            nn.LayerNorm(out_dim),
        )

    def forward(self, qfim_edge: torch.Tensor) -> torch.Tensor:
        q = qfim_edge.view(-1, self.pd, self.pd)
        frob = q.square().sum(dim=(1, 2), keepdim=True).sqrt().view(-1, 1)
        return self.proj(frob)


class _QFIMHeadDiagStats(nn.Module):
    """Very weak encoder from diagonal/off-diagonal summary statistics."""

    def __init__(self, pd: int, out_dim: int = 4):
        super().__init__()
        self.pd = pd
        self.proj = nn.Sequential(
            nn.Linear(4, out_dim),
            nn.LayerNorm(out_dim),
        )

    def forward(self, qfim_edge: torch.Tensor) -> torch.Tensor:
        q = qfim_edge.view(-1, self.pd, self.pd)
        diag = torch.diagonal(q, dim1=1, dim2=2)
        off = q - torch.diag_embed(diag)
        stats = torch.stack(
            [
                diag.mean(dim=1),
                diag.std(dim=1, unbiased=False),
                off.abs().mean(dim=(1, 2)),
                off.std(dim=(1, 2), unbiased=False),
            ],
            dim=1,
        )
        return self.proj(stats)


_QFIM_RESIDUAL_HEADS = {
    **_QFIM_HEADS,
    "frobenius": _QFIMHeadFrobenius,
    "diagstats": _QFIMHeadDiagStats,
}


class _QFIMFullConv2dHead(nn.Module):
    """
    Encode the full molecule QFIM matrix, not just one 6x6 Q_ij block.

    Input qfim_block has shape (B, nq, nq, pd, pd). We reshape it back to
    (B, 1, nq*pd, nq*pd), e.g. 10 qubits * 6 params = 60x60, then use a
    larger Conv2D kernel to see cross-qubit and cross-parameter structure.
    The resulting graph-level vector is broadcast to every bond edge in the
    molecule before the residual message branch.
    """

    def __init__(
        self,
        nq: int,
        pd: int,
        out_dim: int = 16,
        channels: int = 16,
        kernel_size: int = 7,
        activation: str = "relu",
    ):
        super().__init__()
        self.nq = nq
        self.pd = pd
        padding = kernel_size // 2
        self.net = nn.Sequential(
            nn.Conv2d(1, channels, kernel_size=kernel_size, padding=padding),
            nn.LayerNorm([channels, nq * pd, nq * pd]),
            _make_activation(activation),
            nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding),
            nn.LayerNorm([channels, nq * pd, nq * pd]),
            _make_activation(activation),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, out_dim),
            nn.LayerNorm(out_dim),
        )

    def forward(self, qfim_block: torch.Tensor) -> torch.Tensor:
        bsz, nq, _, pd, _ = qfim_block.shape
        if int(nq) != self.nq or int(pd) != self.pd:
            raise ValueError(
                f"expected qfim_block with nq={self.nq}, pd={self.pd}; "
                f"got nq={int(nq)}, pd={int(pd)}"
            )
        qfim_full = qfim_block.permute(0, 1, 3, 2, 4).reshape(
            bsz, 1, self.nq * self.pd, self.nq * self.pd
        )
        return self.net(qfim_full)


class _QFIMResidualMP(MessagePassing):
    """Per-edge QFIM message branch without its own residual skip."""

    def __init__(
        self,
        node_dim: int,
        qfim_dim: int,
        edge_dim: int = 0,
        activation: str = "relu",
        msg_layers: int = 1,
        edge_gate: bool = False,
        branch_dropout: float = 0.0,
    ):
        super().__init__(aggr="add")
        msg_in = 2 * node_dim + qfim_dim + edge_dim
        self.edge_dim = edge_dim
        self.edge_gate = bool(edge_gate)
        self.dropout = nn.Dropout(branch_dropout) if branch_dropout > 0.0 else nn.Identity()
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
        self.gate_mlp = None
        if self.edge_gate:
            gate_in = qfim_dim + edge_dim
            self.gate_mlp = nn.Sequential(
                nn.Linear(gate_in, max(4, gate_in)),
                _make_activation(activation),
                nn.Linear(max(4, gate_in), 1),
            )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        qfim_attr: torch.Tensor,
        edge_attr_geom: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.propagate(
            edge_index, x=x, qfim_attr=qfim_attr, edge_attr_geom=edge_attr_geom
        )

    def message(
        self,
        x_i: torch.Tensor,
        x_j: torch.Tensor,
        qfim_attr: torch.Tensor,
        edge_attr_geom: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        pieces = [x_i, x_j, qfim_attr]
        gate_in = [qfim_attr]
        if self.edge_dim > 0 and edge_attr_geom is not None:
            pieces.append(edge_attr_geom)
            gate_in.append(edge_attr_geom)
        msg = self.msg_mlp(torch.cat(pieces, dim=-1))
        if self.gate_mlp is not None:
            gate = torch.sigmoid(self.gate_mlp(torch.cat(gate_in, dim=-1)))
            msg = gate * msg
        return self.dropout(msg)


class _QFIMGatedBaselineMP(MessagePassing):
    """Baseline messages multiplied by a QFIM-derived edge gate."""

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        qfim_dim: int,
        activation: str = "relu",
        msg_layers: int = 1,
        per_layer_edge_update: bool = False,
    ):
        super().__init__(aggr="add")
        self.edge_dim = edge_dim
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
        gate_in = qfim_dim + edge_dim
        self.gate_mlp = nn.Sequential(
            nn.Linear(gate_in, max(4, gate_in)),
            _make_activation(activation),
            nn.Linear(max(4, gate_in), 1),
        )
        self._edge_update_mlp = None
        if per_layer_edge_update:
            self._edge_update_mlp = nn.Sequential(
                nn.Linear(edge_dim, edge_dim),
                _make_activation(activation),
            )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        qfim_attr: torch.Tensor,
    ) -> torch.Tensor:
        if self._edge_update_mlp is not None:
            edge_attr = edge_attr + self._edge_update_mlp(edge_attr)
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, qfim_attr=qfim_attr)
        return x + out

    def message(
        self,
        x_i: torch.Tensor,
        x_j: torch.Tensor,
        edge_attr: torch.Tensor,
        qfim_attr: torch.Tensor,
    ) -> torch.Tensor:
        msg = self.msg_mlp(torch.cat([x_i, x_j, edge_attr], dim=-1))
        gate = torch.sigmoid(self.gate_mlp(torch.cat([qfim_attr, edge_attr], dim=-1)))
        return gate * msg


class _QFIMRescaledBaselineMP(MessagePassing):
    """Baseline messages plus an alpha-scaled QFIM modulation of themselves."""

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        qfim_dim: int,
        activation: str = "relu",
        msg_layers: int = 1,
        per_layer_edge_update: bool = False,
    ):
        super().__init__(aggr="add")
        self.edge_dim = edge_dim
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
        self.scale_mlp = nn.Sequential(
            nn.Linear(qfim_dim, max(4, qfim_dim)),
            _make_activation(activation),
            nn.Linear(max(4, qfim_dim), 1),
            nn.Tanh(),
        )
        self._edge_update_mlp = None
        if per_layer_edge_update:
            self._edge_update_mlp = nn.Sequential(
                nn.Linear(edge_dim, edge_dim),
                _make_activation(activation),
            )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        qfim_attr: torch.Tensor,
        alpha: torch.Tensor,
    ) -> torch.Tensor:
        if self._edge_update_mlp is not None:
            edge_attr = edge_attr + self._edge_update_mlp(edge_attr)
        out = self.propagate(
            edge_index, x=x, edge_attr=edge_attr, qfim_attr=qfim_attr, alpha=alpha
        )
        return x + out

    def message(
        self,
        x_i: torch.Tensor,
        x_j: torch.Tensor,
        edge_attr: torch.Tensor,
        qfim_attr: torch.Tensor,
        alpha: torch.Tensor,
    ) -> torch.Tensor:
        msg = self.msg_mlp(torch.cat([x_i, x_j, edge_attr], dim=-1))
        scale = self.scale_mlp(qfim_attr)
        return msg + alpha * scale * msg


class QFIMResidualGNN(GNN):
    """
    Baseline GNN plus a gated residual QFIM branch on the same bond graph.

    Unlike QFIMGNN, QFIM features are not concatenated into the baseline edge
    features. The standard geometry message passing remains intact, and QFIM
    contributes through a separate residual update controlled by one learnable
    scalar gate.
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
        activation: str = "relu",
        mlp_residual: bool = False,
        msg_layers: int = 1,
        per_layer_edge_update: bool = False,
        qfim_n_qubits: int = 10,
        qfim_per_qubit_dim: int = 6,
        qfim_embed_op: str = "conv2d",
        qfim_out_dim: int = 8,
        qfim_head_normalize: bool = False,
        qfim_residual_gate_init: float = 0.0,
        qfim_full_conv_kernel: int = 7,
        qfim_full_conv_channels: int = 16,
        qfim_alpha_mode: str = "shared",
        qfim_edge_gate: bool = False,
        qfim_use_geom: bool = False,
        qfim_mode: str = "additive",
        qfim_msg_layers: Optional[int] = None,
        qfim_branch_dropout: float = 0.0,
        qfim_rescale_beta: float = 1.0,
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
        if qfim_embed_op != "full_conv2d" and qfim_embed_op not in _QFIM_RESIDUAL_HEADS:
            raise ValueError(
                "qfim_embed_op must be one of "
                f"{list(_QFIM_RESIDUAL_HEADS) + ['full_conv2d']}; got {qfim_embed_op!r}"
            )
        if qfim_alpha_mode not in ("shared", "per_layer"):
            raise ValueError("qfim_alpha_mode must be 'shared' or 'per_layer'")
        if qfim_mode not in (
            "additive",
            "node_gate",
            "baseline_msg_gate",
            "baseline_msg_rescale",
        ):
            raise ValueError(
                "qfim_mode must be one of: additive, node_gate, "
                "baseline_msg_gate, baseline_msg_rescale"
            )

        self.qfim_nq = qfim_n_qubits
        self.qfim_pd = qfim_per_qubit_dim
        self.qfim_out_dim = qfim_out_dim
        self.qfim_embed_op = qfim_embed_op
        self.qfim_alpha_mode = qfim_alpha_mode
        self.qfim_edge_gate = bool(qfim_edge_gate)
        self.qfim_use_geom = bool(qfim_use_geom)
        self.qfim_mode = qfim_mode
        self.qfim_rescale_beta = float(qfim_rescale_beta)
        qfim_msg_layers = int(qfim_msg_layers or msg_layers)
        self.qfim_msg_layers = qfim_msg_layers
        if qfim_alpha_mode == "shared":
            self.qfim_residual_gate = nn.Parameter(
                torch.tensor(float(qfim_residual_gate_init))
            )
        else:
            self.qfim_residual_gate = nn.Parameter(
                torch.full((num_mp_layers,), float(qfim_residual_gate_init))
            )

        if qfim_embed_op == "full_conv2d":
            self.qfim_head = _QFIMFullConv2dHead(
                nq=qfim_n_qubits,
                pd=qfim_per_qubit_dim,
                out_dim=qfim_out_dim,
                channels=qfim_full_conv_channels,
                kernel_size=qfim_full_conv_kernel,
                activation=activation,
            )
        else:
            head_kwargs = {"pd": qfim_per_qubit_dim, "out_dim": qfim_out_dim}
            if qfim_embed_op in ("mlp", "gated"):
                head_kwargs["head_normalize"] = bool(qfim_head_normalize)
            self.qfim_head = _QFIM_RESIDUAL_HEADS[qfim_embed_op](**head_kwargs)

        if self.qfim_mode in ("baseline_msg_gate", "baseline_msg_rescale"):
            self.qfim_layers = nn.ModuleList()
            self.node_gate_layers = nn.ModuleList()
        else:
            self.qfim_layers = nn.ModuleList([
                _QFIMResidualMP(
                    self.node_dim,
                    qfim_out_dim,
                    edge_dim=self.edge_dim if self.qfim_use_geom else 0,
                    activation=activation,
                    msg_layers=qfim_msg_layers,
                    edge_gate=self.qfim_edge_gate,
                    branch_dropout=qfim_branch_dropout,
                )
                for _ in range(num_mp_layers)
            ])
            if self.qfim_mode == "node_gate":
                self.node_gate_layers = nn.ModuleList([
                    nn.Sequential(
                        nn.Linear(self.node_dim, self.node_dim),
                        nn.Sigmoid(),
                    )
                    for _ in range(num_mp_layers)
                ])
            else:
                self.node_gate_layers = nn.ModuleList()

        # Recreate the baseline MP stack explicitly so its size stays paired
        # with qfim_layers even if a parent default changes later.
        if self.qfim_mode == "baseline_msg_gate":
            self.mp_layers = nn.ModuleList([
                _QFIMGatedBaselineMP(
                    self.node_dim,
                    self.edge_dim,
                    qfim_out_dim,
                    activation=activation,
                    msg_layers=msg_layers,
                    per_layer_edge_update=per_layer_edge_update,
                    rescale_beta=self.qfim_rescale_beta,
                )
                for _ in range(num_mp_layers)
            ])
        elif self.qfim_mode == "baseline_msg_rescale":
            self.mp_layers = nn.ModuleList([
                _QFIMRescaledBaselineMP(
                    self.node_dim,
                    self.edge_dim,
                    qfim_out_dim,
                    activation=activation,
                    msg_layers=msg_layers,
                    per_layer_edge_update=per_layer_edge_update,
                )
                for _ in range(num_mp_layers)
            ])
        else:
            self.mp_layers = nn.ModuleList([
                InvariantMP(
                    self.node_dim,
                    self.edge_dim,
                    activation=activation,
                    msg_layers=msg_layers,
                    per_layer_edge_update=per_layer_edge_update,
                )
                for _ in range(num_mp_layers)
            ])

    def _alpha_at(self, layer_idx: int) -> torch.Tensor:
        if self.qfim_alpha_mode == "shared":
            return self.qfim_residual_gate
        return self.qfim_residual_gate[layer_idx]

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

        if self.qfim_embed_op == "full_conv2d":
            graph_qfim_feat = self.qfim_head(qfim_block)
            qfim_feat = graph_qfim_feat[batch[edge_index[0]]]
        else:
            qfim_edge = _gather_edge_qfim(qfim_block, qfim_nq, edge_index, batch)
            qfim_feat = self.qfim_head(qfim_edge)

        for layer_idx, base_layer in enumerate(self.mp_layers):
            if self.qfim_mode == "baseline_msg_gate":
                h = base_layer(h, edge_index, e, qfim_feat)
                continue
            alpha = self._alpha_at(layer_idx)
            if self.qfim_mode == "baseline_msg_rescale":
                h = base_layer(h, edge_index, e, qfim_feat, alpha)
                continue
            qfim_layer = self.qfim_layers[layer_idx]
            h_base = base_layer(h, edge_index, e)
            q_update = qfim_layer(
                h,
                edge_index,
                qfim_feat,
                edge_attr_geom=e if self.qfim_use_geom else None,
            )
            if self.qfim_mode == "node_gate":
                gate = self.node_gate_layers[layer_idx](q_update)
                h = h_base + alpha * gate * h_base
            else:
                h = h_base + alpha * q_update

        g = self._pool_nodes(h, batch)
        z = self.readout(g).squeeze(-1)
        return z
