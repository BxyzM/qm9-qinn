"""PyG DimeNet++ wrapper for QM9."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch_geometric.nn.models import DimeNetPlusPlus
import torch_geometric.nn.models.dimenet as pyg_dimenet

from ..gnn import _COORD_COLS, _Z_COL
from ..gnn_qfim import _QFIM_HEADS, _gather_edge_qfim


def _radius_graph_fallback(
    x: torch.Tensor,
    r: float,
    batch: Optional[torch.Tensor] = None,
    loop: bool = False,
    max_num_neighbors: int = 32,
    flow: str = "source_to_target",
    num_workers: int = 1,
    batch_size: Optional[int] = None,
) -> torch.Tensor:
    """Small pure-PyTorch radius_graph fallback for QM9-sized molecules."""
    del num_workers, batch_size
    if batch is None:
        batch = x.new_zeros(x.size(0), dtype=torch.long)
    edges = []
    for graph_id in batch.unique(sorted=True):
        nodes = (batch == graph_id).nonzero(as_tuple=False).view(-1)
        pos = x[nodes]
        dist = torch.cdist(pos, pos)
        for target_local in range(nodes.numel()):
            mask = dist[target_local] <= float(r)
            if not loop:
                mask[target_local] = False
            source_local = mask.nonzero(as_tuple=False).view(-1)
            if source_local.numel() > max_num_neighbors:
                d = dist[target_local, source_local]
                source_local = source_local[torch.topk(
                    d, k=max_num_neighbors, largest=False
                ).indices]
            target = nodes[target_local].expand_as(source_local)
            source = nodes[source_local]
            if flow == "source_to_target":
                edges.append(torch.stack([source, target], dim=0))
            else:
                edges.append(torch.stack([target, source], dim=0))
    if not edges:
        return torch.empty((2, 0), dtype=torch.long, device=x.device)
    return torch.cat(edges, dim=1)


def _triplets_fallback(edge_index: torch.Tensor, num_nodes: int):
    """Small torch-sparse-free DimeNet triplet builder for QM9 graphs."""
    del num_nodes
    row, col = edge_index  # j -> i
    idx_i_parts = []
    idx_j_parts = []
    idx_k_parts = []
    idx_kj_parts = []
    idx_ji_parts = []
    for edge_id in range(row.numel()):
        j = row[edge_id]
        i = col[edge_id]
        incoming = (col == j).nonzero(as_tuple=False).view(-1)
        if incoming.numel() == 0:
            continue
        k = row[incoming]
        keep = k != i
        incoming = incoming[keep]
        k = k[keep]
        if incoming.numel() == 0:
            continue
        idx_i_parts.append(i.expand_as(k))
        idx_j_parts.append(j.expand_as(k))
        idx_k_parts.append(k)
        idx_kj_parts.append(incoming)
        idx_ji_parts.append(torch.full_like(incoming, edge_id))

    if not idx_i_parts:
        empty = row.new_empty(0)
        return col, row, empty, empty, empty, empty, empty

    return (
        col,
        row,
        torch.cat(idx_i_parts),
        torch.cat(idx_j_parts),
        torch.cat(idx_k_parts),
        torch.cat(idx_kj_parts),
        torch.cat(idx_ji_parts),
    )


try:
    import torch_cluster  # noqa: F401
except ImportError:
    pyg_dimenet.radius_graph = _radius_graph_fallback

try:
    import torch_sparse  # noqa: F401
except ImportError:
    pyg_dimenet.triplets = _triplets_fallback


class DimeNetPP(nn.Module):
    """Thin wrapper around PyG DimeNet++ using the repo's QM9 node layout."""

    def __init__(
        self,
        hidden_channels: int = 128,
        out_channels: int = 1,
        num_blocks: int = 4,
        int_emb_size: int = 64,
        basis_emb_size: int = 8,
        out_emb_channels: int = 256,
        num_spherical: int = 7,
        num_radial: int = 6,
        cutoff: float = 5.0,
        max_num_neighbors: int = 32,
        envelope_exponent: int = 5,
        num_before_skip: int = 1,
        num_after_skip: int = 2,
        num_output_layers: int = 3,
        act: str = "swish",
        output_initializer: str = "zeros",
    ) -> None:
        super().__init__()
        self.model = DimeNetPlusPlus(
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            num_blocks=num_blocks,
            int_emb_size=int_emb_size,
            basis_emb_size=basis_emb_size,
            out_emb_channels=out_emb_channels,
            num_spherical=num_spherical,
            num_radial=num_radial,
            cutoff=cutoff,
            max_num_neighbors=max_num_neighbors,
            envelope_exponent=envelope_exponent,
            num_before_skip=num_before_skip,
            num_after_skip=num_after_skip,
            num_output_layers=num_output_layers,
            act=act,
            output_initializer=output_initializer,
        )

    def forward(
        self,
        x: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        z = x[:, _Z_COL].long()
        pos = x[:, _COORD_COLS]
        return self.model(z, pos, batch).view(-1)


class DimeNetPPQFIM(nn.Module):
    """DimeNet++ with QFIM rescaling of internal edge embeddings."""

    def __init__(
        self,
        hidden_channels: int = 128,
        out_channels: int = 1,
        num_blocks: int = 4,
        int_emb_size: int = 64,
        basis_emb_size: int = 8,
        out_emb_channels: int = 256,
        num_spherical: int = 7,
        num_radial: int = 6,
        cutoff: float = 5.0,
        max_num_neighbors: int = 32,
        envelope_exponent: int = 5,
        num_before_skip: int = 1,
        num_after_skip: int = 2,
        num_output_layers: int = 3,
        act: str = "swish",
        output_initializer: str = "zeros",
        qfim_per_qubit_dim: int = 6,
        qfim_embed_op: str = "conv2d",
        qfim_out_dim: int = 8,
        qfim_head_normalize: bool = False,
        qfim_residual_gate_init: float = 0.0,
        qfim_rescale_beta: float = 1.0,
    ) -> None:
        super().__init__()
        if qfim_embed_op not in _QFIM_HEADS:
            raise ValueError(f"Unknown qfim_embed_op={qfim_embed_op!r}")
        self.model = DimeNetPlusPlus(
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            num_blocks=num_blocks,
            int_emb_size=int_emb_size,
            basis_emb_size=basis_emb_size,
            out_emb_channels=out_emb_channels,
            num_spherical=num_spherical,
            num_radial=num_radial,
            cutoff=cutoff,
            max_num_neighbors=max_num_neighbors,
            envelope_exponent=envelope_exponent,
            num_before_skip=num_before_skip,
            num_after_skip=num_after_skip,
            num_output_layers=num_output_layers,
            act=act,
            output_initializer=output_initializer,
        )
        head_kwargs = {"pd": qfim_per_qubit_dim, "out_dim": qfim_out_dim}
        if qfim_embed_op in ("mlp", "gated"):
            head_kwargs["head_normalize"] = bool(qfim_head_normalize)
        self.qfim_head = _QFIM_HEADS[qfim_embed_op](**head_kwargs)
        self.qfim_scale = nn.Sequential(
            nn.Linear(qfim_out_dim, max(4, qfim_out_dim)),
            nn.SiLU(),
            nn.Linear(max(4, qfim_out_dim), 1),
            nn.Tanh(),
        )
        self.qfim_residual_gate = nn.Parameter(
            torch.tensor(float(qfim_residual_gate_init))
        )
        self.qfim_rescale_beta = float(qfim_rescale_beta)

    def _rescale_edges(
        self,
        edge_x: torch.Tensor,
        edge_index: torch.Tensor,
        qfim_block: torch.Tensor,
        qfim_nq: int,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        qfim_edge = _gather_edge_qfim(qfim_block, qfim_nq, edge_index, batch)
        qfim_feat = self.qfim_head(qfim_edge)
        scale = self.qfim_scale(qfim_feat)
        return edge_x + self.qfim_rescale_beta * self.qfim_residual_gate * scale * edge_x

    def forward(
        self,
        x: torch.Tensor,
        qfim_block: torch.Tensor,
        qfim_nq: int,
        batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        z = x[:, _Z_COL].long()
        pos = x[:, _COORD_COLS]
        if batch is None:
            batch = torch.zeros(z.size(0), dtype=torch.long, device=z.device)

        edge_index = pyg_dimenet.radius_graph(
            pos,
            r=self.model.cutoff,
            batch=batch,
            max_num_neighbors=self.model.max_num_neighbors,
        )
        i, j, idx_i, idx_j, idx_k, idx_kj, idx_ji = pyg_dimenet.triplets(
            edge_index, num_nodes=z.size(0)
        )

        dist = (pos[i] - pos[j]).pow(2).sum(dim=-1).sqrt()
        pos_jk = pos[idx_j] - pos[idx_k]
        pos_ij = pos[idx_i] - pos[idx_j]
        a = (pos_ij * pos_jk).sum(dim=-1)
        b = torch.cross(pos_ij, pos_jk, dim=1).norm(dim=-1)
        angle = torch.atan2(b, a)

        rbf = self.model.rbf(dist)
        sbf = self.model.sbf(dist, angle, idx_kj)

        edge_x = self.model.emb(z, rbf, i, j)
        edge_x = self._rescale_edges(edge_x, edge_index, qfim_block, qfim_nq, batch)
        out = self.model.output_blocks[0](edge_x, rbf, i, num_nodes=pos.size(0))

        for interaction_block, output_block in zip(
            self.model.interaction_blocks, self.model.output_blocks[1:]
        ):
            edge_x = interaction_block(edge_x, rbf, sbf, idx_kj, idx_ji)
            edge_x = self._rescale_edges(edge_x, edge_index, qfim_block, qfim_nq, batch)
            out = out + output_block(edge_x, rbf, i, num_nodes=pos.size(0))

        return pyg_dimenet.scatter(out, batch, dim=0, reduce="sum").view(-1)
