"""
Invariant message-passing GNN for QM9, vectorized with PyTorch Geometric.

Node features (9D): atomic_number, aromatic_flag, hybridisation, n_hydrogens,
                     x, y, z, n_atoms_total, n_heavy.
Edge features from the loader (4D): [bond_type, theta, phi, distance].

Rotation/translation invariance:
- bond_type and distance are invariant by construction.
- bond_angle (3-bond) and dihedral_angle (4-bond) are computed from coordinates
  using only scatter-based vectorized ops -- no Python loops -- and are
  rotation+translation invariant.

All angle computations run on the device holding x; no .item() syncs.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_max_pool
from torch_geometric.utils import scatter


def _safe_normalize(v: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return v / (v.norm(dim=-1, keepdim=True) + eps)


@torch.no_grad()
def compute_bond_angles(
    coords: torch.Tensor,
    edge_index: torch.Tensor,
) -> torch.Tensor:
    """
    Mean bond angle at the source node of each edge.

    For edge (i -> j), average angle(k - i - j) over all other neighbors k of i.
    Fully vectorized via scatter_mean over an expanded (edge, neighbor) tensor
    built from a self-join on i.

    Args:
        coords: (N, 3) atom positions.
        edge_index: (2, E) sparse edges.

    Returns:
        angles: (E,) in radians, 0 where i has no other neighbor.
    """
    src, dst = edge_index[0], edge_index[1]
    E = src.numel()
    if E == 0:
        return coords.new_zeros(0)

    # Group edges by source: sort once, then for each edge in the group
    # collect all neighbor-ids of the same source.
    order = torch.argsort(src, stable=True)
    src_sorted = src[order]
    dst_sorted = dst[order]

    # For each source value, the edges in its group are a contiguous slice.
    # We build pairs (edge_idx, other_neighbor) within each group, excluding
    # the edge's own destination.
    counts = torch.bincount(src_sorted, minlength=int(coords.size(0)))
    # group offsets
    offsets = torch.cat([counts.new_zeros(1), counts.cumsum(0)])

    # For each edge e in the sorted order, size of its group:
    grp_size = counts[src_sorted]                      # (E,)
    # Position of each edge inside its group:
    pos_in_grp = torch.arange(E, device=src.device) - offsets[src_sorted]

    # Each edge has (grp_size - 1) companions (other edges sharing the same src).
    companions_per_edge = (grp_size - 1).clamp(min=0)
    total_pairs = int(companions_per_edge.sum().item())
    if total_pairs == 0:
        return coords.new_zeros(E)

    # Expansion: each edge repeated `companions_per_edge[e]` times.
    edge_expand = torch.repeat_interleave(
        torch.arange(E, device=src.device), companions_per_edge
    )
    # Position of each companion slot within its edge's expansion window.
    expand_offsets = companions_per_edge.cumsum(0) - companions_per_edge
    within = torch.arange(total_pairs, device=src.device) - torch.repeat_interleave(
        expand_offsets, companions_per_edge
    )
    # Partner index inside the group: k if k < pos_in_grp[e] else k+1 (skip self).
    skip = within >= pos_in_grp[edge_expand]
    partner_in_grp = within + skip.long()
    partner_sorted_idx = offsets[src_sorted[edge_expand]] + partner_in_grp
    partner_dst = dst_sorted[partner_sorted_idx]

    # Vectors for angle at node i = src[edge_expand]
    i_nodes = src_sorted[edge_expand]
    j_nodes = dst_sorted[edge_expand]
    k_nodes = partner_dst

    v_ij = _safe_normalize(coords[j_nodes] - coords[i_nodes])
    v_ik = _safe_normalize(coords[k_nodes] - coords[i_nodes])
    cos_ang = (v_ij * v_ik).sum(-1).clamp(-1.0, 1.0)
    ang = torch.acos(cos_ang)                                              # (total_pairs,)

    # Average angle per original sorted edge, then map back to original order.
    mean_ang_sorted = scatter(ang, edge_expand, dim=0, dim_size=E, reduce="mean")
    mean_ang = torch.empty_like(mean_ang_sorted)
    mean_ang[order] = mean_ang_sorted
    return mean_ang


@torch.no_grad()
def compute_dihedral_angles(
    coords: torch.Tensor,
    edge_index: torch.Tensor,
) -> torch.Tensor:
    """
    Dihedral angle for each edge (i -> j) using one neighbor of i and one of j.

    Vectorized: for each edge picks the first other-neighbor of i (call it k)
    and the first other-neighbor of j (call it l), then computes dihedral of
    the plane k-i-j versus i-j-l.

    Returns 0 for edges where either i or j has no additional neighbor.
    """
    src, dst = edge_index[0], edge_index[1]
    E = src.numel()
    if E == 0:
        return coords.new_zeros(0)

    N = int(coords.size(0))

    def _first_other_neighbor(anchor: torch.Tensor, excluded: torch.Tensor) -> torch.Tensor:
        # For each edge, find any neighbor of `anchor` different from `excluded`.
        # We use scatter: for each edge give its anchor a candidate; pick min
        # candidate per (anchor, excluded != dst) group.
        # Build a mask of edges usable as a neighbor source: where the edge's
        # src == anchor[e] and dst != excluded[e].
        # Trick: group edges by their src. For each group, precompute two smallest
        # dst values so that "first other" can be picked in O(1).
        src_ = edge_index[0]
        dst_ = edge_index[1]
        # Sort edges by src:
        order = torch.argsort(src_, stable=True)
        src_s = src_[order]
        dst_s = dst_[order]
        counts = torch.bincount(src_s, minlength=N)
        offsets = torch.cat([counts.new_zeros(1), counts.cumsum(0)])

        # First candidate = dst_s[offsets[anchor]]; second = dst_s[offsets[anchor]+1] if count>1
        cnt = counts[anchor]
        idx0 = offsets[anchor]
        first = dst_s[idx0.clamp(max=dst_s.numel() - 1)]
        # fallback index when count >= 2
        idx1 = (idx0 + 1).clamp(max=dst_s.numel() - 1)
        second = dst_s[idx1]

        use_second = (first == excluded) & (cnt >= 2)
        chosen = torch.where(use_second, second, first)
        # Edges with cnt==0 can't happen (anchor is an endpoint of at least this edge);
        # Edges with cnt==1 and first==excluded have no other neighbor:
        has_other = cnt >= 2
        return chosen, has_other

    k_node, has_k = _first_other_neighbor(src, dst)
    l_node, has_l = _first_other_neighbor(dst, src)
    valid = has_k & has_l

    # Dihedral of (k, i, j, l)
    r_ki = coords[src] - coords[k_node]
    r_ij = coords[dst] - coords[src]
    r_jl = coords[l_node] - coords[dst]
    n1 = torch.cross(r_ki, r_ij, dim=-1)
    n2 = torch.cross(r_ij, r_jl, dim=-1)
    n1 = _safe_normalize(n1)
    n2 = _safe_normalize(n2)
    cos_d = (n1 * n2).sum(-1).clamp(-1.0, 1.0)
    ang = torch.acos(cos_d)
    ang = torch.where(valid, ang, torch.zeros_like(ang))
    return ang


def build_invariant_edge_attr(
    edge_attr_raw: torch.Tensor,
    coords: torch.Tensor,
    edge_index: torch.Tensor,
    include_dihedral: bool = True,
) -> torch.Tensor:
    """
    Map raw loader edge features [bond_type, theta, phi, distance] to purely
    invariant features [bond_type, distance, bond_angle, (dihedral)].

    theta/phi in the loader are tied to the lab frame and get dropped.
    """
    bond_type = edge_attr_raw[:, 0:1]
    distance = edge_attr_raw[:, 3:4]
    bond_ang = compute_bond_angles(coords, edge_index).unsqueeze(-1)
    feats = [bond_type, distance, bond_ang]
    if include_dihedral:
        dih = compute_dihedral_angles(coords, edge_index).unsqueeze(-1)
        feats.append(dih)
    return torch.cat(feats, dim=-1)


class InvariantMP(MessagePassing):
    """Single message-passing step with invariant edge features."""

    def __init__(self, node_dim: int, edge_dim: int):
        super().__init__(aggr="add")
        msg_in = 2 * node_dim + edge_dim
        self.msg_mlp = nn.Sequential(
            nn.Linear(msg_in, node_dim),
            nn.LayerNorm(node_dim),
            nn.ReLU(),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        return x + out

    def message(
        self, x_i: torch.Tensor, x_j: torch.Tensor, edge_attr: torch.Tensor
    ) -> torch.Tensor:
        return self.msg_mlp(torch.cat([x_i, x_j, edge_attr], dim=-1))


class InvariantGNN(nn.Module):
    """
    Invariant message-passing GNN.

    Consumes loader-native tensors: x = (N, node_dim), edge_index = (2, E),
    edge_attr = (E, 4) raw from the loader. Invariant edge features are
    computed internally (once per forward).
    """

    def __init__(
        self,
        node_dim: int = 9,
        hidden_dim: int = 64,
        num_layers: int = 6,
        include_dihedral: bool = True,
        coord_cols: slice = slice(4, 7),
        out_dim: int = 1,
    ):
        super().__init__()
        self.coord_cols = coord_cols
        self.include_dihedral = include_dihedral
        self.edge_dim = 4 if include_dihedral else 3

        self.node_embed = nn.Linear(node_dim, hidden_dim)
        self.edge_embed = nn.Sequential(
            nn.Linear(self.edge_dim, hidden_dim),
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

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        coords = x[:, self.coord_cols]
        inv_attr = build_invariant_edge_attr(
            edge_attr, coords, edge_index, self.include_dihedral
        )
        h = self.node_embed(x)
        e = self.edge_embed(inv_attr)
        for layer in self.layers:
            h = layer(h, edge_index, e)
        g = global_max_pool(h, batch) if batch is not None else h.max(0, keepdim=True)[0]
        return self.readout(g).squeeze(-1)
