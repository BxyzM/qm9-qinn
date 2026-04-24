"""
Baseline GNN for QM9 HOMO-LUMO gap regression.

Node features (4D per atom):
    [Z, x, y, z]   - atomic number + 3D position.

Edge features per bond (i -> j):
    vec3  : up to 3 bond angles  angle(k - i - j) at the source endpoint i,
            one per bonded neighbor k != j, padded with zeros to length 3.
            **Raw loader order is preserved** -- the HDF5 maker sorts atoms
            by descending atomic number, so position 0 is the heaviest
            bonded neighbor by construction. Position-dependent weights in
            the edge MLP can learn physical meaning like "angle involving
            the heaviest neighbor matters more."
    vec4  : up to 9 unsigned dihedrals  dih(k - i - j - l), one per valid
            (k, l) 4-atom chain where k bonds to i (k != j), l bonds to j
            (l != i, l != k), padded with zeros to length 9.
    dist  : bond distance in Angstroms (scalar).
    ==> concat = 13-dim edge input (3 + 9 + 1).
    edge_mlp(13 -> 6 -> 8 -> 16 -> 8 -> 3)  produces 3 learned edge dims.
    Bond type enters as a learnable multiplicative scalar:
        edge_out = (alpha * bond_type) * edge_mlp(vec3, vec4, dist)

Node embedding (expand-then-compress MLP):
    node_mlp(4 -> 8 -> 16 -> 8 -> 4)

Message passing: 6 x InvariantMP with sum aggregation and residual, operating
in 4-dim node space with 3-dim edge features. Mean pool over nodes, tiny
readout to scalar gap in eV.

Target standardization:
    Stats fit once via model.fit_target_stats(train_loader) before training.
    Buffers (target_mean, target_std) travel with state_dict so checkpoints
    keep consistent normalization.

Permutation note:
    Angles/dihedrals are kept as fixed-length vectors in loader order, not
    averaged, because the downstream position-dependent MLP is meant to
    exploit that the heaviest-neighbor is always in position 0. If the
    loader's atom ordering changes, this model's predictions change -- the
    invariance is not built into the architecture.
"""

from __future__ import annotations

from typing import Iterable, Optional, Tuple

import torch
import torch.nn as nn
from torch_geometric.nn import (
    MessagePassing,
    global_add_pool,
    global_max_pool,
    global_mean_pool,
)
from torch_geometric.utils import scatter


_POOLINGS = {
    "add": global_add_pool,
    "mean": global_mean_pool,
    "max": global_max_pool,
}

# Column indices into the 9-dim HDF5 node feature vector.
_Z_COL = 0
_COORD_COLS = slice(4, 7)

# Column indices into the 4-dim HDF5 edge feature vector: bond_type is [0],
# distance is [3] (theta, phi at [1], [2] are not used; we recompute geometry
# from coords for the richer vector-valued bond/dihedral features).
_BOND_TYPE_COL = 0
_DISTANCE_COL = 3

# Pad-to-max sizes for the edge angle vectors. Values match the physical
# maxima in QM9's CHNOF subset:
#   - max degree of any heavy atom is 4 (carbon, nitrogen)
#     -> at most 3 bond angles per edge (degree(i) - 1)
#     -> at most (deg(i) - 1) * (deg(j) - 1) = 9 dihedral chains per edge
# No edges get truncated; slots beyond the actual count are padded with 0.
MAX_NEIGHBORS = 3
MAX_CHAINS = 9


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _safe_normalize(v: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return v / (v.norm(dim=-1, keepdim=True) + eps)


def _sorted_adjacency(
    edge_index: torch.Tensor, num_nodes: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Group edges by source node. CSR-style."""
    src, dst = edge_index[0], edge_index[1]
    order = torch.argsort(src, stable=True)
    src_s = src[order]
    dst_s = dst[order]
    counts = torch.bincount(src_s, minlength=num_nodes)
    offsets = torch.cat([counts.new_zeros(1), counts.cumsum(0)])
    return order, src_s, dst_s, offsets


# ---------------------------------------------------------------------------
# Vector-valued geometric edge features
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_bond_angle_vec(
    coords: torch.Tensor,
    edge_index: torch.Tensor,
    max_neighbors: int = MAX_NEIGHBORS,
) -> torch.Tensor:
    """
    For each edge (i -> j), collect up to `max_neighbors` bond angles
    angle(k - i - j), one per bonded neighbor k of i with k != j, in the
    order the adjacency enumeration produces them (source-sorted CSR).

    Returns:
        (E, max_neighbors) tensor. Unused slots are zero. Edges whose source
        has no other bonded neighbor come out as all zeros.
    """
    src = edge_index[0]
    E = src.numel()
    if E == 0:
        return coords.new_zeros((0, max_neighbors))

    N = int(coords.size(0))
    order, src_s, dst_s, offsets = _sorted_adjacency(edge_index, N)
    counts = offsets[1:] - offsets[:-1]
    dev = src.device

    grp_size = counts[src_s]                                       # |N(i)|
    pos_in_grp = torch.arange(E, device=dev) - offsets[src_s]       # slot of j in N(i)
    companions_per_edge = (grp_size - 1).clamp(min=0)
    if companions_per_edge.sum() == 0:
        return coords.new_zeros((E, max_neighbors))

    total_pairs = int(companions_per_edge.sum().item())
    edge_expand = torch.repeat_interleave(
        torch.arange(E, device=dev), companions_per_edge
    )
    expand_offsets = companions_per_edge.cumsum(0) - companions_per_edge
    within = torch.arange(total_pairs, device=dev) - torch.repeat_interleave(
        expand_offsets, companions_per_edge
    )
    # The within-group position of the k partner (skipping j's slot).
    skip = within >= pos_in_grp[edge_expand]
    partner_in_grp = within + skip.long()
    partner_sorted_idx = offsets[src_s[edge_expand]] + partner_in_grp
    partner_dst = dst_s[partner_sorted_idx]

    i_nodes = src_s[edge_expand]
    j_nodes = dst_s[edge_expand]
    k_nodes = partner_dst

    v_ij = _safe_normalize(coords[j_nodes] - coords[i_nodes])
    v_ik = _safe_normalize(coords[k_nodes] - coords[i_nodes])
    cos_ang = (v_ij * v_ik).sum(-1).clamp(-1.0, 1.0)
    ang = torch.acos(cos_ang)

    # Scatter each angle into slot (within, edge_expand). Cap at max_neighbors.
    out_sorted = coords.new_zeros((E, max_neighbors))
    keep = within < max_neighbors
    rows = edge_expand[keep]
    slots = within[keep]
    vals = ang[keep]
    out_sorted[rows, slots] = vals

    # Undo the sort back to original edge order.
    out = torch.empty_like(out_sorted)
    out[order] = out_sorted
    return out


@torch.no_grad()
def compute_dihedral_vec(
    coords: torch.Tensor,
    edge_index: torch.Tensor,
    max_chains: int = MAX_CHAINS,
) -> torch.Tensor:
    """
    For each edge (i -> j), collect up to `max_chains` unsigned dihedrals
    dih(k - i - j - l), where k in N(i)\\{j}, l in N(j)\\{i, k}. Chains
    are emitted in the order the adjacency enumeration produces them.

    Returns:
        (E, max_chains) tensor. Unused slots are zero. Edges with no valid
        (k, l) chain come out as all zeros.
    """
    src = edge_index[0]
    E = src.numel()
    if E == 0:
        return coords.new_zeros((0, max_chains))

    N = int(coords.size(0))
    order, src_s, dst_s, offsets = _sorted_adjacency(edge_index, N)
    counts = offsets[1:] - offsets[:-1]
    dev = src.device

    grp_i = counts[src_s]
    pos_j_in_Ni = torch.arange(E, device=dev) - offsets[src_s]
    num_k = (grp_i - 1).clamp(min=0)
    if num_k.sum() == 0:
        return coords.new_zeros((E, max_chains))

    pk = int(num_k.sum().item())
    edge_of_pair = torch.repeat_interleave(torch.arange(E, device=dev), num_k)
    base_k = torch.repeat_interleave(num_k.cumsum(0) - num_k, num_k)
    within_k = torch.arange(pk, device=dev) - base_k
    slot_k = within_k + (within_k >= pos_j_in_Ni[edge_of_pair]).long()
    k_nodes = dst_s[offsets[src_s[edge_of_pair]] + slot_k]

    i_of_pair = src_s[edge_of_pair]
    j_of_pair = dst_s[offsets[src_s[edge_of_pair]] + pos_j_in_Ni[edge_of_pair]]

    grp_j = counts[j_of_pair]
    if grp_j.sum() == 0:
        return coords.new_zeros((E, max_chains))

    pl = int(grp_j.sum().item())
    pair_idx = torch.repeat_interleave(torch.arange(pk, device=dev), grp_j)
    base_l = torch.repeat_interleave(grp_j.cumsum(0) - grp_j, grp_j)
    slot_l = torch.arange(pl, device=dev) - base_l
    l_nodes = dst_s[offsets[j_of_pair[pair_idx]] + slot_l]

    edge_q = edge_of_pair[pair_idx]
    i_q = i_of_pair[pair_idx]
    j_q = j_of_pair[pair_idx]
    k_q = k_nodes[pair_idx]
    l_q = l_nodes

    # Drop invalid chains: l == i (going back) or l == k (3-ring coplanar).
    keep = (l_q != i_q) & (l_q != k_q)
    edge_q = edge_q[keep]; i_q = i_q[keep]; j_q = j_q[keep]
    k_q = k_q[keep];       l_q = l_q[keep]
    if edge_q.numel() == 0:
        return coords.new_zeros((E, max_chains))

    r_ki = coords[i_q] - coords[k_q]
    r_ij = coords[j_q] - coords[i_q]
    r_jl = coords[l_q] - coords[j_q]
    n1 = _safe_normalize(torch.cross(r_ki, r_ij, dim=-1))
    n2 = _safe_normalize(torch.cross(r_ij, r_jl, dim=-1))
    cos_d = (n1 * n2).sum(-1).clamp(-1.0, 1.0)
    ang = torch.acos(cos_d)

    # Per-edge slot assignment: the m-th valid chain belonging to edge q
    # goes into column m (capped at max_chains). We compute "slot within
    # edge group" by sorting entries by (edge_q), using positions within
    # each sorted group as slot indices, then mapping back.
    order_q = torch.argsort(edge_q, stable=True)
    sorted_eq = edge_q[order_q]
    # slot = position_in_sorted_array - first_position_of_this_edge_group
    #      = arange - cummax over (is_new ? arange : 0)
    pos = torch.arange(sorted_eq.numel(), device=dev)
    is_new = torch.ones_like(sorted_eq, dtype=torch.bool)
    is_new[1:] = sorted_eq[1:] != sorted_eq[:-1]
    group_start = torch.where(is_new, pos, torch.zeros_like(pos))
    group_start_running = torch.cummax(group_start, dim=0).values
    slot_sorted = pos - group_start_running
    # Unsort back to original entry order.
    slot_in_group = torch.empty_like(slot_sorted)
    slot_in_group[order_q] = slot_sorted

    out_sorted = coords.new_zeros((E, max_chains))
    keep_slot = slot_in_group < max_chains
    rows = edge_q[keep_slot]
    slots = slot_in_group[keep_slot]
    vals = ang[keep_slot]
    out_sorted[rows, slots] = vals

    out = torch.empty_like(out_sorted)
    out[order] = out_sorted
    return out


def build_edge_raw_features(
    edge_attr_loader: torch.Tensor,
    coords: torch.Tensor,
    edge_index: torch.Tensor,
    max_neighbors: int = MAX_NEIGHBORS,
    max_chains: int = MAX_CHAINS,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build the (E, 9)-dim raw edge features and the (E,) bond-type vector.

    Raw features layout: [vec3_bond (MAX_NEIGHBORS) | vec4_dihedral (MAX_CHAINS) | distance (1)].
    Bond type is returned separately because it enters the model as a
    multiplicative scalar, not as an additive MLP input.
    """
    distance = edge_attr_loader[:, _DISTANCE_COL:_DISTANCE_COL + 1]
    bond_type = edge_attr_loader[:, _BOND_TYPE_COL]
    vec3 = compute_bond_angle_vec(coords, edge_index, max_neighbors)
    vec4 = compute_dihedral_vec(coords, edge_index, max_chains)
    raw = torch.cat([vec3, vec4, distance], dim=-1)
    return raw, bond_type


# ---------------------------------------------------------------------------
# Message passing (small, fixed to node_dim=4, edge_dim as passed in)
# ---------------------------------------------------------------------------

class InvariantMP(MessagePassing):
    """Sum-aggregating MP block with residual update. LayerNorm inside msg."""

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


# ---------------------------------------------------------------------------
# MLP builder
# ---------------------------------------------------------------------------

def _build_mlp(dims: Tuple[int, ...]) -> nn.Sequential:
    """
    Build a Linear-ReLU stack from a sequence of layer widths.

    Example: _build_mlp((4, 8, 16, 8, 4))
             -> Linear(4,8) ReLU Linear(8,16) ReLU Linear(16,8) ReLU Linear(8,4)

    ReLU is inserted between every pair of Linear layers, never after the
    last one, so the final activation is left to the caller.
    """
    layers: list = []
    for idx in range(len(dims) - 1):
        layers.append(nn.Linear(dims[idx], dims[idx + 1]))
        if idx < len(dims) - 2:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


# ---------------------------------------------------------------------------
# Baseline GNN
# ---------------------------------------------------------------------------

class GNN(nn.Module):
    """
    Baseline GNN on 4-dim [Z, x, y, z] node features and 3-dim learned
    edge features (geometry only, no QFIM). This is the post-refactor
    successor to InvariantGNN.
    """

    DEFAULT_TARGET_INDEX = 4           # PyG QM9 layout: 4 = HOMO-LUMO gap (eV)

    def __init__(
        self,
        num_mp_layers: int = 6,
        node_mlp_dims: Tuple[int, ...] = (4, 8, 16, 8, 4),
        edge_mlp_dims: Tuple[int, ...] = (13, 6, 8, 16, 8, 3),
        max_neighbors: int = MAX_NEIGHBORS,
        max_chains: int = MAX_CHAINS,
        pooling: str = "mean",
    ):
        super().__init__()
        if pooling not in _POOLINGS:
            raise ValueError(f"pooling must be one of {list(_POOLINGS)}; got {pooling!r}")
        if node_mlp_dims[0] != 4:
            raise ValueError(f"node_mlp_dims must start at 4 ([Z, x, y, z]); got {node_mlp_dims[0]}")
        if edge_mlp_dims[0] != max_neighbors + max_chains + 1:
            raise ValueError(
                f"edge_mlp_dims[0]={edge_mlp_dims[0]} inconsistent with "
                f"max_neighbors+max_chains+1={max_neighbors + max_chains + 1}"
            )
        self.max_neighbors = max_neighbors
        self.max_chains = max_chains
        self.node_dim = node_mlp_dims[-1]
        self.edge_dim = edge_mlp_dims[-1]
        self._pool = _POOLINGS[pooling]

        self.node_mlp = _build_mlp(node_mlp_dims)
        self.edge_mlp = _build_mlp(edge_mlp_dims)

        # Learnable scalar prefactor on bond_type. bond_type is an integer
        # in {1, 2, 3, 4}; alpha scales it into a learned regime without
        # embedding.
        self.bond_alpha = nn.Parameter(torch.tensor(1.0))

        self.mp_layers = nn.ModuleList(
            [InvariantMP(self.node_dim, self.edge_dim) for _ in range(num_mp_layers)]
        )

        # Readout: 4 -> 16 -> 1. Intentionally small to match the model's
        # overall compact scale.
        self.readout = nn.Sequential(
            nn.Linear(self.node_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

        self.register_buffer("target_mean", torch.zeros(1))
        self.register_buffer("target_std", torch.ones(1))
        self.register_buffer("_stats_fitted", torch.tensor(False))

    # --- target stats ------------------------------------------------------

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
            raise RuntimeError(f"fit_target_stats needs >= 2 samples; got {count}.")
        std = (m2 / (count - 1)) ** 0.5
        if std < 1e-8:
            raise RuntimeError(f"Target std ~0 ({std:.2e}); check target_index.")

        device = self.target_mean.device
        self.target_mean.copy_(torch.tensor([mean], device=device))
        self.target_std.copy_(torch.tensor([std], device=device))
        self._stats_fitted.copy_(torch.tensor(True))
        return float(mean), float(std)

    @property
    def stats_fitted(self) -> bool:
        return bool(self._stats_fitted.item())

    # --- feature prep (reused by QFIMGNN) ---------------------------------

    def build_node_feat(self, x: torch.Tensor) -> torch.Tensor:
        """Extract [Z, x, y, z] from the loader's 9-dim node feature tensor."""
        Z = x[:, _Z_COL:_Z_COL + 1]
        coords = x[:, _COORD_COLS]
        return torch.cat([Z, coords], dim=-1)

    def build_edge_feat(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr_loader: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the 3-dim geometry-only edge feature used in MP."""
        coords = x[:, _COORD_COLS]
        raw, bond_type = build_edge_raw_features(
            edge_attr_loader, coords, edge_index,
            max_neighbors=self.max_neighbors, max_chains=self.max_chains,
        )
        e = self.edge_mlp(raw)                                    # (E, 3)
        e = (self.bond_alpha * bond_type).unsqueeze(-1) * e       # bond-type scaling
        return e

    # --- forward ----------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if not bool(self._stats_fitted.item()):
            raise RuntimeError(
                "GNN.forward called before fit_target_stats. "
                "Call model.fit_target_stats(train_loader) once before training."
            )
        if batch is None:
            batch = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)

        node_in = self.build_node_feat(x)                         # (N, 4)
        h = self.node_mlp(node_in)                                # (N, 4)
        e = self.build_edge_feat(x, edge_index, edge_attr)        # (E, 3)

        for layer in self.mp_layers:
            h = layer(h, edge_index, e)

        g = self._pool(h, batch)                                  # (B, 4)
        z = self.readout(g).squeeze(-1)                           # (B,)
        return z * self.target_std + self.target_mean
