"""
Invariant message-passing GNN for QM9 HOMO-LUMO gap regression.

Node features (9D): atomic_number, aromatic_flag, hybridisation, n_hydrogens,
                     x, y, z, n_atoms_total, n_heavy.
Edge features from the loader (4D): [bond_type, theta, phi, distance].

Rotation/translation invariance:
- bond_type and distance are invariant by construction.
- bond_angle (3-bond) and dihedral_angle (4-bond) are computed from coordinates
  using only scatter-based vectorized ops -- no Python loops -- and are
  rotation+translation invariant.

Target standardization:
- The model stores (target_mean, target_std) as buffers.
- Internally the MLP regresses on a standardized target; `forward` denormalizes so callers always work in physical units (eV).
- Fit statistics on the training split only: `model.fit_target_stats(loader)`
  once before training. Stats save with state_dict so checkpoints / eval runs
  don't drift.

Pooling default is "mean", appropriate for the HOMO-LUMO gap (intensive).
Switch to "add" for extensive targets (U0, H, G).
"""

from __future__ import annotations

from typing import Iterable, Optional, Tuple

import torch
import torch.nn as nn
from torch_geometric.nn import (
    MessagePassing,
    global_add_pool,
    global_mean_pool,
    global_max_pool,
)
from torch_geometric.utils import scatter


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _safe_normalize(v: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return v / (v.norm(dim=-1, keepdim=True) + eps)


def _sorted_adjacency(
    edge_index: torch.Tensor, num_nodes: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Group edges by source node.

    Returns:
        order:   permutation s.t. edge_index[0, order] is sorted.
        src_s:   sorted sources (E,).
        dst_s:   destinations in sorted order (E,).
        offsets: (num_nodes + 1,) CSR-style offsets into src_s / dst_s.
    """
    src, dst = edge_index[0], edge_index[1]
    order = torch.argsort(src, stable=True)
    src_s = src[order]
    dst_s = dst[order]
    counts = torch.bincount(src_s, minlength=num_nodes)
    offsets = torch.cat([counts.new_zeros(1), counts.cumsum(0)])
    return order, src_s, dst_s, offsets


# ---------------------------------------------------------------------------
# Invariant geometric edge features
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_bond_angles(
    coords: torch.Tensor,
    edge_index: torch.Tensor,
) -> torch.Tensor:
    """
    Mean bond angle at the source node of each edge.

    For edge (i -> j), average angle(k - i - j) over all other neighbors k of i.
    Returns 0 for edges whose source has no other neighbor.
    """
    src = edge_index[0]
    E = src.numel()
    if E == 0:
        return coords.new_zeros(0)

    N = int(coords.size(0))
    order, src_s, dst_s, offsets = _sorted_adjacency(edge_index, N)

    counts = offsets[1:] - offsets[:-1]
    grp_size = counts[src_s]
    pos_in_grp = torch.arange(E, device=src.device) - offsets[src_s]

    companions_per_edge = (grp_size - 1).clamp(min=0)
    total_pairs = companions_per_edge.sum()  # 0-dim tensor, no host sync

    edge_expand = torch.repeat_interleave(
        torch.arange(E, device=src.device), companions_per_edge
    )
    expand_offsets = companions_per_edge.cumsum(0) - companions_per_edge
    within = torch.arange(total_pairs, device=src.device) - torch.repeat_interleave(
        expand_offsets, companions_per_edge
    )

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
    Mean unsigned dihedral at each edge (i -> j), averaged over all valid
    4-atom chains k - i - j - l where:
        * k is a bonded neighbor of i, k != j
        * l is a bonded neighbor of j, l != i
        * k != l  (else k, i, j, l are coplanar -- 3-ring degeneracy)

    Returns 0 for edges with no valid (k, l) pair (terminal bonds, or the
    only other neighbor on each side collapses to the same atom).

    Unsigned (O(3)-invariant) is correct for QM9 scalar targets like the
    HOMO-LUMO gap, which are identical for a molecule and its mirror image.
    A signed dihedral would break that reflection symmetry and let the model
    learn a spurious dependence on handedness.

    Averaging over all (k, l) pairs (not just the "first" pair each side)
    makes the feature independent of the neighbor storage order and uses all
    the local geometry the bonds provide.
    """
    src = edge_index[0]
    E = src.numel()
    if E == 0:
        return coords.new_zeros(0)

    N = int(coords.size(0))
    order, src_s, dst_s, offsets = _sorted_adjacency(edge_index, N)
    counts = offsets[1:] - offsets[:-1]
    dev = src.device

    # Enumerate every (edge, k) pair where k in neighbors(i) \ {j}. Same
    # expansion as compute_bond_angles: for each sorted edge e=(i,j), we want
    # all of i's neighbors except the slot occupied by j itself.
    grp_i = counts[src_s]                                       # |N(i)|
    pos_j_in_Ni = torch.arange(E, device=dev) - offsets[src_s]  # slot of j inside N(i)
    num_k = (grp_i - 1).clamp(min=0)                            # |N(i)| - 1
    if num_k.sum() == 0:
        return coords.new_zeros(E)

    pk = num_k.sum().item()
    edge_of_pair = torch.repeat_interleave(torch.arange(E, device=dev), num_k)
    base_k = torch.repeat_interleave(num_k.cumsum(0) - num_k, num_k)
    within_k = torch.arange(pk, device=dev) - base_k
    slot_k = within_k + (within_k >= pos_j_in_Ni[edge_of_pair]).long()
    k_nodes = dst_s[offsets[src_s[edge_of_pair]] + slot_k]

    i_of_pair = src_s[edge_of_pair]
    j_of_pair = dst_s[offsets[src_s[edge_of_pair]] + pos_j_in_Ni[edge_of_pair]]

    # Enumerate every (edge, k, l) triple where l in neighbors(j). We emit
    # all |N(j)| slots per (edge, k) and then mask out l==i and l==k. We don't
    # try to encode the "skip i" trick here because neighbors(j) aren't sorted
    # by destination value -- simpler to mask.
    grp_j = counts[j_of_pair]                                   # |N(j)|
    if grp_j.sum() == 0:
        return coords.new_zeros(E)

    pl = grp_j.sum().item()
    pair_idx = torch.repeat_interleave(torch.arange(pk, device=dev), grp_j)
    base_l = torch.repeat_interleave(grp_j.cumsum(0) - grp_j, grp_j)
    slot_l = torch.arange(pl, device=dev) - base_l
    l_nodes = dst_s[offsets[j_of_pair[pair_idx]] + slot_l]

    # Lift per-pair fields to per-triple, then filter.
    edge_q = edge_of_pair[pair_idx]
    i_q = i_of_pair[pair_idx]
    j_q = j_of_pair[pair_idx]
    k_q = k_nodes[pair_idx]
    l_q = l_nodes

    # Drop l == i (going back along the bond) and l == k (3-ring: k,i,j,l coplanar).
    keep = (l_q != i_q) & (l_q != k_q)
    edge_q = edge_q[keep]; i_q = i_q[keep]; j_q = j_q[keep]
    k_q = k_q[keep];       l_q = l_q[keep]

    if edge_q.numel() == 0:
        return coords.new_zeros(E)

    # Dihedral geometry: angle between normals of planes (k,i,j) and (i,j,l).
    r_ki = coords[i_q] - coords[k_q]
    r_ij = coords[j_q] - coords[i_q]
    r_jl = coords[l_q] - coords[j_q]
    n1 = _safe_normalize(torch.cross(r_ki, r_ij, dim=-1))
    n2 = _safe_normalize(torch.cross(r_ij, r_jl, dim=-1))
    cos_d = (n1 * n2).sum(-1).clamp(-1.0, 1.0)
    ang = torch.acos(cos_d)

    # Mean per sorted edge, then undo the sort back to original edge order.
    # scatter(reduce="mean") of an empty group returns 0 -- exactly what we
    # want for edges with no valid (k, l) triple.
    mean_ang_sorted = scatter(ang, edge_q, dim=0, dim_size=E, reduce="mean")
    mean_ang = torch.empty_like(mean_ang_sorted)
    mean_ang[order] = mean_ang_sorted
    return mean_ang


def build_invariant_edge_attr(
    edge_attr_raw: torch.Tensor,
    coords: torch.Tensor,
    edge_index: torch.Tensor,
    include_dihedral: bool = True,
) -> torch.Tensor:
    """m
    Map raw loader edge features [bond_type, theta, phi, distance] to purely
    invariant features [bond_type, distance, bond_angle, (dihedral)].
    """
    bond_type = edge_attr_raw[:, 0:1]
    distance = edge_attr_raw[:, 3:4]
    bond_ang = compute_bond_angles(coords, edge_index).unsqueeze(-1)
    feats = [bond_type, distance, bond_ang]
    if include_dihedral:
        dih = compute_dihedral_angles(coords, edge_index).unsqueeze(-1)
        feats.append(dih)
    return torch.cat(feats, dim=-1)


# ---------------------------------------------------------------------------
# Message passing
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

_POOLINGS = {
    "add": global_add_pool,
    "mean": global_mean_pool,
    "max": global_max_pool,
}


class InvariantGNN(nn.Module):
    """
    Invariant message-passing GNN for QM9 HOMO-LUMO gap regression.

    Predicts in physical units (eV). Internally the network regresses on a
    standardized target; mean/std are stored as buffers and applied in
    `forward` to return original-scale predictions.

    Workflow:
        model = InvariantGNN(...)
        model.fit_target_stats(train_loader)  # once, on train split only
        # ... train with loss_fn(model(...), batch.y) in eV ...
        # (Huber/MAE/MSE -- see train.py; both pred and y are in eV.)
    """

    # PyG's QM9 target layout: 4 = HOMO-LUMO gap (eV).
    DEFAULT_TARGET_INDEX = 4

    def __init__(
        self,
        node_dim: int = 9,
        hidden_dim: int = 64,
        num_layers: int = 6,
        include_dihedral: bool = True,
        coord_cols: slice = slice(4, 7), # -> x,y,z
        out_dim: int = 1,
        pooling: str = "mean",  # intensive target -> mean
    ):
        super().__init__()
        if pooling not in _POOLINGS:
            raise ValueError(f"pooling must be one of {list(_POOLINGS)}; got {pooling!r}")
        self.coord_cols = coord_cols
        self.include_dihedral = include_dihedral
        self.edge_dim = 4 if include_dihedral else 3
        self._pool = _POOLINGS[pooling]

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

        # Target standardization buffers. Start at identity (0, 1) and get
        # overwritten by fit_target_stats. The _stats_fitted flag is checked
        # in forward so the model refuses to predict before stats are fit --
        # silent use of the identity would produce wrong-scale outputs.
        # All three are buffers: they travel with state_dict, so a loaded
        # checkpoint carries the train-split stats (and "fitted" bit) with it.
        self.register_buffer("target_mean", torch.zeros(1))
        self.register_buffer("target_std", torch.ones(1))
        self.register_buffer("_stats_fitted", torch.tensor(False))

    # ------------------------------------------------------------------
    # Target standardization
    # ------------------------------------------------------------------
    @torch.no_grad()
    def fit_target_stats(
        self,
        loader: Iterable,
        target_index: Optional[int] = None,
    ) -> Tuple[float, float]:
        """
        Compute mean/std of the target on the given loader. This MUST be the
        training split only -- using val/test leaks information.

        Uses a numerically stable streaming (Welford) update so arbitrarily
        large datasets work without loading everything into memory at once.

        `target_index` selects the column of batch.y; defaults to the
        HOMO-LUMO gap (index 4 in PyG's QM9).

        Returns (mean, std) as Python floats.
        """
        if target_index is None:
            target_index = self.DEFAULT_TARGET_INDEX

        # Chan/Welford parallel variance: merges per-batch (count, mean, M2)
        # into a running (count, mean, M2) that is mathematically identical to
        # computing mean/var over the concatenated dataset, in O(1) memory.
        # NOT per-batch standardization -- the final `mean`, `std` are global
        # over the whole loader. Used instead of torch.cat(all_y).std() so this
        # scales to arbitrarily large HDF5 splits without loading every label.
        # Reference: Chan, Golub, LeVeque (1979), "Updating Formulae and a
        # Pairwise Algorithm for Computing Sample Variances."
        count = 0
        mean = 0.0
        m2 = 0.0  # running sum of squared deviations from the running mean

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
            # Merge (count, mean, m2) <- (count, mean, m2) U (n_b, mean_b, m2_b).
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
                f"Target std is ~0 ({std:.2e}); check that target_index={target_index} "
                f"selects the right column."
            )

        device = self.target_mean.device
        self.target_mean.copy_(torch.tensor([mean], device=device))
        self.target_std.copy_(torch.tensor([std], device=device))
        self._stats_fitted.copy_(torch.tensor(True))
        return float(mean), float(std)


    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Guard against using the model before train-split stats were fit
        # (or after loading a checkpoint saved before fit_target_stats ran).
        # Without this, the (0, 1) identity buffers would silently produce
        # predictions on the wrong scale.
        if not bool(self._stats_fitted.item()):
            raise RuntimeError(
                "InvariantGNN.forward called before fit_target_stats. "
                "Call model.fit_target_stats(train_loader) once before "
                "training / evaluation, or load a checkpoint saved after it."
            )
        coords = x[:, self.coord_cols]
        inv_attr = build_invariant_edge_attr(
            edge_attr, coords, edge_index,
            include_dihedral=self.include_dihedral,
        )
        h = self.node_embed(x)
        e = self.edge_embed(inv_attr)
        for layer in self.layers:
            h = layer(h, edge_index, e)
        if batch is None:
            batch = torch.zeros(h.size(0), dtype=torch.long, device=h.device)
        g = self._pool(h, batch)
        z = self.readout(g).squeeze(-1)  # standardized-space prediction

        # Denormalize to physical units (eV). The guard at the top of forward
        # ensures target_mean/target_std hold fitted train-split values.
        return z * self.target_std + self.target_mean