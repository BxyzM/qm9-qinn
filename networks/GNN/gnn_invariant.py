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
- Internally the MLP regresses on a standardized target; `forward` denormalizes
  so callers always work in physical units (eV).
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
    signed: bool = False,
) -> torch.Tensor:
    """
    Dihedral angle for each edge (i -> j) using one other neighbor k of i
    and one other neighbor l of j. Returns 0 for edges where either side has
    no additional neighbor.
    """
    src, dst = edge_index[0], edge_index[1]
    E = src.numel()
    if E == 0:
        return coords.new_zeros(0)

    N = int(coords.size(0))
    _, src_s, dst_s, offsets = _sorted_adjacency(edge_index, N)
    counts = offsets[1:] - offsets[:-1]
    dst_s_len = dst_s.numel()

    def first_other(anchor: torch.Tensor, excluded: torch.Tensor):
        cnt = counts[anchor]
        idx0 = offsets[anchor]
        idx0_safe = idx0.clamp(max=dst_s_len - 1)
        idx1_safe = (idx0 + 1).clamp(max=dst_s_len - 1)
        first = dst_s[idx0_safe]
        second = dst_s[idx1_safe]
        use_second = (first == excluded) & (cnt >= 2)
        chosen = torch.where(use_second, second, first)
        has_other = (cnt >= 2) | ((cnt >= 1) & (first != excluded))
        return chosen, has_other

    k_node, has_k = first_other(src, dst)
    l_node, has_l = first_other(dst, src)
    valid = has_k & has_l

    r_ki = coords[src] - coords[k_node]
    r_ij = coords[dst] - coords[src]
    r_jl = coords[l_node] - coords[dst]
    n1 = torch.cross(r_ki, r_ij, dim=-1)
    n2 = torch.cross(r_ij, r_jl, dim=-1)

    if signed:
        n1n = _safe_normalize(n1)
        n2n = _safe_normalize(n2)
        r_ij_n = _safe_normalize(r_ij)
        m1 = torch.cross(n1n, r_ij_n, dim=-1)
        x = (n1n * n2n).sum(-1)
        y = (m1 * n2n).sum(-1)
        ang = torch.atan2(y, x)
    else:
        n1n = _safe_normalize(n1)
        n2n = _safe_normalize(n2)
        cos_d = (n1n * n2n).sum(-1).clamp(-1.0, 1.0)
        ang = torch.acos(cos_d)

    return torch.where(valid, ang, torch.zeros_like(ang))


def build_invariant_edge_attr(
    edge_attr_raw: torch.Tensor,
    coords: torch.Tensor,
    edge_index: torch.Tensor,
    include_dihedral: bool = True,
    signed_dihedral: bool = False,
) -> torch.Tensor:
    """
    Map raw loader edge features [bond_type, theta, phi, distance] to purely
    invariant features [bond_type, distance, bond_angle, (dihedral)].
    """
    bond_type = edge_attr_raw[:, 0:1]
    distance = edge_attr_raw[:, 3:4]
    bond_ang = compute_bond_angles(coords, edge_index).unsqueeze(-1)
    feats = [bond_type, distance, bond_ang]
    if include_dihedral:
        dih = compute_dihedral_angles(coords, edge_index, signed=signed_dihedral).unsqueeze(-1)
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
        model.fit_target_stats(train_loader, target_index=4)  # 4 = gap in PyG QM9
        # ... train with MSE between model(...) and batch.y[:, 4] ...
    """

    # PyG's QM9 target layout: 4 = HOMO-LUMO gap (eV).
    DEFAULT_TARGET_INDEX = 4

    def __init__(
        self,
        node_dim: int = 9,
        hidden_dim: int = 64,
        num_layers: int = 6,
        include_dihedral: bool = True,
        signed_dihedral: bool = False,
        coord_cols: slice = slice(4, 7),
        out_dim: int = 1,
        pooling: str = "mean",  # intensive target -> mean
    ):
        super().__init__()
        if pooling not in _POOLINGS:
            raise ValueError(f"pooling must be one of {list(_POOLINGS)}; got {pooling!r}")
        self.coord_cols = coord_cols
        self.include_dihedral = include_dihedral
        self.signed_dihedral = signed_dihedral
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
            signed_dihedral=self.signed_dihedral,
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