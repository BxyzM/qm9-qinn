"""
Filters QM9 to molecules with 7 or 8 heavy atoms, splits 50-10-40
train/val/test, and writes each split to a separate HDF5 file.

Node features are stored as (N, MAX_NODES, 9), where columns 7 and 8
are the total atom count and heavy atom count respectively, broadcast
identically across all atom rows within a molecule.

Atoms within each molecule are sorted in descending order of atomic
number prior to writing. Padding rows (atomic number 0) are naturally
placed last by this sort. Edge features are reordered consistently.

Edge features are stored as (N, MAX_NODES, MAX_NODES, 4) arrays where
the four channels per atom pair are:
    [bond_type_integer, theta_polar_rad, phi_azimuthal_rad, distance_angstrom]
Geometric quantities are zeroed for non-bonded pairs.

Author: Dr. Aritra Bal (ETP)
Date: March 11, 2026
"""

import pathlib
import numpy as np
import torch
import h5py
from torch_geometric.utils import to_dense_adj
from loguru import logger

from data_handlers.qm9_dataloader import QM9DenseDataset

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SAVE_ROOT   = pathlib.Path("/ceph/mbinder/BIO/QM9")
QM9_ROOT    = "./data/qm9"
MAX_NODES   = 36
SEED        = 42
SPLIT_RATIO = (0.50, 0.10, 0.40)   # train, val, test
_HEAVY_ATOM_COUNTS = (5, 6, 7, 8, 9)
NODE_FEATURE_NAMES = np.array(
    [
        "atomic_number",
        "aromatic_flag",
        "hybridisation_scalar",
        "num_attached_hydrogens",
        "x_coord_angstrom",
        "y_coord_angstrom",
        "z_coord_angstrom",
        "n_atoms_total",
        "n_heavy_atoms",
    ],
    dtype=h5py.special_dtype(vlen=str),
)

EDGE_FEATURE_NAMES = np.array(
    [
        "bond_type_integer",
        "theta_polar_rad",
        "phi_azimuthal_rad",
        "distance_angstrom",
    ],
    dtype=h5py.special_dtype(vlen=str),
)

ATOM_NUMBER_INFO = np.array(
    [
        "dim0: total atom count including hydrogens",
        "dim1: heavy atom count (atomic number > 1)",
    ],
    dtype=h5py.special_dtype(vlen=str),
)

TARGET_INFO = np.array(
    [
        "mu: dipole moment (Debye)",
        "alpha: isotropic polarizability (a0^3)",
        "homo: HOMO energy (Hartree)",
        "lumo: LUMO energy (Hartree)",
        "gap: HOMO-LUMO gap (Hartree)",
        "r2: electronic spatial extent (a0^2)",
        "zpve: zero-point vibrational energy (Hartree)",
        "U0: internal energy at 0K (Hartree)",
        "U: internal energy at 298.15K (Hartree)",
        "H: enthalpy at 298.15K (Hartree)",
        "G: free energy at 298.15K (Hartree)",
        "Cv: heat capacity at 298.15K (cal/mol/K)",
        "U0_atom: atomisation energy at 0K (Hartree)",
        "U_atom: atomisation energy at 298.15K (Hartree)",
        "H_atom: atomisation enthalpy at 298.15K (Hartree)",
        "G_atom: atomisation free energy at 298.15K (Hartree)",
        "A: rotational constant A (GHz)",
        "B: rotational constant B (GHz)",
        "C: rotational constant C (GHz)",
    ],
    dtype=h5py.special_dtype(vlen=str),
)

TARGET_IDX = np.array(
    [
        "mu", "alpha", "homo", "lumo", "gap", "r2", "zpve",
        "U0", "U", "H", "G", "Cv",
        "U0_atom", "U_atom", "H_atom", "G_atom",
        "rot_A", "rot_B", "rot_C",
    ],
    dtype=h5py.special_dtype(vlen=str),
)


# ---------------------------------------------------------------------------
# Pair-feature computation
# ---------------------------------------------------------------------------
def compute_pair_features(
    node_feat: np.ndarray,
    edge_feat: np.ndarray,
    max_nodes: int,
) -> np.ndarray:
    """
    Compute pairwise geometric features for all atom pairs.

    For each pair (i, j) the four channels are:
        [0] bond_type : integer bond type (0=none, 1=single, 2=double,
                        3=triple, 4=aromatic)
        [1] theta     : polar angle of displacement vector r_j - r_i  [0, pi]
        [2] phi       : azimuthal angle of displacement r_j - r_i  [-pi, pi]
        [3] d         : Euclidean distance ||r_j - r_i|| in Angstrom

    Geometric quantities are zeroed for non-bonded pairs (bond_type == 0).
    Coordinates are read from node_feat columns 4:7; appended scalar
    columns (7, 8) are ignored here.

    Args:
        node_feat : (max_nodes, 9); columns 4:7 are xyz coordinates.
        edge_feat : (max_nodes, max_nodes); integer bond type, 0 if absent.
        max_nodes : padded node dimension (MAX_NODES).

    Returns:
        pair_feat : (max_nodes, max_nodes, 4) float32.
    """
    coords = node_feat[:max_nodes, 4:7].astype(np.float32)                    # (N, 3)
    bond   = edge_feat[:max_nodes, :max_nodes].astype(np.float32)             # (N, N)

    # diff[i, j] = r_j - r_i
    diff = coords[np.newaxis, :, :] - coords[:, np.newaxis, :]                # (N, N, 3)
    dx, dy, dz = diff[..., 0], diff[..., 1], diff[..., 2]

    dist  = np.sqrt(np.sum(diff ** 2, axis=-1) + 1e-8)                        # (N, N)
    theta = np.arccos(np.clip(dz / (dist + 1e-8), -1.0, 1.0))                # (N, N)
    phi   = np.arctan2(dy, dx)                                                 # (N, N)

    # Zero geometric values for non-bonded pairs
    bonded = bond > 0.5                                                        # (N, N) bool
    theta  = np.where(bonded, theta, 0.0)
    phi    = np.where(bonded, phi,   0.0)
    dist   = np.where(bonded, dist,  0.0)

    return np.stack([bond, theta, phi, dist], axis=-1).astype(np.float32)     # (N, N, 4)


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------
def get_heavy_atom_filtered_indices(
    dataset: QM9DenseDataset,
    heavy_atom_counts: tuple = (7, 8),
) -> list:
    """
    Return dataset indices where heavy atom count is in heavy_atom_counts.

    Args:
        dataset           : QM9DenseDataset instance.
        heavy_atom_counts : Tuple of accepted heavy atom counts.

    Returns:
        List of integer indices into dataset.
    """
    indices = []
    for i in range(len(dataset.dataset)):
        data    = dataset.dataset[i]
        n_heavy = int((data.x[:, 5] > 1).sum())
        if n_heavy in heavy_atom_counts:
            indices.append(i)
    logger.info(
        f"Filtered to {len(indices)} molecules with "
        f"{heavy_atom_counts} heavy atoms"
    )
    return indices


# ---------------------------------------------------------------------------
# Dense conversion
# ---------------------------------------------------------------------------
def sample_to_dense(data) -> tuple:
    """
    Convert a single PyG Data sample to dense, sorted numpy arrays.

    Node feature layout (9 columns):
        0  atomic_number
        1  aromatic_flag
        2  hybridisation_scalar  (0=unknown, 1=SP, 2=SP2, 3=SP3)
        3  num_attached_hydrogens
        4  x_coord_angstrom
        5  y_coord_angstrom
        6  z_coord_angstrom
        7  n_atoms_total          (broadcast scalar, same for all rows)
        8  n_heavy_atoms          (broadcast scalar, same for all rows)

    Atoms are sorted in descending order of atomic number. Padding rows
    (atomic number == 0) are placed last by this sort. The bond adjacency
    matrix is reordered by the same permutation on both axes.

    Args:
        data : PyG Data object for one molecule.

    Returns:
        node_feat : (MAX_NODES, 9)             float32
        pair_feat : (MAX_NODES, MAX_NODES, 4)  float32  [bond, theta, phi, d]
        targets   : (19,)                       float32
        n_atoms   : (2,)                        float32  [total, heavy]
    """
    n = data.x.shape[0]

    # --- Hybridisation scalar -------------------------------------------
    hyb     = data.x[:, 7:10]                                                 # (n, 3) one-hot
    hyb_any = hyb.sum(dim=-1, keepdim=True).bool()
    hyb_s   = (hyb.argmax(dim=-1, keepdim=True).float() + 1.0) * hyb_any.float()

    node_raw = torch.cat(
        [data.x[:, 5:6], data.x[:, 6:7], hyb_s, data.x[:, 10:11], data.pos],
        dim=-1,
    )                                                                          # (n, 7)

    n_total = n
    n_heavy = int((data.x[:, 5] > 1).sum())

    # --- Pad to MAX_NODES -------------------------------------------------
    node_feat_7        = torch.zeros(MAX_NODES, 7)
    node_feat_7[:n]    = node_raw
    node_np            = node_feat_7.numpy().astype(np.float32)               # (MAX_NODES, 7)

    # --- Bond adjacency (bond type integers) ------------------------------
    bond_types = data.edge_attr.argmax(dim=-1).float() + 1
    edge_np    = to_dense_adj(
        data.edge_index,
        edge_attr=bond_types,
        max_num_nodes=MAX_NODES,
    ).squeeze(0).numpy().astype(np.float32)                                    # (MAX_NODES, MAX_NODES)

    # --- Sort atoms by descending atomic number ---------------------------
    # Padding rows have atomic_number == 0 and therefore sort last naturally.
    perm    = np.argsort(-node_np[:, 0])                                       # (MAX_NODES,)
    node_np = node_np[perm]                                                    # reorder rows
    edge_np = edge_np[np.ix_(perm, perm)]                                      # reorder rows and cols

    # --- Append n_atoms_total and n_heavy as broadcast columns -----------
    scalar_cols        = np.zeros((MAX_NODES, 2), dtype=np.float32)
    scalar_cols[:, 0]  = float(n_total)
    scalar_cols[:, 1]  = float(n_heavy)
    node_np            = np.concatenate([node_np, scalar_cols], axis=-1)      # (MAX_NODES, 9)

    # --- Pairwise geometric features ------------------------------------
    pair_feat = compute_pair_features(node_np, edge_np, MAX_NODES)            # (MAX_NODES, MAX_NODES, 4)

    n_atoms = np.array([n_total, n_heavy], dtype=np.float32)

    return (
        node_np,
        pair_feat,
        data.y.squeeze(0).numpy().astype(np.float32),
        n_atoms,
    )


# ---------------------------------------------------------------------------
# HDF5 writer
# ---------------------------------------------------------------------------
def write_split(
    path: pathlib.Path,
    dataset: QM9DenseDataset,
    indices: list,
) -> None:
    """
    Write one split to an HDF5 file at path.

    Datasets are written sample-by-sample into pre-allocated arrays to
    avoid RAM spikes.

    Schema
    ------
    node_features    : (n, MAX_NODES, 9)
    edge_features    : (n, MAX_NODES, MAX_NODES, 4)
    targets          : (n, 19)
    n_atoms          : (n, 2)
    nodeFeatureNames : string metadata  (9 entries)
    edgeFeatureNames : string metadata  (4 entries)
    atomNumberInfo   : string metadata
    targetInfo       : string metadata
    targetIndex      : string metadata

    Args:
        path    : Full path to the output .h5 file.
        dataset : QM9DenseDataset instance.
        indices : Molecule indices for this split.
    """
    n = len(indices)
    path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Writing {n} samples to {path}")

    with h5py.File(path, "w") as f:
        # Pre-allocate numeric datasets
        ds_node   = f.create_dataset(
            "node_features",
            shape=(n, MAX_NODES, 9),
            dtype=np.float32,
        )
        ds_edge   = f.create_dataset(
            "edge_features",
            shape=(n, MAX_NODES, MAX_NODES, 4),
            dtype=np.float32,
        )
        ds_target = f.create_dataset(
            "targets",
            shape=(n, 19),
            dtype=np.float32,
        )
        ds_natoms = f.create_dataset(
            "n_atoms",
            shape=(n, 2),
            dtype=np.float32,
        )

        # String metadata
        f.create_dataset("nodeFeatureNames", data=NODE_FEATURE_NAMES)
        f.create_dataset("edgeFeatureNames", data=EDGE_FEATURE_NAMES)
        f.create_dataset("atomNumberInfo",   data=ATOM_NUMBER_INFO)
        f.create_dataset("targetInfo",       data=TARGET_INFO)
        f.create_dataset("targetIndex",      data=TARGET_IDX)

        # Sample-by-sample write
        for out_idx, src_idx in enumerate(indices):
            node, pair, tgt, nat = sample_to_dense(dataset.dataset[src_idx])
            ds_node[out_idx]   = node
            ds_edge[out_idx]   = pair
            ds_target[out_idx] = tgt
            ds_natoms[out_idx] = nat

        # Split-level metadata attributes
        f.attrs["n_samples"]         = n
        f.attrs["max_nodes"]         = MAX_NODES
        f.attrs["seed"]              = SEED
        f.attrs["split_ratio"]       = str(SPLIT_RATIO)
        f.attrs["heavy_atom_filter"] = str((7, 8))
        f.attrs["node_channels"]     = (
            "atomic_number | aromatic_flag | hybridisation_scalar | "
            "num_attached_hydrogens | x | y | z | n_atoms_total | n_heavy_atoms"
        )
        f.attrs["edge_channels"]     = (
            "bond_type_integer | theta_polar_rad | "
            "phi_azimuthal_rad | distance_angstrom"
        )
        f.attrs["atom_sort_order"]   = "descending atomic number; padding (Z=0) last"

    logger.info(f"Wrote {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    dataset = QM9DenseDataset(root=QM9_ROOT)

    # Filter to 7 and 8 heavy-atom molecules
    filtered = get_heavy_atom_filtered_indices(dataset, heavy_atom_counts=_HEAVY_ATOM_COUNTS)
    total    = len(filtered)

    # Deterministic shuffle then split
    rng      = np.random.default_rng(SEED)
    shuffled = rng.permutation(filtered).tolist()

    n_train   = int(SPLIT_RATIO[0] * total)
    n_val     = int(SPLIT_RATIO[1] * total)
    train_idx = shuffled[:n_train]
    val_idx   = shuffled[n_train : n_train + n_val]
    test_idx  = shuffled[n_train + n_val :]           # remainder avoids rounding loss

    logger.info(
        f"Split sizes | train={len(train_idx)} | "
        f"val={len(val_idx)} | test={len(test_idx)}"
    )

    write_split(SAVE_ROOT / "train" / "qm9_train.h5", dataset, train_idx)
    write_split(SAVE_ROOT / "val"   / "qm9_val.h5",   dataset, val_idx)
    write_split(SAVE_ROOT / "test"  / "qm9_test.h5",  dataset, test_idx)

    logger.info("All splits written successfully")