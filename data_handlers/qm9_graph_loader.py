"""
Lazy, map-style HDF5 dataloader for QM9 graph data.

Returns torch_geometric `Data` objects per molecule. Sparsification of the
dense bond matrix is fully vectorized (torch.nonzero) so per-batch CPU cost
scales with edges, not with max_nodes**2 * batch_size.

HDF5 layout expected (written by data_processors/h5_maker_qm9.py):
    node_features : (N, MAX_NODES, 9)   float
    edge_features : (N, MAX_NODES, MAX_NODES, 4) float
        channels = [bond_type, theta, phi, distance]
    targets       : (N, 19)             float
    n_atoms       : (N, 2)              int   [total, heavy]
    qfim          : (N, 101, 101)       float64   (optional)

The QFIM is stored as an additional dataset in the same HDF5 file. The
first (n_qubits * per_qubit_dim) x (n_qubits * per_qubit_dim) block is the
rot-gate parameter block (n_qubits=10, per_qubit_dim=num_layers *
operations_per_layer=6, so 60x60). It is reshaped to
(n_qubits, n_qubits, per_qubit_dim, per_qubit_dim) and sliced per bonded
atom pair to form edge-level QFIM features for gnn_qfim.

Design:
- One `h5py.File` handle per worker process, opened lazily in __getitem__ and
  reused for the lifetime of the worker. No reopen per sample.
- `worker_init_fn` resets the handle so forked workers don't share it.
- Vectorized collate via `torch_geometric.data.Batch.from_data_list`.

Target indices follow the 19-dim QM9 target layout; resolved once in __init__.
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence, Tuple

import h5py
import numpy as np
import torch
from loguru import logger
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Batch, Data


TARGET_IDX: Tuple[str, ...] = (
    "mu", "alpha", "homo", "lumo", "gap", "r2", "zpve",
    "U0", "U", "H", "G", "Cv",
    "U0_atom", "U_atom", "H_atom", "G_atom",
    "rot_A", "rot_B", "rot_C",
)

TARGET_MEAN_ATTR = "target_mean"
TARGET_STD_ATTR = "target_std"
TARGET_COUNT_ATTR = "target_count"


def _resolve_target_indices(keys: List[str]) -> List[int]:
    invalid = [k for k in keys if k not in TARGET_IDX]
    if invalid:
        raise ValueError(f"Unknown targets {invalid}. Valid: {list(TARGET_IDX)}")
    return [TARGET_IDX.index(k) for k in keys]


def ensure_target_stats_metadata(h5_path: str) -> Tuple[np.ndarray, np.ndarray, int]:
    """Read target stats from HDF5 attrs, or compute and store them once."""
    h5_path = str(h5_path)
    attr_names = (TARGET_MEAN_ATTR, TARGET_STD_ATTR, TARGET_COUNT_ATTR)

    with h5py.File(h5_path, "r") as f:
        if all(name in f.attrs for name in attr_names):
            mean = np.asarray(f.attrs[TARGET_MEAN_ATTR], dtype=np.float64)
            std = np.asarray(f.attrs[TARGET_STD_ATTR], dtype=np.float64)
            count = int(f.attrs[TARGET_COUNT_ATTR])
            return mean, std, count

    with h5py.File(h5_path, "r+") as f:
        if all(name in f.attrs for name in attr_names):
            mean = np.asarray(f.attrs[TARGET_MEAN_ATTR], dtype=np.float64)
            std = np.asarray(f.attrs[TARGET_STD_ATTR], dtype=np.float64)
            count = int(f.attrs[TARGET_COUNT_ATTR])
            return mean, std, count

        targets = np.asarray(f["targets"], dtype=np.float64)
        if targets.ndim == 1:
            targets = targets[:, None]
        if targets.shape[0] < 2:
            raise RuntimeError(
                f"Target stats need >= 2 samples; got {targets.shape[0]}."
            )

        mean = targets.mean(axis=0)
        std = targets.std(axis=0, ddof=1)
        count = int(targets.shape[0])
        if np.any(std < 1e-8):
            raise RuntimeError("Target std is ~0 for at least one target column.")

        f.attrs[TARGET_MEAN_ATTR] = mean
        f.attrs[TARGET_STD_ATTR] = std
        f.attrs[TARGET_COUNT_ATTR] = count
        logger.info(f"wrote target stats metadata | count={count} | path={h5_path}")
        return mean, std, count


def target_stats_for_keys(
    h5_path: str,
    target_keys: List[str],
) -> Tuple[np.ndarray, np.ndarray, int]:
    mean, std, count = ensure_target_stats_metadata(h5_path)
    indices = _resolve_target_indices(target_keys)
    return mean[indices], std[indices], count


def _worker_init_fn(worker_id: int) -> None:
    """Reset the cached HDF5 handle in each worker so they open fresh."""
    info = torch.utils.data.get_worker_info()
    if info is None:
        return
    dataset = info.dataset
    if hasattr(dataset, "_reset_handles"):
        dataset._reset_handles()


class QM9GraphDataset(Dataset):
    """Map-style QM9 graph dataset with lazy per-worker HDF5 access."""

    def __init__(
        self,
        h5_path: str,
        target_keys: List[str],
        qfim_shape: Optional[Tuple[int, int]] = None,
        target_mean: Optional[Sequence[float]] = None,
        target_std: Optional[Sequence[float]] = None,
        qfim_ablation_mode: str = "none",
        qfim_ablation_seed: int = 42,
        qfim_random_scale: float = 0.25,
        use_all_atoms: bool = False,
    ) -> None:
        """
        Args:
            h5_path:     Path to QM9 HDF5 split file.
            target_keys: Subset of TARGET_IDX to predict.
            qfim_shape:  (n_qubits, per_qubit_dim) to slice the in-file qfim
                         rot-gate block. Required to enable QFIM features.
            target_mean: Train-split mean for selected targets.
            target_std:  Train-split std for selected targets.
            qfim_ablation_mode:
                         none | row_shuffle | random | zero. Controls only the
                         QFIM tensor; graph rows and targets stay unchanged.
            use_all_atoms: If true, include hydrogens and use n_atoms[:, 0].
                           Default false preserves the heavy-atom-only setup.
        """
        self.h5_path = str(h5_path)
        self.target_indices = _resolve_target_indices(target_keys)
        self.qfim_shape = qfim_shape
        self.use_all_atoms = bool(use_all_atoms)
        self.qfim_ablation_mode = str(qfim_ablation_mode).lower()
        if self.qfim_ablation_mode not in ("none", "row_shuffle", "random", "zero"):
            raise ValueError(
                "qfim_ablation_mode must be one of: none, row_shuffle, random, zero"
            )
        self.qfim_ablation_seed = int(qfim_ablation_seed)
        self.qfim_random_scale = float(qfim_random_scale)
        self.target_mean = (
            np.asarray(target_mean, dtype=np.float32).reshape(-1)
            if target_mean is not None
            else None
        )
        self.target_std = (
            np.asarray(target_std, dtype=np.float32).reshape(-1)
            if target_std is not None
            else None
        )
        if (self.target_mean is None) != (self.target_std is None):
            raise ValueError("target_mean and target_std must be provided together.")
        if self.target_mean is not None:
            n_targets = len(self.target_indices)
            if (
                self.target_mean.shape != (n_targets,)
                or self.target_std.shape != (n_targets,)
            ):
                raise ValueError(
                    "target_mean and target_std must match the selected target count."
                )
            if np.any(self.target_std < 1e-8):
                raise ValueError("target_std is ~0 for at least one selected target.")

        with h5py.File(self.h5_path, "r") as f:
            self.n_samples = int(f["node_features"].shape[0])
            self.max_nodes = int(f["node_features"].shape[1])
            self.node_dim = int(f["node_features"].shape[2])
            self._qfim_in_h5 = "qfim" in f

        self._h5: Optional[h5py.File] = None
        self._indices: Optional[np.ndarray] = None
        self._qfim_row_perm: Optional[np.ndarray] = None

        if self._qfim_in_h5 and self.qfim_shape is None:
            self._qfim_in_h5 = False
        if self.use_all_atoms and self._qfim_in_h5:
            nq = int(self.qfim_shape[0]) if self.qfim_shape is not None else 0
            if nq < self.max_nodes:
                raise ValueError(
                    "use_all_atoms=True is not compatible with edge-level QFIM "
                    f"for this file: qfim.n_qubits={nq}, max_nodes={self.max_nodes}. "
                    "Run all-atom baselines with qfim disabled, or provide an "
                    "all-atom QFIM tensor."
                )
        if self._qfim_in_h5 and self.qfim_ablation_mode == "row_shuffle":
            rng = np.random.default_rng(self.qfim_ablation_seed)
            perm = rng.permutation(self.n_samples)
            self._qfim_row_perm = perm.astype(np.int64, copy=False)
        logger.info(
            f"QM9GraphDataset: {len(self)} molecules"
            f"{f' ({self.n_samples} raw)' if self._indices is not None else ''} | "
            f"node_dim={self.node_dim} | max_nodes={self.max_nodes} | "
            f"atoms={'all' if self.use_all_atoms else 'heavy'} | "
            f"qfim={'yes' if self._qfim_in_h5 else 'no'}"
            f"{f' ({self.qfim_ablation_mode})' if self._qfim_in_h5 else ''} | "
            f"path={self.h5_path}"
        )

    def _reset_handles(self) -> None:
        if self._h5 is not None:
            try:
                self._h5.close()
            except Exception:
                pass
        self._h5 = None

    def _ensure_open(self) -> None:
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, "r", libver="latest", swmr=True)

    def __len__(self) -> int:
        if self._indices is not None:
            return int(self._indices.shape[0])
        return self.n_samples

    def __getitem__(self, idx: int) -> Data:
        self._ensure_open()
        if self._indices is not None:
            idx = int(self._indices[idx])

        n_total = int(self._h5["n_atoms"][idx, 0])
        n_heavy = int(self._h5["n_atoms"][idx, 1])
        n_nodes = n_total if self.use_all_atoms else n_heavy
        nodes = self._h5["node_features"][idx, :n_nodes]                     # (N, 9)
        edges = self._h5["edge_features"][idx, :n_nodes, :n_nodes]           # (N, N, 4)
        raw_target = np.asarray(self._h5["targets"][idx])
        if raw_target.ndim == 0:
            if len(self.target_indices) != 1:
                raise ValueError(
                    "Scalar targets found in HDF5, but multiple target keys were requested."
                )
            target = raw_target.reshape(1)
        else:
            target = raw_target[self.target_indices]
        if self.target_mean is not None and self.target_std is not None:
            target = (target - self.target_mean) / self.target_std

        nodes_t = torch.from_numpy(np.asarray(nodes, dtype=np.float32))
        edges_t = torch.from_numpy(np.asarray(edges, dtype=np.float32))
        target_t = torch.from_numpy(np.asarray(target, dtype=np.float32))

        # Vectorized sparsification: one GPU-eligible op, no Python loops.
        bond_mask = edges_t[..., 0] > 0                                       # (H, H)
        edge_idx = bond_mask.nonzero(as_tuple=False).t().contiguous()         # (2, E)
        edge_attr = edges_t[bond_mask]                                        # (E, 4)

        data = Data(
            x=nodes_t,
            edge_index=edge_idx.long(),
            edge_attr=edge_attr,
            y=target_t,
            n_total=torch.tensor([n_total], dtype=torch.long),
            n_heavy=torch.tensor([n_heavy], dtype=torch.long),
            n_nodes=torch.tensor([n_nodes], dtype=torch.long),
        )

        # Full rot-gate QFIM sub-block per molecule: (n_qubits, n_qubits, pd, pd).
        # Off-diagonal (i != j) sub-blocks are the parameter couplings between
        # qubits; used as per-edge quantum features by gnn_qfim.
        if self._qfim_in_h5 and self.qfim_shape is not None:
            nq, pd = self.qfim_shape
            n_rot = nq * pd
            qfim_idx = idx
            if self._qfim_row_perm is not None:
                qfim_idx = int(self._qfim_row_perm[idx])

            if self.qfim_ablation_mode == "zero":
                rot_block = np.zeros((n_rot, n_rot), dtype=np.float32)
            elif self.qfim_ablation_mode == "random":
                rng = np.random.default_rng(self.qfim_ablation_seed + idx)
                rot_block = rng.uniform(
                    low=-self.qfim_random_scale,
                    high=self.qfim_random_scale,
                    size=(n_rot, n_rot),
                ).astype(np.float32)
                rot_block = 0.5 * (rot_block + rot_block.T)
            else:
                rot_block = np.asarray(
                    self._h5["qfim"][qfim_idx, :n_rot, :n_rot], dtype=np.float32
                )
            # bioQINN stores rot-gate weights as (num_layers, ops_per_layer,
            # n_qubits) and PennyLane's metric_tensor flattens in C order, so
            # the qubit axis is the FASTEST-varying one along the 60-dim flat
            # parameter vector. Verified empirically by probe_qfim_reshape.py
            # against a recomputed QFIM -- see commit log.
            # (n_rot, n_rot) -> (pd, nq, pd, nq) -> (nq, nq, pd, pd)
            rot_block = rot_block.reshape(pd, nq, pd, nq).transpose(1, 3, 0, 2)
            # Add leading batch axis so PyG's concat-along-dim-0 produces the
            # desired (B, nq, nq, pd, pd) stacked tensor in the batched Data.
            data.qfim_block = torch.from_numpy(
                np.ascontiguousarray(rot_block)
            ).unsqueeze(0)
            data.qfim_nq = torch.tensor([nq], dtype=torch.long)

        return data

    def __del__(self):
        self._reset_handles()


def _collate(batch: List[Data]) -> Batch:
    return Batch.from_data_list(batch)


def build_loader(
    h5_path: str,
    target_keys: List[str],
    batch_size: int,
    shuffle: bool,
    num_workers: int = 4,
    qfim_shape: Optional[Tuple[int, int]] = None,
    target_mean: Optional[Sequence[float]] = None,
    target_std: Optional[Sequence[float]] = None,
    qfim_ablation_mode: str = "none",
    qfim_ablation_seed: int = 42,
    qfim_random_scale: float = 0.25,
    use_all_atoms: bool = False,
    max_samples: Optional[int] = None,
    seed: int = 42,
    pin_memory: bool = True,
) -> DataLoader:
    """Build a DataLoader with the map-style dataset and vectorized collate."""
    dataset: Dataset = QM9GraphDataset(
        h5_path=h5_path,
        target_keys=target_keys,
        qfim_shape=qfim_shape,
        target_mean=target_mean,
        target_std=target_std,
        qfim_ablation_mode=qfim_ablation_mode,
        qfim_ablation_seed=qfim_ablation_seed,
        qfim_random_scale=qfim_random_scale,
        use_all_atoms=use_all_atoms,
    )
    if max_samples is not None and max_samples < len(dataset):
        rng = np.random.default_rng(seed)
        indices = rng.choice(len(dataset), size=max_samples, replace=False).tolist()
        dataset = torch.utils.data.Subset(dataset, indices)
        logger.info(f"Subsampled to {max_samples} molecules (seed={seed})")

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        worker_init_fn=_worker_init_fn,
        collate_fn=_collate,
    )

def build_loaders_from_config(config: Any) -> Any:
    """
    Build train+val loaders (if config.setup.train) or test loader otherwise.

    Reads optional config.qfim sub-namespace:
        qfim.n_qubits      : int
        qfim.per_qubit_dim : int (num_layers * ops_per_layer)
    QFIM data is read from the `qfim` dataset inside the HDF5 split file.
    """
    target_keys = list(config.setup.targets)
    batch_size = int(config.setup.batch_size)
    num_workers = int(getattr(config.setup, "num_workers", 4))
    seed = int(getattr(config.setup, "seed", 42))
    shuffle = bool(getattr(config.setup, "shuffle", True))
    data_cfg = getattr(config, "data", None)
    use_all_atoms = bool(
        getattr(
            data_cfg,
            "use_all_atoms",
            getattr(config.setup, "use_all_atoms", False),
        )
    )

    qfim_cfg = getattr(config, "qfim", None)
    qfim_shape: Optional[Tuple[int, int]] = None
    if qfim_cfg is not None:
        qfim_shape = (int(qfim_cfg.n_qubits), int(qfim_cfg.per_qubit_dim))
    qfim_ablation_mode = (
        str(getattr(qfim_cfg, "ablation_mode", "none"))
        if qfim_cfg is not None
        else "none"
    )
    qfim_ablation_seed = (
        int(getattr(qfim_cfg, "ablation_seed", seed))
        if qfim_cfg is not None
        else seed
    )
    qfim_random_scale = (
        float(getattr(qfim_cfg, "random_scale", 0.25))
        if qfim_cfg is not None
        else 0.25
    )

    target_mean, target_std, target_count = target_stats_for_keys(
        config.paths.train,
        target_keys=target_keys,
    )

    def _attach_target_stats(loader: DataLoader) -> None:
        loader.target_mean = target_mean
        loader.target_std = target_std
        loader.target_stats_count = target_count

    if config.setup.train:
        train_loader = build_loader(
            h5_path=config.paths.train,
            target_keys=target_keys,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            qfim_shape=qfim_shape,
            target_mean=target_mean,
            target_std=target_std,
            qfim_ablation_mode=qfim_ablation_mode,
            qfim_ablation_seed=qfim_ablation_seed,
            qfim_random_scale=qfim_random_scale,
            use_all_atoms=use_all_atoms,
            max_samples=getattr(config.setup, "train_n", None),
            seed=seed,
        )
        _attach_target_stats(train_loader)
        val_loader = build_loader(
            h5_path=config.paths.val,
            target_keys=target_keys,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            qfim_shape=qfim_shape,
            target_mean=target_mean,
            target_std=target_std,
            qfim_ablation_mode=qfim_ablation_mode,
            qfim_ablation_seed=qfim_ablation_seed + 1,
            qfim_random_scale=qfim_random_scale,
            use_all_atoms=use_all_atoms,
            max_samples=getattr(config.setup, "val_n", None),
            seed=seed,
        )
        _attach_target_stats(val_loader)
        logger.info(
            f"train batches={len(train_loader)} | val batches={len(val_loader)}"
        )
        return train_loader, val_loader

    test_loader = build_loader(
        h5_path=config.paths.test,
        target_keys=target_keys,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        qfim_shape=qfim_shape,
        target_mean=target_mean,
        target_std=target_std,
        qfim_ablation_mode=qfim_ablation_mode,
        qfim_ablation_seed=qfim_ablation_seed + 2,
        qfim_random_scale=qfim_random_scale,
        use_all_atoms=use_all_atoms,
        max_samples=getattr(config.setup, "test_n", None),
        seed=seed,
    )
    _attach_target_stats(test_loader)
    logger.info(f"test batches={len(test_loader)}")
    return test_loader
