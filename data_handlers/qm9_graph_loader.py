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

import pathlib
from typing import Any, List, Optional, Tuple

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


def _resolve_target_indices(keys: List[str]) -> List[int]:
    invalid = [k for k in keys if k not in TARGET_IDX]
    if invalid:
        raise ValueError(f"Unknown targets {invalid}. Valid: {list(TARGET_IDX)}")
    return [TARGET_IDX.index(k) for k in keys]


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
    ) -> None:
        """
        Args:
            h5_path:     Path to QM9 HDF5 split file.
            target_keys: Subset of TARGET_IDX to predict.
            qfim_shape:  (n_qubits, per_qubit_dim) to slice the in-file qfim
                         rot-gate block. Required to enable QFIM features.
        """
        self.h5_path = str(h5_path)
        self.target_indices = _resolve_target_indices(target_keys)
        self.qfim_shape = qfim_shape

        with h5py.File(self.h5_path, "r") as f:
            self.n_samples = int(f["node_features"].shape[0])
            self.max_nodes = int(f["node_features"].shape[1])
            self.node_dim = int(f["node_features"].shape[2])
            self._qfim_in_h5 = "qfim" in f

        self._h5: Optional[h5py.File] = None

        if self._qfim_in_h5 and self.qfim_shape is None:
            # qfim present in the file but user did not request it -- ignore.
            self._qfim_in_h5 = False

        logger.info(
            f"QM9GraphDataset: {self.n_samples} molecules | "
            f"node_dim={self.node_dim} | max_nodes={self.max_nodes} | "
            f"qfim={'yes' if self._qfim_in_h5 else 'no'} | path={self.h5_path}"
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
        return self.n_samples

    def __getitem__(self, idx: int) -> Data:
        self._ensure_open()

        n_heavy = int(self._h5["n_atoms"][idx, 1])
        # Slice only the heavy-atom block; dense zeros beyond are padding.
        nodes = self._h5["node_features"][idx, :n_heavy]                     # (H, 9)
        edges = self._h5["edge_features"][idx, :n_heavy, :n_heavy]           # (H, H, 4)
        raw_target = np.asarray(self._h5["targets"][idx])
        if raw_target.ndim == 0:
            if len(self.target_indices) != 1:
                raise ValueError(
                    "Scalar targets found in HDF5, but multiple target keys were requested."
                )
            target = raw_target.reshape(1)
        else:
            target = raw_target[self.target_indices]

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
            n_heavy=torch.tensor([n_heavy], dtype=torch.long),
        )

        # Full rot-gate QFIM sub-block per molecule: (n_qubits, n_qubits, pd, pd).
        # Off-diagonal (i != j) sub-blocks are the parameter couplings between
        # qubits; used as per-edge quantum features by gnn_qfim.
        if self._qfim_in_h5 and self.qfim_shape is not None:
            nq, pd = self.qfim_shape
            n_rot = nq * pd
            rot_block = np.asarray(
                self._h5["qfim"][idx, :n_rot, :n_rot], dtype=np.float32
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
    max_samples: Optional[int] = None,
    seed: int = 42,
    pin_memory: bool = True,
) -> DataLoader:
    """Build a DataLoader with the map-style dataset and vectorized collate."""
    dataset: Dataset = QM9GraphDataset(
        h5_path=h5_path,
        target_keys=target_keys,
        qfim_shape=qfim_shape,
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

    qfim_cfg = getattr(config, "qfim", None)
    qfim_shape: Optional[Tuple[int, int]] = None
    if qfim_cfg is not None:
        qfim_shape = (int(qfim_cfg.n_qubits), int(qfim_cfg.per_qubit_dim))

    if config.setup.train:
        train_loader = build_loader(
            h5_path=config.paths.train,
            target_keys=target_keys,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            qfim_shape=qfim_shape,
            max_samples=getattr(config.setup, "train_n", None),
            seed=seed,
        )
        val_loader = build_loader(
            h5_path=config.paths.val,
            target_keys=target_keys,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            qfim_shape=qfim_shape,
            max_samples=getattr(config.setup, "val_n", None),
            seed=seed,
        )
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
        max_samples=getattr(config.setup, "test_n", None),
        seed=seed,
    )
    logger.info(f"test batches={len(test_loader)}")
    return test_loader
