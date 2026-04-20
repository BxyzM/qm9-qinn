"""
Minimal dataloader for QM9 HDF5 files.

Author: Dr. Aritra Bal (ETP)
Date: March 03, 2026
"""

import h5py

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, Subset
from loguru import logger
from typing import Any, Tuple, Union, List
import pennylane.numpy as pnp
try:
    import pennylane.numpy as pnp
    _PNP_AVAILABLE = True
except ImportError:
    _PNP_AVAILABLE = False
    import numpy as pnp

TARGET_IDX = [
    "mu", "alpha", "homo", "lumo", "gap", "r2", "zpve",
    "U0", "U", "H", "G", "Cv",
    "U0_atom", "U_atom", "H_atom", "G_atom",
    "rot_A", "rot_B", "rot_C",
]


class QM9HDF5Dataset(Dataset):
    """Lazy-loading dataset backed by a single QM9 HDF5 split file."""

    def __init__(
        self,
        path: str,
        target_indices: List[int],
    ) -> None:
        """
        Args:
            path           : Path to the HDF5 file.
            target_indices : Column indices into the 19-dim target array
                             to select and return.
        """
        self.path           = path
        self.target_indices = target_indices

        # Open once to read length; keep closed otherwise.
        with h5py.File(path, "r") as f:
            self.n_samples = f["node_features"].shape[0]

        logger.info(f"HDF5 dataset ready: {self.n_samples} samples from {path}")

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(
        self, idx: int
    ) -> Tuple[pnp.ndarray, pnp.ndarray, pnp.ndarray, pnp.ndarray]:
        """
        Returns:
            node_features : (max_nodes, 7)
            edge_features : (max_nodes, max_nodes)
            targets       : (n_targets,) or scalar if n_targets==1
            n_atoms       : (2,)  [total, heavy]
        """
        with h5py.File(self.path, "r") as f:
            nodes   = pnp.array(f["node_features"][idx])
            edges   = pnp.array(f["edge_features"][idx])
            targets = pnp.array(f["targets"][idx][self.target_indices])
            n_atoms = pnp.array(f["n_atoms"][idx])

        if targets.shape[0] == 1:
            targets = targets.squeeze(0)

        return nodes, edges, targets, n_atoms


def _resolve_target_indices(target_keys: List[str]) -> List[int]:
    """
    Map target key strings to column indices in the 19-dim target array.

    Args:
        target_keys: List of strings from TARGET_IDX.

    Returns:
        List of integer column indices.

    Raises:
        ValueError if any key is not found in TARGET_IDX.
    """
    invalid = [k for k in target_keys if k not in TARGET_IDX]
    if invalid:
        raise ValueError(f"Unknown target keys: {invalid}. Valid: {TARGET_IDX}")
    return [TARGET_IDX.index(k) for k in target_keys]


def _make_pnp_collate(requires_grad: bool):
    """Return a collate_fn that converts tensors to pennylane.numpy arrays."""
    def collate_fn(batch):
        nodes, edges, targets, n_atoms = torch.utils.data.default_collate(batch)
        return (
            pnp.array(nodes.numpy(),   requires_grad=requires_grad),
            pnp.array(edges.numpy(),   requires_grad=requires_grad),
            pnp.array(targets.numpy(), requires_grad=requires_grad),
            pnp.array(n_atoms.numpy(), requires_grad=False),
        )
    return collate_fn


def _build_single_loader(
    path: str,
    target_indices: List[int],
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    convert_pnp: bool,
    max_samples: int = None,
    seed: int = 42,
) -> DataLoader:
    """Construct one DataLoader from an HDF5 path."""
    dataset = QM9HDF5Dataset(path, target_indices)    
    collate_fn  = None
    if max_samples is not None and max_samples < len(dataset):
        rng     = np.random.default_rng(seed)
        indices = rng.choice(len(dataset), size=max_samples, replace=False).tolist()
        dataset = Subset(dataset, indices)
        logger.info(f"Dataset randomly subsampled to {max_samples} samples (seed={seed})")
    if convert_pnp:
        if not _PNP_AVAILABLE:
            raise ImportError("pennylane not installed; cannot use convert_pnp=True")
        if num_workers > 0:
            logger.warning("convert_pnp=True requires num_workers=0; overriding.")
            num_workers = 0
        collate_fn = _make_pnp_collate(requires_grad=False)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )


def build_loaders_from_config(
    config: Any,
) -> Union[Tuple[DataLoader, DataLoader], DataLoader]:
    """
    Build train+val or test DataLoader(s) from a config object.

    Args:
        config: Config object with dot-notation access. See module docstring
                for required fields.

    Returns:
        If config.setup.train is True : (train_loader, val_loader)
        If config.setup.train is False: test_loader
    """
    target_indices = _resolve_target_indices(config.setup.targets)
    batch_size     = config.setup.batch_size
    shuffle        = getattr(config.setup, "shuffle",      True)
    num_workers    = getattr(config.setup, "num_workers",  0)
    convert_pnp    = getattr(config.setup, "convert_pnp",  False)
    train_n = getattr(config.setup, "train_n", None)
    val_n   = getattr(config.setup, "val_n",   None)
    test_n  = getattr(config.setup, "test_n",  None)
    seed    = getattr(config.setup, "seed",    42)
    logger.info(f"Targets: {config.setup.targets} -> indices {target_indices}")

    if config.setup.train:
        train_loader = _build_single_loader(
            config.paths.train, target_indices,
            batch_size, shuffle, num_workers, convert_pnp,
            max_samples=train_n,
            seed=seed,
        )
        val_loader = _build_single_loader(
            config.paths.val, target_indices,
            batch_size, False, num_workers, convert_pnp,
            max_samples=val_n,
            seed=seed,
        )
        logger.info(
            f"Train loader: {len(train_loader)} batches | "
            f"Val loader: {len(val_loader)} batches"
        )
        return train_loader, val_loader
    else:
        test_loader = _build_single_loader(
            config.paths.test, target_indices,
            batch_size, False, num_workers, convert_pnp,
            max_samples=test_n,
            seed=seed,
        )
        logger.info(f"Test loader: {len(test_loader)} batches")
        return test_loader

if __name__ == "__main__":
    import argparse
    from types import SimpleNamespace

    parser = argparse.ArgumentParser(description="Test QM9 HDF5 dataloader")
    parser.add_argument("--train-path",   type=str, default="/ceph/mbinder/BIO/QM9/train/qm9_train.h5")
    parser.add_argument("--val-path",     type=str, default="/ceph/mbinder/BIO/QM9/val/qm9_val.h5")
    parser.add_argument("--test-path",    type=str, default="/ceph/mbinder/BIO/QM9/test/qm9_test.h5")
    parser.add_argument("--train",        action="store_true", default=False)
    parser.add_argument("--batch-size",   type=int, default=128)
    parser.add_argument("--num-workers",  type=int, default=0)
    parser.add_argument("--targets",      type=str, nargs="+", default=["gap"])
    args = parser.parse_args()

    config = SimpleNamespace(
        paths=SimpleNamespace(
            train=args.train_path,
            val=args.val_path,
            test=args.test_path,
        ),
        setup=SimpleNamespace(
            train=args.train,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            targets=args.targets,
            convert_pnp=True,
        ),
    )
    result = build_loaders_from_config(config)
    loaders = result if isinstance(result, tuple) else (result,)
    names   = ("train", "val") if config.setup.train else ("test",)
    
    for name, loader in zip(names, loaders):
        logger.info(f"--- {name} loader ---")
        all_targets, all_natoms = [], []

        for i, (nodes, edges, targets, n_atoms) in enumerate(loader):
            if i == 10:
                break
            all_targets.append(targets if targets.ndim > 0 else targets.unsqueeze(-1))
            all_natoms.append(n_atoms)
        #import pdb; pdb.set_trace()
    
        targets_cat = torch.cat([t.reshape(len(t), -1) for t in all_targets], dim=0)
        natoms_cat  = torch.cat(all_natoms, dim=0)

        logger.info(f"  batches loaded      : {i}")
        logger.info(f"  samples seen        : {targets_cat.shape[0]}")
        logger.info(f"  node shape (last)   : {nodes.shape}")
        logger.info(f"  edge shape (last)   : {edges.shape}")
        logger.info(f"  target shape (last) : {targets.shape}")
        logger.info(f"  target mean         : {targets_cat.mean(dim=0).numpy()}")
        logger.info(f"  target std          : {targets_cat.std(dim=0).numpy()}")
        logger.info(f"  target min          : {targets_cat.min(dim=0).values.numpy()}")
        logger.info(f"  target max          : {targets_cat.max(dim=0).values.numpy()}")
        logger.info(f"  total atoms  mean   : {natoms_cat[:, 0].float().mean():.2f}")
        logger.info(f"  heavy atoms  mean   : {natoms_cat[:, 1].float().mean():.2f}")