"""
Repair the `targets` dataset in the bioqinn `_with_qfim.h5` files.

Background
----------
The `_with_qfim.h5` splits under /ceph/mbinder/bioqinn/classical/data/ were
built by copying the abal originals and merging QFIM features in. During that
merge the 19-D `targets` array was reduced to 1-D (column 0 = dipole moment).
That silently made every downstream trainer fit mu instead of whatever column
the YAML config asked for, because the map-style loader's
`raw_target[self.target_indices]` is a no-op when `raw_target` is a scalar.

Row order in the bioqinn files matches the abal originals byte-for-byte
(verified via n_atoms and node_features). We therefore replace `targets` in
place with the full (N, 19) array from the abal source.

As a second pass we audit the QFIM dataset for rows that are all zeros (the
QFIM inference log mentioned skipping ~5k indices, and we want to know if any
rows ended up with uninitialized values).

Usage
-----
    python -m data_processors.repair_targets_and_audit_qfim \
        --splits train val test \
        [--dry-run]

A .bak copy of each file is created next to it before any write. If the .bak
already exists the script refuses to overwrite it (re-running this script on
files that already have a .bak would lose the pre-repair version).
"""

from __future__ import annotations

import argparse
import pathlib
import shutil
import sys
from typing import Dict, Tuple

import h5py
import numpy as np

TARGET_IDX = (
    "mu", "alpha", "homo", "lumo", "gap", "r2", "zpve",
    "U0", "U", "H", "G", "Cv",
    "U0_atom", "U_atom", "H_atom", "G_atom",
    "rot_A", "rot_B", "rot_C",
)

TARGET_INFO = (
    "mu: dipole moment (Debye)",
    "alpha: isotropic polarizability (a0^3)",
    "homo: HOMO energy (eV)",
    "lumo: LUMO energy (eV)",
    "gap: HOMO-LUMO gap (eV)",
    "r2: electronic spatial extent (a0^2)",
    "zpve: zero-point vibrational energy (eV)",
    "U0: internal energy at 0K (eV)",
    "U: internal energy at 298.15K (eV)",
    "H: enthalpy at 298.15K (eV)",
    "G: free energy at 298.15K (eV)",
    "Cv: heat capacity at 298.15K (cal/mol/K)",
    "U0_atom: atomisation energy at 0K (eV)",
    "U_atom: atomisation energy at 298.15K (eV)",
    "H_atom: atomisation enthalpy at 298.15K (eV)",
    "G_atom: atomisation free energy at 298.15K (eV)",
    "A: rotational constant A (GHz)",
    "B: rotational constant B (GHz)",
    "C: rotational constant C (GHz)",
)

SPLITS: Dict[str, Tuple[pathlib.Path, pathlib.Path]] = {
    "train": (
        pathlib.Path("/ceph/abal/BIO/QM9/heavymol_5-9/train/qm9_train.h5"),
        pathlib.Path("/ceph/mbinder/bioqinn/classical/data/train/qm9_train_with_qfim.h5"),
    ),
    "val": (
        pathlib.Path("/ceph/abal/BIO/QM9/heavymol_5-9/val/qm9_val.h5"),
        pathlib.Path("/ceph/mbinder/bioqinn/classical/data/val/qm9_val_with_qfim.h5"),
    ),
    "test": (
        pathlib.Path("/ceph/abal/BIO/QM9/heavymol_5-9/test/qm9_test.h5"),
        pathlib.Path("/ceph/mbinder/bioqinn/classical/data/test/qm9_test_with_qfim.h5"),
    ),
}


def _verify_alignment(src: pathlib.Path, dst: pathlib.Path) -> int:
    """Confirm row order matches. Returns N (sample count)."""
    with h5py.File(src, "r") as fs, h5py.File(dst, "r") as fd:
        ns = int(fs["node_features"].shape[0])
        nd = int(fd["node_features"].shape[0])
        if ns != nd:
            raise RuntimeError(f"sample count mismatch: src={ns} dst={nd}")
        # Cheap first-10 byte-equality check on node_features and n_atoms.
        if not np.array_equal(
            np.asarray(fs["node_features"][:10]),
            np.asarray(fd["node_features"][:10]),
        ):
            raise RuntimeError("node_features[:10] mismatch -- row order differs")
        if not np.array_equal(
            np.asarray(fs["n_atoms"]),
            np.asarray(fd["n_atoms"]),
        ):
            raise RuntimeError("n_atoms mismatch -- row order differs")
    return ns


def _audit_qfim(dst: pathlib.Path) -> None:
    """Scan QFIM for all-zero rows or NaN/Inf."""
    with h5py.File(dst, "r") as f:
        if "qfim" not in f:
            print(f"  qfim dataset absent -- skipping audit")
            return
        qfim = f["qfim"]
        N = qfim.shape[0]
        chunk = 1024
        n_zero = 0
        n_nan = 0
        n_inf = 0
        zero_indices: list[int] = []
        for start in range(0, N, chunk):
            end = min(start + chunk, N)
            block = np.asarray(qfim[start:end])
            flat = block.reshape(block.shape[0], -1)
            zero_mask = np.all(flat == 0.0, axis=1)
            nan_mask = np.any(np.isnan(flat), axis=1)
            inf_mask = np.any(np.isinf(flat), axis=1)
            n_zero += int(zero_mask.sum())
            n_nan += int(nan_mask.sum())
            n_inf += int(inf_mask.sum())
            if zero_mask.any():
                for off in np.where(zero_mask)[0][: max(0, 20 - len(zero_indices))]:
                    zero_indices.append(start + int(off))
        print(
            f"  qfim audit: N={N}  all-zero={n_zero}  nan_rows={n_nan}  inf_rows={n_inf}"
        )
        if zero_indices:
            preview = zero_indices[:20]
            print(f"    first zero-row indices (up to 20): {preview}")


def _repair_split(split: str, dry_run: bool) -> None:
    src, dst = SPLITS[split]
    print(f"\n=== split={split} ===")
    print(f"  src = {src}")
    print(f"  dst = {dst}")

    if not src.exists():
        raise FileNotFoundError(src)
    if not dst.exists():
        raise FileNotFoundError(dst)

    N = _verify_alignment(src, dst)
    print(f"  row order verified, N={N}")

    with h5py.File(src, "r") as fs:
        src_targets = np.asarray(fs["targets"], dtype=np.float32)
    if src_targets.shape != (N, 19):
        raise RuntimeError(
            f"unexpected src targets shape {src_targets.shape}, want ({N}, 19)"
        )
    print(
        f"  src targets: shape={src_targets.shape}  "
        f"gap[col4] mean={src_targets[:,4].mean():.4f} std={src_targets[:,4].std():.4f}"
    )

    with h5py.File(dst, "r") as fd:
        dst_targets = np.asarray(fd["targets"])
    print(f"  dst targets (current): shape={dst_targets.shape}")

    _audit_qfim(dst)

    if dry_run:
        print("  DRY RUN -- skipping write")
        return

    bak = dst.with_suffix(dst.suffix + ".bak")
    if bak.exists():
        raise RuntimeError(
            f"backup already exists: {bak}. Refusing to overwrite it. "
            f"Delete or rename the .bak manually if you really want to re-run."
        )
    print(f"  copying {dst} -> {bak}  (this takes a bit; ~file size)")
    shutil.copy2(dst, bak)

    print(f"  rewriting targets in {dst}")
    with h5py.File(dst, "a") as fd:
        if "targets" in fd:
            del fd["targets"]
        fd.create_dataset("targets", data=src_targets)
        for name, payload in (
            ("targetIndex", np.array(TARGET_IDX, dtype=h5py.string_dtype())),
            ("targetInfo",  np.array(TARGET_INFO, dtype=h5py.string_dtype())),
        ):
            if name in fd:
                del fd[name]
            fd.create_dataset(name, data=payload)
    print("  done.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", choices=list(SPLITS), default=list(SPLITS))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    for split in args.splits:
        _repair_split(split, dry_run=args.dry_run)
    print("\nall requested splits processed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
