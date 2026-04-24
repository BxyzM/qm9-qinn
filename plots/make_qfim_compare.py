"""
Compare BaselineGNN / QFIMGNN / QFIMGNNStructured over 100 epochs.

Row 2 in each epoch_stats.csv is corrupted by a race between the initial
header write and the first-epoch row append (same defect in every run).
We skip it: the 100-epoch curves start at epoch 2.
"""

from __future__ import annotations

import csv
import pathlib
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt


RUNS: Dict[str, pathlib.Path] = {
    "Baseline": pathlib.Path(
        "/ceph/mbinder/qm9-qinn/classical/saved_models/gnn_invariant/"
        "gnn_inv_huber/epoch_stats.csv"
    ),
    "QFIM": pathlib.Path(
        "/ceph/mbinder/qm9-qinn/classical/saved_models/gnn_qfim/"
        "gnn_qfim_huber/epoch_stats.csv"
    ),
    "QFIM structured": pathlib.Path(
        "/ceph/mbinder/qm9-qinn/classical/saved_models/gnn_qfim_structured/"
        "gnn_qfim_structured/epoch_stats.csv"
    ),
}


def load_curve(path: pathlib.Path) -> Tuple[List[int], List[float], List[float], List[float]]:
    epochs: List[int] = []
    val_loss: List[float] = []
    val_mae: List[float] = []
    seconds: List[float] = []
    with path.open() as f:
        reader = csv.reader(f)
        header = next(reader)
        idx = {name: i for i, name in enumerate(header)}
        for row in reader:
            # Skip the corrupted row 2 (only 2 cols) and any other short row.
            if len(row) < len(header):
                continue
            try:
                ep = int(row[idx["epoch"]])
                vl = float(row[idx["val_loss"]])
                vm = float(row[idx["val_mae"]])
                s = float(row[idx["seconds"]])
            except ValueError:
                continue
            epochs.append(ep)
            val_loss.append(vl)
            val_mae.append(vm)
            seconds.append(s)
    return epochs, val_loss, val_mae, seconds


def main() -> None:
    curves = {name: load_curve(p) for name, p in RUNS.items()}
    for name, (ep, vl, vm, s) in curves.items():
        print(
            f"{name:20s} | epochs={len(ep)} "
            f"| best_val_loss={min(vl):.4f} "
            f"| final_mae={vm[-1]:.4f} "
            f"| total_seconds={sum(s):.0f}"
        )

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    fig.suptitle(
        "QM9 gap regression: Baseline vs QFIM vs QFIM structured",
        fontsize=12,
    )

    colors = {"Baseline": "tab:orange", "QFIM": "tab:blue", "QFIM structured": "tab:green"}

    # Panel 1: val_loss (log y)
    ax = axes[0]
    for name, (ep, vl, _, _) in curves.items():
        ax.plot(ep, vl, label=name, color=colors[name], linewidth=1.4)
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("val_loss (huber, log)")
    ax.set_title("Validation loss")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend()

    # Panel 2: val_mae (log y)
    ax = axes[1]
    for name, (ep, _, vm, _) in curves.items():
        ax.plot(ep, vm, label=name, color=colors[name], linewidth=1.4)
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("val_mae (eV, log)")
    ax.set_title("Validation MAE")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend()

    # Panel 3: seconds per epoch
    ax = axes[2]
    for name, (ep, _, _, s) in curves.items():
        ax.plot(ep, s, label=name, color=colors[name], linewidth=1.0, alpha=0.85)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("seconds")
    ax.set_title("Seconds per epoch")
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    out = pathlib.Path(__file__).with_suffix(".png")
    plt.savefig(out, dpi=140)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
