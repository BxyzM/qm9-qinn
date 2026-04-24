"""
Standalone log-y val_loss plot: Baseline vs QFIM vs QFIM structured.

Row 2 of each epoch_stats.csv is corrupted (same artifact in all three runs);
skipped via the short-row guard so curves start at epoch 2.
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

COLORS = {
    "Baseline": "tab:orange",
    "QFIM": "tab:blue",
    "QFIM structured": "tab:green",
}


def load_val_loss(path: pathlib.Path) -> Tuple[List[int], List[float]]:
    epochs: List[int] = []
    val_loss: List[float] = []
    with path.open() as f:
        reader = csv.reader(f)
        header = next(reader)
        idx = {name: i for i, name in enumerate(header)}
        for row in reader:
            if len(row) < len(header):
                continue
            try:
                epochs.append(int(row[idx["epoch"]]))
                val_loss.append(float(row[idx["val_loss"]]))
            except ValueError:
                continue
    return epochs, val_loss


def main() -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))

    for name, path in RUNS.items():
        ep, vl = load_val_loss(path)
        ax.plot(ep, vl, label=name, color=COLORS[name], linewidth=1.6)
        print(f"{name:18s} | best_val_loss={min(vl):.6f}")

    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("val_loss (huber, log scale)")
    ax.set_title("QM9 gap regression — validation loss")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()

    plt.tight_layout()
    out = pathlib.Path(__file__).with_suffix(".png")
    plt.savefig(out, dpi=140)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
