"""
Unified GNN training entry for QM9.

Selects model via config.model.type in {"gnn", "gnn_invariant", "gnn_qfim"}.
Uses the map-style vectorized loader in data_handlers.qm9_graph_loader.

Usage:
    python -m networks.GNN.train --config configs/YAML/qm9_gnn.yaml
"""

from __future__ import annotations

import argparse
import csv
import pathlib
import time
from typing import Tuple

import torch
import torch.nn as nn
from loguru import logger
from tqdm import tqdm

from configs.configuration import Config
from data_handlers.qm9_graph_loader import build_loaders_from_config
from networks.GNN import GNN, InvariantGNN, QFIMGNN, QFIMGNNStructured, QFIMGNNConv


def _build_model(config) -> nn.Module:
    mt = config.model.type
    node_dim = int(getattr(config.model, "node_dim", 9))
    hidden = int(getattr(config.model, "hidden_dim", 64))
    layers = int(getattr(config.model, "num_layers", 6))
    include_dihedral = bool(getattr(config.model, "include_dihedral", True))
    pooling = getattr(config.model, "pooling", None)

    if mt == "gnn":
        return GNN(
            node_dim=node_dim,
            edge_dim=4,
            hidden_dim=hidden,
            num_layers=layers,
        )
    if mt == "gnn_invariant":
        kwargs = dict(
            node_dim=node_dim,
            hidden_dim=hidden,
            num_layers=layers,
            include_dihedral=include_dihedral,
        )
        if pooling is not None:
            kwargs["pooling"] = pooling
        return InvariantGNN(**kwargs)
    if mt == "gnn_qfim":
        pd = int(config.qfim.per_qubit_dim)
        return QFIMGNN(
            node_dim=node_dim,
            qfim_per_qubit_dim=pd,
            hidden_dim=hidden,
            num_layers=layers,
            include_dihedral=include_dihedral,
            pooling=pooling or "mean",
        )
    if mt == "gnn_qfim_structured":
        pd = int(config.qfim.per_qubit_dim)
        qnl = int(getattr(config.qfim, "num_layers", 2))
        qops = int(getattr(config.qfim, "ops_per_layer", 3))
        return QFIMGNNStructured(
            node_dim=node_dim,
            qfim_per_qubit_dim=pd,
            qfim_num_layers=qnl,
            qfim_ops_per_layer=qops,
            hidden_dim=hidden,
            num_layers=layers,
            include_dihedral=include_dihedral,
            pooling=pooling or "mean",
        )
    if mt == "gnn_qfim_conv":
        pd = int(config.qfim.per_qubit_dim)
        return QFIMGNNConv(
            node_dim=node_dim,
            qfim_per_qubit_dim=pd,
            qfim_conv_channels=int(getattr(config.qfim, "conv_channels", 16)),
            qfim_kernel_size=int(getattr(config.qfim, "kernel_size", 3)),
            qfim_out_dim=int(getattr(config.qfim, "out_dim", 8)),
            hidden_dim=hidden,
            num_layers=layers,
            include_dihedral=include_dihedral,
            pooling=pooling or "mean",
        )
    raise ValueError(f"Unknown model.type={mt!r}")


def _forward(model: nn.Module, batch, model_type: str) -> torch.Tensor:
    if model_type in ("gnn_qfim", "gnn_qfim_structured", "gnn_qfim_conv"):
        nq = int(batch.qfim_nq[0].item())
        return model(
            batch.x, batch.edge_index, batch.edge_attr,
            batch.qfim_block, nq, batch.batch,
        )
    return model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)


def _run_epoch(
    model: nn.Module,
    loader,
    loss_fn,
    optimizer,
    device: torch.device,
    model_type: str,
    train: bool,
    desc: str = "",
) -> Tuple[float, float]:
    model.train(train)
    total_loss = 0.0
    total_mae = 0.0
    n = 0
    pbar = tqdm(loader, desc=desc, leave=False, dynamic_ncols=True)
    for batch in pbar:
        batch = batch.to(device, non_blocking=True)
        y = batch.y.view(-1)
        with torch.set_grad_enabled(train):
            pred = _forward(model, batch, model_type).view(-1)
            loss = loss_fn(pred, y)
            mae = (pred.detach() - y).abs().mean()
        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        bs = y.numel()
        total_loss += float(loss.item()) * bs
        total_mae += float(mae.item()) * bs
        n += bs
        pbar.set_postfix(loss=f"{total_loss/max(n,1):.4f}", mae=f"{total_mae/max(n,1):.4f}")
    return total_loss / max(n, 1), total_mae / max(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    config = Config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"device={device} | building loaders...")

    train_loader, val_loader = build_loaders_from_config(config)
    logger.info(
        f"loaders ready | train_batches={len(train_loader)} "
        f"val_batches={len(val_loader)} batch_size={config.setup.batch_size} "
        f"num_workers={getattr(config.setup, 'num_workers', 0)}"
    )
    model = _build_model(config).to(device)
    logger.info(
        f"Model {config.model.type} | params={sum(p.numel() for p in model.parameters()):,}"
    )

    # Fit target standardization stats on the training split only. For
    # InvariantGNN these live as buffers that travel with state_dict so
    # checkpoints / infer runs don't need a separate stats file. For other
    # models fit_target_stats is a no-op.
    if hasattr(model, "fit_target_stats"):
        mean, std = model.fit_target_stats(train_loader)
        logger.info("=" * 72)
        logger.info(f"target normalization | mean={mean:.6f} | std={std:.6f}")
        logger.info("=" * 72)

    loss_name = getattr(config.loss, "name", "huber")
    if loss_name == "mse":
        loss_fn = nn.MSELoss()
    elif loss_name == "mae":
        loss_fn = nn.L1Loss()
    else:
        loss_fn = nn.HuberLoss(delta=float(getattr(config.loss, "delta", 0.1)))

    lr = float(config.optimizer.lr)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt,
        factor=float(getattr(config.optimizer, "decay_factor", 0.5)),
        patience=int(getattr(config.optimizer, "decay_patience", 5)),
        threshold=float(getattr(config.optimizer, "decay_threshold", 1e-3)),
    )

    model_dir = pathlib.Path(config.paths.model_dir) / config.setup.run_id
    model_dir.mkdir(parents=True, exist_ok=True)
    config.save(str(model_dir / "config.yaml"))

    stats_path = model_dir / "epoch_stats.csv"
    with stats_path.open("w", newline="") as stats_file:
        writer = csv.writer(stats_file)
        writer.writerow([
            "epoch",
            "train_loss",
            "train_mae",
            "val_loss",
            "val_mae",
            "lr",
            "seconds",
            "best_val",
        ])

        best_val = float("inf")
        best_val_mae = float("inf")
        epochs_since_mae_improved = 0
        # ------------------------ Early stopping --------------------------
        #
        # Early stopping on val_mae. Patience = N epochs with no improvement
        # over the best-so-far val_mae before we stop. Disable by setting
        # setup.early_stop_patience to 0 (or leaving it absent with <=0).
        #
        # ------------------------------------------------------------------
        early_stop_patience = int(getattr(config.setup, "early_stop_patience", 10))
        epochs = int(config.setup.epochs)
        save_every_epoch = bool(getattr(config.setup, "save_every_epoch", False))
        for epoch in range(1, epochs + 1):
            t0 = time.time()
            train_loss, train_mae = _run_epoch(
                model, train_loader, loss_fn, opt, device, config.model.type,
                train=True, desc=f"train {epoch:03d}",
            )
            val_loss, val_mae = _run_epoch(
                model, val_loader, loss_fn, opt, device, config.model.type,
                train=False, desc=f"val   {epoch:03d}",
            )
            scheduler.step(val_loss)
            dt = time.time() - t0
            current_lr = opt.param_groups[0]["lr"]

            logger.info(
                f"epoch {epoch:03d}/{epochs} | "
                f"train_loss={train_loss:.4f} mae={train_mae:.4f} | "
                f"val_loss={val_loss:.4f} mae={val_mae:.4f} | "
                f"lr={current_lr:.2e} | {dt:.1f}s"
            )
            torch.save(model.state_dict(), model_dir / "last.pt")
            if save_every_epoch:
                epoch_dir = model_dir / "epochs"
                epoch_dir.mkdir(exist_ok=True)
                torch.save(model.state_dict(), epoch_dir / f"epoch_{epoch:03d}.pt")
            if val_loss < best_val:
                best_val = val_loss
                torch.save(model.state_dict(), model_dir / "best.pt")
                logger.info(f"  new best val_loss={val_loss:.4f}")

            with stats_path.open("a", newline="") as stats_file:
                writer = csv.writer(stats_file)
                writer.writerow([
                    epoch,
                    f"{train_loss:.8f}",
                    f"{train_mae:.8f}",
                    f"{val_loss:.8f}",
                    f"{val_mae:.8f}",
                    f"{current_lr:.8e}",
                    f"{dt:.4f}",
                    f"{best_val:.8f}",
                ])

            # ---------------------------------------------------------------
            # Early stopping
            # ---------------------------------------------------------------
            # Track the best val_mae seen so far. If it improves this epoch,
            # reset the stale-counter; otherwise increment it. When the
            # counter reaches `early_stop_patience`, stop training — further
            # epochs are unlikely to improve val_mae. Checkpoints stay as
            # they were (best.pt still tracks val_loss); this only controls
            # the training loop termination.
            # ---------------------------------------------------------------
            if val_mae < best_val_mae:
                best_val_mae = val_mae
                epochs_since_mae_improved = 0
            else:
                epochs_since_mae_improved += 1

            if (
                early_stop_patience > 0
                and epochs_since_mae_improved >= early_stop_patience
            ):
                logger.info(
                    f"early stopping at epoch {epoch} | "
                    f"val_mae did not improve for {early_stop_patience} epochs | "
                    f"best_val_mae={best_val_mae:.4f}"
                )
                break

    logger.info(
        f"done | best_val={best_val:.4f} | best_val_mae={best_val_mae:.4f} "
        f"| artifacts in {model_dir}"
    )


if __name__ == "__main__":
    main()
