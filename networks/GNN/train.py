"""
Unified GNN training entry for QM9.

Selects model via config.model.type:
    "gnn"      -> baseline GNN (geometry only, no QFIM)
    "gnn_qfim" -> baseline + QFIM edge head (qfim.embed_op picks the head)

Uses the map-style vectorized loader in data_handlers.qm9_graph_loader.

Usage:
    python -m networks.GNN.train --config configs/YAML/qm9.yaml
    python -m networks.GNN.train --config configs/YAML/qm9_qfim.yaml
"""

from __future__ import annotations

import argparse
import csv
import math
import pathlib
import time
from typing import Tuple

import torch
import torch.nn as nn
from loguru import logger
from tqdm import tqdm

from configs.configuration import Config
from data_handlers.qm9_graph_loader import build_loaders_from_config
from networks.GNN import (
    DimeNetPP,
    DimeNetPPQFIM,
    GNN,
    QFIMGNN,
    QFIMAttnGNN,
    QFIMBondAttnGNN,
    QFIMBondGateGNN,
    QFIMResidualGNN,
)


def _build_model(config) -> nn.Module:
    mt = config.model.type
    num_mp_layers = int(getattr(config.model, "num_layers", 6))
    pooling = getattr(config.model, "pooling", None) or "mean"
    activation = str(getattr(config.model, "activation", "relu"))
    mlp_residual = bool(getattr(config.model, "mlp_residual", False))
    msg_layers = int(getattr(config.model, "msg_layers", 1))
    per_layer_edge_update = bool(getattr(config.model, "per_layer_edge_update", False))
    node_mlp_dims = tuple(getattr(config.model, "node_mlp_dims", (19, 32, 64, 64, 32)))
    edge_mlp_dims = tuple(getattr(config.model, "edge_mlp_dims", (28, 32, 32, 16, 8)))
    max_neighbors = int(getattr(config.model, "max_neighbors", 3))
    max_chains = int(getattr(config.model, "max_chains", 9))

    if mt == "gnn":
        return GNN(
            num_mp_layers=num_mp_layers,
            node_mlp_dims=node_mlp_dims,
            edge_mlp_dims=edge_mlp_dims,
            max_neighbors=max_neighbors,
            max_chains=max_chains,
            pooling=pooling,
            activation=activation,
            mlp_residual=mlp_residual,
            msg_layers=msg_layers,
            per_layer_edge_update=per_layer_edge_update,
        )
    if mt == "dimenet_pp":
        return DimeNetPP(
            hidden_channels=int(getattr(config.model, "hidden_channels", 128)),
            out_channels=int(getattr(config.model, "out_channels", 1)),
            num_blocks=int(getattr(config.model, "num_blocks", 4)),
            int_emb_size=int(getattr(config.model, "int_emb_size", 64)),
            basis_emb_size=int(getattr(config.model, "basis_emb_size", 8)),
            out_emb_channels=int(getattr(config.model, "out_emb_channels", 256)),
            num_spherical=int(getattr(config.model, "num_spherical", 7)),
            num_radial=int(getattr(config.model, "num_radial", 6)),
            cutoff=float(getattr(config.model, "cutoff", 5.0)),
            max_num_neighbors=int(getattr(config.model, "max_num_neighbors", 32)),
            envelope_exponent=int(getattr(config.model, "envelope_exponent", 5)),
            num_before_skip=int(getattr(config.model, "num_before_skip", 1)),
            num_after_skip=int(getattr(config.model, "num_after_skip", 2)),
            num_output_layers=int(getattr(config.model, "num_output_layers", 3)),
            act=str(getattr(config.model, "act", "swish")),
            output_initializer=str(getattr(config.model, "output_initializer", "zeros")),
        )
    if mt == "dimenet_pp_qfim":
        return DimeNetPPQFIM(
            hidden_channels=int(getattr(config.model, "hidden_channels", 128)),
            out_channels=int(getattr(config.model, "out_channels", 1)),
            num_blocks=int(getattr(config.model, "num_blocks", 4)),
            int_emb_size=int(getattr(config.model, "int_emb_size", 64)),
            basis_emb_size=int(getattr(config.model, "basis_emb_size", 8)),
            out_emb_channels=int(getattr(config.model, "out_emb_channels", 256)),
            num_spherical=int(getattr(config.model, "num_spherical", 7)),
            num_radial=int(getattr(config.model, "num_radial", 6)),
            cutoff=float(getattr(config.model, "cutoff", 5.0)),
            max_num_neighbors=int(getattr(config.model, "max_num_neighbors", 32)),
            envelope_exponent=int(getattr(config.model, "envelope_exponent", 5)),
            num_before_skip=int(getattr(config.model, "num_before_skip", 1)),
            num_after_skip=int(getattr(config.model, "num_after_skip", 2)),
            num_output_layers=int(getattr(config.model, "num_output_layers", 3)),
            act=str(getattr(config.model, "act", "swish")),
            output_initializer=str(getattr(config.model, "output_initializer", "zeros")),
            qfim_per_qubit_dim=int(config.qfim.per_qubit_dim),
            qfim_embed_op=str(getattr(config.qfim, "embed_op", "conv2d")),
            qfim_out_dim=int(getattr(config.qfim, "out_dim", 8)),
            qfim_head_normalize=bool(getattr(config.qfim, "head_normalize", False)),
            qfim_residual_gate_init=float(getattr(config.qfim, "residual_gate_init", 0.0)),
            qfim_rescale_beta=float(getattr(config.qfim, "rescale_beta", 1.0)),
        )
    if mt == "gnn_qfim":
        pd = int(config.qfim.per_qubit_dim)
        embed_op = str(getattr(config.qfim, "embed_op", "mlp"))
        out_dim = int(getattr(config.qfim, "out_dim", 4))
        head_normalize = bool(getattr(config.qfim, "head_normalize", False))
        return QFIMGNN(
            num_mp_layers=num_mp_layers,
            node_mlp_dims=node_mlp_dims,
            edge_mlp_dims=edge_mlp_dims,
            pooling=pooling,
            activation=activation,
            mlp_residual=mlp_residual,
            msg_layers=msg_layers,
            per_layer_edge_update=per_layer_edge_update,
            qfim_per_qubit_dim=pd,
            qfim_embed_op=embed_op,
            qfim_out_dim=out_dim,
            qfim_head_normalize=head_normalize,
        )
    if mt == "gnn_qfim_residual":
        pd = int(config.qfim.per_qubit_dim)
        embed_op = str(getattr(config.qfim, "embed_op", "conv2d"))
        out_dim = int(getattr(config.qfim, "out_dim", 8))
        head_normalize = bool(getattr(config.qfim, "head_normalize", False))
        residual_gate_init = float(getattr(config.qfim, "residual_gate_init", 0.0))
        full_conv_kernel = int(getattr(config.qfim, "full_conv_kernel", 7))
        full_conv_channels = int(getattr(config.qfim, "full_conv_channels", 16))
        alpha_mode = str(getattr(config.qfim, "alpha_mode", "shared"))
        edge_gate = bool(getattr(config.qfim, "edge_gate", False))
        use_geom = bool(getattr(config.qfim, "use_geom", False))
        qfim_mode = str(getattr(config.qfim, "mode", "additive"))
        qfim_msg_layers = getattr(config.qfim, "msg_layers", None)
        if qfim_msg_layers is not None:
            qfim_msg_layers = int(qfim_msg_layers)
        qfim_branch_dropout = float(getattr(config.qfim, "branch_dropout", 0.0))
        qfim_rescale_beta = float(getattr(config.qfim, "rescale_beta", 1.0))
        return QFIMResidualGNN(
            num_mp_layers=num_mp_layers,
            node_mlp_dims=node_mlp_dims,
            edge_mlp_dims=edge_mlp_dims,
            max_neighbors=max_neighbors,
            max_chains=max_chains,
            pooling=pooling,
            activation=activation,
            mlp_residual=mlp_residual,
            msg_layers=msg_layers,
            per_layer_edge_update=per_layer_edge_update,
            qfim_n_qubits=int(config.qfim.n_qubits),
            qfim_per_qubit_dim=pd,
            qfim_embed_op=embed_op,
            qfim_out_dim=out_dim,
            qfim_head_normalize=head_normalize,
            qfim_residual_gate_init=residual_gate_init,
            qfim_full_conv_kernel=full_conv_kernel,
            qfim_full_conv_channels=full_conv_channels,
            qfim_alpha_mode=alpha_mode,
            qfim_edge_gate=edge_gate,
            qfim_use_geom=use_geom,
            qfim_mode=qfim_mode,
            qfim_msg_layers=qfim_msg_layers,
            qfim_branch_dropout=qfim_branch_dropout,
            qfim_rescale_beta=qfim_rescale_beta,
        )
    if mt == "gnn_qfim_attn":
        pd = int(config.qfim.per_qubit_dim)
        beta_init = float(getattr(config.qfim, "attn_beta_init", 1.0))
        edge_dim = int(getattr(config.qfim, "edge_dim", 16))
        attn_uniform = bool(getattr(config.qfim, "attn_uniform", False))
        gate_mode = str(getattr(config.qfim, "gate_mode", "softmax"))
        gate_alpha_init = float(getattr(config.qfim, "gate_alpha_init", 1.0))
        gate_theta_init = float(getattr(config.qfim, "gate_theta_init", 0.0))
        return QFIMAttnGNN(
            num_mp_layers=num_mp_layers,
            node_mlp_dims=node_mlp_dims,
            edge_mlp_dims=edge_mlp_dims,
            pooling=pooling,
            activation=activation,
            mlp_residual=mlp_residual,
            msg_layers=msg_layers,
            per_layer_edge_update=per_layer_edge_update,
            qfim_per_qubit_dim=pd,
            qfim_attn_beta_init=beta_init,
            qfim_edge_dim=edge_dim,
            attn_uniform=attn_uniform,
            gate_mode=gate_mode,
            gate_alpha_init=gate_alpha_init,
            gate_theta_init=gate_theta_init,
        )
    if mt == "gnn_qfim_bond_attn":
        pd = int(config.qfim.per_qubit_dim)
        beta_init = float(getattr(config.qfim, "attn_beta_init", 1.0))
        return QFIMBondAttnGNN(
            num_mp_layers=num_mp_layers,
            node_mlp_dims=node_mlp_dims,
            edge_mlp_dims=edge_mlp_dims,
            pooling=pooling,
            activation=activation,
            mlp_residual=mlp_residual,
            msg_layers=msg_layers,
            per_layer_edge_update=per_layer_edge_update,
            qfim_per_qubit_dim=pd,
            qfim_attn_beta_init=beta_init,
        )
    if mt == "gnn_qfim_bond_gate":
        pd = int(config.qfim.per_qubit_dim)
        beta_init = float(getattr(config.qfim, "gate_beta_init", 1.0))
        alpha_init = float(getattr(config.qfim, "gate_alpha_init", 0.0))
        theta_init = float(getattr(config.qfim, "gate_theta_init", 0.0))
        return QFIMBondGateGNN(
            num_mp_layers=num_mp_layers,
            node_mlp_dims=node_mlp_dims,
            edge_mlp_dims=edge_mlp_dims,
            pooling=pooling,
            activation=activation,
            mlp_residual=mlp_residual,
            msg_layers=msg_layers,
            per_layer_edge_update=per_layer_edge_update,
            qfim_per_qubit_dim=pd,
            qfim_gate_beta_init=beta_init,
            qfim_gate_alpha_init=alpha_init,
            qfim_gate_theta_init=theta_init,
        )
    raise ValueError(f"Unknown model.type={mt!r}")


def _forward(model: nn.Module, batch, model_type: str) -> torch.Tensor:
    if model_type == "dimenet_pp":
        return model(batch.x, batch.batch)
    if model_type == "dimenet_pp_qfim":
        nq = int(batch.qfim_nq[0].item())
        return model(batch.x, batch.qfim_block, nq, batch.batch)
    if model_type in (
        "gnn_qfim",
        "gnn_qfim_attn",
        "gnn_qfim_bond_attn",
        "gnn_qfim_bond_gate",
        "gnn_qfim_residual",
    ):
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

    logger.info(
        "targets standardized in dataloader | "
        f"mean={train_loader.target_mean} | std={train_loader.target_std} "
        f"| count={train_loader.target_stats_count}"
    )

    loss_name = getattr(config.loss, "name", "huber")
    if loss_name == "mse":
        loss_fn = nn.MSELoss()
    elif loss_name == "mae":
        loss_fn = nn.L1Loss()
    else:
        loss_fn = nn.HuberLoss(delta=float(getattr(config.loss, "delta", 0.1)))

    lr = float(config.optimizer.lr)
    weight_decay = float(getattr(config.optimizer, "weight_decay", 0.0))
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    schedule_kind = str(getattr(config.optimizer, "schedule", "plateau")).lower()
    if schedule_kind == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt,
            factor=float(getattr(config.optimizer, "decay_factor", 0.5)),
            patience=int(getattr(config.optimizer, "decay_patience", 5)),
            threshold=float(getattr(config.optimizer, "decay_threshold", 1e-3)),
        )
    elif schedule_kind == "cosine":
        warmup_epochs = int(getattr(config.optimizer, "warmup_epochs", 5))
        total_epochs = int(config.setup.epochs)
        min_lr = float(getattr(config.optimizer, "min_lr", 1e-5))
        min_ratio = min_lr / lr
        cos_epochs = max(1, total_epochs - warmup_epochs)

        def _lr_lambda(epoch_idx: int) -> float:
            if epoch_idx < warmup_epochs:
                return (epoch_idx + 1) / max(1, warmup_epochs)
            t = (epoch_idx - warmup_epochs) / cos_epochs
            t = min(max(t, 0.0), 1.0)
            cos = 0.5 * (1.0 + math.cos(math.pi * t))
            return min_ratio + (1.0 - min_ratio) * cos

        scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=_lr_lambda)
    elif schedule_kind in ("none", "constant"):
        scheduler = None
    else:
        raise ValueError(
            "optimizer.schedule must be 'plateau', 'cosine', or 'none'; "
            f"got {schedule_kind!r}"
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
        epochs = int(config.setup.epochs)
        # ------------------------ Early stopping --------------------------
        #
        # Stop on val_mae if it does not improve by at least
        # setup.early_stop_min_delta_mev. Patience is capped at 10% of the
        # allowed epochs. Disable by setting setup.early_stop_patience <= 0.
        #
        # ------------------------------------------------------------------
        configured_patience = int(getattr(config.setup, "early_stop_patience", 10))
        max_patience = max(1, math.ceil(0.10 * epochs))
        early_stop_patience = (
            min(configured_patience, max_patience)
            if configured_patience > 0
            else 0
        )
        min_delta_mev = float(getattr(config.setup, "early_stop_min_delta_mev", 0.0))
        if min_delta_mev < 0.0:
            raise ValueError("setup.early_stop_min_delta_mev must be >= 0")
        if min_delta_mev > 15.0:
            logger.warning(
                f"early_stop_min_delta_mev={min_delta_mev:.3f} exceeds 15 meV; "
                "capping to 15 meV"
            )
            min_delta_mev = 15.0
        target_std0 = float(torch.as_tensor(train_loader.target_std).view(-1)[0].item())
        early_stop_min_delta = min_delta_mev / (target_std0 * 1000.0)
        logger.info(
            "early stopping | "
            f"patience={early_stop_patience} "
            f"(configured={configured_patience}, max_10pct={max_patience}) | "
            f"min_delta={min_delta_mev:.3f} meV "
            f"({early_stop_min_delta:.6f} normalized)"
        )
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
            if schedule_kind == "plateau":
                scheduler.step(val_loss)
            elif scheduler is not None:
                scheduler.step()
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
            if val_mae < best_val_mae - early_stop_min_delta:
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
                    f"val_mae did not improve by at least {min_delta_mev:.3f} meV "
                    f"for {early_stop_patience} epochs | best_val_mae={best_val_mae:.4f}"
                )
                break

    logger.info(
        f"done | best_val={best_val:.4f} | best_val_mae={best_val_mae:.4f} "
        f"| artifacts in {model_dir}"
    )


if __name__ == "__main__":
    main()
