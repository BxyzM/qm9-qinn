"""
Unified GNN inference for QM9.

Usage:
    python -m networks.GNN.infer --config configs/YAML/qm9_gnn.yaml \
        --weights /path/to/best.pt --out predictions.npz
"""

from __future__ import annotations

import argparse
import pathlib

import numpy as np
import torch
from loguru import logger

from configs.configuration import Config
from data_handlers.qm9_graph_loader import build_loaders_from_config
from networks.GNN.train import _build_model, _forward


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--out", default="predictions.npz")
    args = ap.parse_args()

    config = Config(args.config)
    # Force test-only loader path regardless of YAML.
    config.setup.train = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_loader = build_loaders_from_config(config)

    model = _build_model(config).to(device)
    state = torch.load(args.weights, map_location=device)
    model.load_state_dict(state)
    model.eval()
    logger.info(f"Loaded weights from {args.weights}")

    preds, trues = [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device, non_blocking=True)
            p = _forward(model, batch, config.model.type).view(-1).cpu().numpy()
            preds.append(p)
            trues.append(batch.y.view(-1).cpu().numpy())

    preds = np.concatenate(preds)
    trues = np.concatenate(trues)
    mae = float(np.mean(np.abs(preds - trues)))
    rmse = float(np.sqrt(np.mean((preds - trues) ** 2)))
    logger.info(f"test MAE={mae:.4f} | RMSE={rmse:.4f} | N={len(preds)}")

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, pred=preds, true=trues, mae=mae, rmse=rmse)
    logger.info(f"wrote {out}")


if __name__ == "__main__":
    main()
