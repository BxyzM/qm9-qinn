# qm9-gnn — Classical GNNs for QM9 Property Prediction

Classical graph neural networks for molecular property regression on the [QM9](https://www.nature.com/articles/sdata201422) dataset. Split out from the bioQINN project; this repository contains only the classical (and QFIM-informed classical) components.

Primary regression target: **HOMO-LUMO gap** (Δε between HOMO and LUMO).

---

## Repository Structure

```
qm9-gnn/
├── configs/
│   ├── YAML/
│   │   ├── qm9.yaml            # Example run config
│   │   ├── qm9_gnn.yaml        # GNN / QFIM-GNN run config
│   │   └── qm9_run005.yaml
│   ├── configuration.py        # Config loader: defaults + YAML overrides
│   └── defaults.py             # Default parameter values
├── data_handlers/
│   ├── qm9_graph_loader.py     # PyG graph-based loader (supports optional QFIM features)
│   ├── qm9_h5_dataloader.py    # HDF5-backed dense loader
│   └── file_paths.py
├── data_processors/
│   └── h5_maker_qm9.py         # One-time HDF5 dataset creation
└── networks/
    ├── GNN/                    # GNN, InvariantGNN, QFIMGNN (QFIM-informed classical GNN)
    ├── GAT/
    └── MultiplexGNN/
```

---

## Models

- **GNN** — standard message-passing GNN baseline.
- **InvariantGNN** — rotation/permutation-invariant variant.
- **QFIMGNN** — classical GNN augmented with per-edge Quantum Fisher Information Matrix features precomputed from a variational quantum circuit. The QFIM features are read from the HDF5 input; no quantum computation is performed at train time.
- **GAT**, **MultiplexGNN** — additional classical architectures.

---

## Dependencies

See `requirements.txt`.

---

## Usage

### Step 1: Prepare the Dataset (run once)

```bash
python3 data_processors/h5_maker_qm9.py
```

Writes train/val/test HDF5 splits with node features, edge features, targets, and atom counts.

### Step 2: Configure

Defaults live in `configs/defaults.py`; override via a YAML file under `configs/YAML/`.

### Step 3: Train

```bash
python3 -m networks.GNN.train --config configs/YAML/qm9_gnn.yaml
```

Select model type via `config.model.type` ∈ {`gnn`, `gnn_invariant`, `gnn_qfim`}.

---

## Reference

Embedding strategy derived from:

```bibtex
@article{bal2025,
  title = {One particle - one qubit: Particle physics data encoding for quantum machine learning},
  author = {Bal, Aritra and Klute, Markus and Maier, Benedikt and Oughton, Melik and Pezone, Eric and Spannowsky, Michael},
  journal = {Phys. Rev. D},
  volume = {112},
  issue = {7},
  pages = {076004},
  year = {2025},
  doi = {10.1103/l8y2-87vq},
  url = {https://link.aps.org/doi/10.1103/l8y2-87vq}
}
```
