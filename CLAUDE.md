# CLAUDE.md — bioQINN working memory

This file is the canonical working agreement for Claude in this repo. Read
top-to-bottom on every session. Treat it as overriding general defaults.

---

## Master prompt: how we work here

**Role.** You are the lead engineer on a scientific codebase for quantum +
classical neural networks on QM9 molecular property prediction. The code is
Python + PyTorch + PyTorch Geometric + PennyLane (for the quantum circuit).
There is no frontend, no web stack, no TypeScript — ignore any generic
"frontend/backend/shared" advice.

**Operating principles.**

1. **Read before writing.** Before generating code, inspect the relevant
   file(s) so the change fits the existing style, shapes, and conventions.
   Never guess a function signature or tensor shape you have not verified.
2. **Keep it small.** This repo was recently cleaned up from 25+ stray
   markdown files and ~10 duplicate scripts. Do not reintroduce clutter.
   Prefer editing an existing file to creating a new one. Do not create
   tutorial `.md` files in the root.
3. **No docstrings explaining obvious code.** One short line where it adds
   value (a non-obvious shape transform, a physics convention, a
   performance trick). Never multi-paragraph.
4. **Vectorize first.** CPU hot paths in this repo have repeatedly been
   killed by Python `for` loops over atoms/edges. New code that touches
   per-edge or per-atom data must use tensor ops (`torch.nonzero`,
   scatter, index_select) — not Python loops.
5. **Fair ablations.** When adding a new model variant, match
   `hidden_dim`, `num_layers`, and the message-MLP width of the baseline
   so parameter counts are comparable. The existing baseline sits at
   ~21k params (hidden=32, 6 layers).
6. **Do not guess file paths.** The QM9 HDF5 splits live at
   `/ceph/mbinder/bioqinn/classical/data/{train,val,test}/qm9_*_with_qfim.h5`.
   Verify paths by running `ls` before relying on them.
7. **Destructive actions require confirmation.** Do not `rm -rf`, delete
   git branches, force-push, or drop HDF5 datasets without an explicit go.
8. **When in doubt, ask.** Surface conflicts (architecture vs. request,
   shape mismatch, missing dependency) before writing code.

**What Dr. Aritra Bal's files do.** `train.py` and `test.py` at the repo
root drive the PennyLane variational circuit training/inference. They
depend on [data_handlers/qm9_h5_dataloader.py](data_handlers/qm9_h5_dataloader.py)
with `convert_pnp=True` to return `pennylane.numpy` arrays. Do not touch
these unless asked; the classical GNN stack is the active work area.

---

## Repo layout (current)

```
bioQINN/
├── train.py                        # Dr. Bal — PennyLane quantum circuit training
├── test.py                         # Dr. Bal — PennyLane inference
├── README.md                       # project overview
├── requirements.txt
├── configs/
│   ├── configuration.py            # YAML loader -> dot-notation Config
│   ├── defaults.py
│   └── YAML/
│       ├── qm9.yaml                # quantum-circuit config
│       ├── qm9_run005.yaml
│       └── qm9_gnn.yaml            # classical GNN stack config
├── data/                           # local QM9 cache (small)
├── data_handlers/
│   ├── file_paths.py
│   ├── qm9_h5_dataloader.py        # pennylane path (quantum circuit only)
│   └── qm9_graph_loader.py         # GNN path (map-style, torch_geometric)
├── data_processors/
│   └── h5_maker_qm9.py             # builds the QM9 HDF5 splits
├── networks/
│   ├── DNN/
│   ├── GAT/  MultiplexGNN/  QuantumGAT/   # legacy stubs, inactive
│   └── GNN/
│       ├── __init__.py             # exports GNN, InvariantGNN, QFIMGNN
│       ├── gnn.py                  # plain MP baseline
│       ├── gnn_invariant.py        # invariant MP (3-bond + 4-bond)
│       ├── gnn_qfim.py             # invariant MP + per-edge QFIM
│       ├── train.py                # unified training entry
│       ├── infer.py                # unified inference entry
│       └── README.md               # per-model notes
├── quantum/
│   ├── architectures.py            # pennylane circuit + QFIM computation
│   ├── qfim_parallel.py
│   └── trainer.py
├── plotters/
└── src/
```

---

## How to run things

All GNN work runs as modules from the repo root so imports resolve. Activate
the `ParT` conda env first (`conda activate ParT`).

### Train a classical GNN

```bash
cd /work/mbinder/src/bioQINN
python -m networks.GNN.train --config configs/YAML/qm9_gnn.yaml
```

Switch models by editing `model.type` in [qm9_gnn.yaml](configs/YAML/qm9_gnn.yaml):

- `"gnn"` — plain baseline (no invariance).
- `"gnn_invariant"` — rotation/translation-invariant GNN.
- `"gnn_qfim"` — invariant GNN + per-edge QFIM features.

Artifacts land in `paths.model_dir/<setup.run_id>/`:
- `config.yaml` — resolved config (defaults merged with user YAML).
- `last.pt` — most recent weights.
- `best.pt` — best val-loss weights.

### Inference

```bash
python -m networks.GNN.infer \
    --config  configs/YAML/qm9_gnn.yaml \
    --weights /ceph/mbinder/bioqinn/classical/saved_models/gnn/gnn_001/best.pt \
    --out     predictions.npz
```

Writes `pred`, `true`, `mae`, `rmse` to the npz.

### Quantum circuit (Dr. Bal's pipeline)

```bash
python train.py        # uses configs/YAML/qm9.yaml
python test.py
```

---

## Current state (as of 2026-04-17)

- **Cleanup done**: deleted 25 loose root-level .md files, 10+ duplicate
  qfim/train scripts, three redundant dataloaders, old scratch result dirs.
  Keep it lean.
- **Classical GNN stack is live**: one loader, three models, one
  train/infer entry. See [networks/GNN/README.md](networks/GNN/README.md).
- **QFIM is available in-file**: the HDF5 splits at
  `/ceph/mbinder/bioqinn/classical/data/` already contain a
  `(N, 101, 101)` `qfim` dataset per split. First 60×60 = rot-gate block
  (10 qubits × 2 layers × 3 ops). No separate preprocessing step is
  required for `gnn_qfim`.
- **Known quirks**:
  - `torch_scatter` is intentionally NOT installed. The code uses
    `torch_geometric.utils.scatter` instead (pure PyTorch).
  - First epoch is slow because 7 GB + 1.5 GB files need to warm the OS
    page cache; subsequent epochs are fast. Set `num_workers >= 4` in
    the YAML to parallelize the cold read.

---

## Networks

All three models live in [networks/GNN/](networks/GNN/) and share:
- node features: 9D `[atomic_number, aromatic, hybridisation, n_H, x, y, z,
  n_atoms_total, n_heavy]`
- 6 message-passing layers, hidden dim 32, global max-pool, 32→32→1
  readout MLP
- forward signature: `(x, edge_index, edge_attr, batch)` for `gnn` /
  `gnn_invariant`; `(x, edge_index, edge_attr, qfim_block, qfim_nq, batch)`
  for `gnn_qfim`

### [gnn.py](networks/GNN/gnn.py) — plain baseline

- Message: `Linear(2·hidden + edge → hidden) + LayerNorm + ReLU` per layer.
- Residual add: `h ← h + propagate(...)`.
- **Uses raw loader edge features** `[bond_type, theta, phi, distance]`.
  theta/phi are lab-frame, so this model is NOT rotation-invariant. Serves
  as a lower-bound baseline.

### [gnn_invariant.py](networks/GNN/gnn_invariant.py) — invariant MP

- Drops theta/phi. Edge features become `[bond_type, distance, bond_angle,
  (dihedral)]` — all rotation/translation invariant.
- `compute_bond_angles` and `compute_dihedral_angles` are fully vectorized
  via `torch_geometric.utils.scatter`. No `.item()` calls, no Python loops
  — runs on whichever device the input sits on.
- `include_dihedral: true` enables 4-bond condition (dihedrals). `false`
  keeps only the 3-bond condition.

### [gnn_qfim.py](networks/GNN/gnn_qfim.py) — QFIM via edges only

- Nodes: same as `gnn_invariant` (no QFIM on nodes, no `qfim_embed`, no
  `fuse`).
- Edges: for each bonded pair `(atom_i, atom_j)` fetch the `(pd, pd)`
  sub-block of the QFIM that couples qubit-i's and qubit-j's rotation
  parameters, flatten to `pd*pd` = 36, and concatenate to the invariant
  geometric edge features.
- Consequence: the QFIM enters every message-passing layer (as edge
  features are re-consumed per layer), not just once at input.
- Atoms with local index ≥ `n_qubits` get zero blocks.

---

## Data loader

[data_handlers/qm9_graph_loader.py](data_handlers/qm9_graph_loader.py) —
map-style `torch.utils.data.Dataset` returning `torch_geometric.data.Data`
objects. Batching via `torch_geometric.data.Batch.from_data_list`.

### HDF5 schema (expected)

```
node_features : (N, MAX_NODES=36, 9)          float32
edge_features : (N, MAX_NODES, MAX_NODES, 4)  float32
    channels = [bond_type, theta, phi, distance]
n_atoms       : (N, 2)                         int32   [total, heavy]
targets       : (N,) or (N, 19)                float32
qfim          : (N, 101, 101)                  float64  (optional)
    first 60x60 block = rot-gate parameters
    (n_qubits=10) x (num_layers=2) x (ops_per_layer=3) = 60
```

The loader auto-detects whether `targets` is 1-D (single target, e.g. gap
only) or 2-D (19-target vector) and indexes accordingly.

### Design choices

1. **Per-worker HDF5 handle.** Each DataLoader worker opens its own
   `h5py.File` once in `_ensure_open()` and reuses it for the lifetime
   of the worker. **Never reopen per-sample** — that pathological
   pattern is what pinned CPU at 5000% in the old code.
2. **Lazy slice reads.** `__getitem__` only slices the heavy-atom block
   `[:n_heavy, :n_heavy]`, not the full `(36, 36)` dense pad.
3. **Vectorized edge sparsification.** Per sample:
   ```python
   bond_mask = edges[..., 0] > 0            # (H, H)
   edge_idx  = bond_mask.nonzero().t()      # (2, E)
   edge_attr = edges[bond_mask]             # (E, 4)
   ```
   One `nonzero` call per sample. No Python loops over atom pairs.
4. **Worker init**: `_worker_init_fn` resets the cached `_h5` /
   `_qfim` handles on each worker fork. Required because child
   processes can't inherit an open HDF5 handle safely.
5. **QFIM loading**: auto-detected from the `qfim` dataset in the same
   HDF5. The loader reads the `(n_rot, n_rot)` rot-gate block per
   molecule, reshapes to `(n_qubits, n_qubits, pd, pd)`, and stores it
   with a leading singleton so PyG's concat-along-dim-0 batching gives
   `(B, n_qubits, n_qubits, pd, pd)` in the batched Data.
6. **External override**: `qfim.{train,val,test}_path` in the YAML
   points to an optional standalone `.npy` memmap of shape
   `(N, n_rot)`. Used if you pre-extracted the rot-gate diagonals.
   Otherwise, the in-file QFIM wins.

### Config keys consumed

```yaml
setup:
  batch_size: 256
  num_workers: 4          # >= 4 recommended for cold-cache speed
  shuffle: true
  targets: ["gap"]        # list of keys from TARGET_IDX
  train_n / val_n / test_n: optional subsample caps
  seed: 42

qfim:                     # only needed for gnn_qfim
  n_qubits: 10
  per_qubit_dim: 6        # = num_layers * operations_per_layer

paths:
  train: /ceph/mbinder/bioqinn/classical/data/train/qm9_train_with_qfim.h5
  val:   /ceph/mbinder/bioqinn/classical/data/val/qm9_val_with_qfim.h5
  test:  /ceph/mbinder/bioqinn/classical/data/test/qm9_test_with_qfim.h5
```

### Common pitfalls

- **num_workers=0**: fine for debugging, but serializes all HDF5 reads.
  Use ≥ 4 for real training.
- **Setting `convert_pnp: true`**: only for the quantum-circuit pipeline
  (`qm9_h5_dataloader.py`). The GNN loader doesn't support it and
  doesn't need it.
- **Mismatched `qfim.n_qubits` / `per_qubit_dim`**: must satisfy
  `n_qubits * per_qubit_dim == 60` for the in-file layout. Wrong values
  raise `ValueError` at the first sample.

---

## When adding a new model or loader feature

1. Follow the existing forward signatures — either pure graph
   `(x, edge_index, edge_attr, batch)` or the QFIM-augmented form.
2. Register in [networks/GNN/__init__.py](networks/GNN/__init__.py).
3. Add a branch in [`_build_model`](networks/GNN/train.py) keyed on
   `config.model.type`.
4. Update [networks/GNN/README.md](networks/GNN/README.md) with one
   table row describing the variant.
5. Do not create a parallel `train_<variant>.py` — extend the shared
   entry instead.
