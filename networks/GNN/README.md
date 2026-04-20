# GNN models for QM9 property prediction

This folder contains three graph neural networks, all sharing a common
loader, training script, and inference script. Select between them with
`model.type` in the YAML config.

## Models

| File | `model.type` | Notes |
|---|---|---|
| [gnn.py](gnn.py) | `gnn` | Plain message-passing baseline. Uses raw edge features `[bond_type, theta, phi, distance]`. **Not rotation invariant.** Lower-bound baseline. |
| [gnn_invariant.py](gnn_invariant.py) | `gnn_invariant` | Rotation- and translation-invariant. Replaces lab-frame theta/phi with vectorized bond angles (3-bond) and optional dihedral angles (4-bond), computed on-device via `torch_scatter`. |
| [gnn_qfim.py](gnn_qfim.py) | `gnn_qfim` | Invariant geometric edges + **per-edge QFIM coupling features**: the (pd, pd) off-diagonal rot-gate sub-block linking qubit-i's and qubit-j's rotation parameters is flattened and concatenated to each edge feature. Pure EdgeConv; QFIM never enters node features. |

All three share:
- Node features: 9D (atomic number, aromatic, hybridisation, n_H, x, y, z, n_atoms, n_heavy).
- 6 message-passing layers, hidden dim 64 (configurable).
- Global max pooling + 2-layer MLP readout.

## Data loader

[data_handlers/qm9_graph_loader.py](../../data_handlers/qm9_graph_loader.py) —
map-style `torch.utils.data.Dataset` returning `torch_geometric.data.Data`
objects.

Key properties:
- **Per-worker HDF5 handles**, opened lazily once, reused across samples. No
  reopen-per-item pathology.
- **Vectorized edge sparsification** inside `__getitem__` (`torch.nonzero` on
  the bond-type matrix). No Python loops over atom pairs.
- **`num_workers >= 2` + `persistent_workers=True` + `pin_memory=True`**.
- Optional per-molecule QFIM rot-gate block loaded from a `.npy` memmap.
- Target subset selected by name from the 19-dim QM9 target vector.

## Training

```bash
# plain GNN baseline
python -m networks.GNN.train --config configs/YAML/qm9_gnn.yaml  # set model.type: gnn

# invariant GNN (3-bond + 4-bond)
python -m networks.GNN.train --config configs/YAML/qm9_gnn.yaml  # model.type: gnn_invariant

# QFIM-enhanced GNN (requires a qfim.*_path and qfim.n_qubits / qfim.per_qubit_dim)
python -m networks.GNN.train --config configs/YAML/qm9_gnn.yaml  # model.type: gnn_qfim
```

Artifacts are written to `paths.model_dir/<setup.run_id>/`:
- `config.yaml` — resolved configuration.
- `last.pt` — most-recent model weights.
- `best.pt` — weights at the best validation loss.

## Inference

```bash
python -m networks.GNN.infer \
    --config  configs/YAML/qm9_gnn.yaml \
    --weights /work/mbinder/bioqinn/saved_models/gnn/gnn_inv_001/best.pt \
    --out     predictions.npz
```

Writes `pred`, `true`, `mae`, `rmse` to the `.npz` file.

## QFIM feature extraction (for `model.type: gnn_qfim`)

The QFIM is computed by the PennyLane variational circuit in
[quantum/architectures.py](../../quantum/architectures.py) via
`QuantumCircuit.quantum_fisher(node_feat, edge_feat)`. For the GNN we use
only the **rot-gate diagonal block** — the QFIM diagonal entries for the
trainable `RX/RY/RZ` parameters applied at each qubit in each layer.

### In-file layout (current)

The HDF5 splits at
`/ceph/mbinder/bioqinn/classical/data/{train,val,test}/qm9_*_with_qfim.h5`
already contain a `qfim` dataset of shape `(N, 101, 101)` float64 per split,
where the first `60 × 60` block holds the rot-gate parameters (10 qubits ×
2 layers × 3 ops), and the remaining 41 rows/cols hold the `extra_weights`.

The loader auto-detects the `qfim` dataset, reads the full
`(n_rot × n_rot)` rot-gate block per molecule, and reshapes it to
`(n_qubits, n_qubits, per_qubit_dim, per_qubit_dim)`. For each bonded
edge `(atom_i, atom_j)` the GNN fetches the `(per_qubit_dim,
per_qubit_dim)` sub-block coupling qubit-i's and qubit-j's rotation
parameters, flattens it to `per_qubit_dim**2` and appends it to the
edge attribute. Atoms beyond the qubit budget get zero blocks.

### Config YAML

```yaml
qfim:
  n_qubits: 10                # matches model.n_qubits of the quantum circuit
  per_qubit_dim: 6            # = num_layers * operations_per_layer
```

No separate `.npy` files are needed when the HDF5 contains `qfim`. To
override with an external memmap, set `qfim.train_path / val_path /
test_path` to `.npy` files of shape `(N, n_qubits * per_qubit_dim)`.

## Config keys

- `setup.batch_size`, `setup.num_workers`, `setup.epochs`, `setup.targets` (list).
- `setup.train_n / val_n / test_n` — optional subsample caps.
- `model.type` ∈ {`gnn`, `gnn_invariant`, `gnn_qfim`}.
- `model.hidden_dim`, `model.num_layers`, `model.include_dihedral`.
- `qfim.*` — only required for `gnn_qfim`.
- `loss.name` ∈ {`huber`, `mse`, `mae`}, `loss.delta` for huber.
- `optimizer.lr`, `optimizer.decay_factor`, `optimizer.decay_patience`.

## Adding a new model

1. Drop `networks/GNN/my_model.py` exporting a `nn.Module` whose `forward`
   signature matches either the plain `(x, edge_index, edge_attr, batch)`
   or the QFIM-augmented `(x, edge_index, edge_attr, qfim, batch)`.
2. Register in [__init__.py](__init__.py).
3. Add a branch in `_build_model` in [train.py](train.py) keyed on
   `config.model.type`.
