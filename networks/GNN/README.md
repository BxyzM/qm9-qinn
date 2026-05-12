# GNN models for QM9 property prediction

This folder contains the QM9 graph neural networks, all sharing a common
loader, training script, and inference script. Select between them with
`model.type` in the YAML config.

## Models

| File | `model.type` | Notes |
|---|---|---|
| [gnn.py](gnn.py) | `gnn` | Compact message-passing baseline. |
| [gnn_qfim.py](gnn_qfim.py) | `gnn_qfim` | Baseline GNN with per-edge QFIM features. |
| [gnn_qfim_attn.py](gnn_qfim_attn.py) | `gnn_qfim_attn`, `gnn_qfim_bond_attn`, `gnn_qfim_bond_gate` | QFIM attention/gating variants. |
| [gnn_qfim_residual.py](gnn_qfim_residual.py) | `gnn_qfim_residual` | Residual QFIM message-rescaling GNN. |
| [dimenet/](dimenet/) | `dimenet_pp`, `dimenet_pp_qfim` | DimeNet++ baseline and QFIM edge-state rescaling model used for the paper. |

The compact GNN variants share:
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
python -m networks.GNN.train --config configs/YAML/qm9.yaml

# QFIM-enhanced GNN
python -m networks.GNN.train --config configs/YAML/qm9_qfim.yaml
```

### DimeNet++ run005v2 config templates

These templates match the corrected DimeNet++ runs used for the paper. Change
`setup.run_id` and `setup.seed` for each repeat.

Baseline DimeNet++:

```yaml
setup:
  run_id: "dimenet_pp_heavy_pat30_delta025_run005v2_seed42"
  train: true
  batch_size: 128
  epochs: 300
  shuffle: true
  num_workers: 1
  targets: ["gap"]
  seed: 42
  train_n: null
  val_n: null
  test_n: null
  save_every_epoch: true
  early_stop_patience: 30
  early_stop_min_delta_mev: 0.25

data:
  use_all_atoms: false

model:
  type: "dimenet_pp"
  hidden_channels: 128
  out_channels: 1
  num_blocks: 4
  int_emb_size: 64
  basis_emb_size: 8
  out_emb_channels: 256
  num_spherical: 7
  num_radial: 6
  cutoff: 5.0
  max_num_neighbors: 32
  envelope_exponent: 5
  num_before_skip: 1
  num_after_skip: 2
  num_output_layers: 3
  act: "swish"
  output_initializer: "zeros"

optimizer:
  name: "adam"
  lr: 1.0e-3
  weight_decay: 0.0
  schedule: "none"

loss:
  name: "mae"

paths:
  train: "/ceph/mbinder/bioqinn/classical/data_no_bioqinn_overlap_uncompressed_2/train/qm9_train_with_qfim_no_bioqinn_overlap_2.h5"
  val:   "/ceph/mbinder/bioqinn/classical/data_no_bioqinn_overlap_uncompressed_2/val/qm9_val_with_qfim_no_bioqinn_overlap_2.h5"
  test:  "/ceph/mbinder/bioqinn/classical/data_no_bioqinn_overlap_uncompressed_2/test/qm9_test_with_qfim_no_bioqinn_overlap_2.h5"
  model_dir: "/ceph/mbinder/qm9-qinn/classical/saved_models/dimenet_pp"
```

QFIM edge-state rescaling DimeNet++:

```yaml
setup:
  run_id: "dimenet_pp_heavy_qfim_pat30_delta025_run005v2_seed42"
  train: true
  batch_size: 128
  epochs: 300
  shuffle: true
  num_workers: 1
  targets: ["gap"]
  seed: 42
  train_n: null
  val_n: null
  test_n: null
  save_every_epoch: true
  early_stop_patience: 30
  early_stop_min_delta_mev: 0.25

data:
  use_all_atoms: false

model:
  type: "dimenet_pp_qfim"
  hidden_channels: 128
  out_channels: 1
  num_blocks: 4
  int_emb_size: 64
  basis_emb_size: 8
  out_emb_channels: 256
  num_spherical: 7
  num_radial: 6
  cutoff: 5.0
  max_num_neighbors: 32
  envelope_exponent: 5
  num_before_skip: 1
  num_after_skip: 2
  num_output_layers: 3
  act: "swish"
  output_initializer: "zeros"

qfim:
  n_qubits: 10
  per_qubit_dim: 6
  embed_op: "conv2d"
  out_dim: 8
  residual_gate_init: 0.0
  rescale_beta: 1.0

optimizer:
  name: "adam"
  lr: 1.0e-3
  weight_decay: 0.0
  schedule: "none"

loss:
  name: "mae"

paths:
  train: "/ceph/mbinder/bioqinn/classical/data_no_bioqinn_overlap_uncompressed_2/train/qm9_train_with_qfim_no_bioqinn_overlap_2.h5"
  val:   "/ceph/mbinder/bioqinn/classical/data_no_bioqinn_overlap_uncompressed_2/val/qm9_val_with_qfim_no_bioqinn_overlap_2.h5"
  test:  "/ceph/mbinder/bioqinn/classical/data_no_bioqinn_overlap_uncompressed_2/test/qm9_test_with_qfim_no_bioqinn_overlap_2.h5"
  model_dir: "/ceph/mbinder/qm9-qinn/classical/saved_models/dimenet_pp_qfim"
```

Artifacts are written to `paths.model_dir/<setup.run_id>/`:
- `config.yaml` — resolved configuration.
- `last.pt` — most-recent model weights.
- `best.pt` — weights at the best validation loss.

## Inference

```bash
python -m networks.GNN.infer \
    --config  configs/YAML/qm9_dimenet_pp_heavy.yaml \
    --weights /ceph/mbinder/qm9-qinn/classical/saved_models/dimenet_pp/<run_id>/best.pt \
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
- `model.type` ∈ {`gnn`, `gnn_qfim`, `gnn_qfim_attn`, `gnn_qfim_bond_attn`,
  `gnn_qfim_bond_gate`, `gnn_qfim_residual`, `dimenet_pp`,
  `dimenet_pp_qfim`}.
- `model.*` — architecture-specific hyperparameters.
- `qfim.*` — required for QFIM-enabled models.
- `loss.name` ∈ {`huber`, `mse`, `mae`}, `loss.delta` for huber.
- `optimizer.lr`, `optimizer.decay_factor`, `optimizer.decay_patience`.

## Adding a new model

1. Drop `networks/GNN/my_model.py` exporting a `nn.Module` whose `forward`
   signature matches either the plain `(x, edge_index, edge_attr, batch)`
   or the QFIM-augmented `(x, edge_index, edge_attr, qfim, batch)`.
2. Register in [__init__.py](__init__.py).
3. Add a branch in `_build_model` in [train.py](train.py) keyed on
   `config.model.type`.
