# qm9-qinn

Classical graph neural networks for HOMO-LUMO gap regression on QM9, with an
optional Quantum Fisher Information Matrix (QFIM) edge feature derived from
a variational quantum circuit trained upstream in the ..


The goal is to measure whether quantum information, extracted as the QFIM of
a trained variational ansatz, improves a purely classical message-passing
GNN over a geometry-only baseline.

---

## Quick start

```bash
# Baseline (geometry only)
python -m networks.GNN.train --config configs/YAML/qm9.yaml

# QFIM-informed, MLP head (default). Change qfim.embed_op for other heads.
python -m networks.GNN.train --config configs/YAML/qm9_qfim.yaml
```

Checkpoints, stats, and configs are saved under `paths.model_dir/<run_id>/`
on `/ceph`. Early stopping fires on `val_mae` with patience 10 by default.

---

## Repo layout

```
qm9-qinn/
├── configs/
│   ├── configuration.py           # YAML + dataclass loader
│   ├── defaults.py
│   └── YAML/
│       ├── qm9.yaml               # baseline config
│       ├── qm9_qfim.yaml          # QFIM config (edit qfim.embed_op to swap heads)
│       └── legacy/                # pre-refactor configs, kept for reproducibility
├── data_handlers/
│   └── qm9_graph_loader.py        # map-style PyG loader, vectorized QFIM gather
├── data_processors/
│   ├── h5_maker_qm9.py            # one-time HDF5 build from raw QM9
│   └── repair_targets_and_audit_qfim.py
├── networks/
│   └── GNN/
│       ├── gnn.py                 # baseline GNN (geometry only)
│       ├── gnn_qfim.py            # baseline + QFIM edge head (4 swappable heads)
│       ├── train.py               # unified training entry
│       ├── probe_qfim_permutation.py  # alignment diagnostic
│       ├── probe_qfim_reshape.py      # reshape-convention diagnostic
│       └── legacy/                    # pre-refactor models (reference only, not dispatched)
└── plots/
    └── make_qfim_compare*.py      # comparison plots over saved_models/
```

---

## Models

### `GNN` (baseline, `gnn.py`)

SOTA-flavored compact GNN. Only atomic number is embedded on nodes; bond
distance is encoded with a Gaussian RBF basis; hybridization, aromatic
flag, n_H, and bond type are dropped (left to be learned implicitly by
the MP stack from geometry).

<!-- ---------- changed (v3): chemistry annotations dropped, only Z embedded ---------- -->
**Node features** (9 dims):
- `Z` (atomic number) → `nn.Embedding(10, 6)`
- `x, y, z` — passed through directly.
- Concatenated to 9 dims.

Hybridization, aromatic_flag, and n_H are no longer consumed. SOTA
chemistry GNNs (SchNet, DimeNet, PaiNN) use only Z and rely on the MP
stack to learn local chemistry from geometry.
<!-- ---------- /changed ---------- -->

**Edge features** (3 dims output, built from 28 raw dims):
- `vec3_bond` (3): up to 3 bond angles `∠(k - i - j)` at source atom i,
  padded with zeros. Neighbors are emitted in raw HDF5-loader order, which
  is sorted by descending atomic number (`h5_maker_qm9.py` permutation), so
  position 0 is the heaviest bonded neighbor.
- `vec4_dihedral` (9): up to 9 unsigned dihedrals `dih(k - i - j - l)`,
  filtered against `l = i` and `l = k` (3-ring degeneracy).
<!-- ---------- changed (v3): scalar distance -> Gaussian RBF expansion ---------- -->
- `rbf_distance` (16): Gaussian RBF expansion of the bond distance:
  `exp(-γ (d - μ_k)²)` for 16 evenly-spaced centers in [0, 5] Å with
  γ = 4. Replaces the v2 scalar distance. The localized basis lets the
  edge MLP learn distance-regime-specific features (single bond, double
  bond, 1,3 contact) rather than decoding meaning from a single scalar.
<!-- ---------- /changed ---------- -->
<!-- ---------- changed (v3): bond_type embedding removed ---------- -->
- Bond type is no longer used. The edge MLP output is fed to MP as-is.
<!-- ---------- /changed ---------- -->

<!-- ---------- changed (v3): node MLP shrunk, edge MLP widened for RBF input ---------- -->
**MLP shapes**:
- Node MLP: `9 → 16 → 32 → 16 → 8`
- Edge MLP: `28 → 16 → 16 → 8 → 3`
<!-- ---------- /changed ---------- -->

**Message passing**: 6 layers of `msg_mlp([x_i, x_j, edge_attr]) → 8`,
sum-aggregated, residual. Operates in 8-dim node space.

**Readout**: pooled (mean / max / add, configurable via `model.pooling`)
→ `Linear(8, 32) → ReLU → Linear(32, 1)`. Mean is correct for intensive
targets like the gap.

**Target standardization**: `fit_target_stats(train_loader)` must run once
before training. Stats live in state_dict buffers.

<!-- ---------- changed (v3): param count smaller despite RBF (chemistry features dropped) ---------- -->
~3.7k parameters total (v1: 1.2k, v2: 7.5k, v3: 3.7k).
<!-- ---------- /changed ---------- -->

**v2 → v3 changes summary:**
- **Node features**: dropped hybridization, aromatic_flag, n_H. Only Z
  embedded; xyz still passed. Node input: 21 dims → 9 dims.
- **Edge features**: scalar distance replaced with 16-dim Gaussian RBF
  expansion. Edge raw input: 13 dims → 28 dims.
- **Bond type**: dropped entirely. The `bond_embed` multiplicative gate
  is removed.
- **MLP shapes**: node MLP shrunk to match smaller node input; edge MLP
  widened to match larger edge input.
- **Pooling**: mean (intensive target, default).

### `QFIMGNN` (`gnn_qfim.py`)

Extends `GNN` with a 4-dim per-edge QFIM summary concatenated onto the
geometric edge feature (7-dim edges in MP, alongside the 8-dim node
space). The QFIM head is swappable via config:

| `qfim.embed_op` | Mechanism |
|---|---|
| `mlp`     | Autoencoder MLP `36 → 64 → 32 → 16 → 8 → 4` on the flattened 6×6 QFIM sub-block |
| `conv1d`  | Conv1d over one qubit-param axis, symmetrized by averaging over both axis assignments, pooled to 4 |
| `conv2d`  | Conv2d on the (6, 6) block, pooled to 4 |
| `gated`   | `C_ij = ‖Q[i,j]‖_F` scalar, tiny MLP projecting to 4 dims |

Only off-diagonal QFIM sub-blocks are used (bond edges never have i = j).
Diagonal self-coupling `Q[i, i]` is not consumed by any top-level model;
a legacy `QFIMGNNNode` variant that adds it as a node feature lives under
`networks/GNN/legacy/` for reference.

1.3k – 6.5k parameters depending on head.

---

## QFIM alignment

The bioQINN circuit stores rot-gate weights as shape
`(num_layers, ops_per_layer, n_qubits) = (2, 3, 10)` and PennyLane's
`metric_tensor` flattens them in C order, so **qubit is the fastest-varying
axis** along the 60-dim flat parameter vector. The loader reshape must
therefore be:

```python
rot_block.reshape(pd, nq, pd, nq).transpose(1, 3, 0, 2)
```

A qubit-major interpretation (the initial implementation) silently
misaligned every QFIM run before 2026-04-22. See
`networks/GNN/probe_qfim_reshape.py` for the empirical verification
against a re-computed QFIM.

---

## Target and data

Primary regression target is **HOMO-LUMO gap** (eV). Targets are
standardized by the training split's mean/std; buffers live in the model
and travel with `state_dict`, so val/test and checkpoint reloads always
denormalize consistently.

HDF5 input schema (written by `data_processors/h5_maker_qm9.py` and
extended with QFIM by the bioQINN pipeline):

| dataset | shape | notes |
|---|---|---|
| `node_features` | `(N, 36, 9)` | atom features, sorted by descending Z |
| `edge_features` | `(N, 36, 36, 4)` | `[bond_type, theta, phi, distance]` |
| `targets` | `(N, 19)` | PyG QM9 target layout; column 4 = gap (eV) |
| `n_atoms` | `(N, 2)` | `[total, heavy]` |
| `qfim` | `(N, 101, 101)` | rot-gate block = first 60×60 sub-block |

Train/val/test splits live at
`/ceph/mbinder/bioqinn/classical/data/{train,val,test}/qm9_*_with_qfim.h5`.

---

## Diagnostics

- `networks/GNN/probe_qfim_reshape.py` — recomputes one QFIM via bioQINN
  and verifies the loader's reshape matches ground truth. Run once when
  changing anything in the loader's QFIM path.
- `networks/GNN/probe_qfim_permutation.py` — permutes qubit axes per
  molecule at training time. If a trained QFIM model's val_loss is
  indistinguishable between real and permuted QFIM, the model is ignoring
  the QFIM content.

---

## Legacy

`networks/GNN/legacy/` and `configs/YAML/legacy/` hold pre-refactor code
for reproducibility with the existing runs under
`/ceph/mbinder/qm9-qinn/classical/saved_models/`:

- `gnn_plain.py` — non-invariant MPNN baseline
- `gnn_invariant.py` — invariant GNN with mean-reduced angle/dihedral
  features and 128-dim hidden space
- `gnn_qfim.py`, `gnn_qfim_structured.py`, `gnn_qfim_conv.py`,
  `gnn_qfim_node.py`, `gnn_qfim_cij.py` — five QFIM injection variants
  explored before unification

Legacy modules are not wired into `train.py`. To run one, import from
`networks.GNN.legacy.<module>` and dispatch manually.

---

## References

Embedding strategy derived from:

```bibtex
@article{bal2025,
  title = {One particle - one qubit: Particle physics data encoding for quantum machine learning},
  author = {Bal, Aritra and Klute, Markus and Maier, Benedikt and Oughton, Melik and Pezone, Eric and Spannowsky, Michael},
  journal = {Phys. Rev. D},
  volume = {112}, issue = {7}, pages = {076004}, year = {2025},
  doi = {10.1103/l8y2-87vq},
}
```
