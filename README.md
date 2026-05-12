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
│       └── probe_qfim_reshape.py      # reshape-convention diagnostic
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

**Target standardization**: the train HDF5 stores `target_mean`, `target_std`,
and `target_count` attrs. The dataloader standardizes `y` before batching.

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
Diagonal self-coupling `Q[i, i]` is not consumed by any top-level model.

1.3k – 6.5k parameters depending on head.

### `QFIMAttnGNN` — Option A: dense QFIM graph + attention (`gnn_qfim_attn.py`)

A different way to use QFIM. Instead of concatenating QFIM as an edge
feature, QFIM defines **both the graph topology and the attention
weights** that scale messages.

**Graph topology — dense, not bond-based.** The bond graph is replaced
with a fully-connected directed graph over the qubit-budgeted atoms:
every directed pair `(i, j)` with `i != j` and `i, j < qfim_nq` is an
edge. Typical QM9 molecule (9 heavy atoms): bond graph ≈ 22 edges,
dense graph = 72 edges. Atoms beyond the qubit budget have no edges.

**Edge features — RBF distance only.** Bond angles and dihedrals do not
extend naturally to non-bonded pairs, so they are dropped here. Each
edge gets a 16-dim Gaussian RBF expansion of the Euclidean distance
(centers in [0, 5] Å), projected through `Linear(16) → SiLU → LayerNorm`
to a 16-dim learned edge representation.

**Attention scores from QFIM coupling.**

```
C_ij  = ||Q[i, j]||_F                   # scalar per edge
s_ij  = beta * C_ij                     # learnable temperature beta per layer
α_ij  = softmax_dst( s_ij )             # GAT convention: incoming weights of i sum to 1
m_ij  = msg_mlp( cat(h_i, h_j, e_ij) )  # 2-layer message MLP, same as v36
h_i  := h_i + Σ_j α_ij * m_ij           # residual update
```

The Frobenius norm `C_ij` collapses each (6, 6) QFIM sub-block to one
positive scalar — "how strongly are qubits i and j coupled?". The
softmax over the destination's neighbours gives a unit-budget attention:
each node redistributes a fixed amount of attention across its
neighbours, with QFIM-strong neighbours getting more weight. Six MP
layers, each with its own learnable `beta`, so different layers can
sharpen attention differently.

**Why this works (vs concat-on-edge).** The MLP / conv heads inject QFIM
as one channel inside the edge-feature vector that the message MLP must
learn to read alongside geometric features. They land at or near the v36
baseline — QFIM as a feature is redundant with what geometry already
encodes for bonded pairs. Option A is structurally different: it gives
the model 1-hop access to non-bonded couplings (long-range pairs the
bond graph would only reach after many MP hops) and uses QFIM to *weight*
how loudly each pair speaks. On the no-bioQINN-overlap split this beats
v36 baseline by ~7% (val MAE 0.108 vs 0.116 eV at 100 epochs).

**Caveat / ablations.** Going from baseline to Option A changes two
things simultaneously:
1. graph topology (sparse bond → dense)
2. aggregation rule (sum → QFIM-weighted softmax attention)

To disambiguate which contributes:

| Variant | `model.type` | Graph | Aggregation | Tests |
|---|---|---|---|---|
| Option A | `gnn_qfim_attn` (default) | dense | softmax(β·‖Q‖) | combined effect (current best) |
| Option D | `gnn_qfim_bond_attn` | bond | softmax(β·‖Q‖) | QFIM weights only — at v36 baseline |
| Uniform | `gnn_qfim_attn` with `qfim.attn_uniform: true` | dense | uniform 1/N_i | dense graph alone, no QFIM signal |

Running all four tells the full story: Option D ≈ baseline says QFIM
weights on the bond graph don't help. The Uniform run determines whether
the dense graph alone carries Option A's lift, or whether QFIM coupling
is genuinely doing work.

**Config keys (under `qfim:`):**

```yaml
qfim:
  per_qubit_dim: 6           # pd from bioQINN: num_layers * ops_per_layer
  attn_beta_init: 1.0        # initial β; learned per layer
  edge_dim: 16               # MP-layer edge-feature dim after RBF projection
  attn_uniform: false        # true → ablation: uniform attention, no QFIM
```

~30k parameters (close to baseline). Plus 6 learnable `β` scalars (one
per MP layer).

------------------------- E ----------------

### Option E: dense graph + multiplicative softplus gate

Same `QFIMAttnGNN` architecture as Option A — same dense graph, same
RBF-distance edges, same node path — but the message-weighting mechanism
is changed from softmax-attention to a **multiplicative gate**:

```
g_ij = 1 + alpha * softplus( beta * C_ij - theta )
m_ij = msg_mlp( cat(h_i, h_j, e_ij) )
h_i := h_i + Σ_j  g_ij * m_ij           # NO softmax normalisation
```

with `alpha`, `beta`, `theta` learnable scalars per MP layer.

**Why this is different from Option A.** Softmax forces incoming weights
into each node to sum to 1: messages compete for a fixed budget, and
the *magnitude* of QFIM coupling is discarded after normalisation. The
gate keeps the magnitude — a node with 9 strongly-coupled neighbours
receives ~9× the signal of a node with 1 strongly-coupled neighbour,
which matches the physical intuition "more coupling -> stronger message".

**Properties.**
- `g_ij >= 1` always (softplus >= 0), so the gate can only amplify,
  never suppress. QFIM-uninformative edges contribute exactly the
  baseline message; QFIM-informative edges contribute extra.
- Falls back to baseline gracefully: as `alpha -> 0`, `g_ij -> 1` for
  every edge, and the model degenerates to "v36 architecture on the
  dense graph with sum aggregation."
- Bounded gradient through softplus -- stable training.

**Config (under `qfim:`):**

```yaml
qfim:
  gate_mode: "softplus_gate"
  attn_beta_init: 1.0       # learnable scale on coupling
  gate_alpha_init: 1.0      # learnable amplification factor (init 1.0)
  gate_theta_init: 0.0      # learnable bias inside softplus (init 0.0)
```

**Run:**

```bash
python -m networks.GNN.train --config configs/YAML/qm9_qfim_attn_gate.yaml
```

--------------------------------------------

------------------------- F ----------------

### Option F: dense graph + raw multiplicative weight

The simplest possible multiplicative form:

```
g_ij = beta * C_ij                       # no softplus, no offset, no clamp
m_ij = msg_mlp( cat(h_i, h_j, e_ij) )
h_i := h_i + Σ_j  g_ij * m_ij
```

Only one learnable scalar per MP layer (`beta`), no normalisation, no
saturation. Each edge's contribution scales linearly with QFIM coupling.

**Properties.**
- Most direct test of the "high coupling → strong message" intuition.
- Magnitude unbounded: `‖Q‖_F` varies across molecules, so the loss
  landscape sees per-batch scale variation. Optimisation may be less
  stable than Option E.
- No clean fall-back to baseline: if QFIM is uninformative, the model
  has to learn to drive `beta -> 0` to neutralise the gate, which
  removes *all* messages -- not the same as the baseline path.

**Config (under `qfim:`):**

```yaml
qfim:
  gate_mode: "raw"
  attn_beta_init: 1.0       # only learnable scalar; init at 1.0
```

**Run:**

```bash
python -m networks.GNN.train --config configs/YAML/qm9_qfim_attn_raw.yaml
```

--------------------------------------------

The full `gate_mode` registry on `gnn_qfim_attn`:

| `gate_mode`        | Form                                        | Sums to 1 | Bounded |
|---|---|---|---|
| `softmax`          | `softmax_dst(beta * C_ij)`                  | yes       | [0, 1]  |
| `uniform`          | `1 / in_degree(dst)` (no QFIM)              | yes       | [0, 1]  |
| `softplus_gate` (E) | `1 + alpha * softplus(beta * C_ij - theta)` | no        | [1, ∞)  |
| `raw` (F)          | `beta * C_ij`                               | no        | unbounded |

All four use the same dense QFIM-adjacency graph and the same RBF-distance
edge features.

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
standardized in the dataloader by the training split's mean/std stored in
the train HDF5 metadata. Inference converts predictions back to physical
units before reporting metrics.

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

## Legacy Configs

`configs/YAML/legacy/` holds pre-refactor configuration files for
reproducibility with existing archived runs under
`/ceph/mbinder/qm9-qinn/classical/saved_models/`. The corresponding
pre-refactor model implementations have been removed from the active source
tree; current experiments use the unified top-level modules in
`networks/GNN/`.

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
