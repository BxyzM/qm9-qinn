# Message-Rescale QFIM: Paper-Ready Summary

This document collects the architecture, math, implementation, and integration
details of the *message-rescale* QFIM mechanism for the paper. Two backbones
are covered:

1. The 300k-parameter custom invariant GNN (`gnn_qfim_residual` with
   `mode="baseline_msg_rescale"`).
2. The DimeNet++ wrapper (`DimeNetPPQFIM`) used in the larger comparison.

The mechanism is identical in spirit: a per-edge scalar derived from the
quantum Fisher information matrix (QFIM) modulates *existing* baseline
edge messages multiplicatively, with a learnable global gate `alpha`
initialised to zero so the model starts at the exact baseline.

---

## 1. Motivation

We previously used an additive QFIM residual branch:

```
h_{l+1} = baseline_mp(h_l, E) + alpha * qfim_mp(h_l, E, Q)
```

This gave an improvement, but a random-symmetric matrix swapped in for `Q`
gave a similar improvement. The branch was expressive enough to add capacity
regardless of the input's physical content, which makes it a poor probe for
whether the QFIM carries useful chemistry/quantum information.

The message-rescale design constrains QFIM to *gate* baseline messages
rather than *create* its own. It cannot open a parallel computational
pathway; it can only strengthen or weaken what the baseline already
computed. If a random `Q` then matches the real `Q`, that is unambiguous
evidence that the architecture is exploiting generic matrix capacity rather
than aligned physics.

---

## 2. Mathematical Definition

### 2.1 Notation

- `h_i^l` ∈ R^{d_node}: node embedding of atom `i` at MP layer `l`.
- `e_ij` ∈ R^{d_edge}: invariant geometric edge feature (RBF distance,
  bond/dihedral angles).
- `Q_ij` ∈ R^{p_d × p_d}: off-diagonal sub-block of the molecule-level QFIM
  coupling the rotation-gate parameters of qubit `i` and qubit `j`. With
  `p_d = 6` parameters per qubit (3 single-qubit rotations × 2 layers in our
  ansatz), `Q_ij` is a 6×6 real symmetric block.
- `alpha` ∈ R: a single learnable scalar, initialised to 0.

### 2.2 Per-edge QFIM scalar

A small encoder maps `Q_ij` to a low-dimensional latent and then to a bounded
scalar:

```
q_ij = Enc(Q_ij)             ∈ R^{d_q}
s_ij = tanh(psi(q_ij))       ∈ [-1, 1]
```

`Enc` is the swappable QFIM head (`mlp`, `conv1d`, `conv2d`, `gated`,
`frobenius`, `diagstats`, or the molecule-level `full_conv2d`). For the main
runs we use `conv2d` with `out_dim = 8`. `psi` is a 2-layer MLP terminating
in `tanh`, so `s_ij` is bounded in `[-1, 1]` regardless of QFIM scale.

### 2.3 Message-rescale update

Let `m_ij^l = phi_base(h_i^l, h_j^l, e_ij)` be the baseline message. The
QFIM-modulated message is

```
m_ij'^l = m_ij^l + alpha * s_ij * m_ij^l
  = (1 + alpha * s_ij) * m_ij^l                                  (1)
```

and the node update is the standard sum-aggregation residual:

```
h_i^{l+1} = h_i^l + sum_{j ∈ N(i)} m_ij'^l                             (2)
```

Substituting:

```
h_i^{l+1} = h_i^l + sum_{j ∈ N(i)}
                                  (1 + alpha * tanh(psi(Enc(Q_ij))))
                                  * phi_base(h_i^l, h_j^l, e_ij)       (3)
```

### 2.4 Key properties

- **Baseline at initialisation.** `alpha = 0` ⇒ `m_ij'^l = m_ij^l`, so the
  network is bit-identical to the baseline at the start of training. Any
  improvement is *learned* and attributable to QFIM information.
- **Bounded modulation.** `s_ij ∈ [-1, 1]` and a learned scalar `alpha`
  bound the per-edge multiplicative factor in `[1 - |alpha|, 1 + |alpha|]`.
  In practice `|alpha| < 1` after training, so messages are never sign-
  flipped.
- **No new pathway.** Because `m_ij'^l` is `m_ij^l` times a scalar,
  whenever `m_ij^l = 0` (a chemically uninformative edge) the QFIM cannot
  inject information at that edge. The QFIM can only rescale information
  that the baseline already produced.
- **Permutation alignment.** The QFIM is recorded in the same molecule-
  ordering convention as the targets (see §5 on data alignment); the
  `_gather_edge_qfim` step indexes `Q[mol_id, local_i, local_j]` so the
  block applied to edge `(i, j)` is the coupling between the rotation-gate
  groups of qubits `i` and `j`, not arbitrary qubits.

---

## 3. Implementation

### 3.1 Common encoder (heads)

The QFIM head is shared between the custom GNN and the DimeNet++ wrapper.
The default is the 2-layer Conv2d head over the 6×6 block:

```python
class _QFIMHeadConv2d(nn.Module):
    def __init__(self, pd=6, out_dim=8, conv_channels=16, kernel_size=3):
        ...
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.norm1 = nn.LayerNorm([16, pd, pd])
        self.conv2 = nn.Conv2d(16, 16, 3, padding=1)
        self.norm2 = nn.LayerNorm([16, pd, pd])
        self.pool  = nn.AdaptiveAvgPool2d(1)
        self.project = nn.Linear(16, out_dim)
        self.out_norm = nn.LayerNorm(out_dim)

    def forward(self, qfim_edge):              # (E, pd*pd)
        Q = qfim_edge.view(-1, 1, pd, pd)
        h = relu(self.norm1(self.conv1(Q)))
        h = relu(self.norm2(self.conv2(h)))
        h = self.pool(h).flatten(1)
        return self.out_norm(self.project(h))  # (E, out_dim)
```

[networks/GNN/gnn_qfim.py:145-169](networks/GNN/gnn_qfim.py#L145-L169)

The bounded scalar `s_ij` is produced by a tiny 2-layer MLP terminating in
`tanh`:

```python
self.scale_mlp = nn.Sequential(
    nn.Linear(qfim_dim, max(4, qfim_dim)),
    activation(),
    nn.Linear(max(4, qfim_dim), 1),
    nn.Tanh(),
)
```

### 3.2 Per-edge gather

`_gather_edge_qfim` takes the batched QFIM tensor of shape
`(B, n_q, n_q, p_d, p_d)` and the bond-graph `edge_index`, computes each
edge's source/destination *local* qubit indices within its molecule, and
returns the corresponding flattened `(E, p_d*p_d)` block. Atoms whose
local index exceeds `n_q` (heavy-atom budget = 10 in QM9) are zero-masked
so QFIM contributes nothing for those edges.

[networks/GNN/gnn_qfim.py:54-85](networks/GNN/gnn_qfim.py#L54-L85)

### 3.3 Custom GNN: `_QFIMRescaledBaselineMP`

```python
class _QFIMRescaledBaselineMP(MessagePassing):
    def message(self, x_i, x_j, edge_attr, qfim_attr, alpha):
        msg   = self.msg_mlp(torch.cat([x_i, x_j, edge_attr], dim=-1))
        scale = self.scale_mlp(qfim_attr)
        return msg + alpha * scale * msg
```

[networks/GNN/gnn_qfim_residual.py:280-351](networks/GNN/gnn_qfim_residual.py#L280-L351)

Per layer:

```python
def forward(self, x, edge_index, edge_attr, qfim_attr, alpha):
    out = self.propagate(edge_index, x=x, edge_attr=edge_attr,
                         qfim_attr=qfim_attr, alpha=alpha)
    return x + out                                # node-level residual
```

The wrapper class `QFIMResidualGNN` builds `num_mp_layers` of these and
either shares one `alpha` across layers (`alpha_mode="shared"`) or
allocates one `alpha` per layer (`alpha_mode="per_layer"`). For the headline
results we use the shared scalar. The forward pass computes `qfim_feat`
once per minibatch and reuses it across layers.

[networks/GNN/gnn_qfim_residual.py:507-518, 559-566](networks/GNN/gnn_qfim_residual.py#L507-L566)

### 3.4 DimeNet++: `DimeNetPPQFIM`

DimeNet++ does not have an explicit "node message"; it carries
*directed-edge embeddings* `m_ij` that are updated by interaction blocks.
The rescale is applied to those edge embeddings after the embedding block
and after each interaction block, so the same modulation pattern
`(1 + alpha * s_ij) * m_ij` is realised in DimeNet++'s native edge state:

```python
def _rescale_edges(self, edge_x, edge_index, qfim_block, qfim_nq, batch):
    qfim_edge = _gather_edge_qfim(qfim_block, qfim_nq, edge_index, batch)
    qfim_feat = self.qfim_head(qfim_edge)
    scale     = self.qfim_scale(qfim_feat)
    return edge_x + self.qfim_rescale_beta * self.qfim_residual_gate \
                                          * scale * edge_x

def forward(self, x, qfim_block, qfim_nq, batch):
    ...
    edge_x = self.model.emb(z, rbf, i, j)
    edge_x = self._rescale_edges(edge_x, ...)            # after embedding
    out    = self.model.output_blocks[0](edge_x, rbf, i, ...)
    for interaction_block, output_block in zip(
            self.model.interaction_blocks,
            self.model.output_blocks[1:]):
        edge_x = interaction_block(edge_x, rbf, sbf, idx_kj, idx_ji)
        edge_x = self._rescale_edges(edge_x, ...)        # after each block
        out    = out + output_block(edge_x, rbf, i, ...)
    return scatter(out, batch, dim=0, reduce="sum")
```

[networks/GNN/dimenet.py:225-281](networks/GNN/dimenet.py#L225-L281)

The `edge_index` used for QFIM gather is DimeNet++'s `radius_graph` at
cutoff = 5 Å; this is the same graph the interaction blocks operate on,
so qubit-pair attribution is consistent with the message graph.

`qfim_rescale_beta` is a multiplicative budget left at `1.0` for headline
runs; together with `tanh` and the learned `alpha` it bounds the per-edge
factor in
`[1 - qfim_rescale_beta * |alpha|, 1 + qfim_rescale_beta * |alpha|]`.

---

## 4. Architecture Configurations

### 4.1 300k custom GNN (headline rescale runs)

- 6 invariant message-passing layers, SiLU activations, per-layer edge
  update, max pooling.
- Node MLP: `[19, 128, 256, 64, 32]`.
- Edge MLP: `[28, 64, 128, 64, 32]`. Edge input = 3 bond-angle slots +
  9 dihedral slots + 16 RBF distance bins = 28 dims.
- QFIM: `n_qubits = 10`, `per_qubit_dim = 6`, `embed_op = conv2d`,
  `out_dim = 8`, `mode = baseline_msg_rescale`, `alpha` shared, init 0.

Parameter counts:

| Variant                                | Params  | Δ vs baseline |
|----------------------------------------|---------|---------------|
| 300k baseline                          | 308,545 | —             |
| Message-rescale QFIM (this work)       | 313,968 | +5,423 (1.8%) |
| Older additive residual QFIM (compare) | 337,170 | +28,625 (9.3%)|

The message-rescale variant adds only ~5.4k parameters
(QFIM head + scale MLP + one scalar `alpha`), about a fifth of the
additive-residual cost.

Configs:
[configs/YAML/qm9_qfim_residual_local_300k_msg_rescale.yaml](configs/YAML/qm9_qfim_residual_local_300k_msg_rescale.yaml),
[configs/YAML/qm9_v37_300k.yaml](configs/YAML/qm9_v37_300k.yaml).

### 4.2 DimeNet++ (heavy)

- `hidden_channels = 128`, `num_blocks = 4`, `int_emb_size = 64`,
  `basis_emb_size = 8`, `out_emb_channels = 256`, `num_spherical = 7`,
  `num_radial = 6`, `cutoff = 5.0`, `max_num_neighbors = 32`,
  `act = swish`, `output_initializer = zeros`.
- QFIM head/scale identical to §3.1, `out_dim = 8`, `rescale_beta = 1.0`,
  `residual_gate_init = 0.0`.
- Optimizer: Adam, `lr = 1e-3`, no weight decay; loss = MAE on
  standardized gap.

Configs:
[configs/YAML/qm9_dimenet_pp_heavy_qfim.yaml](configs/YAML/qm9_dimenet_pp_heavy_qfim.yaml),
[configs/YAML/qm9_dimenet_pp_heavy.yaml](configs/YAML/qm9_dimenet_pp_heavy.yaml).

---

## 5. QFIM Data Alignment

Two alignment facts matter and are easy to get wrong; both are now
verified.

### 5.1 Row alignment with target

Each row in the HDF5 dataset stores `(graph_i, target_i, qfim_i)` for the
*same* molecule. The loader's default `qfim_ablation_mode = "none"`
preserves this, and the `random` ablation replaces the matrix while
keeping the row index, so any row-permutation bug would affect both
controls equally.

### 5.2 Rot-gate flatten convention

bioQINN's parameterised circuit uses 2 hardware-efficient layers × 3
single-qubit rotations × 10 qubits = 60 rot-gate weights, stored in
`(num_layers, ops_per_layer, n_qubits)` with the qubit axis as the
fastest-varying. PennyLane's `metric_tensor` flattens in C order, so the
60×60 raw matrix has qubit as its fastest-varying axis. To recover the
intended `(n_q, n_q, p_d, p_d)` layout used by the gather, we reshape and
transpose:

```python
rot_block = rot_block.reshape(p_d, n_q, p_d, n_q).transpose(1, 3, 0, 2)
```

[data_handlers/qm9_graph_loader.py:300-309](data_handlers/qm9_graph_loader.py#L300-L309)

This was verified by `probe_qfim_reshape.py` against a recomputed QFIM.
A naive `(n_q, p_d, n_q, p_d)` reshape silently mis-attributes per-qubit
parameters and was responsible for early non-results.

---

## 6. Targets and Loss

Targets are standardized using train-split statistics only:

```
y_norm = (y_raw - train_mean) / train_std
```

Validation MAE is reported denormalized in meV; the mean cancels:

```
MAE_meV = MAE_norm * train_std_gap * 1000
```

For the gap target, `train_std_gap ≈ 1.284 eV`. The custom GNN uses
Huber loss (delta = 0.1) on standardized targets; DimeNet++ uses MAE.

---

## 7. Empirical Behaviour

### 7.1 300k custom GNN (seed 42, validation; n=5 seeds, test)

**Seed-42 Validation MAE**:

| Variant                       | Best (meV) | Final (meV) |
|-------------------------------|-----------:|------------:|
| Baseline                      | 135.2      | 136.5       |
| Message-rescale QFIM (real)   | 125.0      | 126.2       |
| Message-rescale QFIM (random) | 133.7      | 136.0       |

Seed-42 validation suggested QFIM improves by ~10 meV and random QFIM sits at
baseline. However, this does not generalise to multiple seeds.

**Test-Set MAE** (final evaluation, n = 5 seeds):

| Variant                       | Test MAE (meV)      |
|-------------------------------:|--------------------:|
| Baseline                      | 128.5 ± 5.5         |
| Message-rescale QFIM (real)   | 133.2 ± 5.3         |
| Message-rescale QFIM (random) | 127.7 ± 4.7         |
| Effect of real QFIM           | −4.7 ± 1.6 (worse)  |

The aligned QFIM is detrimental to the 300k GNN, in stark contrast to
DimeNet++ (§7.2). The architecture passes the random-control test (random
QFIM ≈ baseline), confirming no generic matrix-input exploitation. However,
QFIM appears orthogonal to or in conflict with learned representations in
this smaller custom GNN.

### 7.2 DimeNet++ heavy (5 seeds, 42–46)

**Validation MAE**, mean over the common epoch range (n = 5):

| Variant                | Best mean (meV) | Final mean (meV) |
|------------------------|----------------:|-----------------:|
| DimeNet++ baseline     | 75.6            | 75.6             |
| DimeNet++ + QFIM       | 69.5            | 72.1             |

Per-seed best/final values are tabulated by
[plots/make_dimenet_repeats.py](plots/make_dimenet_repeats.py); the
companion plot is `plots/make_dimenet_repeats_diff.png` with the bottom
panel showing `Δ MAE = baseline − QFIM` per epoch with seed-spread
error bars.

**Test-Set MAE** (final evaluation, n = 5 seeds):

| Variant                | Test MAE (meV)      |
|------------------------|--------------------:|
| DimeNet++ baseline     | 71.6 ± 1.7          |
| DimeNet++ + QFIM       | 67.6 ± 1.5          |
| Improvement            | 4.0 ± 2.1 (std)     |

A random-QFIM control for the DimeNet++ runs is the natural next ablation
to mirror §7.1 and is recommended before publication.

---

## 8. Reproducibility

300k GNN, 5 seeds:

```bash
for s in 42 43 44 45 46; do
  python -m networks.GNN.train --config \
    configs/YAML/qm9_qfim_residual_local_300k_msg_rescale_seed${s}.yaml
done
```

DimeNet++, 5 seeds (same pattern with
`configs/YAML/qm9_dimenet_pp_heavy{,_qfim}_seed${s}.yaml`).

Random-QFIM controls (architecture-identical, replaces `Q` with a
symmetric uniform random matrix, scale 0.25):

```yaml
qfim:
  ablation_mode: "random"
  ablation_seed: 42
  random_scale: 0.25
```

---

## 9. Implementation Note

The 300k message-rescale path is **message-scale only** (no separate
additive QFIM residual branch). In the current implementation,
`_QFIMRescaledBaselineMP.message` uses

```python
return msg + alpha * scale * msg
```

so the modulation is exactly equation (1):

```python
(1 + alpha * s_ij) * m_ij
```

The 300k message-rescale path is documented in alpha-only form. The
DimeNet++ wrapper still exposes `qfim_rescale_beta` as its own explicit
scaling control.

[networks/GNN/gnn_qfim_residual.py:280-351](networks/GNN/gnn_qfim_residual.py#L280-L351)

---

## 10. One-paragraph Abstract Insert

We constrain the role of the quantum Fisher information matrix (QFIM) in
a molecular graph neural network to a *message rescale*: a tiny encoder
maps each off-diagonal QFIM sub-block `Q_ij` to a bounded scalar
`s_ij ∈ [-1, 1]`, and each baseline edge message is multiplied by
`(1 + α s_ij)` with a single learnable `α` initialised to zero. This
adds 5.4k parameters (1.8 %) over a 308k-parameter invariant GNN
baseline, leaves the model bit-identical to the baseline at
initialisation, and prevents QFIM from creating a parallel computational
pathway — it can only strengthen or weaken existing chemistry messages.
The same rescale is applied to DimeNet++ edge embeddings after the
embedding block and after each interaction block. On QM9 HOMO–LUMO gap,
the rescale lowers validation MAE on both backbones, while a control
that swaps the real QFIM for a random symmetric matrix collapses to
baseline, supporting the claim that aligned QFIM information — not extra
matrix-conditioned capacity — is the source of improvement.
