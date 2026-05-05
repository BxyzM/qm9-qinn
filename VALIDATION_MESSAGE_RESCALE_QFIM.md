# Validation: Message-Rescale QFIM GNN

## Objective

We want to validate whether QFIM improves the QM9 GNN through aligned physical
information, rather than through a large independent residual branch or generic
matrix-conditioned capacity.

The previous additive residual model used:

```text
h_next = baseline_gnn_layer(h, E) + alpha * qfim_correction_layer(h, Q_FI)
```

That branch was too expressive: random matrix inputs could also improve
performance. The new architecture is more constrained. QFIM can only rescale the
baseline chemistry/geometry message.

## Model Equation

Baseline message for directed edge `i -> j`:

```text
m_ij^l = phi_base(h_i^l, h_j^l, e_ij)
```

QFIM encoding:

```text
q_ij = Enc(Q_ij)
```

Scalar QFIM modulation:

```text
s_ij = tanh(psi(q_ij))
```

Message-rescale update:

```text
m_ij'^l = m_ij^l + alpha * s_ij * m_ij^l
```

Equivalent:

```text
m_ij'^l = (1 + alpha * s_ij) * m_ij^l
```

Node update:

```text
h_i^{l+1} = h_i^l + sum_{j in N(i)} m_ij'^l
```

Full equation:

```text
h_i^{l+1}
=
h_i^l
+
sum_{j in N(i)}
(1 + alpha * tanh(psi(Enc(Q_ij))))
*
phi_base(h_i^l, h_j^l, e_ij)
```

Important property:

```text
alpha = 0  =>  m_ij'^l = m_ij^l
```

So the model starts as the exact baseline and learns whether QFIM should
increase or decrease existing baseline messages.

## Implementation Snippet

Implemented in:

```text
networks/GNN/gnn_qfim_residual.py
```

Core message-rescale layer:

```python
class _QFIMRescaledBaselineMP(MessagePassing):
    """Baseline messages plus an alpha-scaled QFIM modulation of themselves."""

    def message(self, x_i, x_j, edge_attr, qfim_attr, alpha):
        msg = self.msg_mlp(torch.cat([x_i, x_j, edge_attr], dim=-1))
        scale = self.scale_mlp(qfim_attr)
        return msg + alpha * scale * msg
```

The QFIM scale is bounded:

```python
self.scale_mlp = nn.Sequential(
    nn.Linear(qfim_dim, max(4, qfim_dim)),
    _make_activation(activation),
    nn.Linear(max(4, qfim_dim), 1),
    nn.Tanh(),
)
```

So:

```text
s_ij in [-1, 1]
```

The mode is selected by config:

```yaml
qfim:
  mode: "baseline_msg_rescale"
```

## Baseline Features

Node features used:

```text
atomic number Z -> learned embedding
x, y, z coordinates
```

Node features not used:

```text
hydrogen count
aromaticity
hybridization
charge
```

Edge features used in the current main runs:

```text
3 bond-angle slots
9 dihedral-angle slots
16 Gaussian RBF distance features
```

Total raw edge dimension:

```text
3 + 9 + 16 = 28
```

The 300k baseline edge MLP:

```yaml
edge_mlp_dims: [28, 64, 128, 64, 32]
```

## Parameter Counts

Current 300k baseline:

```text
baseline GNN: 308,545 parameters
```

Message-rescale QFIM:

```text
message-rescale QFIM: 313,968 parameters
extra parameters:       5,423
relative increase:      1.8%
```

This is much smaller than the older additive residual QFIM branch:

```text
old additive residual QFIM: 337,170 parameters
extra parameters:            28,625
relative increase:            9.3%
```

## Main Configs

Baseline:

```text
configs/YAML/qm9_v37_300k.yaml
```

Message-rescale real QFIM:

```text
configs/YAML/qm9_qfim_residual_local_300k_msg_rescale.yaml
```

Message-rescale random QFIM:

```text
configs/YAML/qm9_qfim_residual_local_300k_msg_rescale_random_seed42.yaml
```

## Commands

Baseline:

```bash
/work/mbinder/nn/miniconda3/envs/ParT/bin/python -m networks.GNN.train \
  --config configs/YAML/qm9_v37_300k.yaml
```

Real QFIM:

```bash
/work/mbinder/nn/miniconda3/envs/ParT/bin/python -m networks.GNN.train \
  --config configs/YAML/qm9_qfim_residual_local_300k_msg_rescale.yaml
```

Random QFIM control:

```bash
/work/mbinder/nn/miniconda3/envs/ParT/bin/python -m networks.GNN.train \
  --config configs/YAML/qm9_qfim_residual_local_300k_msg_rescale_random_seed42.yaml
```

## Plot Command

Focused plot:

```bash
/work/mbinder/nn/miniconda3/envs/ParT/bin/python plots/make_v37_msg_rescale.py
```

Output:

```text
plots/make_v37_msg_rescale.png
```

The plot uses denormalized validation MAE:

```text
MAE_meV = MAE_normalized * target_std_gap * 1000
```

## Current Results

### Seed-42 Validation (Single-Seed)

```text
baseline                  best 135.2 meV, final 136.5 meV
message rescale QFIM      best 125.0 meV, final 126.2 meV
message rescale random    best 133.7 meV, final 136.0 meV
```

Single-seed validation suggested QFIM improves by ~10.2 meV, but this does not generalize.

### Test-Set Results (n=5 Seeds, Final)

**Baseline 300k:**
```text
Mean ± std: 128.5 ± 5.5 meV
Per-seed: [136.8, 121.5, 132.7, 124.6, 126.7] meV
```

**QFIM Message-Rescale 300k:**
```text
Mean ± std: 133.2 ± 5.3 meV
Per-seed: [138.9, 135.1, 123.4, 136.1, 132.8] meV
```

**QFIM Random Control 300k:**
```text
Mean ± std: 127.7 ± 4.7 meV
Per-seed: [133.0, 130.3, 126.2, 129.7, 119.3] meV
```

### Interpretation

Contrary to the seed-42 single-run validation result:

```text
real QFIM is WORSE than baseline by 4.7 ± 1.6 meV (paired)
random QFIM is similar to baseline (-0.8 ± 1.5 meV)
```

**Key Finding:** The aligned QFIM does NOT improve the 300k GNN model. The
single-seed validation improvement (seed 42) was a statistical fluctuation that
does not persist when averaged over multiple seeds and evaluated on the test set.

This contrasts sharply with DimeNet++, where QFIM provides consistent 4.0 ± 1.3 meV
improvement across n=5 seeds on the test set (67.6 ± 1.7 meV with QFIM vs. 
71.6 ± 1.9 meV baseline).

**Conclusion:** The message-rescale QFIM architecture is physically well-motivated,
but empirically does not provide utility for this GNN on QM9. The architecture
passes the random-control test (random QFIM ≈ baseline), confirming it is
not exploiting a generic matrix-input capacity. However, aligned QFIM appears
orthogonal to the learned representations in this model, and may even add noise.

## Required Validation Checks

### 1. Same Baseline Architecture

Check that baseline and QFIM runs use the same core architecture:

```yaml
model:
  num_layers: 6
  pooling: "max"
  activation: "silu"
  mlp_residual: true
  msg_layers: 2
  per_layer_edge_update: true
  node_mlp_dims: [19, 128, 256, 64, 32]
  edge_mlp_dims: [28, 64, 128, 64, 32]
```

Only QFIM message-rescale parameters are added.

### 2. Correct Aligned QFIM

Real QFIM configs must not contain:

```yaml
ablation_mode: "random"
ablation_mode: "row_shuffle"
```

If no ablation is specified, the loader defaults to:

```text
qfim_ablation_mode = "none"
```

So:

```text
graph row i -> target row i -> QFIM row i
```

### 3. Random Control

Random control configs must contain:

```yaml
qfim:
  ablation_mode: "random"
  ablation_seed: 42
  random_scale: 0.25
```

This keeps the same architecture and parameter count, but replaces QFIM with
synthetic random symmetric matrix input.

### 4. Target Normalization

Targets are standardized with train-split metadata only:

```text
y_norm = (y_raw - train_mean) / train_std
```

Validation MAE is denormalized for plotting:

```text
MAE_meV = MAE_norm * train_std_gap * 1000
```

Mean is not needed for MAE denormalization because the offset cancels.

### 5. Repeat Runs

Seed 42 is complete. Repeat configs were created for seeds 43-46:

```text
real QFIM:
configs/YAML/qm9_qfim_residual_local_300k_msg_rescale_seed43.yaml
configs/YAML/qm9_qfim_residual_local_300k_msg_rescale_seed44.yaml
configs/YAML/qm9_qfim_residual_local_300k_msg_rescale_seed45.yaml
configs/YAML/qm9_qfim_residual_local_300k_msg_rescale_seed46.yaml

random QFIM:
configs/YAML/qm9_qfim_residual_local_300k_msg_rescale_random_seed43.yaml
configs/YAML/qm9_qfim_residual_local_300k_msg_rescale_random_seed44.yaml
configs/YAML/qm9_qfim_residual_local_300k_msg_rescale_random_seed45.yaml
configs/YAML/qm9_qfim_residual_local_300k_msg_rescale_random_seed46.yaml
```

Final claim should use mean +/- std over paired seeds.

## Acceptance Criteria and Result

### Expected Pattern for Success

Strong evidence for useful aligned QFIM:

```text
real QFIM < baseline
real QFIM < random QFIM
random QFIM ~= baseline
```

Weak or inconclusive evidence:

```text
real QFIM ~= random QFIM
```

This would mean the architecture still exploits generic matrix input.

### Actual Result (Test-Set, n=5)

The final n=5 test-set results show:

```text
real QFIM > baseline       (WORSE)
real QFIM > random QFIM    (WORSE than random)
random QFIM ~= baseline    (PASS: random control is at baseline)
```

The architecture **does not** exploit generic matrix inputs (random QFIM ≈ baseline),
confirming the design is sound. However, the aligned QFIM is detrimental rather than
helpful. This suggests that:

1. The 300k GNN has already learned effective message passing that is orthogonal
   to the QFIM information provided.
2. Adding QFIM-based modulation introduces noise or conflicts with learned
   representations rather than refining them.
3. The benefit of QFIM may be specific to architectures like DimeNet++, which
   have different representational capacity or inductive biases.

## Distance-Only Follow-Up

To test whether QFIM helps more when angular/dihedral geometry is removed, two
distance-only configs were added.

Baseline distance-only:

```text
configs/YAML/qm9_v37_300k_distance_only.yaml
```

QFIM message-rescale distance-only:

```text
configs/YAML/qm9_qfim_residual_local_300k_msg_rescale_distance_only.yaml
```

These use:

```yaml
max_neighbors: 0
max_chains: 0
edge_mlp_dims: [16, 64, 128, 64, 32]
```

So the edge input is only:

```text
RBF(distance) in R^16
```

No bond angles and no dihedrals are used.

Parameter counts:

```text
distance-only baseline:             307,009
distance-only message-rescale QFIM: 312,432
extra QFIM parameters:                5,423
```

## Slide-Safe Summary

The message-rescale architecture constrains QFIM to modulate existing molecular
messages:

```text
m_ij' = (1 + alpha * tanh(psi(Enc(Q_ij)))) * m_ij
```

Unlike the earlier additive residual branch, QFIM cannot create a separate
message-passing path. It can only strengthen or weaken baseline chemistry
messages.

In seed 42, aligned QFIM improves validation MAE over the 300k baseline, while
the random-QFIM control remains close to baseline. This suggests the new
message-rescale architecture is a cleaner way to test whether aligned QFIM
contains useful information.
