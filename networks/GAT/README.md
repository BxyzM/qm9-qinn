# GAT — Graph Attention Network

## Theory

A Graph Attention Network (Veličković et al., ICLR 2018) performs message passing over
a molecular graph where **atoms are nodes** and **bonds are edges**. Unlike GCN which
uses fixed normalisation, GAT learns a scalar attention coefficient `α_ij` for each
directed edge (i → j), so the model decides how much atom j should listen to atom i.

### Attention mechanism

For a node pair (i, j) sharing an edge, the attention logit is:

```
e_ij = LeakyReLU( a^T [ W h_i || W h_j ] )
```

where `h_i` is the node feature of atom i, `W` is a shared linear projection, `a` is
a learned vector, and `||` denotes concatenation.

The logits are normalised over all neighbours of node i via softmax:

```
α_ij = softmax_j( e_ij ) = exp(e_ij) / Σ_{k ∈ N(i)} exp(e_ik)
```

The updated node embedding is:

```
h_i' = σ( Σ_{j ∈ N(i)} α_ij W h_j )
```

With **multi-head attention** (K heads), K independent attention mechanisms run in
parallel and their outputs are concatenated (intermediate layers) or averaged (final layer):

```
h_i' = ||_{k=1}^{K} σ( Σ_j α_ij^k W^k h_j )
```

### Why GAT for molecules

- Attention weights capture bond importance — a double bond should contribute differently
  from a single bond without hand-engineering that rule.
- Edge features (bond type, geometry) can be injected into the attention logit,
  making `e_ij` a function of the bond, not just the atoms.
- The learned `α_ij` values are interpretable and can be compared against QFIM-derived
  coupling strengths as a sanity check.

---

## Implementation for this project

### Graph construction

Build a PyTorch Geometric `Data` object per molecule:

```python
# node_feat: (max_nodes, 7), edge_feat: (max_nodes, max_nodes, 4)
# n_heavy = n_atoms[1]  ← use heavy-atom count only

x = node_feat[:n_heavy]                  # (n_heavy, 7)  node features

# Build edge index from non-zero bond entries
src, dst = [], []
edge_attrs = []
for i in range(n_heavy):
    for j in range(n_heavy):
        if edge_feat[i, j, 0] > 0:      # bond_type > 0 means a bond exists
            src.append(i); dst.append(j)
            edge_attrs.append(edge_feat[i, j])   # (4,): bond_type, θ, φ, d

edge_index = torch.tensor([src, dst], dtype=torch.long)   # (2, n_edges)
edge_attr  = torch.tensor(edge_attrs, dtype=torch.float)  # (n_edges, 4)
```

Add self-loops so each atom attends to itself (standard practice in GAT).

### Architecture

```
Input node features: (n_heavy, 7)
Input edge features: (n_edges, 4)

GATConv layer 1:  in=7,  out=64, heads=4, edge_dim=4  → (n_heavy, 256)  + ELU
GATConv layer 2:  in=256, out=64, heads=4, edge_dim=4 → (n_heavy, 256)  + ELU
GATConv layer 3:  in=256, out=64, heads=1, edge_dim=4 → (n_heavy, 64)   + ELU

Global mean pool: (n_heavy, 64) → (64,)   ← aggregate over atoms

MLP head:
  Linear(64, 32) → ReLU
  Linear(32, 1)  → gap prediction (eV)
```

Use `torch_geometric.nn.GATConv` with `edge_dim=4` to incorporate bond features
into the attention computation. This maps edge features through a separate linear
layer that is added to the attention logit before the LeakyReLU.

### Training details

- Optimizer: Adam, lr=5e-4, weight decay=1e-5
- Batch size: 32 (PyG handles variable-size graphs via batch pointers)
- Epochs: 200 with early stopping on val MAE (patience=20)
- Loss: MAE (L1Loss)
- Normalise gap target (same mean/std as DNN baseline for fair comparison)

### What to record for comparison

- Val and test MAE in meV
- Per-edge attention weights `α_ij` for a sample of molecules — visualise as a
  heatmap over the molecular graph (compare against QFIM coupling matrix later)
- Training curves (MAE vs epoch)

### Expected performance

A 3-layer GAT with edge features on QM9 gap should reach **400–650 meV MAE**,
roughly SchNet level without the continuous-filter convolutions. This is your main
classical graph baseline.
