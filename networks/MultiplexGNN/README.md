# MultiplexGNN — Multi-Layer Quantum-Informed Hypergraph Network

## Theory

### Motivation

Your quantum circuit has `num_layers=2`, each containing entanglement gates and
trainable `Rot(α, β, γ)` rotations. The QFIM naturally decomposes into a
**block structure** that mirrors this layered architecture:

```
F = [ F_11  F_12 ]
    [ F_21  F_22 ]
```

where `F_ll'` is the `(3·n_qubits × 3·n_qubits)` cross-correlation block between
circuit layer `l` and layer `l'`. `F_11`, `F_22` are within-layer blocks; `F_12`,
`F_21` are cross-layer blocks capturing how parameter sensitivity in layer 1
influences layer 2.

A standard GNN processes a single graph. A **multiplex graph** is a stack of
graphs — one per circuit layer — with additional inter-layer edges connecting the
same node (atom) across layers. This directly represents the layered QFIM structure.

### Formal definition

A multiplex graph `G = {G^1, G^2, ..., G^L}` has:
- L copies of the node set (one per layer): atom i in layer l → node `(i, l)`
- **Intra-layer edges**: bonds between atoms within layer l, weighted by `F_ll` block
- **Inter-layer edges**: for each atom i, connect `(i, 1)` to `(i, 2)` (same atom,
  adjacent layers), weighted by the diagonal of the `F_12` cross block

### Why this goes beyond a standard GNN

A standard GNN on the molecular graph cannot represent the fact that the same bond
(i,j) has **different coupling strengths** at different depths of the circuit. The
multiplex structure encodes this explicitly: the intra-layer edge (i,j) in layer 1
carries `F_11[i,j]` and in layer 2 carries `F_22[i,j]`, and these can differ.

The inter-layer edges represent **information flow between circuit depths** — a
concept with no analogue in classical molecular GNNs.

### Hypergraph extension

A hyperedge connects more than two nodes. In this context:
- Define a hyperedge over all atoms `{i, j, k, ...}` whose QFIM entries exceed
  a threshold `τ` within a given layer block.
- These capture **multi-body quantum correlations** induced by the entanglement
  structure — the IsingXX/IsingYY/IsingZZ gates entangle multiple qubits simultaneously.

For the initial implementation, start with the pairwise multiplex (tractable).
Add hyperedges as an extension once the baseline multiplex result is established.

---

## Implementation for this project

### Step 1 — compute per-layer QFIM blocks

From `fisher_matrices: (N, n_rot, n_rot)` with `n_rot = num_layers × 3 × n_qubits`:

```python
num_layers = 2
n_ops      = 3
n_qubits   = 10
block_size = n_ops * n_qubits   # = 30

# For molecule m:
F = fisher_matrices[m]          # (60, 60)

# Within-layer blocks
F_11 = F[:block_size, :block_size]                    # (30, 30) layer 1
F_22 = F[block_size:, block_size:]                    # (30, 30) layer 2

# Cross-layer blocks
F_12 = F[:block_size, block_size:]                    # (30, 30)

# Marginalise over operations → atom-atom coupling per layer
# Reshape each block to (n_ops, n_qubits, n_ops, n_qubits)
def block_to_coupling(B, n_ops, n_qubits):
    B_r = B.reshape(n_ops, n_qubits, n_ops, n_qubits)
    return np.abs(B_r).mean(axis=(0, 2))              # (n_qubits, n_qubits)

C_11 = block_to_coupling(F_11, n_ops, n_qubits)      # intra-layer 1 coupling
C_22 = block_to_coupling(F_22, n_ops, n_qubits)      # intra-layer 2 coupling
C_12 = block_to_coupling(F_12, n_ops, n_qubits)      # cross-layer coupling
```

Save `C_11`, `C_22`, `C_12` per molecule alongside the HDF5 data.

### Step 2 — graph construction

For each molecule, build a PyG `HeteroData` object with two node types
(`layer_1`, `layer_2`) and three edge types:

```python
from torch_geometric.data import HeteroData

data = HeteroData()

# Node features: same atom features for both layers
data['layer_1'].x = node_feat_tensor          # (n_heavy, 7)
data['layer_2'].x = node_feat_tensor          # (n_heavy, 7) — same atoms

# Intra-layer edges (bonds that exist in the molecular graph)
# Layer 1 — edge weight from C_11
for i, j where bond(i,j) exists:
    data['layer_1', 'bond', 'layer_1'].edge_index  # (2, n_edges)
    data['layer_1', 'bond', 'layer_1'].edge_attr   # (n_edges, 5): bond feats + C_11[i,j]

# Same for layer 2 with C_22
data['layer_2', 'bond', 'layer_2'].edge_index  ...
data['layer_2', 'bond', 'layer_2'].edge_attr   # + C_22[i,j]

# Inter-layer edges: every atom connects to itself in the other layer
inter_idx = torch.arange(n_heavy)
data['layer_1', 'cross', 'layer_2'].edge_index = torch.stack([inter_idx, inter_idx])
data['layer_1', 'cross', 'layer_2'].edge_attr  = C_12_diag    # (n_heavy, 1)
# Add reverse direction too
data['layer_2', 'cross', 'layer_1'].edge_index = torch.stack([inter_idx, inter_idx])
data['layer_2', 'cross', 'layer_1'].edge_attr  = C_12_diag
```

### Step 3 — architecture

Use `torch_geometric.nn.HeteroConv` to wrap separate `GATConv` modules for each
edge type, then sum contributions at each node:

```
--- Per layer: intra-layer message passing ---
GATConv( in=7,   out=64, heads=4, edge_dim=5 ) → h_l1, h_l2   shape (n_heavy, 256)

--- Cross-layer: inter-layer message passing ---
GATConv( in=256, out=64, heads=4, edge_dim=1 ) on (layer_1 → layer_2) edges
GATConv( in=256, out=64, heads=4, edge_dim=1 ) on (layer_2 → layer_1) edges
→ updated h_l1, h_l2   shape (n_heavy, 256)

Repeat for 2 rounds of message passing.

--- Readout ---
# Concatenate layer representations per atom
h_atom = concat(h_l1, h_l2)              # (n_heavy, 512)

# Global mean pool over atoms
h_graph = mean_pool(h_atom)              # (512,)

MLP head:
  Linear(512, 128) → ReLU
  Linear(128, 32)  → ReLU
  Linear(32,  1)   → gap (eV)
```

### Step 4 — training

- Adam, lr=3e-4, weight decay=1e-5
- Batch size: 16 (HeteroData objects are larger)
- MAE loss, normalised target
- Early stopping patience=20
- Clip gradients at norm 1.0 (heterogeneous message passing can produce spikes)

### Hyperedge extension (follow-up)

Once the pairwise multiplex is working, add hyperedges using
`torch_geometric.nn.HypergraphConv`:

1. For each layer l, threshold `C_ll` at τ (e.g. τ = mean + 1 std).
2. Each connected component in the thresholded graph becomes a hyperedge.
3. Build the incidence matrix `H ∈ {0,1}^{n_atoms × n_hyperedges}`.
4. Replace the intra-layer `GATConv` with `HypergraphConv(in, out, use_attention=True)`.

This captures the genuine multi-body entanglement structure — atoms that are all
mutually strongly coupled in the QFIM form a single hyperedge rather than many
pairwise edges.

### Ablation structure

| Variant | Description |
|---|---|
| **Multiplex-noQFIM** | Two-layer graph, uniform edge weights |
| **Multiplex-QFIM** | Two-layer graph, QFIM-weighted edges |
| **Multiplex-cross** | Add inter-layer edges, QFIM-weighted |
| **Multiplex-hyper** | Replace intra-layer edges with hyperedges |

Run in sequence — each variant adds one component. This isolates what each
QFIM-derived structural element contributes to the final MAE.

### What makes this novel

No existing molecular GNN uses a quantum circuit's layered parameter geometry to
define a multiplex graph topology. The inter-layer edges are a direct translation
of cross-layer quantum correlations into a classical graph structure — a bridge
between quantum information geometry and geometric deep learning.
