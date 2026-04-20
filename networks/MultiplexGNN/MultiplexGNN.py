"""
MultiplexGNN scaffold for QM9 gap prediction.

This file intentionally contains comments only.
Use it as a guided implementation checklist.
"""

# ============================================================
# 1) Imports
# ============================================================
# Import torch / torch.nn as nn.
# Import PyG heterogeneous graph modules:
# - HeteroData
# - HeteroConv
# - GATConv
# - global_mean_pool (or hetero-compatible pooling strategy)


# ============================================================
# 2) QFIM block decomposition utilities
# ============================================================
# Implement helper:
#   split_qfim_blocks(F, num_layers=2, n_ops=3, n_qubits=10)
#
# Given F: (n_rot, n_rot), with n_rot=num_layers*n_ops*n_qubits
# - block_size = n_ops * n_qubits = 30
# - F_11 = F[:block_size, :block_size]
# - F_22 = F[block_size:, block_size:]
# - F_12 = F[:block_size, block_size:]
#
# Implement helper:
#   block_to_coupling(B, n_ops=3, n_qubits=10)
# - reshape B -> (n_ops, n_qubits, n_ops, n_qubits)
# - return abs(B).mean(axis=(0,2))  # (n_qubits, n_qubits)
#
# Final per-molecule outputs:
# - C_11, C_22, C_12  each shape (10, 10)


# ============================================================
# 3) HeteroData construction
# ============================================================
# Build helper:
#   build_multiplex_graph(node_feat, edge_feat, n_atoms, C_11, C_22, C_12)
#
# Node types:
# - 'layer_1' with x: (n_heavy, 7)
# - 'layer_2' with x: (n_heavy, 7)  # same atom features duplicated
#
# Edge types:
# A) ('layer_1', 'bond',  'layer_1')
#    edge_attr dim=5: [bond_4, C_11[i,j]] for bonded pairs
# B) ('layer_2', 'bond',  'layer_2')
#    edge_attr dim=5: [bond_4, C_22[i,j]] for bonded pairs
# C) ('layer_1', 'cross', 'layer_2')
#    edges i->i for all atoms, edge_attr dim=1 from diag(C_12)
# D) ('layer_2', 'cross', 'layer_1')
#    reverse edges i->i, same dim=1
#
# Use n_heavy = n_atoms[1] and ignore padded atoms.


# ============================================================
# 4) Multiplex model architecture
# ============================================================
# Create class MultiplexGNN(nn.Module):
# - Use HeteroConv with per-edge-type GATConv modules.
# - Two rounds of message passing as described in README.
#
# Round design target:
# - Intra-layer bond conv: in=7 -> out=64, heads=4, edge_dim=5  (=> 256)
# - Cross-layer conv:      in=256 -> out=64, heads=4, edge_dim=1 (=> 256)
# - Repeat message passing round a second time.
#
# After message passing:
# - h_atom = concat(h_layer1, h_layer2)   # (n_heavy, 512)
# - graph readout via mean pooling over atoms
# - MLP: Linear(512,128)->ReLU, Linear(128,32)->ReLU, Linear(32,1)


# ============================================================
# 5) Pooling and batching notes
# ============================================================
# For batched HeteroData, ensure pooling uses correct batch vector(s)
# per node type. Common pattern:
# - pool layer_1 and layer_2 separately
# - then concatenate pooled graph embeddings
# OR
# - concatenate node embeddings per atom before pooling if alignment is preserved
#
# Keep implementation simple and deterministic for first version.


# ============================================================
# 6) Training settings
# ============================================================
# Target settings from README:
# - Adam lr=3e-4, weight_decay=1e-5
# - batch_size=16
# - L1Loss / MAE
# - normalized targets
# - early stopping patience=20
# - gradient clipping: clip_grad_norm_(model.parameters(), 1.0)


# ============================================================
# 7) Optional hypergraph extension placeholder
# ============================================================
# Later extension (not in initial baseline):
# - derive hyperedges from thresholded C_ll
# - build incidence matrix H
# - replace intra-layer GATConv with HypergraphConv(use_attention=True)
#
# Keep this as a later experiment after pairwise multiplex baseline works.


# ============================================================
# 8) Ablation variants checklist
# ============================================================
# TODO: Multiplex-noQFIM   (uniform edge weights)
# TODO: Multiplex-QFIM     (QFIM-weighted intra-layer edges)
# TODO: Multiplex-cross    (add inter-layer cross edges)
# TODO: Multiplex-hyper    (replace pairwise with hyperedges)
#
# Log each variant separately and compare MAE.


# End of comment-only scaffold.
