"""
Legacy GNN models kept for reference.

These are not wired into networks.GNN's top-level package and are not
dispatched from train.py. To run one, import it directly from
`networks.GNN.legacy.<module>` and manually wire it up.

Files:
    gnn_plain.py            -- non-invariant MPNN baseline (pre-refactor)
    gnn_invariant.py        -- invariant GNN with mean-over-neighbors
                               angle/dihedral reductions
    gnn_qfim.py             -- QFIM on edges via flattened 36-dim block
    gnn_qfim_structured.py  -- QFIM + 12 per-3x3-sub-block scalar summaries
    gnn_qfim_conv.py        -- QFIM via symmetric Conv1d compressor
    gnn_qfim_node.py        -- QFIM diagonal on nodes + off-diagonal on edges
    gnn_qfim_cij.py         -- QFIM reduced to single C_ij = ||Q[i,j]||_F

Kept because some were trained to completion and the checkpoints / stats
at /ceph/mbinder/qm9-qinn/classical/saved_models/ are referenced by plot
scripts. Keeping the source next to the runs makes the legacy state
reproducible.
"""
