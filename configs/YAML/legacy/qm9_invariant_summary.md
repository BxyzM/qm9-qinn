# QM9 Invariant GNN Summary

This config runs the invariant QM9 GNN with dihedral features enabled.

## Config

- Model type: `gnn_invariant`
- Node feature dim: `9`
- Hidden dim: `256`
- Message-passing layers: `6`
- Dihedral: `on`
- Target: `gap`

## Parameter Structure

The model is built from four trainable parts:

1. `node_embed`: `Linear(9, 256)`
2. `edge_embed`: `Linear(4, 256) -> ReLU -> Linear(256, 256)`
3. `6 x InvariantMP` blocks, each with:
   - `Linear(516, 256)` for the message MLP input `[x_i, x_j, edge_attr]`
   - `LayerNorm(256)`
4. `readout`: `Linear(256, 32) -> ReLU -> Linear(32, 1)`

## Approximate Trainable Parameters

With `hidden_dim=256`, the model has about `875,073` trainable parameters.

## How to Run

```bash
python -m networks.GNN.train --config configs/YAML/qm9_invariant_300k.yaml
```

## Notes

- Targets are standardized from the training split before optimization; MAE/RMSE are inverse-transformed back to the original QM9 scale for reporting.
- `include_dihedral: true` adds the dihedral angle as an extra invariant edge feature.
- The invariant edge input is `bond_type + distance + bond_angle + dihedral`.
- Dihedral does not replace the other pairwise geometric features; it is added on top of them.