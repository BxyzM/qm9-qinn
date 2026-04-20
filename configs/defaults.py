"""
Shared default configuration parameters for the QM9 quantum regression pipeline.

Author: Dr. Aritra Bal (ETP)
Date: March 03, 2026
"""

DEFAULTS = {
    "setup": {
        "run_id":       "run_001",
        "train":        True,
        "batch_size":   1,
        "epochs":       50,
        "shuffle":      True,
        "num_workers":  0,
        "targets":      ["gap"],
        "convert_pnp":  False,
        "seed":         42,
        "train_n":      1000,
        "val_n":        200,
        "test_n":       3000,
    },
    "model": {
        "n_qubits":     8,
        "num_layers":   1,
        "device":       "default.gpu",
        "backend":      "autograd",
        "shots":        None,
        "operations_per_layer": 3,
    },
    "optimizer": {
        "name":             "Adam",
        "lr":               0.01,
        "lr_decay":         True,
        "decay_factor":     0.2,
        "decay_patience":   2,
        "decay_threshold":  0.1,
        "patience":         3,
    },
    "paths": {
        "train":        "/ceph/abal/BIO/QM9/train/qm9_train.h5",
        "val":          "/ceph/abal/BIO/QM9/val/qm9_val.h5",
        "test":         "/ceph/abal/BIO/QM9/test/qm9_test.h5",
        "model_dir":    "/ceph/abal/BIO/QM9/models",
    },
}