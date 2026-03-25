"""
Centralized configuration for the MMP9 sequence-generation codebase.

NOTE: Values are copied from current legacy scripts to avoid behavior drift.
"""

from __future__ import annotations

import torch


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


# Encoder (masked token recovery) defaults: see legacy_deprecated/train.py
ENCODER_PARAMS = {
    # relative to repository root when running `python run.py ...`
    "path": "data/MMP9.tsv",
    "max_length": 12,
    "mask_prob": 0.2,
    "n_mask": 4,
    "batch_size": 16,
    "embed_dim": 1024,
    "hidden_dim": 512,
    "num_heads": 8,
    "num_layers": 8,
    "dropout": 0.1,
    "epochs": 200,
    "lr": 1e-7,
    "device": get_device(),
    "split_ratio": (0.8, 0.1, 0.1),
    "random_seed": 42,
    "save_path": "models/best_model_spec_masked.pth",
    "resume": True,
    "masked_loss": True,
    "mask_specials": True,
}


def infer_gflownet_params() -> dict:
    """
    GFlowNet training/inference is not a direct 1:1 mapping from a single params dict,
    but we keep the most commonly referenced defaults here.
    """

    return {
        "device": get_device(),
        "max_length": 12,
        "embed_dim": ENCODER_PARAMS["embed_dim"],
        "hidden_dim": 512,
        "num_heads": 8,
    }

