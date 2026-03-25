"""
Evaluation: mask recovery curve (Phase 2 migration).

Skeleton.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from ..config import ENCODER_PARAMS
from ..tokenizer import build_vocab
from ..dataset import SequenceDataset, collate_fn
from ..models.transformer_encoder import TransformerEncoderModel


def load_and_split_data(path: str, seed: int = 42, split_ratio=(0.8, 0.1, 0.1)):
    np.random.seed(seed)
    import pandas as pd

    df = pd.read_csv(path, sep="\t")
    df_filtered = df[df["Kobs (M-1 s-1)"] > 0].reset_index(drop=True)
    n = len(df_filtered)
    indices = np.arange(n)
    np.random.shuffle(indices)

    n_train = int(split_ratio[0] * n)
    n_valid = int(split_ratio[1] * n)
    # n_test = n - n_train - n_valid

    train_idx = indices[:n_train]
    valid_idx = indices[n_train : n_train + n_valid]
    test_idx = indices[n_train + n_valid :]

    train_df = df_filtered.iloc[train_idx].reset_index(drop=True)
    valid_df = df_filtered.iloc[valid_idx].reset_index(drop=True)
    test_df = df_filtered.iloc[test_idx].reset_index(drop=True)
    return train_df, valid_df, test_df


def evaluate_mask_recovery(mask_prob: float, *, model, test_df, vocab, inv_vocab, device: str):
    test_dataset = SequenceDataset(
        test_df,
        vocab,
        inv_vocab=inv_vocab,
        mask_prob=mask_prob,
        n_mask=1,
        max_length=ENCODER_PARAMS["max_length"],
        mask_specials=ENCODER_PARAMS["mask_specials"],
    )
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn)

    total_correct_mask = 0
    total_tokens_mask = 0

    with torch.no_grad():
        for masked_tokens, target_tokens, mask_positions in test_loader:
            masked_tokens = masked_tokens.to(device)
            target_tokens = target_tokens.to(device)
            logits = model(masked_tokens)
            preds = logits.argmax(-1).cpu()

            for i, pos_list in enumerate(mask_positions):
                for pos in pos_list:
                    if preds[i, pos].item() == target_tokens[i, pos].item():
                        total_correct_mask += 1
                    total_tokens_mask += 1

    if total_tokens_mask == 0:
        return 1.0
    return total_correct_mask / total_tokens_mask


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate mask recovery curve.")
    parser.add_argument("--model_path", type=str, default=ENCODER_PARAMS["save_path"])
    parser.add_argument("--out", type=str, default="plot/mask_recovery_curve.png")
    parser.add_argument("--min_mask_prob", type=float, default=0.1)
    parser.add_argument("--max_mask_prob", type=float, default=1.0)
    parser.add_argument("--steps", type=int, default=10)
    args = parser.parse_args()

    vocab, inv_vocab, vocab_size = build_vocab()
    device = ENCODER_PARAMS["device"]

    model_path = args.model_path
    model = TransformerEncoderModel(
        vocab_size=vocab_size,
        embed_dim=ENCODER_PARAMS["embed_dim"],
        num_heads=ENCODER_PARAMS["num_heads"],
        num_layers=ENCODER_PARAMS["num_layers"],
        dropout=ENCODER_PARAMS["dropout"],
        hidden_dim=ENCODER_PARAMS["hidden_dim"],
        max_len=ENCODER_PARAMS["max_length"] + 3,
        max_relative_distance=16,
    )
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()

    _, _, test_df = load_and_split_data(
        ENCODER_PARAMS["path"], seed=ENCODER_PARAMS["random_seed"], split_ratio=ENCODER_PARAMS["split_ratio"]
    )

    mask_probs = np.linspace(args.min_mask_prob, args.max_mask_prob, args.steps)
    recovery_rates = []

    print("Evaluating mask recovery across probabilities:")
    for p in mask_probs:
        rate = evaluate_mask_recovery(p, model=model, test_df=test_df, vocab=vocab, inv_vocab=inv_vocab, device=device)
        recovery_rates.append(rate)
        print(f"mask_prob={p:.2f} → Masked Token Recovery Rate = {rate:.4f}")

    plt.figure(figsize=(7, 5))
    plt.plot(mask_probs, recovery_rates, marker="o", linewidth=2)
    plt.title("Masked Token Recovery Rate vs Mask Probability")
    plt.xlabel("Mask Probability")
    plt.ylabel("Recovery Rate")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out, dpi=300)
    plt.show()
    print(f"\n✅ 曲线已保存到: {args.out}")


if __name__ == "__main__":
    main()

