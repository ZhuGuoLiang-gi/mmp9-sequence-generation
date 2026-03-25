"""
Diagnostic evaluation for GFlowNet training:
Compare model-assigned sequence log-prob with dataset reward.

It measures whether the trained decoder actually gives higher probability
to higher-reward sequences under the (sequence -> encoder embedding) condition.
"""

from __future__ import annotations

import argparse
import math
import random
from pathlib import Path
from typing import Tuple

import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import torch  # type: ignore

from ..config import ENCODER_PARAMS
from ..models.gflownet_transformer import GFlowNetTransformer
from ..models.transformer_encoder import TransformerEncoderModel
from ..tokenizer import build_vocab
from ..training.train_gflownet import build_sequence_embeddings, build_seq_idx


def seq_logp_from_logits(
    logits: torch.Tensor,
    sequences: torch.Tensor,
    *,
    pad_id: int,
) -> torch.Tensor:
    # logits: [B, L, V], sequences: [B, L]
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    token_logp = log_probs.gather(2, sequences.unsqueeze(-1)).squeeze(-1)  # [B, L]
    pad_mask = sequences != pad_id  # [B, L]
    denom = pad_mask.sum(dim=1).clamp(min=1)  # [B]
    seq_logp = (token_logp * pad_mask).sum(dim=1) / denom  # [B] average logp per non-pad token
    return seq_logp


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description="Eval GFlowNet: logp vs reward correlation.")
    parser.add_argument("--gflownet_checkpoint", type=str, default="models/best_gflownet.pth")
    parser.add_argument("--encoder_checkpoint", type=str, default=ENCODER_PARAMS["save_path"])
    parser.add_argument("--data_path", type=str, default=ENCODER_PARAMS["path"])
    parser.add_argument("--max_seqs", type=int, default=2000, help="Max sequences to evaluate.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_kobs_log", action="store_true", help="Use reward=log(Kobs+eps). If off, use raw Kobs.")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = ENCODER_PARAMS["device"]

    vocab, inv_vocab, vocab_size = build_vocab()
    pad_id = vocab["<pad>"]

    df = pd.read_csv(args.data_path, sep="\t")
    df = df[df["Kobs (M-1 s-1)"] >= 0].reset_index(drop=True)
    if len(df) > args.max_seqs:
        df = df.sample(n=args.max_seqs, random_state=args.seed).reset_index(drop=True)

    # Build encoder condition embeddings for the same sequences (matches training pipeline).
    encoder_model = TransformerEncoderModel(
        vocab_size=len(vocab),
        embed_dim=ENCODER_PARAMS["embed_dim"],
        num_heads=ENCODER_PARAMS["num_heads"],
        num_layers=ENCODER_PARAMS["num_layers"],
        dropout=ENCODER_PARAMS["dropout"],
        hidden_dim=ENCODER_PARAMS["hidden_dim"],
        max_len=ENCODER_PARAMS["max_length"] + 3,
        max_relative_distance=16,
    ).to(device)
    encoder_ckpt = torch.load(args.encoder_checkpoint, map_location=device)
    encoder_model.load_state_dict(encoder_ckpt["model_state"])
    encoder_model.eval()

    X = build_sequence_embeddings(df, encoder_model, vocab, inv_vocab, device=device, spec_token=False)
    seq_idx = build_seq_idx(df, vocab, X.shape[1], device=device)

    rewards_kobs = df["Kobs (M-1 s-1)"].values.astype(np.float64)
    eps = 1e-8
    if args.use_kobs_log:
        rewards = np.log(rewards_kobs + eps)
    else:
        rewards = rewards_kobs

    decode_model = GFlowNetTransformer(
        embed_dim=ENCODER_PARAMS["embed_dim"],
        hidden_dim=512,
        num_layers=4,
        num_heads=8,
        vocab_size=len(vocab),
        max_len=X.shape[1],
    ).to(device)
    decode_model.load_state_dict(torch.load(args.gflownet_checkpoint, map_location=device))
    decode_model.eval()

    logits = decode_model(X)
    seq_logp = seq_logp_from_logits(logits, seq_idx, pad_id=pad_id).detach().cpu().numpy()

    # Correlations
    def pearson(a: np.ndarray, b: np.ndarray) -> float:
        a = a.astype(np.float64)
        b = b.astype(np.float64)
        a = a - a.mean()
        b = b - b.mean()
        denom = math.sqrt((a * a).sum() * (b * b).sum()) + 1e-12
        return float((a * b).sum() / denom)

    def spearman(a: np.ndarray, b: np.ndarray) -> float:
        ra = a.argsort().argsort().astype(np.float64)
        rb = b.argsort().argsort().astype(np.float64)
        return pearson(ra, rb)

    p = pearson(seq_logp, rewards)
    s = spearman(seq_logp, rewards)

    # Quartile separation
    q = np.quantile(rewards, [0.25, 0.75])
    low_mask = rewards <= q[0]
    high_mask = rewards >= q[1]
    low_mean = float(seq_logp[low_mask].mean()) if low_mask.any() else float("nan")
    high_mean = float(seq_logp[high_mask].mean()) if high_mask.any() else float("nan")

    print(f"Evaluated sequences: {len(df)} (device={device})")
    print(f"Reward type: {'log(Kobs+eps)' if args.use_kobs_log else 'Kobs raw'}")
    print(f"Pearson(seq_logp, reward)={p:.4f}")
    print(f"Spearman(seq_logp, reward)={s:.4f}")
    print(f"Seq_logp low-reward mean={low_mean:.4f}, high-reward mean={high_mean:.4f}")


if __name__ == "__main__":
    main()

