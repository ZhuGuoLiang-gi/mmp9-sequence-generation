"""
Diagnostic evaluation for GFlowNet:
Compute top-K reward among dataset candidates using model-assigned seq_logp.

This isolates:
  - whether training aligned model ranking with reward (on candidate set)
  - from whether stochastic sampling produces exact TSV matches
"""

from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import torch  # type: ignore
from Bio import SeqIO  # type: ignore  # noqa: F401

from ..config import ENCODER_PARAMS
from ..inference.infer_gflownet_fasta import is_valid_sequence
from ..models.gflownet_transformer import GFlowNetTransformer
from ..models.transformer_encoder import TransformerEncoderModel
from ..tokenizer import build_vocab
from ..training.train_gflownet import build_sequence_embeddings, build_seq_idx


@dataclass(frozen=True)
class TopKSummary:
    n_conditions: int
    n_candidates: int
    topk: int
    mean_topk_mean_reward: float
    mean_topk_max_reward: float
    mean_top1_reward: float
    random_mean_topk_mean_reward: float


def reward_from_kobs(kobs: float, *, use_log: bool) -> float:
    if use_log:
        return float(np.log(float(kobs) + 1e-8))
    return float(kobs)


@torch.no_grad()
def seq_logp_for_condition(
    *,
    decode_model: GFlowNetTransformer,
    X_condition: torch.Tensor,  # [1, L, E]
    candidate_seq_idx: torch.Tensor,  # [N, L]
    pad_id: int,
) -> torch.Tensor:
    """
    Compute seq_logp for candidates given a single condition embedding.
    Returns:
      seq_logp: [N]
    """
    logits = decode_model(X_condition)  # [1, L, V]
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)  # [1, L, V]

    # Expand log_probs to [N, L, V] for gather.
    n = candidate_seq_idx.shape[0]
    log_probs = log_probs.expand(n, -1, -1)  # [N, L, V]

    token_logp = log_probs.gather(2, candidate_seq_idx.unsqueeze(-1)).squeeze(-1)  # [N, L]
    pad_mask = candidate_seq_idx != pad_id  # [N, L]
    denom = pad_mask.sum(dim=1).clamp(min=1)  # [N]
    seq_logp = (token_logp * pad_mask).sum(dim=1) / denom  # [N]
    return seq_logp


def sample_random_topk_rewards(
    rng: random.Random,
    rewards: np.ndarray,
    *,
    topk: int,
    n_trials: int,
) -> np.ndarray:
    n = len(rewards)
    out = np.empty(n_trials, dtype=np.float64)
    for t in range(n_trials):
        idx = rng.sample(range(n), k=min(topk, n)) if topk <= n else [rng.choice(range(n)) for _ in range(topk)]
        chosen = rewards[idx]
        out[t] = float(np.mean(np.sort(chosen)[::-1]))  # mean of topk within that random draw
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Eval GFlowNet top-K reward ranking on dataset candidates.")
    parser.add_argument("--gflownet_checkpoint", type=str, default="models/best_gflownet.pth")
    parser.add_argument("--encoder_checkpoint", type=str, default=ENCODER_PARAMS["save_path"])
    parser.add_argument("--data_path", type=str, default=ENCODER_PARAMS["path"])
    parser.add_argument("--cond_n", type=int, default=50, help="Number of condition sequences to score (subset).")
    parser.add_argument("--topk", type=int, default=50, help="Top-K to compute per condition.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_kobs_log", action="store_true", help="Use reward=log(Kobs+eps) (default).")
    parser.add_argument("--n_random_trials", type=int, default=200, help="Random baseline trials for top-K mean reward.")
    parser.add_argument("--max_candidates", type=int, default=0, help="Optionally cap number of candidate sequences (0=no cap).")
    args = parser.parse_args()

    use_log = args.use_kobs_log or True  # default to log(Kobs+eps)

    rng = random.Random(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = ENCODER_PARAMS["device"]

    df = pd.read_csv(args.data_path, sep="\t")
    df_filtered = df[df["Kobs (M-1 s-1)"] >= 0].reset_index(drop=True)

    # Keep only sequences that satisfy inference legality rules.
    seqs_all = [str(s) for s in df_filtered["Sequences"].values]
    valid_mask = [is_valid_sequence(s) for s in seqs_all]
    df_filtered = df_filtered.loc[valid_mask].reset_index(drop=True)

    if len(df_filtered) == 0:
        raise RuntimeError("No valid candidate sequences found after filtering.")

    # Candidates = the full (filtered) dataset.
    if args.max_candidates and args.max_candidates > 0:
        df_candidates = df_filtered.sample(n=args.max_candidates, random_state=args.seed).reset_index(drop=True)
    else:
        df_candidates = df_filtered

    candidate_seqs = [str(s) for s in df_candidates["Sequences"].values]
    candidate_rewards = np.array([reward_from_kobs(k, use_log=use_log) for k in df_candidates["Kobs (M-1 s-1)"].values], dtype=np.float64)

    # Conditions = subset of filtered dataset.
    if args.cond_n >= len(df_filtered):
        df_conditions = df_filtered
    else:
        df_conditions = df_filtered.sample(n=args.cond_n, random_state=args.seed).reset_index(drop=True)

    # Build vocab + models.
    vocab, inv_vocab, vocab_size = build_vocab()
    pad_id = vocab["<pad>"]

    encoder_model = TransformerEncoderModel(
        vocab_size=vocab_size,
        embed_dim=ENCODER_PARAMS["embed_dim"],
        num_heads=ENCODER_PARAMS["num_heads"],
        num_layers=ENCODER_PARAMS["num_layers"],
        dropout=ENCODER_PARAMS["dropout"],
        hidden_dim=ENCODER_PARAMS["hidden_dim"],
        max_len=ENCODER_PARAMS["max_length"] + 3,
        max_relative_distance=16,
    ).to(device)
    enc_ckpt = torch.load(args.encoder_checkpoint, map_location=device)
    encoder_model.load_state_dict(enc_ckpt["model_state"])
    encoder_model.eval()

    decode_model = GFlowNetTransformer(
        embed_dim=ENCODER_PARAMS["embed_dim"],
        hidden_dim=512,
        num_layers=4,
        num_heads=8,
        vocab_size=vocab_size,
        max_len=ENCODER_PARAMS["max_length"] + 3,
    ).to(device)
    decode_model.load_state_dict(torch.load(args.gflownet_checkpoint, map_location=device))
    decode_model.eval()

    # Build candidate seq_idx aligned to the condition embedding length.
    # We determine seq_len from condition embeddings.
    X_conditions = build_sequence_embeddings(
        df_conditions,
        encoder_model,
        vocab,
        inv_vocab,
        device=device,
        spec_token=False,
        cond_mask_prob=0.0,
        cond_mask_specials=False,
    )
    seq_len = X_conditions.shape[1]
    candidate_seq_idx = build_seq_idx(df_candidates, vocab, seq_len=seq_len, device=device)  # [N, L]

    topk_means: List[float] = []
    topk_maxs: List[float] = []
    top1s: List[float] = []

    for b in range(X_conditions.shape[0]):
        Xb = X_conditions[b : b + 1]  # [1, L, E]
        seq_logp = seq_logp_for_condition(
            decode_model=decode_model,
            X_condition=Xb,
            candidate_seq_idx=candidate_seq_idx,
            pad_id=pad_id,
        ).detach().cpu().numpy()

        order = np.argsort(seq_logp)[::-1]
        k = min(args.topk, len(order))
        top_idx = order[:k]
        chosen_rewards = candidate_rewards[top_idx]
        topk_means.append(float(np.mean(chosen_rewards)))
        topk_maxs.append(float(np.max(chosen_rewards)))
        top1s.append(float(chosen_rewards[0]))

    # Random baseline for "topk mean reward": pick topk from random sample of candidates.
    random_trials = sample_random_topk_rewards(rng, candidate_rewards, topk=args.topk, n_trials=args.n_random_trials)

    summary = TopKSummary(
        n_conditions=int(len(topk_means)),
        n_candidates=int(len(candidate_seqs)),
        topk=int(args.topk),
        mean_topk_mean_reward=float(np.mean(topk_means)),
        mean_topk_max_reward=float(np.mean(topk_maxs)),
        mean_top1_reward=float(np.mean(top1s)),
        random_mean_topk_mean_reward=float(np.mean(random_trials)),
    )

    print("\n=== Top-K reward ranking summary ===")
    print(f"Conditions: {summary.n_conditions} / Candidates: {summary.n_candidates} / topK={summary.topk}")
    print(f"Model mean(topk mean reward)  = {summary.mean_topk_mean_reward:.6f}")
    print(f"Model mean(topk max reward)   = {summary.mean_topk_max_reward:.6f}")
    print(f"Model mean(top1 reward)       = {summary.mean_top1_reward:.6f}")
    print(f"Random baseline mean(topk mean reward) = {summary.random_mean_topk_mean_reward:.6f}")


if __name__ == "__main__":
    main()

