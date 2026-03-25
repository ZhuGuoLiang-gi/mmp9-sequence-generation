"""
Evaluation: GFlowNet (Phase 2) reward statistics for generated FASTA.

This script compares:
  1) generated sequences' reward distribution
  2) a random sampling baseline from the same dataset (MMP9.tsv)
"""

from __future__ import annotations

import argparse
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from Bio import SeqIO

from ..config import ENCODER_PARAMS


DEFAULT_GEN_FASTA = "eval_gflownet/gen_seq.fasta"


def is_valid_sequence(seq: str) -> bool:
    # Keep same filtering logic as legacy inference:
    # - exactly one '-' not at ends
    # - only uppercase letters and '-'
    if not re.fullmatch(r"[A-Z-]+", seq):
        return False
    if seq.startswith("-") or seq.endswith("-"):
        return False
    if seq.count("-") != 1:
        return False
    return True


def load_reward_table(tsv_path: str) -> Tuple[Dict[str, float], pd.DataFrame]:
    """
    Returns:
      - seq_to_reward: sequence -> reward (log(Kobs + 1e-8))
      - df: filtered dataframe (Kobs > 0) with original columns
    """
    df = pd.read_csv(tsv_path, sep="\t")
    # Reward definition should be consistent with training.
    df_filtered = df[df["Kobs (M-1 s-1)"] >= 0].reset_index(drop=True)
    seq_to_reward: Dict[str, float] = {}
    for seq, kobs in zip(df_filtered["Sequences"].values, df_filtered["Kobs (M-1 s-1)"].values):
        # Reward shaping matches training: log(Kobs + 1e-8)
        seq_to_reward[str(seq)] = float(np.log(float(kobs) + 1e-8))
    return seq_to_reward, df_filtered


def read_fasta_sequences(fasta_path: str) -> List[str]:
    records = list(SeqIO.parse(fasta_path, "fasta"))
    return [str(rec.seq) for rec in records]


@dataclass(frozen=True)
class RewardStats:
    n_total: int
    n_valid: int
    validity: float
    n_unique: int
    uniqueness: float
    n_novel: int
    novelty: float
    n_found: int
    coverage: float
    mean: float
    median: float
    max: float
    topk_mean: float


def compute_reward_stats(
    sequences: Sequence[str],
    *,
    seq_to_reward: Dict[str, float],
    topk: int,
) -> Tuple[RewardStats, List[float]]:
    valid = [s for s in sequences if is_valid_sequence(s)]
    unique_valid = list(set(valid))
    known_set = set(seq_to_reward.keys())
    n_novel = sum(1 for s in unique_valid if s not in known_set)
    found_rewards: List[float] = []
    n_found = 0
    for s in valid:
        if s in seq_to_reward:
            n_found += 1
            found_rewards.append(seq_to_reward[s])

    if not found_rewards:
        # Avoid crashes; return NaNs and empty list.
        stats = RewardStats(
            n_total=len(sequences),
            n_valid=len(valid),
            validity=float(len(valid)) / float(len(sequences)) if len(sequences) > 0 else 0.0,
            n_unique=len(unique_valid),
            uniqueness=float(len(unique_valid)) / float(len(valid)) if len(valid) > 0 else 0.0,
            n_novel=n_novel,
            novelty=float(n_novel) / float(len(unique_valid)) if len(unique_valid) > 0 else 0.0,
            n_found=0,
            coverage=0.0,
            mean=float("nan"),
            median=float("nan"),
            max=float("nan"),
            topk_mean=float("nan"),
        )
        return stats, []

    rewards = np.array(found_rewards, dtype=np.float64)
    rewards_sorted = np.sort(rewards)[::-1]  # descending
    k = min(topk, len(rewards_sorted))
    topk_mean = float(np.mean(rewards_sorted[:k])) if k > 0 else float("nan")

    stats = RewardStats(
        n_total=len(sequences),
        n_valid=len(valid),
        validity=float(len(valid)) / float(len(sequences)) if len(sequences) > 0 else 0.0,
        n_unique=len(unique_valid),
        uniqueness=float(len(unique_valid)) / float(len(valid)) if len(valid) > 0 else 0.0,
        n_novel=n_novel,
        novelty=float(n_novel) / float(len(unique_valid)) if len(unique_valid) > 0 else 0.0,
        n_found=n_found,
        coverage=float(n_found) / float(len(valid)) if len(valid) > 0 else 0.0,
        mean=float(np.mean(rewards)),
        median=float(np.median(rewards)),
        max=float(np.max(rewards)),
        topk_mean=topk_mean,
    )
    return stats, found_rewards


def sample_random_baseline(
    df_filtered: pd.DataFrame,
    *,
    n_random: int,
    seed: int,
) -> List[str]:
    # Baseline should be comparable to generation outputs: apply the same validity filter.
    seqs = [str(s) for s in df_filtered["Sequences"].values if is_valid_sequence(str(s))]
    if not seqs:
        return []

    rng = random.Random(seed)
    if n_random <= len(seqs):
        return rng.sample(seqs, n_random)
    # If dataset is smaller than requested, sample with replacement.
    return [rng.choice(seqs) for _ in range(n_random)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate GFlowNet reward for generated FASTA.")
    parser.add_argument(
        "--gen_fasta",
        type=str,
        default=DEFAULT_GEN_FASTA,
        help=f"Generated FASTA path (default: {DEFAULT_GEN_FASTA}).",
    )
    parser.add_argument("--n_random", type=int, default=1000, help="Number of random baseline samples.")
    parser.add_argument(
        "--random_baseline",
        type=str,
        default="dataset",
        choices=["dataset"],
        help="Random baseline source.",
    )
    parser.add_argument("--topk", type=int, default=20, help="Top-K used for Top-K mean statistic.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")

    parser.add_argument("--out_plot", type=str, default="plot/gflownet_reward_compare.png", help="Output plot path.")
    parser.add_argument("--out_csv", type=str, default="plot/gflownet_reward_summary.csv", help="Output CSV path.")

    args = parser.parse_args()

    # Allow a convenience fallback when the provided default directory doesn't exist.
    if args.gen_fasta == DEFAULT_GEN_FASTA and not os.path.exists(args.gen_fasta) and os.path.exists("gen_seq.fasta"):
        print(f"[warn] default gen_fasta not found: {args.gen_fasta}, fallback to: gen_seq.fasta")
        args.gen_fasta = "gen_seq.fasta"

    if not os.path.exists(args.gen_fasta):
        raise FileNotFoundError(f"gen_fasta not found: {args.gen_fasta}")

    np.random.seed(args.seed)
    random.seed(args.seed)

    # 1) Load reward table from dataset (MMP9.tsv)
    seq_to_reward, df_filtered = load_reward_table(ENCODER_PARAMS["path"])
    print(f"Loaded reward table from {ENCODER_PARAMS['path']}: {len(seq_to_reward)} sequences (Kobs>=0).")

    # 2) Load generated sequences
    gen_sequences = read_fasta_sequences(args.gen_fasta)
    print(f"Loaded generated FASTA: {len(gen_sequences)} sequences from {args.gen_fasta}")

    # 3) Random baseline
    if args.random_baseline == "dataset":
        baseline_sequences = sample_random_baseline(df_filtered, n_random=args.n_random, seed=args.seed)
    else:
        baseline_sequences = []

    print(f"Random baseline: {len(baseline_sequences)} sequences sampled (n_random={args.n_random}).")

    # 4) Compute stats
    gen_stats, gen_rewards = compute_reward_stats(gen_sequences, seq_to_reward=seq_to_reward, topk=args.topk)
    base_stats, base_rewards = compute_reward_stats(
        baseline_sequences, seq_to_reward=seq_to_reward, topk=args.topk
    )

    print("\n=== Reward Stats ===")
    print(
        f"[Gen]  total={gen_stats.n_total} valid={gen_stats.n_valid} validity={gen_stats.validity:.3f} "
        f"unique={gen_stats.n_unique} uniqueness={gen_stats.uniqueness:.3f} "
        f"novel={gen_stats.n_novel} novelty={gen_stats.novelty:.3f} "
        f"found={gen_stats.n_found} "
        f"coverage={gen_stats.coverage:.3f} mean={gen_stats.mean:.4f} median={gen_stats.median:.4f} "
        f"max={gen_stats.max:.4f} top{args.topk}_mean={gen_stats.topk_mean:.4f}"
    )
    print(
        f"[Rnd]  total={base_stats.n_total} valid={base_stats.n_valid} validity={base_stats.validity:.3f} "
        f"unique={base_stats.n_unique} uniqueness={base_stats.uniqueness:.3f} "
        f"novel={base_stats.n_novel} novelty={base_stats.novelty:.3f} "
        f"found={base_stats.n_found} "
        f"coverage={base_stats.coverage:.3f} mean={base_stats.mean:.4f} median={base_stats.median:.4f} "
        f"max={base_stats.max:.4f} top{args.topk}_mean={base_stats.topk_mean:.4f}"
    )

    # 5) Save CSV summary
    out_csv_path = Path(args.out_csv)
    out_csv_path.parent.mkdir(parents=True, exist_ok=True)

    df_out = pd.DataFrame(
        [
            {
                "group": "gen",
                "n_total": gen_stats.n_total,
                "n_valid": gen_stats.n_valid,
                "validity": gen_stats.validity,
                "n_unique": gen_stats.n_unique,
                "uniqueness": gen_stats.uniqueness,
                "n_novel": gen_stats.n_novel,
                "novelty": gen_stats.novelty,
                "n_found": gen_stats.n_found,
                "coverage": gen_stats.coverage,
                "mean": gen_stats.mean,
                "median": gen_stats.median,
                "max": gen_stats.max,
                "topk_mean": gen_stats.topk_mean,
                "topk": args.topk,
            },
            {
                "group": "random",
                "n_total": base_stats.n_total,
                "n_valid": base_stats.n_valid,
                "validity": base_stats.validity,
                "n_unique": base_stats.n_unique,
                "uniqueness": base_stats.uniqueness,
                "n_novel": base_stats.n_novel,
                "novelty": base_stats.novelty,
                "n_found": base_stats.n_found,
                "coverage": base_stats.coverage,
                "mean": base_stats.mean,
                "median": base_stats.median,
                "max": base_stats.max,
                "topk_mean": base_stats.topk_mean,
                "topk": args.topk,
            },
        ]
    )
    df_out.to_csv(out_csv_path, index=False)
    print(f"\nSaved summary CSV: {args.out_csv}")

    # 6) Plot: reward distributions
    out_plot_path = Path(args.out_plot)
    out_plot_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(7.5, 5.5))
    if gen_rewards:
        plt.hist(gen_rewards, bins=40, alpha=0.6, label="GFlowNet gen", density=True)
    if base_rewards:
        plt.hist(base_rewards, bins=40, alpha=0.6, label="Random baseline", density=True)
    plt.title("Reward distribution comparison")
    plt.xlabel("reward = log(Kobs + 1e-8)")
    plt.ylabel("density")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out_plot, dpi=300)
    plt.show()
    print(f"Saved plot: {args.out_plot}")


if __name__ == "__main__":
    main()

