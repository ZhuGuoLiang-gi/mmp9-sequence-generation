"""
Fusion command:
  1) Auto-generate condition sequences from MMP9.tsv (Kobs>=0).
  2) Run GFlowNet inference to generate FASTA.
  3) Immediately evaluate reward vs random baseline using eval_gflownet_reward logic.

This is meant to avoid manual "infer -> eval" steps.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import List, Tuple

import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import torch  # type: ignore
from Bio import SeqIO  # type: ignore
from Bio.Seq import Seq  # type: ignore
from Bio.SeqRecord import SeqRecord  # type: ignore

from ..config import ENCODER_PARAMS
from ..inference.infer_gflownet_fasta import is_valid_sequence
from ..models.gflownet_transformer import GFlowNetTransformer
from ..models.transformer_encoder import TransformerEncoderModel
from ..tokenizer import build_vocab
from ..training.train_gflownet import build_sequence_embeddings

# Reuse reward evaluation helpers for consistent reward definition + plotting.
from .eval_gflownet_reward import (
    compute_reward_stats,
    load_reward_table,
    sample_random_baseline,
)


def choose_condition_sequences(
    df_filtered: pd.DataFrame,
    *,
    cond_strategy: str,
    cond_n: int,
    seed: int,
) -> List[str]:
    # Keep only sequences that satisfy the inference validity filter; this reduces wasted condition encoding.
    all_candidates = [str(s) for s in df_filtered["Sequences"].values if is_valid_sequence(str(s))]
    if not all_candidates:
        return []

    rng = random.Random(seed)
    if cond_strategy == "random":
        if cond_n <= len(all_candidates):
            return rng.sample(all_candidates, cond_n)
        # If dataset is small, sample with replacement.
        return [rng.choice(all_candidates) for _ in range(cond_n)]

    if cond_strategy == "topk":
        # Sort by Kobs descending, then take the top N (after validity filtering).
        # df_filtered is already filtered by reward range; use its ordering based on Kobs.
        df_sorted = df_filtered.sort_values("Kobs (M-1 s-1)", ascending=False)
        out = []
        for s in df_sorted["Sequences"].values:
            s = str(s)
            if is_valid_sequence(s):
                out.append(s)
            if len(out) >= cond_n:
                break
        return out

    if cond_strategy == "first":
        out = []
        for s in df_filtered["Sequences"].values:
            s = str(s)
            if is_valid_sequence(s):
                out.append(s)
            if len(out) >= cond_n:
                break
        return out

    raise ValueError(f"Unknown cond_strategy: {cond_strategy}")


def same_dash_position(ref_seq: str, pred_seq: str) -> bool:
    try:
        return ref_seq.index("-") == pred_seq.index("-")
    except ValueError:
        return False


def same_sequence_length(ref_seq: str, pred_seq: str) -> bool:
    return len(ref_seq) == len(pred_seq)


@torch.no_grad()
def run_inference_generate(
    *,
    input_sequences: List[str],
    encoder_checkpoint: str,
    model_path: str,
    n_sample: int,
    temperature: float,
    spec_token: bool,
    device: str,
    constrain_to_tsv_candidates: bool = False,
    candidate_tsv: str | None = None,
    candidate_kobs_min: float = 0.0,
    candidate_topk_by_logp: int = 0,
    disable_dash_pos_constraint: bool = False,
) -> List[str]:
    vocab, inv_vocab, vocab_size = build_vocab()

    # Encoder model: used only to build input embeddings.
    encoder_model = TransformerEncoderModel(
        vocab_size=vocab_size,
        embed_dim=ENCODER_PARAMS["embed_dim"],
        num_heads=ENCODER_PARAMS["num_heads"],
        num_layers=ENCODER_PARAMS["num_layers"],
        dropout=ENCODER_PARAMS["dropout"],
        hidden_dim=ENCODER_PARAMS["hidden_dim"],
        max_len=ENCODER_PARAMS["max_length"] + 3,
        max_relative_distance=16,
    )
    checkpoint = torch.load(encoder_checkpoint, map_location=device)
    encoder_model.load_state_dict(checkpoint["model_state"])
    encoder_model.to(device).eval()

    # Decoder model: GFlowNet transformer.
    decode_model = GFlowNetTransformer(
        embed_dim=ENCODER_PARAMS["embed_dim"],
        hidden_dim=512,
        num_layers=4,
        num_heads=8,
        vocab_size=vocab_size,
        max_len=ENCODER_PARAMS["max_length"] + 3,
    )
    decode_model.load_state_dict(torch.load(model_path, map_location=device))
    decode_model.to(device).eval()

    df_input = pd.DataFrame({"Sequences": input_sequences})
    existing = set()
    valid_out: List[str] = []

    if constrain_to_tsv_candidates:
        if spec_token:
            raise ValueError("constrain_to_tsv_candidates currently requires spec_token=False.")

        tsv_path = candidate_tsv or ENCODER_PARAMS["path"]
        df_candidates = pd.read_csv(tsv_path, sep="\t")
        df_candidates = df_candidates[df_candidates["Kobs (M-1 s-1)"] >= candidate_kobs_min].reset_index(drop=True)
        candidate_seqs = [str(s) for s in df_candidates["Sequences"].values if is_valid_sequence(str(s))]
        candidate_seqs = sorted(set(candidate_seqs))
        print(f"Candidate pool: {len(candidate_seqs)} sequences (Kobs>={candidate_kobs_min}).")

        pad_id = vocab["<pad>"]

        def build_candidate_seq_idx(seq_len: int, ref_dash_pos: int) -> tuple[torch.Tensor, List[str]]:
            idx_rows: List[List[int]] = []
            seq_out: List[str] = []
            for s in candidate_seqs:
                if s.index("-") != ref_dash_pos:
                    continue
                idx_seq = [vocab.get(tok, vocab["<unk>"]) for tok in s]
                if len(idx_seq) > seq_len:
                    continue
                pad_len = seq_len - len(idx_seq)
                idx_seq = idx_seq + [pad_id] * pad_len
                idx_rows.append(idx_seq)
                seq_out.append(s)
            if not idx_rows:
                return torch.empty(0, seq_len, dtype=torch.long, device=device), []
            return torch.tensor(idx_rows, dtype=torch.long, device=device), seq_out

        for b, cond_seq in enumerate(input_sequences):
            df_input_b = pd.DataFrame({"Sequences": [cond_seq]})
            X_input = build_sequence_embeddings(
                df_input_b,
                encoder_model,
                vocab,
                inv_vocab,
                device=device,
                spec_token=spec_token,
            )
            seq_len = X_input.shape[1]
            ref_dash_pos = cond_seq.index("-")
            candidate_seq_idx, candidate_seq_out = build_candidate_seq_idx(seq_len, ref_dash_pos)
            if candidate_seq_idx.numel() == 0:
                continue

            logits = decode_model(X_input)  # [1, L, V]
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)  # [1, L, V]

            n = candidate_seq_idx.shape[0]
            log_probs_n = log_probs.expand(n, -1, -1)  # [N, L, V]
            token_logp = log_probs_n.gather(2, candidate_seq_idx.unsqueeze(-1)).squeeze(-1)  # [N, L]
            pad_mask = candidate_seq_idx != pad_id  # [N, L]
            denom = pad_mask.sum(dim=1).clamp(min=1)
            seq_logp = (token_logp * pad_mask).sum(dim=1) / denom  # [N]

            if candidate_topk_by_logp and candidate_topk_by_logp > 0 and candidate_topk_by_logp < n:
                topk = candidate_topk_by_logp
                top_idx = torch.topk(seq_logp, k=topk, largest=True).indices
                seq_logp = seq_logp[top_idx]
                candidate_seq_out = [candidate_seq_out[i] for i in top_idx.tolist()]

            probs = torch.softmax(seq_logp / max(temperature, 1e-8), dim=0)
            if n_sample <= len(candidate_seq_out):
                chosen_idx = torch.multinomial(probs, num_samples=n_sample, replacement=False)
            else:
                chosen_idx = torch.multinomial(probs, num_samples=n_sample, replacement=True)

            for idx in chosen_idx.tolist():
                seq_str = candidate_seq_out[idx]
                if not same_sequence_length(cond_seq, seq_str):
                    continue
                if (not disable_dash_pos_constraint) and (not same_dash_position(cond_seq, seq_str)):
                    continue
                if seq_str in existing:
                    continue
                existing.add(seq_str)
                valid_out.append(seq_str)

        return valid_out

    # Default behavior: unconstrained stochastic token sampling.
    X_input = build_sequence_embeddings(
        df_input,
        encoder_model,
        vocab,
        inv_vocab,
        device=device,
        spec_token=spec_token,
    )
    gen_sequences = decode_model.sample(X_input, n_sample=n_sample, temperature=temperature)

    # Flatten + filter validity + de-duplicate.
    for cond_idx, seqs in enumerate(gen_sequences):
        for seq in seqs:
            seq_str = "".join(seq)
            if not same_sequence_length(input_sequences[cond_idx], seq_str):
                continue
            if (not disable_dash_pos_constraint) and (not same_dash_position(input_sequences[cond_idx], seq_str)):
                continue
            if seq_str in existing:
                continue
            existing.add(seq_str)
            if is_valid_sequence(seq_str):
                valid_out.append(seq_str)

    return valid_out


def write_fasta(seqs: List[str], fasta_path: str) -> None:
    out_path = Path(fasta_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    records = [SeqRecord(Seq(seq_str), id=f"generated_seq_{i+1}", description="generated_by_GFlowNet") for i, seq_str in enumerate(seqs)]
    SeqIO.write(records, str(out_path), "fasta")


def main() -> None:
    parser = argparse.ArgumentParser(description="Eval GFlowNet reward (auto condition -> generate -> eval).")

    # Condition generation (auto FASTA)
    parser.add_argument("--cond_strategy", type=str, default="random", choices=["random", "topk", "first"], help="How to pick condition sequences from MMP9.tsv.")
    parser.add_argument("--cond_n", type=int, default=50, help="Number of condition sequences.")

    # Inference parameters
    parser.add_argument("--n_sample", type=int, default=20, help="Number of samples per condition sequence.")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature.")
    parser.add_argument("--spec_token", action="store_true", help="Whether to include special tokens in embeddings.")

    parser.add_argument("--encoder_checkpoint", type=str, default=ENCODER_PARAMS["save_path"], help="Path to encoder checkpoint.")
    parser.add_argument("--model_path", type=str, default="models/best_gflownet.pth", help="Path to GFlowNet checkpoint.")

    # Candidate constraint (to improve strict found/coverage in reward eval).
    parser.add_argument(
        "--constrain_to_tsv_candidates",
        action="store_true",
        help="If set, sample only from (MMP9.tsv, Kobs>=candidate_kobs_min, valid sequence) candidates.",
    )
    parser.add_argument("--candidate_tsv", type=str, default=ENCODER_PARAMS["path"], help="Candidate TSV path (default: data/MMP9.tsv).")
    parser.add_argument("--candidate_kobs_min", type=float, default=0.0, help="Min Kobs for candidate pool (default: 0.0).")
    parser.add_argument("--candidate_topk_by_logp", type=int, default=0, help="If >0, restrict sampling to top-K logp candidates per condition.")
    parser.add_argument(
        "--disable_dash_pos_constraint",
        action="store_true",
        help="Disable the default constraint that generated '-' position must match the condition sequence.",
    )

    # Reward evaluation parameters
    parser.add_argument("--n_random", type=int, default=1000, help="Random baseline sample count.")
    parser.add_argument("--topk", type=int, default=20, help="Top-K mean statistic.")
    parser.add_argument("--seed", type=int, default=42, help="Seed.")

    parser.add_argument("--gen_fasta_out", type=str, default="eval_gflownet/gen_seq_from_condition.fasta", help="Generated FASTA path.")
    parser.add_argument("--out_plot", type=str, default="plot/gflownet_reward_compare_from_condition.png", help="Output plot path.")
    parser.add_argument("--out_csv", type=str, default="plot/gflownet_reward_summary_from_condition.csv", help="Output CSV path.")

    args = parser.parse_args()

    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = ENCODER_PARAMS["device"]

    # 1) Load dataset reward table and pick conditions.
    seq_to_reward, df_filtered = load_reward_table(ENCODER_PARAMS["path"])
    cond_sequences = choose_condition_sequences(
        df_filtered,
        cond_strategy=args.cond_strategy,
        cond_n=args.cond_n,
        seed=args.seed,
    )
    if not cond_sequences:
        raise RuntimeError("No valid condition sequences found after filtering; cannot run inference.")

    print(f"Condition sequences: {len(cond_sequences)} (strategy={args.cond_strategy}, cond_n={args.cond_n})")

    # 2) Inference: generate sequences from conditions.
    generated_valid = run_inference_generate(
        input_sequences=cond_sequences,
        encoder_checkpoint=args.encoder_checkpoint,
        model_path=args.model_path,
        n_sample=args.n_sample,
        temperature=args.temperature,
        spec_token=args.spec_token,
        device=device,
        constrain_to_tsv_candidates=args.constrain_to_tsv_candidates,
        candidate_tsv=args.candidate_tsv,
        candidate_kobs_min=args.candidate_kobs_min,
        candidate_topk_by_logp=args.candidate_topk_by_logp,
        disable_dash_pos_constraint=args.disable_dash_pos_constraint,
    )
    print(f"Generated valid sequences: {len(generated_valid)}")

    # 3) Write generated FASTA and evaluate reward.
    write_fasta(generated_valid, args.gen_fasta_out)
    gen_sequences = generated_valid

    baseline_sequences = sample_random_baseline(
        df_filtered,
        n_random=args.n_random,
        seed=args.seed,
    )

    gen_stats, gen_rewards = compute_reward_stats(gen_sequences, seq_to_reward=seq_to_reward, topk=args.topk)
    base_stats, base_rewards = compute_reward_stats(baseline_sequences, seq_to_reward=seq_to_reward, topk=args.topk)

    print("\n=== Reward Stats ===")
    print(
        f"[Gen]  total={gen_stats.n_total} valid={gen_stats.n_valid} found={gen_stats.n_found} "
        f"coverage={gen_stats.coverage:.3f} mean={gen_stats.mean:.4f} median={gen_stats.median:.4f} "
        f"max={gen_stats.max:.4f} top{args.topk}_mean={gen_stats.topk_mean:.4f}"
    )
    print(
        f"[Rnd]  total={base_stats.n_total} valid={base_stats.n_valid} found={base_stats.n_found} "
        f"coverage={base_stats.coverage:.3f} mean={base_stats.mean:.4f} median={base_stats.median:.4f} "
        f"max={base_stats.max:.4f} top{args.topk}_mean={base_stats.topk_mean:.4f}"
    )

    # Save CSV summary
    out_csv_path = Path(args.out_csv)
    out_csv_path.parent.mkdir(parents=True, exist_ok=True)

    df_out = pd.DataFrame(
        [
            {
                "group": "gen",
                "n_total": gen_stats.n_total,
                "n_valid": gen_stats.n_valid,
                "n_found": gen_stats.n_found,
                "coverage": gen_stats.coverage,
                "mean": gen_stats.mean,
                "median": gen_stats.median,
                "max": gen_stats.max,
                "topk_mean": gen_stats.topk_mean,
                "topk": args.topk,
                "cond_n": args.cond_n,
                "n_sample": args.n_sample,
                "temperature": args.temperature,
                "cond_strategy": args.cond_strategy,
            },
            {
                "group": "random",
                "n_total": base_stats.n_total,
                "n_valid": base_stats.n_valid,
                "n_found": base_stats.n_found,
                "coverage": base_stats.coverage,
                "mean": base_stats.mean,
                "median": base_stats.median,
                "max": base_stats.max,
                "topk_mean": base_stats.topk_mean,
                "topk": args.topk,
                "n_random": args.n_random,
            },
        ]
    )
    df_out.to_csv(str(out_csv_path), index=False)
    print(f"\nSaved summary CSV: {args.out_csv}")

    # Plot reward distributions
    import matplotlib.pyplot as plt  # type: ignore

    out_plot_path = Path(args.out_plot)
    out_plot_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(7.5, 5.5))
    if gen_rewards:
        plt.hist(gen_rewards, bins=40, alpha=0.6, label="GFlowNet gen", density=True)
    if base_rewards:
        plt.hist(base_rewards, bins=40, alpha=0.6, label="Random baseline", density=True)
    plt.title("Reward distribution comparison (from condition generation)")
    plt.xlabel("reward = log(Kobs + 1e-8)")
    plt.ylabel("density")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(str(out_plot_path), dpi=300)
    plt.show()
    print(f"Saved plot: {args.out_plot}")


if __name__ == "__main__":
    main()

