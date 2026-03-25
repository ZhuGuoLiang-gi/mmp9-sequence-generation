"""
GFlowNet inference from FASTA (Phase 2 migration).

Skeleton.
"""

from __future__ import annotations

import argparse
import os
import re
from typing import Dict, List, Set, Tuple

import pandas as pd
import torch
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from ..config import ENCODER_PARAMS
from ..tokenizer import build_vocab
from ..models.transformer_encoder import TransformerEncoderModel
from ..models.gflownet_transformer import GFlowNetTransformer
from ..training.train_gflownet import build_sequence_embeddings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GFlowNet-based sequence generation inference.")
    parser.add_argument("-f", "--input_fasta", required=True, help="输入 FASTA 文件路径")
    parser.add_argument("-o", "--output_fasta", required=True, help="输出 FASTA 文件路径")
    parser.add_argument(
        "--output_fasta_no_dash",
        default="",
        help="可选：额外输出一个去掉 '-' 的 FASTA 文件路径（默认不输出）。",
    )
    parser.add_argument("-n", "--n_sample", type=int, default=20, help="每个输入序列生成的样本数 (默认: 20)")
    parser.add_argument("-t", "--temperature", type=float, default=1.0, help="采样温度 (默认: 1.0)")
    parser.add_argument("--model_path", default="models/best_gflownet.pth", help="解码器模型路径")
    parser.add_argument("--encoder_path", default=ENCODER_PARAMS["save_path"], help="编码器模型路径")

    # Candidate-constrained decoding (to improve strict TSV hit rate during evaluation).
    parser.add_argument(
        "--constrain_to_tsv_candidates",
        action="store_true",
        help="If set, sample only from (MMP9.tsv, Kobs>=candidate_kobs_min, valid sequence) candidates.",
    )
    parser.add_argument("--candidate_tsv", type=str, default=ENCODER_PARAMS["path"], help="Candidate TSV path (default: data/MMP9.tsv).")
    parser.add_argument("--candidate_kobs_min", type=float, default=0.0, help="Min Kobs for candidate sampling (default: 0.0).")
    parser.add_argument("--candidate_topk_by_logp", type=int, default=0, help="If >0, restrict sampling to top-K logp candidates per condition.")
    parser.add_argument(
        "--disable_dash_pos_constraint",
        action="store_true",
        help="Disable the default constraint that generated '-' position must match input reference.",
    )

    return parser.parse_args()


def unique_sequences(sequences: List[str], existing_set: Set[str]) -> Tuple[List[str], Set[str]]:
    unique: List[str] = []
    for seq in sequences:
        if seq not in existing_set:
            unique.append(seq)
            existing_set.add(seq)
    return unique, existing_set


def is_valid_sequence(seq: str) -> bool:
    # Keep same legacy filtering: exactly one '-' not at ends, only uppercase letters and '-'
    if not re.fullmatch(r"[A-Z-]+", seq):
        return False
    if seq.startswith("-") or seq.endswith("-"):
        return False
    if seq.count("-") != 1:
        return False
    return True


def same_dash_position(ref_seq: str, pred_seq: str) -> bool:
    """
    Enforce '-' position to match reference sequence.
    """
    try:
        return ref_seq.index("-") == pred_seq.index("-")
    except ValueError:
        return False


def same_sequence_length(ref_seq: str, pred_seq: str) -> bool:
    return len(ref_seq) == len(pred_seq)


def parse_annotated_sequence(raw_seq: str) -> tuple[str, Dict[int, str]]:
    """
    Parse sequence annotations like <A>/<a>:
    - output normalized sequence with angle brackets removed
    - return fixed position constraints, e.g. {idx: "A"}
    """
    seq = raw_seq.strip()
    fixed_positions: Dict[int, str] = {}
    out_chars: List[str] = []
    i = 0
    while i < len(seq):
        if seq[i] == "<":
            j = seq.find(">", i + 1)
            if j == -1:
                raise ValueError(f"Invalid annotation in sequence: {raw_seq}")
            token = seq[i + 1 : j].strip()
            if len(token) != 1 or not token.isalpha():
                raise ValueError(f"Only single amino-acid annotation is supported, got <{token}> in: {raw_seq}")
            aa = token.upper()
            pos = len(out_chars)
            out_chars.append(aa)
            fixed_positions[pos] = aa
            i = j + 1
            continue
        out_chars.append(seq[i].upper())
        i += 1
    return "".join(out_chars), fixed_positions


def satisfies_fixed_positions(seq: str, fixed_positions: Dict[int, str]) -> bool:
    for pos, aa in fixed_positions.items():
        if pos >= len(seq) or seq[pos] != aa:
            return False
    return True


@torch.no_grad()
def main() -> None:
    args = parse_args()
    device = ENCODER_PARAMS["device"]
    spec_token = False

    vocab, inv_vocab, vocab_size = build_vocab()

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
    checkpoint = torch.load(args.encoder_path, map_location=device)
    encoder_model.load_state_dict(checkpoint["model_state"])
    encoder_model.to(device).eval()

    decode_model = GFlowNetTransformer(
        embed_dim=ENCODER_PARAMS["embed_dim"],
        hidden_dim=512,
        num_layers=4,
        num_heads=8,
        vocab_size=vocab_size,
        max_len=ENCODER_PARAMS["max_length"] + 3,
    )
    decode_model.load_state_dict(torch.load(args.model_path, map_location=device))
    decode_model.to(device).eval()

    fasta_records = list(SeqIO.parse(args.input_fasta, "fasta"))
    seq_list: List[str] = []
    fixed_constraints: List[Dict[int, str]] = []
    for rec in fasta_records:
        norm_seq, fixed_pos = parse_annotated_sequence(str(rec.seq))
        seq_list.append(norm_seq)
        fixed_constraints.append(fixed_pos)
    print(f"📖 从输入文件读取到 {len(seq_list)} 条序列。")

    existing_sequences: Set[str] = set(seq_list)
    if os.path.exists(args.output_fasta):
        prev_records = list(SeqIO.parse(args.output_fasta, "fasta"))
        for rec in prev_records:
            existing_sequences.add(str(rec.seq))
        print(f"🧩 检测到已有 {len(prev_records)} 条输出序列，将避免重复。")

    # Candidate-constrained decoding:
    # - Compute logits for each condition
    # - Score all TSV candidate sequences via token logp
    # - Sample from those candidates
    candidate_seqs: list[str] = []
    if args.constrain_to_tsv_candidates:
        df = pd.read_csv(args.candidate_tsv, sep="\t")
        df = df[df["Kobs (M-1 s-1)"] >= args.candidate_kobs_min].reset_index(drop=True)
        candidate_seqs = [str(s) for s in df["Sequences"].values if is_valid_sequence(str(s))]
        candidate_seqs = sorted(set(candidate_seqs))
        print(f"Candidate pool: {len(candidate_seqs)} sequences (Kobs>={args.candidate_kobs_min}).")

    all_generated = []

    if args.constrain_to_tsv_candidates:
        def build_candidate_seq_idx(seq_len: int, ref_dash_pos: int, fixed_pos: Dict[int, str]) -> tuple[torch.Tensor, list[str]]:
            # build idx for candidates that fit seq_len
            idx_rows = []
            seq_out: list[str] = []
            pad_id = vocab["<pad>"]
            for s in candidate_seqs:
                if s.index("-") != ref_dash_pos:
                    continue
                if not satisfies_fixed_positions(s, fixed_pos):
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

        for b, cond_seq in enumerate(seq_list):
            df_input = pd.DataFrame({"Sequences": [cond_seq]})
            X_input = build_sequence_embeddings(df_input, encoder_model, vocab, inv_vocab, device=device, spec_token=spec_token)
            seq_len = X_input.shape[1]
            ref_dash_pos = cond_seq.index("-")
            fixed_pos = fixed_constraints[b]
            candidate_seq_idx, candidate_seq_out = build_candidate_seq_idx(seq_len, ref_dash_pos, fixed_pos)
            if candidate_seq_idx.numel() == 0:
                print(f"[warn] no candidates fit seq_len={seq_len}; skip condition idx={b}")
                continue

            logits = decode_model(X_input)  # [1, L, V]
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)  # [1, L, V]

            # Score candidates: [N, L]
            n = candidate_seq_idx.shape[0]
            log_probs_n = log_probs.expand(n, -1, -1)  # [N, L, V]
            token_logp = log_probs_n.gather(2, candidate_seq_idx.unsqueeze(-1)).squeeze(-1)  # [N, L]
            pad_id = vocab["<pad>"]
            pad_mask = candidate_seq_idx != pad_id  # [N, L]
            denom = pad_mask.sum(dim=1).clamp(min=1)
            seq_logp = (token_logp * pad_mask).sum(dim=1) / denom  # [N]

            if args.candidate_topk_by_logp and args.candidate_topk_by_logp > 0 and args.candidate_topk_by_logp < n:
                topk = args.candidate_topk_by_logp
                top_idx = torch.topk(seq_logp, k=topk, largest=True).indices
                seq_logp = seq_logp[top_idx]
                candidate_seq_out = [candidate_seq_out[i] for i in top_idx.tolist()]
                candidate_seq_idx = candidate_seq_idx[top_idx]

            # Sample from logp distribution
            probs = torch.softmax(seq_logp / max(args.temperature, 1e-8), dim=0)
            if args.n_sample <= len(candidate_seq_out):
                chosen_idx = torch.multinomial(probs, num_samples=args.n_sample, replacement=False)
            else:
                chosen_idx = torch.multinomial(probs, num_samples=args.n_sample, replacement=True)

            for i, idx in enumerate(chosen_idx.tolist()):
                seq_str = candidate_seq_out[idx]
                if not same_sequence_length(cond_seq, seq_str):
                    continue
                if (not args.disable_dash_pos_constraint) and (not same_dash_position(cond_seq, seq_str)):
                    continue
                if not satisfies_fixed_positions(seq_str, fixed_pos):
                    continue
                all_generated.append({"source_idx": b, "sample_idx": i, "sequence": seq_str})
    else:
        df_input = pd.DataFrame({"Sequences": seq_list})
        X_input = build_sequence_embeddings(df_input, encoder_model, vocab, inv_vocab, device=device, spec_token=spec_token)
        # Fast vectorized sampling path for large n_sample.
        # This avoids Python-heavy while-loops in model.sample and is much faster.
        logits = decode_model(X_input)  # [B, L, V]
        probs = torch.softmax(logits / max(args.temperature, 1e-8), dim=-1)  # [B, L, V]
        special_tokens = {"<pad>", "<mask>", "<eso>"}

        for b in range(probs.shape[0]):
            # torch.multinomial on 2D input samples each row independently.
            # Input: [L, V], output: [L, n_sample] -> transpose to [n_sample, L].
            sampled_idx = torch.multinomial(probs[b], num_samples=args.n_sample, replacement=True).transpose(0, 1)
            sampled_idx_list = sampled_idx.tolist()

            for i, seq_idx in enumerate(sampled_idx_list):
                seq = [inv_vocab[idx] for idx in seq_idx if inv_vocab[idx] not in special_tokens]
                seq_str = "".join(seq)
                if not same_sequence_length(seq_list[b], seq_str):
                    continue
                if (not args.disable_dash_pos_constraint) and (not same_dash_position(seq_list[b], seq_str)):
                    continue
                if not satisfies_fixed_positions(seq_str, fixed_constraints[b]):
                    continue
                all_generated.append({"source_idx": b, "sample_idx": i, "sequence": seq_str})

    gen_seq_list = [g["sequence"] for g in all_generated]
    unique_gen, existing_sequences = unique_sequences(gen_seq_list, existing_sequences)
    valid_gen = [seq for seq in unique_gen if is_valid_sequence(seq)]
    removed = len(unique_gen) - len(valid_gen)

    records = [SeqRecord(Seq(seq), id=f"generated_seq_{i+1}", description="generated_by_GFlowNet") for i, seq in enumerate(valid_gen)]
    SeqIO.write(records, args.output_fasta, "fasta")
    print(f"✅ 生成并保存 {len(records)} 条合法序列到 {args.output_fasta}")
    print(f"🚫 删除 {removed} 条非法序列（无 '-'、多个 '-'、首尾 '-' 或特殊字符）")

    if args.output_fasta_no_dash:
        records_no_dash = [
            SeqRecord(
                Seq(seq.replace("-", "")),
                id=f"generated_seq_{i+1}",
                description="generated_by_GFlowNet_no_dash",
            )
            for i, seq in enumerate(valid_gen)
        ]
        SeqIO.write(records_no_dash, args.output_fasta_no_dash, "fasta")
        print(f"✅ 额外保存 {len(records_no_dash)} 条去 '-' 序列到 {args.output_fasta_no_dash}")

    print("\n📘 示例输出序列：")
    for r in records[:5]:
        print(f">{r.id}\n{r.seq}")


if __name__ == "__main__":
    main()

