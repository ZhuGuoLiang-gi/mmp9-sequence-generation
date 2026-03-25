"""
Score FASTA sequences by GFlowNet seq_logp (model preference).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd  # type: ignore
import torch  # type: ignore
from Bio import SeqIO  # type: ignore
from Bio.Seq import Seq  # type: ignore
from Bio.SeqRecord import SeqRecord  # type: ignore

from ..config import ENCODER_PARAMS
from ..models.gflownet_transformer import GFlowNetTransformer
from ..models.transformer_encoder import TransformerEncoderModel
from ..tokenizer import build_vocab, encode_sequence
from ..training.train_gflownet import build_seq_idx


def seq_logp_from_logits(logits: torch.Tensor, sequences: torch.Tensor, *, pad_id: int) -> torch.Tensor:
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    token_logp = log_probs.gather(2, sequences.unsqueeze(-1)).squeeze(-1)  # [B, L]
    pad_mask = sequences != pad_id
    denom = pad_mask.sum(dim=1).clamp(min=1)
    return (token_logp * pad_mask).sum(dim=1) / denom


@torch.no_grad()
def build_sequence_embeddings_fast(
    seqs: list[str],
    model: TransformerEncoderModel,
    vocab,
    inv_vocab,
    *,
    device: str,
    max_length: int,
) -> torch.Tensor:
    """
    Faster batched variant of sequence embedding extraction:
    - tokenize all sequences once
    - one encoder forward on [B, L]
    - remove special tokens (<pad>, <eso>) per sequence
    - pad to [B, L_aligned, D]
    """
    token_rows = [encode_sequence(seq, vocab, max_length=max_length) for seq in seqs]
    tokens = torch.tensor(token_rows, dtype=torch.long, device=device)  # [B, L]

    x = model.embedding(tokens)
    x = model.dropout(x)
    rel_bias = model.compute_relative_bias(tokens.shape[1])
    for layer in model.layers:
        attn_out = layer["self_attn"](x, rel_bias)
        x = x + layer["dropout"](attn_out)
        x = layer["norm1"](x)
        ff_out = layer["linear2"](torch.relu(layer["linear1"](x)))
        x = x + layer["dropout"](ff_out)
        x = layer["norm2"](x)

    aligned_rows = []
    max_aligned_len = 0
    for b, seq in enumerate(seqs):
        keep_idx = [i for i, tok_id in enumerate(token_rows[b]) if inv_vocab[tok_id] not in ("<pad>", "<eso>")]
        aligned = x[b, keep_idx, :]
        if aligned.shape[0] > len(seq):
            aligned = aligned[: len(seq), :]
        aligned_rows.append(aligned)
        max_aligned_len = max(max_aligned_len, aligned.shape[0])

    embed_dim = x.shape[-1]
    padded = []
    for emb in aligned_rows:
        pad_len = max_aligned_len - emb.shape[0]
        if pad_len > 0:
            emb = torch.cat([emb, torch.zeros(pad_len, embed_dim, device=device)], dim=0)
        padded.append(emb)
    return torch.stack(padded, dim=0)


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description="Eval GFlowNet: score FASTA sequences by seq_logp.")
    parser.add_argument("--fasta", required=True, help="Input FASTA path.")
    parser.add_argument("--gflownet_checkpoint", type=str, default="models/best_gflownet.pth", help="Path to GFlowNet checkpoint.")
    parser.add_argument("--encoder_checkpoint", type=str, default=ENCODER_PARAMS["save_path"], help="Path to encoder checkpoint.")
    parser.add_argument(
        "--out_fasta",
        type=str,
        default="eval_gflownet/gflownet_seq_logp_scored.fasta",
        help="Output FASTA with seq_logp in header.",
    )
    parser.add_argument("--out_csv", type=str, default="", help="Optional CSV path. Empty means no CSV output.")
    parser.add_argument("--topn", type=int, default=20, help="Print top-N sequences by seq_logp.")
    args = parser.parse_args()

    records = list(SeqIO.parse(args.fasta, "fasta"))
    seqs = [str(r.seq).upper() for r in records]
    ids = [r.id for r in records]
    if not seqs:
        raise RuntimeError(f"No sequences found in FASTA: {args.fasta}")

    device = ENCODER_PARAMS["device"]
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
    encoder_ckpt = torch.load(args.encoder_checkpoint, map_location=device)
    encoder_model.load_state_dict(encoder_ckpt["model_state"])
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

    df = pd.DataFrame({"Sequences": seqs})
    X = build_sequence_embeddings_fast(
        seqs,
        encoder_model,
        vocab,
        inv_vocab,
        device=device,
        max_length=ENCODER_PARAMS["max_length"],
    )
    seq_idx = build_seq_idx(df, vocab, X.shape[1], device=device)
    logits = decode_model(X)
    seq_logp = seq_logp_from_logits(logits, seq_idx, pad_id=pad_id).detach().cpu().numpy()

    out_df = pd.DataFrame({"id": ids, "sequence": seqs, "seq_logp": seq_logp})
    out_df = out_df.sort_values("seq_logp", ascending=False).reset_index(drop=True)

    scored_records = []
    for _, row in out_df.iterrows():
        seq_id = str(row["id"])
        seq_str = str(row["sequence"])
        logp = float(row["seq_logp"])
        scored_records.append(
            SeqRecord(
                Seq(seq_str),
                id=seq_id,
                description=f"seq_logp={logp:.6f}",
            )
        )

    out_fasta = Path(args.out_fasta)
    out_fasta.parent.mkdir(parents=True, exist_ok=True)
    SeqIO.write(scored_records, str(out_fasta), "fasta")

    out_csv = args.out_csv.strip()
    if out_csv:
        out_csv_path = Path(out_csv)
        out_csv_path.parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(out_csv_path, index=False)

    print(f"Scored sequences: {len(out_df)}")
    print(f"Saved FASTA: {out_fasta}")
    if out_csv:
        print(f"Saved CSV: {out_csv}")
    print(f"Top {min(args.topn, len(out_df))}:")
    for i, row in out_df.head(args.topn).iterrows():
        print(f"{i+1:>3d}. {row['id']}\t{row['sequence']}\tseq_logp={row['seq_logp']:.6f}")


if __name__ == "__main__":
    main()

