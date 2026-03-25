"""
Tokenizer / vocab / sequence encoding utilities.

Migrated from legacy:
  - legacy_deprecated/train.py
"""

from __future__ import annotations

import random
from typing import Dict, List, Tuple, Optional


AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")
SPECIAL_TOKENS_DEFAULT = ["<pad>", "<eso>", "<unk>", "<mask>"]


def build_vocab(include_dash: bool = True, add_specials: bool = True) -> Tuple[Dict[str, int], Dict[int, str], int]:
    special_tokens = SPECIAL_TOKENS_DEFAULT if add_specials else []
    vocab_tokens = special_tokens + AMINO_ACIDS
    if include_dash:
        vocab_tokens.append("-")

    vocab = {tok: idx for idx, tok in enumerate(vocab_tokens)}
    inv_vocab = {i: t for t, i in vocab.items()}
    return vocab, inv_vocab, len(vocab)


def encode_sequence(seq: str, vocab: Dict[str, int], max_length: int = 12) -> List[int]:
    if seq.count("-") != 1:
        raise ValueError(f"序列 '{seq}' 必须包含且仅包含一个 '-'")

    left, right = seq.split("-")
    core_len = len(left) + 1 + len(right)
    max_core_len = max_length + 1
    if core_len > max_core_len:
        left_len = int(len(left) / (len(left) + len(right)) * max_length)
        right_len = max_length - left_len
        left = left[:left_len]
        right = right[:right_len]

    # total_len = max_length + 1 ('-') + 2 ('<eso>')
    total_len = max_length + 1 + 2
    center_idx = total_len // 2
    left_seq_len = center_idx - 1
    # '-' and right part ends with '<eso>'
    right_seq_len = total_len - center_idx - 2

    left_seq = list(left)[-left_seq_len:] if left_seq_len > 0 else []
    right_seq = list(right)[:right_seq_len] if right_seq_len > 0 else []

    left_pad = center_idx - 1 - len(left_seq)
    right_pad = total_len - center_idx - 2 - len(right_seq)

    seq_tokens = ["<pad>"] * left_pad + ["<eso>"] + left_seq + ["-"] + right_seq + ["<eso>"] + ["<pad>"] * right_pad
    return [vocab.get(tok, vocab["<unk>"]) for tok in seq_tokens]


def apply_random_mask(
    tokens: List[int],
    mask_prob: float = 0.15,
    mask_specials: bool = False,
    inv_vocab: Optional[Dict[int, str]] = None,
) -> Tuple[List[int], List[int]]:
    if inv_vocab is None:
        raise ValueError("apply_random_mask requires inv_vocab (id->token).")

    mask_id = None
    for k, v in inv_vocab.items():
        if v == "<mask>":
            mask_id = k
            break
    if mask_id is None:
        raise KeyError('Missing "<mask>" in inv_vocab')

    masked: List[int] = []
    mask_positions: List[int] = []
    for i, t in enumerate(tokens):
        tok = inv_vocab[t]
        if tok == "<pad>":
            masked.append(t)
        elif tok in ["<eso>", "-"] and not mask_specials:
            masked.append(t)
        else:
            if random.random() < mask_prob:
                masked.append(mask_id)
                mask_positions.append(i)
            else:
                masked.append(t)
    return masked, mask_positions


