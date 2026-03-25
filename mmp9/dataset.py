"""
Dataset / collate utilities.

Migrated from legacy:
  - legacy_deprecated/train.py
"""

from __future__ import annotations

from typing import Dict, List

import torch
from torch.utils.data import Dataset

from .tokenizer import apply_random_mask, encode_sequence


class SequenceDataset(Dataset):
    def __init__(
        self,
        df,
        vocab: Dict[str, int],
        inv_vocab: Dict[int, str] | None = None,
        mask_prob: float = 0.15,
        n_mask: int = 5,
        max_length: int = 12,
        mask_specials: bool = False,
    ):
        self.df = df
        self.vocab = vocab
        self.inv_vocab = inv_vocab if inv_vocab is not None else {i: t for t, i in vocab.items()}
        self.mask_prob = mask_prob
        self.n_mask = n_mask
        self.max_length = max_length
        self.mask_specials = mask_specials

        # Pre-encode for speed.
        self.encoded_seqs = [encode_sequence(seq, vocab, max_length) for seq in df["Sequences"]]

    def __len__(self) -> int:
        return len(self.df) * self.n_mask

    def __getitem__(self, idx: int):
        seq_idx = idx // self.n_mask
        tokens = self.encoded_seqs[seq_idx]
        masked_tokens, mask_positions = apply_random_mask(
            tokens, mask_prob=self.mask_prob, mask_specials=self.mask_specials, inv_vocab=self.inv_vocab
        )
        return torch.tensor(masked_tokens), torch.tensor(tokens), mask_positions


def collate_fn(batch):
    masked_batch, target_batch, maskpos_batch = zip(*batch)
    masked_batch = torch.stack(masked_batch, dim=0)
    target_batch = torch.stack(target_batch, dim=0)
    return masked_batch, target_batch, maskpos_batch

