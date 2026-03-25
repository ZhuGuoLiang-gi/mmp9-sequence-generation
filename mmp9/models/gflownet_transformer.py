"""
GFlowNet transformer decoder.

Skeleton for Phase 2.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..tokenizer import build_vocab


class GFlowNetTransformer(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int = 512,
        num_layers: int = 4,
        num_heads: int = 8,
        vocab_size: int | None = None,
        max_len: int = 15,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.max_len = max_len
        self.vocab_size = vocab_size

        self.input_proj = nn.Linear(embed_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        if vocab_size is None:
            raise ValueError("vocab_size must be provided for GFlowNetTransformer.")
        self.output_proj = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, seq_len, embed_dim]
        x = self.input_proj(x)
        x = x.transpose(0, 1)  # [seq_len, B, hidden]
        x = self.transformer(x)
        x = x.transpose(0, 1)  # [B, seq_len, hidden]
        logits = self.output_proj(x)  # [B, seq_len, vocab]
        return logits

    @torch.no_grad()
    def sample(self, x: torch.Tensor, n_sample: int = 5, temperature: float = 1.0):
        vocab, inv_vocab, vocab_size = build_vocab()

        self.eval()
        batch_size, seq_len, _ = x.shape
        all_samples = []

        special_tokens = {"<pad>", "<mask>", "<eso>"}

        logits = self.forward(x)  # [B, seq_len, vocab]
        probs = F.softmax(logits / temperature, dim=-1)  # [B, seq_len, vocab]

        for b in range(batch_size):
            sampled_set = set()
            samples_b = []
            attempts = 0
            while len(samples_b) < n_sample and attempts < n_sample * 10:
                seq_idx = torch.multinomial(probs[b], 1).squeeze(-1)  # [seq_len]
                seq_tokens = [inv_vocab[idx.item()] for idx in seq_idx if inv_vocab[idx.item()] not in special_tokens]
                seq_tuple = tuple(seq_tokens)
                if seq_tuple not in sampled_set:
                    sampled_set.add(seq_tuple)
                    samples_b.append(seq_tokens)
                attempts += 1

            while len(samples_b) < n_sample:
                seq_idx = torch.multinomial(probs[b], 1).squeeze(-1)
                seq_tokens = [inv_vocab[idx.item()] for idx in seq_idx if inv_vocab[idx.item()] not in special_tokens]
                samples_b.append(seq_tokens)

            all_samples.append(samples_b)

        return all_samples

