"""
Transformer encoder model (relative positional bias) for masked token recovery.

Migrated from legacy:
  - legacy_deprecated/train.py
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn


class MultiheadAttentionWithBias(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, rel_bias: torch.Tensor | None = None) -> torch.Tensor:
        # x: [B, L, D]
        B, L, D = x.size()
        H = self.num_heads
        q = self.q_proj(x).view(B, L, H, -1).transpose(1, 2)  # [B,H,L,d]
        k = self.k_proj(x).view(B, L, H, -1).transpose(1, 2)
        v = self.v_proj(x).view(B, L, H, -1).transpose(1, 2)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # [B,H,L,L]
        if rel_bias is not None:
            attn_scores = attn_scores + rel_bias.unsqueeze(0)  # broadcast over batch

        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        out = torch.matmul(attn_weights, v)  # [B,H,L,d]
        out = out.transpose(1, 2).contiguous().view(B, L, D)
        return self.out_proj(out)


class TransformerEncoderModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        hidden_dim: int = 128,
        max_len: int = 512,
        dropout: float = 0.1,
        max_relative_distance: int = 8,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "self_attn": MultiheadAttentionWithBias(embed_dim, num_heads, dropout),
                        "norm1": nn.LayerNorm(embed_dim),
                        "linear1": nn.Linear(embed_dim, hidden_dim),
                        "linear2": nn.Linear(hidden_dim, embed_dim),
                        "norm2": nn.LayerNorm(embed_dim),
                        "dropout": nn.Dropout(dropout),
                    }
                )
                for _ in range(num_layers)
            ]
        )

        # relative position bias table
        self.max_relative_distance = max_relative_distance
        self.relative_bias = nn.Embedding(2 * max_relative_distance + 1, num_heads)

        self.fc = nn.Linear(embed_dim, vocab_size)

    def _relative_position_bucket(self, relative_position: torch.Tensor) -> torch.Tensor:
        max_d = self.max_relative_distance
        relative_position = relative_position.clamp(-max_d, max_d)
        relative_position_bucket = relative_position + max_d
        return relative_position_bucket

    def compute_relative_bias(self, seq_len: int) -> torch.Tensor:
        # output shape: [H, L, L]
        device = self.relative_bias.weight.device
        q_pos = torch.arange(seq_len, device=device)
        k_pos = torch.arange(seq_len, device=device)
        rel_pos = q_pos.unsqueeze(1) - k_pos.unsqueeze(0)  # [L, L]
        rel_bucket = self._relative_position_bucket(rel_pos)
        values = self.relative_bias(rel_bucket)  # [L, L, num_heads]
        values = values.permute(2, 0, 1)  # [num_heads, L, L]
        return values

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L] token ids
        B, L = x.shape
        x = self.embedding(x)
        x = self.dropout(x)

        rel_bias = self.compute_relative_bias(L)  # [H, L, L]
        for layer in self.layers:
            attn_out = layer["self_attn"](x, rel_bias)
            x = x + layer["dropout"](attn_out)
            x = layer["norm1"](x)

            ff_out = layer["linear2"](torch.relu(layer["linear1"](x)))
            x = x + layer["dropout"](ff_out)
            x = layer["norm2"](x)

        return self.fc(x)  # [B, L, vocab_size]

