"""
Embedding utilities.

Migrated from legacy:
  - legacy layout: extract_embedding.py
"""

from __future__ import annotations

import random

import torch  # type: ignore

from ..tokenizer import encode_sequence


def get_sequence_embedding(
    model,
    sequence: str,
    vocab,
    inv_vocab,
    max_length: int = 12,
    device: str = "cpu",
    spec_token: bool = False,
    mask_prob: float = 0.0,
    mask_specials: bool = False,
):
    """
    Input: raw sequence string containing exactly one '-'.
    Output: Transformer encoder hidden states.

    If spec_token=False, returns embedding aligned only to non-special tokens.
    If spec_token=True, returns full embedding including special tokens.
    """

    model.eval()
    model.to(device)

    with torch.no_grad():
        tokens = encode_sequence(sequence, vocab, max_length)
        if mask_prob > 0:
            mask_id = None
            for tok_id, tok in inv_vocab.items():
                if tok == "<mask>":
                    mask_id = tok_id
                    break
            if mask_id is None:
                raise KeyError('Missing "<mask>" token in inv_vocab')

            # token-level masking on the condition side.
            # Follow encoder default: if mask_specials=False, do not mask '-' and '<eso>'.
            masked_tokens = list(tokens)
            for i, tok_id in enumerate(tokens):
                tok = inv_vocab[tok_id]
                if tok in ["<pad>", "<eso>"]:
                    continue
                if (not mask_specials) and tok == "-":
                    continue
                if random.random() < mask_prob:
                    masked_tokens[i] = mask_id
            tokens = masked_tokens

        tokens_tensor = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)  # [1,L]

        x = model.embedding(tokens_tensor)  # [1,L,D]
        x = model.dropout(x)
        L = tokens_tensor.size(1)
        rel_bias = model.compute_relative_bias(L)  # [H,L,L]

        for layer in model.layers:
            attn_out = layer["self_attn"](x, rel_bias)
            x = x + layer["dropout"](attn_out)
            x = layer["norm1"](x)
            ff_out = layer["linear2"](torch.relu(layer["linear1"](x)))
            x = x + layer["dropout"](ff_out)
            x = layer["norm2"](x)

        x = x.squeeze(0)  # [L, D]

        if spec_token:
            return x

        # aligned embeddings: exclude <pad>, <eso>
        aligned_embeddings = []
        seq_idx = 0
        for i, tok_id in enumerate(tokens):
            tok = inv_vocab[tok_id]
            if tok in ["<pad>", "<eso>"]:
                continue
            aligned_embeddings.append(x[i])
            seq_idx += 1
            if seq_idx >= len(sequence):
                break

        return torch.stack(aligned_embeddings, dim=0)

