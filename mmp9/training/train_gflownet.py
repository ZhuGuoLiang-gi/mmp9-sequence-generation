"""
Train GFlowNet (Phase 2).

Skeleton.
"""

from __future__ import annotations

import argparse
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from ..config import ENCODER_PARAMS
from ..tokenizer import build_vocab
from ..models.transformer_encoder import TransformerEncoderModel
from ..models.gflownet_transformer import GFlowNetTransformer
from ..utils.embedding import get_sequence_embedding


DEVICE = ENCODER_PARAMS["device"]
EPOCHS = 100
LR = 1e-4
DECAY_STEP = 100
DECAY_GAMMA = 0.9
SAVE_PATH = "models/best_gflownet.pth"
STATE_PATH = "models/gflownet_training_state.pth"
DATA_PATH = ENCODER_PARAMS["path"]
SPEC_TOKEN = False


def load_and_split_data(path: str, seed: int = 42, split_ratio=(0.8, 0.1, 0.1)):
    np.random.seed(seed)
    df = pd.read_csv(path, sep="\t")
    # Reward definition should be consistent with evaluation:
    # keep Kobs >= 0 so that log(Kobs + eps) is always finite.
    df = df[df["Kobs (M-1 s-1)"] >= 0].reset_index(drop=True)
    print(f"筛选后共有 {len(df)} 条数据。")
    n = len(df)
    indices = np.arange(n)
    np.random.shuffle(indices)
    n_train = int(split_ratio[0] * n)
    n_valid = int(split_ratio[1] * n)
    # n_test = n - n_train - n_valid
    train_idx = indices[:n_train]
    valid_idx = indices[n_train : n_train + n_valid]
    test_idx = indices[n_train + n_valid :]
    train_df = df.iloc[train_idx].reset_index(drop=True)
    valid_df = df.iloc[valid_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)
    return train_df, valid_df, test_df


def build_sequence_embeddings(
    df: pd.DataFrame,
    model: TransformerEncoderModel,
    vocab,
    inv_vocab,
    device: str = DEVICE,
    spec_token: bool = SPEC_TOKEN,
    cond_mask_prob: float = 0.0,
    cond_mask_specials: bool = False,
):
    embeddings = []
    max_seq_len = 0
    for seq in df["Sequences"]:
        emb = get_sequence_embedding(
            model,
            seq,
            vocab,
            inv_vocab,
            max_length=ENCODER_PARAMS["max_length"],
            device=device,
            spec_token=spec_token,
            mask_prob=cond_mask_prob,
            mask_specials=cond_mask_specials,
        )
        embeddings.append(emb)
        if emb.shape[0] > max_seq_len:
            max_seq_len = emb.shape[0]

    embed_dim = embeddings[0].shape[1]
    padded_embeddings = []
    for emb in embeddings:
        pad_len = max_seq_len - emb.shape[0]
        if pad_len > 0:
            pad_tensor = torch.zeros(pad_len, embed_dim, device=device)
            emb_padded = torch.cat([emb, pad_tensor], dim=0)
        else:
            emb_padded = emb
        padded_embeddings.append(emb_padded)

    return torch.stack(padded_embeddings, dim=0)  # [batch, seq_len, embed_dim]


def gflownet_loss(
    logits: torch.Tensor,
    sequences: torch.Tensor,
    rewards: torch.Tensor,
    *,
    pad_id: int,
) -> torch.Tensor:
    # Backward-compatible wrapper: keep existing weighted-NLL behavior.
    # Prefer using `loss_type='weighted_nll'` explicitly after this refactor.
    logp = get_seq_logp(logits, sequences, pad_id=pad_id)
    return weighted_nll_from_seq_logp(logp, rewards)


def get_seq_logp(logits: torch.Tensor, sequences: torch.Tensor, *, pad_id: int) -> torch.Tensor:
    """
    seq_logp: [B] average token log-prob over non-pad positions.
    """
    log_probs = F.log_softmax(logits, dim=-1)
    token_logp = log_probs.gather(2, sequences.unsqueeze(-1)).squeeze(-1)  # [B, L]

    pad_mask = sequences != pad_id  # [B, L] boolean
    denom = pad_mask.sum(dim=1).clamp(min=1)  # [B]
    seq_logp = (token_logp * pad_mask).sum(dim=1) / denom
    return seq_logp


def weighted_nll_from_seq_logp(seq_logp: torch.Tensor, rewards: torch.Tensor) -> torch.Tensor:
    """
    Reward-weighted negative log-likelihood surrogate.
    rewards: log(Kobs + eps). weight = exp(rewards) ~ Kobs.
    """
    weights = torch.exp(rewards)  # [B]
    weights = weights / weights.mean().clamp(min=1e-8)
    return -(weights * seq_logp).mean()


def pairwise_ranking_loss(
    seq_logp: torch.Tensor,
    rewards: torch.Tensor,
    *,
    n_pairs: int,
    tau: float,
) -> torch.Tensor:
    """
    Pairwise logistic ranking loss.

    For sampled pairs (i, j):
      label = sign(reward_i - reward_j) in {-1, 0, +1}
      want label * (logp_i - logp_j) > 0
      loss_ij = softplus(-(logp_diff/tau) * label)
    """
    n = seq_logp.shape[0]
    if n < 2 or n_pairs <= 0:
        return seq_logp.sum() * 0.0

    tau = float(tau)
    tau = max(tau, 1e-8)

    device = seq_logp.device
    i = torch.randint(0, n, (n_pairs,), device=device)
    j = torch.randint(0, n, (n_pairs,), device=device)

    diff = seq_logp[i] - seq_logp[j]  # [n_pairs]
    reward_diff = rewards[i] - rewards[j]
    label = torch.sign(reward_diff)  # {-1,0,+1}

    mask = label != 0
    if mask.sum().item() == 0:
        return seq_logp.sum() * 0.0

    scaled = (diff[mask] / tau) * label[mask].to(dtype=diff.dtype)
    loss_pairs = F.softplus(-scaled)
    return loss_pairs.mean()


def build_seq_idx(df: pd.DataFrame, vocab, seq_len: int, device: str = DEVICE):
    seq_idx = []
    for seq in df["Sequences"]:
        idx_seq = [vocab.get(tok, vocab["<unk>"]) for tok in seq]
        pad_len = seq_len - len(idx_seq)
        idx_seq += [vocab["<pad>"]] * pad_len
        seq_idx.append(idx_seq)
    return torch.tensor(seq_idx, dtype=torch.long, device=device)


def train_gflownet_resume(
    decode_model: GFlowNetTransformer,
    X_train: torch.Tensor,
    seq_train: torch.Tensor,
    rewards_train: torch.Tensor,
    X_valid: torch.Tensor,
    seq_valid: torch.Tensor,
    rewards_valid: torch.Tensor,
    device: str = DEVICE,
    epochs: int = EPOCHS,
    lr: float = LR,
    save_path: str = SAVE_PATH,
    state_path: str = STATE_PATH,
    resume: bool = True,
    pad_id: int = 0,
    loss_type: str = "pairwise_ranking",
    pairwise_tau: float = 1.0,
    pairwise_n_pairs: int = 5000,
    decay_step: int = DECAY_STEP,
    decay_gamma: float = DECAY_GAMMA,
):
    optimizer = optim.Adam(decode_model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=decay_step, gamma=decay_gamma)
    start_epoch = 1
    best_valid_loss = float("inf")

    if resume and os.path.exists(state_path):
        checkpoint = torch.load(state_path, map_location=device)
        decode_model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        start_epoch = checkpoint["epoch"] + 1
        best_valid_loss = checkpoint["best_valid_loss"]
        print(f"发现训练断点，继续从 epoch {start_epoch} 训练")

    for epoch in range(start_epoch, epochs + 1):
        decode_model.train()
        optimizer.zero_grad()
        logits = decode_model(X_train)
        seq_logp = get_seq_logp(logits, seq_train, pad_id=pad_id)
        if loss_type == "weighted_nll":
            loss = weighted_nll_from_seq_logp(seq_logp, rewards_train)
        elif loss_type == "pairwise_ranking":
            loss = pairwise_ranking_loss(seq_logp, rewards_train, n_pairs=pairwise_n_pairs, tau=pairwise_tau)
        else:
            raise ValueError(f"Unknown loss_type: {loss_type}")
        loss.backward()
        optimizer.step()
        scheduler.step()

        decode_model.eval()
        with torch.no_grad():
            val_logits = decode_model(X_valid)
            val_seq_logp = get_seq_logp(val_logits, seq_valid, pad_id=pad_id)
            if loss_type == "weighted_nll":
                val_loss = weighted_nll_from_seq_logp(val_seq_logp, rewards_valid)
            elif loss_type == "pairwise_ranking":
                val_loss = pairwise_ranking_loss(
                    val_seq_logp, rewards_valid, n_pairs=pairwise_n_pairs, tau=pairwise_tau
                )
            else:
                raise ValueError(f"Unknown loss_type: {loss_type}")

        current_lr = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch}: Train loss={loss.item():.4f}, Valid loss={val_loss.item():.4f}, lr={current_lr:.6f}")

        if val_loss.item() < best_valid_loss:
            best_valid_loss = val_loss.item()
            torch.save(decode_model.state_dict(), save_path)
            print(f"  --> 保存最优模型: {save_path}")

        torch.save(
            {
                "epoch": epoch,
                "model_state": decode_model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "best_valid_loss": best_valid_loss,
            },
            state_path,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train GFlowNet decoder (Phase 2).")
    parser.add_argument("--data_path", type=str, default=DATA_PATH, help="Training TSV path.")
    parser.add_argument("--encoder_checkpoint", type=str, default=ENCODER_PARAMS["save_path"], help="Path to trained encoder checkpoint.")
    parser.add_argument("--seed", type=int, default=ENCODER_PARAMS.get("random_seed", 42))

    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Training epochs.")
    parser.add_argument("--lr", type=float, default=LR, help="Adam learning rate.")
    parser.add_argument("--decay_step", type=int, default=DECAY_STEP, help="StepLR step size.")
    parser.add_argument("--decay_gamma", type=float, default=DECAY_GAMMA, help="StepLR gamma.")
    parser.add_argument("--no_resume", action="store_true", help="Do not resume from --state_path even if it exists.")

    parser.add_argument("--save_path", type=str, default=SAVE_PATH, help="Best model checkpoint path.")
    parser.add_argument("--state_path", type=str, default=STATE_PATH, help="Training state checkpoint path for resuming.")

    parser.add_argument("--spec_token", action="store_true", help="Whether to include special token embedding in encoder->sequence embedding.")
    parser.add_argument("--max_length", type=int, default=ENCODER_PARAMS["max_length"], help="Encoder tokenizer max length.")

    # Condition-side noise (to prevent trivial "condition==target" copying).
    parser.add_argument("--cond_mask_prob", type=float, default=0.2, help="Mask probability applied to condition sequences when building embeddings.")
    parser.add_argument(
        "--cond_mask_specials",
        action="store_true",
        help="If set, allow masking special tokens like '-' and '<eso>' in the condition embedding stage.",
    )

    # Decoder architecture knobs
    parser.add_argument("--gflownet_hidden_dim", type=int, default=512)
    parser.add_argument("--gflownet_num_layers", type=int, default=4)
    parser.add_argument("--gflownet_num_heads", type=int, default=8)

    # Loss alignment knobs
    parser.add_argument(
        "--loss_type",
        type=str,
        default="pairwise_ranking",
        choices=["pairwise_ranking", "weighted_nll"],
        help="Training loss type.",
    )
    parser.add_argument("--pairwise_tau", type=float, default=1.0, help="Pairwise ranking temperature tau.")
    parser.add_argument("--pairwise_n_pairs", type=int, default=5000, help="Number of sampled pairs per epoch.")

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    vocab, inv_vocab, vocab_size = build_vocab()
    pad_id = vocab["<pad>"]

    # Ensure checkpoint directory exists (e.g. models/)
    Path(args.save_path).parent.mkdir(parents=True, exist_ok=True)
    Path(args.state_path).parent.mkdir(parents=True, exist_ok=True)

    # Use the trained encoder to build sequence embeddings (reward input).
    encoder_model = TransformerEncoderModel(
        vocab_size=len(vocab),
        embed_dim=ENCODER_PARAMS["embed_dim"],
        num_heads=ENCODER_PARAMS["num_heads"],
        num_layers=ENCODER_PARAMS["num_layers"],
        dropout=ENCODER_PARAMS["dropout"],
        hidden_dim=ENCODER_PARAMS["hidden_dim"],
        max_len=args.max_length + 3,
        max_relative_distance=16,
    )
    checkpoint = torch.load(args.encoder_checkpoint, map_location=DEVICE)
    encoder_model.load_state_dict(checkpoint["model_state"])
    encoder_model.to(DEVICE).eval()

    train_df, valid_df, _test_df = load_and_split_data(args.data_path)

    X_train = build_sequence_embeddings(
        train_df,
        encoder_model,
        vocab,
        inv_vocab,
        device=DEVICE,
        spec_token=args.spec_token,
        cond_mask_prob=args.cond_mask_prob,
        cond_mask_specials=args.cond_mask_specials,
    )
    X_valid = build_sequence_embeddings(
        valid_df,
        encoder_model,
        vocab,
        inv_vocab,
        device=DEVICE,
        spec_token=args.spec_token,
        cond_mask_prob=args.cond_mask_prob,
        cond_mask_specials=args.cond_mask_specials,
    )

    seq_train_idx = build_seq_idx(train_df, vocab, X_train.shape[1], device=DEVICE)
    seq_valid_idx = build_seq_idx(valid_df, vocab, X_valid.shape[1], device=DEVICE)

    rewards_train = torch.log(torch.tensor(train_df["Kobs (M-1 s-1)"].values, dtype=torch.float32, device=DEVICE) + 1e-8)
    rewards_valid = torch.log(torch.tensor(valid_df["Kobs (M-1 s-1)"].values, dtype=torch.float32, device=DEVICE) + 1e-8)

    decode_model = GFlowNetTransformer(
        embed_dim=ENCODER_PARAMS["embed_dim"],
        hidden_dim=args.gflownet_hidden_dim,
        num_layers=args.gflownet_num_layers,
        num_heads=args.gflownet_num_heads,
        vocab_size=len(vocab),
        max_len=X_train.shape[1],
    ).to(DEVICE)

    train_gflownet_resume(
        decode_model,
        X_train,
        seq_train_idx,
        rewards_train,
        X_valid,
        seq_valid_idx,
        rewards_valid,
        device=DEVICE,
        epochs=args.epochs,
        lr=args.lr,
        save_path=args.save_path,
        state_path=args.state_path,
        resume=not args.no_resume,
        pad_id=pad_id,
        loss_type=args.loss_type,
        pairwise_tau=args.pairwise_tau,
        pairwise_n_pairs=args.pairwise_n_pairs,
        decay_step=args.decay_step,
        decay_gamma=args.decay_gamma,
    )


if __name__ == "__main__":
    main()

