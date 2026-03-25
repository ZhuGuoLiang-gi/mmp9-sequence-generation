"""
Train masked-token encoder (Phase 2).

Migrated from legacy `legacy_deprecated/train.py`.

This module implements the encoder training used to produce:
  - `best_model_spec_masked.pth`
"""

from __future__ import annotations

import argparse
import os
import random
from typing import Optional, Tuple
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..config import ENCODER_PARAMS
from ..dataset import SequenceDataset, collate_fn
from ..models.transformer_encoder import TransformerEncoderModel
from ..tokenizer import build_vocab


def load_and_split_data(path: str, seed: int, split_ratio: Tuple[float, float, float]):
    np.random.seed(seed)
    df = pd.read_csv(path, sep="\t")
    df_filtered = df[df["Kobs (M-1 s-1)"] > 0].reset_index(drop=True)

    indices = np.arange(len(df_filtered))
    np.random.shuffle(indices)

    n_train = int(split_ratio[0] * len(df_filtered))
    n_valid = int(split_ratio[1] * len(df_filtered))
    n_test = len(df_filtered) - n_train - n_valid

    train_idx = indices[:n_train]
    valid_idx = indices[n_train : n_train + n_valid]
    test_idx = indices[n_train + n_valid :]

    train_df = df_filtered.iloc[train_idx].reset_index(drop=True)
    valid_df = df_filtered.iloc[valid_idx].reset_index(drop=True)
    test_df = df_filtered.iloc[test_idx].reset_index(drop=True)

    print(f"筛选后共有 {len(df_filtered)} 条数据。")
    print(f"划分结果：train={len(train_df)}, valid={len(valid_df)}, test={len(test_df)}")
    return train_df, valid_df, test_df


def train_model(
    model: TransformerEncoderModel,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    *,
    params: dict,
    vocab: dict,
    vocab_size: int,
    max_train_batches: Optional[int] = None,
    max_valid_batches: Optional[int] = None,
):
    masked_loss = params.get("masked_loss", True)
    model.to(params["device"])
    optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"])
    criterion = nn.CrossEntropyLoss(ignore_index=vocab["<pad>"])

    best_valid_loss = float("inf")
    start_epoch = 0

    if params.get("resume", False) and os.path.exists(params["save_path"]):
        checkpoint = torch.load(params["save_path"], map_location=params["device"])
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_epoch = checkpoint.get("epoch", 0) + 1
        best_valid_loss = checkpoint.get("best_valid_loss", float("inf"))
        print(f"Resuming training from epoch {start_epoch}")

    for epoch in range(start_epoch, params["epochs"]):
        model.train()
        total_loss = 0.0

        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{params['epochs']} [Train]", leave=False)
        for batch_idx, (masked_tokens, target_tokens, mask_positions) in enumerate(train_bar):
            if max_train_batches is not None and batch_idx >= max_train_batches:
                break
            masked_tokens = masked_tokens.to(params["device"])
            target_tokens = target_tokens.to(params["device"])
            logits = model(masked_tokens)

            if masked_loss:
                mask_index = torch.zeros_like(target_tokens, dtype=torch.bool)
                for i, pos_list in enumerate(mask_positions):
                    mask_index[i, pos_list] = True
                if mask_index.any():
                    loss = criterion(logits[mask_index], target_tokens[mask_index])
                else:
                    loss = torch.tensor(0.0, device=masked_tokens.device)
            else:
                loss = criterion(logits.view(-1, vocab_size), target_tokens.view(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            train_bar.set_postfix(loss=total_loss / max(1, len(train_bar)))

        model.eval()
        total_correct = 0
        total_masked = 0
        valid_loss = 0.0

        valid_bar = tqdm(valid_loader, desc=f"Epoch {epoch+1}/{params['epochs']} [Valid]", leave=False)
        with torch.no_grad():
            for batch_idx, (masked_tokens, target_tokens, mask_positions) in enumerate(valid_bar):
                if max_valid_batches is not None and batch_idx >= max_valid_batches:
                    break
                masked_tokens = masked_tokens.to(params["device"])
                target_tokens = target_tokens.to(params["device"])
                logits = model(masked_tokens)

                if masked_loss:
                    mask_index = torch.zeros_like(target_tokens, dtype=torch.bool)
                    for i, pos_list in enumerate(mask_positions):
                        mask_index[i, pos_list] = True
                    if mask_index.any():
                        loss = criterion(logits[mask_index], target_tokens[mask_index])
                    else:
                        loss = torch.tensor(0.0, device=masked_tokens.device)
                else:
                    loss = criterion(logits.view(-1, vocab_size), target_tokens.view(-1))

                valid_loss += loss.item()

                preds = logits.argmax(-1)
                for i in range(masked_tokens.size(0)):
                    positions = mask_positions[i]
                    for pos in positions:
                        if preds[i, pos] == target_tokens[i, pos]:
                            total_correct += 1
                        total_masked += 1

                valid_bar.set_postfix(
                    recovery_acc=total_correct / (total_masked + 1e-9),
                    valid_loss=valid_loss / max(1, len(valid_loader)),
                )

        valid_loss /= max(1, len(valid_loader))
        acc = total_correct / (total_masked + 1e-9)
        print(
            f"Epoch {epoch+1}/{params['epochs']} | Train Loss={total_loss/len(train_loader):.4f} | "
            f"Valid Loss={valid_loss:.4f} | Recovery Acc={acc:.4f}"
        )

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "best_valid_loss": best_valid_loss,
                },
                params["save_path"],
            )
            print(f"  --> Saved new best model with valid_loss={best_valid_loss:.4f} at epoch {epoch+1}")


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Train encoder (masked token recovery).")
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="训练轮数（覆盖配置里的 epochs）",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Adam 学习率（覆盖配置里的 lr）",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="DataLoader batch_size（覆盖配置里的 batch_size）",
    )
    parser.add_argument(
        "--mask_prob",
        type=float,
        default=None,
        help="mask 概率：token 被替换为 <mask> 的概率（loss 只在 <mask> 位置计算）",
    )
    parser.add_argument(
        "--n_mask",
        type=int,
        default=None,
        help="每条原始序列生成多少条不同 mask 版本（dataset 长度=原始行数*n_mask）",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=None,
        help="续跑训练：若 save_path 存在则加载 checkpoint 并继续",
    )
    parser.add_argument(
        "--no_resume",
        action="store_true",
        default=None,
        help="从头训练：不加载 checkpoint（覆盖 resume）",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default=None,
        help="最优验证 loss 对应的 checkpoint 保存路径（以及 resume 时的加载路径）",
    )
    parser.add_argument(
        "--max_train_batches",
        type=int,
        default=None,
        help="快速 sanity-check：限制每个 epoch 最多跑多少个训练 batch（None=不限制）",
    )
    parser.add_argument(
        "--max_valid_batches",
        type=int,
        default=None,
        help="快速 sanity-check：限制每个 epoch 最多跑多少个验证 batch（None=不限制）",
    )
    args = parser.parse_args(argv)

    params = dict(ENCODER_PARAMS)
    if args.epochs is not None:
        params["epochs"] = args.epochs
    if args.lr is not None:
        params["lr"] = args.lr
    if args.batch_size is not None:
        params["batch_size"] = args.batch_size
    if args.mask_prob is not None:
        params["mask_prob"] = args.mask_prob
    if args.n_mask is not None:
        params["n_mask"] = args.n_mask
    if args.save_path is not None:
        params["save_path"] = args.save_path
    if args.resume is True:
        params["resume"] = True
    if args.no_resume is True:
        params["resume"] = False

    # Ensure checkpoint directory exists (e.g. models/)
    Path(params["save_path"]).parent.mkdir(parents=True, exist_ok=True)

    # Seed
    torch.manual_seed(params["random_seed"])
    random.seed(params["random_seed"])
    np.random.seed(params["random_seed"])

    vocab, inv_vocab, vocab_size = build_vocab()

    train_df, valid_df, _test_df = load_and_split_data(
        params["path"], seed=params["random_seed"], split_ratio=params["split_ratio"]
    )

    train_dataset = SequenceDataset(
        train_df,
        vocab,
        inv_vocab=inv_vocab,
        mask_prob=params["mask_prob"],
        n_mask=params["n_mask"],
        max_length=params["max_length"],
        mask_specials=params["mask_specials"],
    )
    valid_dataset = SequenceDataset(
        valid_df,
        vocab,
        inv_vocab=inv_vocab,
        mask_prob=params["mask_prob"],
        n_mask=1,
        max_length=params["max_length"],
        mask_specials=params["mask_specials"],
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=params["batch_size"],
        shuffle=True,
        collate_fn=collate_fn,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=params["batch_size"],
        shuffle=False,
        collate_fn=collate_fn,
    )

    model = TransformerEncoderModel(
        vocab_size=vocab_size,
        embed_dim=params["embed_dim"],
        num_heads=params["num_heads"],
        num_layers=params["num_layers"],
        dropout=params["dropout"],
        hidden_dim=params["hidden_dim"],
        max_len=params["max_length"] + 3,
        max_relative_distance=16,
    )

    train_model(
        model,
        train_loader,
        valid_loader,
        params=params,
        vocab=vocab,
        vocab_size=vocab_size,
        max_train_batches=args.max_train_batches,
        max_valid_batches=args.max_valid_batches,
    )


if __name__ == "__main__":
    main()

