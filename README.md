# MMP9 Sequence Generation Model

This directory contains a modular (refactored) implementation for training and inference on the **MMP9** sequence dataset.

Core idea: use an **encoder** (masked-token Transformer) to learn a representation over fixed-length tokenized sequences, and then learn a **GFlowNet decoder** (`GFlowNetTransformer`).

## Quick navigation

- Entrypoint CLI: `run.py`
- Refactored modules: `mmp9/`
- Model weights (default): `models/`
- Legacy moved for reference: `legacy_deprecated/`

## Data / token format

Each sequence in `MMP9.tsv` (and FASTA used for generation) must contain **exactly one** center separator `-`.

Tokenization wraps the left/right parts with special tokens (e.g. `<eso>`) and uses a fixed length of:

- `max_length + 3` tokens

Special tokens in the vocabulary:

- `<pad>`, `<eso>`, `<unk>`, `<mask>` plus the amino acids `ACDEFGHIKLMNPQRSTVWY` and `-`.

## Training data

- Default dataset file: `data/MMP9.tsv`
- Expected columns:
  - `Sequences`: sequence strings containing exactly one `-`
  - `Kobs (M-1 s-1)`: experimental/label value used for filtering and (for GFlowNet) reward
- Filtering rule (used by encoder/GFlowNet):
  - keep only rows where `Kobs (M-1 s-1) > 0`
- Masking / training objective:
  - Encoder (`train encoder`): BERT-like masked token recovery
  - GFlowNet (`train gflownet`): reward-driven training using `log(Kobs)` (see legacy implementation for exact formulation)

## Model weights

Default save/load paths use the repository root `models/` directory:

- `models/best_model_spec_masked.pth` (encoder)
- `models/best_gflownet.pth` / `models/best_gflownet_transformer.pth` (GFlowNet)

## How to run

### Train

#### 1. Train encoder (masked token recovery):

```bash
python run.py train encoder
```

You can tune these flags (all are forwarded to `python -m mmp9.training.train_encoder`):

- `--epochs` (default: 200): 训练轮数。
- `--lr` (default: 1e-7): 学习率。因为模型较大且 loss 很“平”，通常从较小的值开始再逐步增大。
- `--batch_size` (default: 16): batch 大小（显存不够就减小）。
- `--mask_prob` (default: 0.2): 将 token 替换为 `<mask>` 的概率（不会 mask `<pad>`，可选是否 mask specials）。
  - loss 只在被 `<mask>` 的位置计算：`mask_prob` 太小会导致每个 batch 的有效监督太少，太大则可能让任务过难。
- `--n_mask` (default: 4): 每条原始序列生成多少条不同的 mask 版本（会线性增加训练样本数/训练步数）。
- `--no_resume`：不续跑，从头训练。
- `--resume`：续跑（会加载 `--save_path` 对应的 checkpoint）。
- `--save_path` (default: `models/best_model_spec_masked.pth`): encoder checkpoint 保存位置。
- `--max_train_batches` / `--max_valid_batches`：快速 sanity-check 用（例如都设为 1，可以只跑极少步）。

Encoder tuning workflow:
- 先用小规模验证链路是否通：
  ```bash
  python run.py train encoder --no_resume --epochs 1 --batch_size 2 --n_mask 1 --mask_prob 0.1 --max_train_batches 1 --max_valid_batches 1
  ```
- 再用中等设置训练：
  - `--mask_prob`：0.15 ~ 0.3 之间试
  - `--n_mask`：4 ~ 10 之间试
- 如果收敛很慢，再考虑把 `--lr` 从 `1e-7` 小幅提高（需小步验证）。

After training encoder:

Mask recovery curve:
```bash
python run.py eval mask_recovery
```

Image (preview may not inline local files):
[mask_recovery_curve.png](./plot/mask_recovery_curve.png)

Common parameters:
- `--model_path`：encoder checkpoint 路径（默认：`models/best_model_spec_masked.pth`）
- `--out`：输出图片文件名（默认：`plot/mask_recovery_curve.png`）
- `--min_mask_prob`：扫描最小 `mask_prob`（默认：`0.1`）
- `--max_mask_prob`：扫描最大 `mask_prob`（默认：`1.0`）
- `--steps`：扫描步数（默认：`10`）

#### 2. Train GFlowNet decoder (decoder training):
```bash
python run.py train gflownet
```
常用参数（参数名与 `mmp9/training/train_gflownet.py` 的 `argparse` 一致）：
- `--data_path`：训练数据（默认：`data/MMP9.tsv`）
- `--encoder_checkpoint`：用来生成 reward 输入的 encoder checkpoint（默认：`models/best_model_spec_masked.pth`）
- `--epochs`：训练轮数（默认：`100`）
- `--lr`：Adam 学习率（默认：`1e-4`）
- `--decay_step` / `--decay_gamma`：StepLR 参数（默认：`100` / `0.9`）
- `--save_path`：最优模型保存位置（默认：`models/best_gflownet.pth`）
- `--state_path`：断点训练状态保存/加载（默认：`models/gflownet_training_state.pth`）
- `--no_resume`：不续跑（不加载 `--state_path`）
- `--spec_token`：是否在 embedding 构建时包含特殊 token
- `--cond_mask_prob`：条件侧 `<mask>` 替换概率（默认：`0.2`）
- `--cond_mask_specials`：允许在条件侧 masking 中包含 `-` / `<eso>` 等特殊 token
- `--loss_type`：训练目标（默认：`pairwise_ranking`；可选 `weighted_nll`）
- `--pairwise_tau`：pairwise ranking loss 的温度（默认：`1.0`）
- `--pairwise_n_pairs`：每个 epoch 随机采样的 pairs 数（默认：`5000`）
- `--max_length`：encoder tokenizer 的最大长度
- `--gflownet_hidden_dim` / `--gflownet_num_layers` / `--gflownet_num_heads`：GFlowNet 解码器结构参数

### Inference (generation)

#### GFlowNet inference from FASTA (reads input sequences and writes generated FASTA):

```bash
python run.py infer gflownet_fasta -f input.fasta -o output.fasta -n 20 -t 1.0
```

默认会约束生成序列的 `-` 位置与输入参考序列一致。
默认也会约束生成序列长度与输入参考序列一致。
输入 FASTA 还支持位点固定语法：`<A>` / `<a>`（表示该位点必须保持为该氨基酸）。
例如：`ACD-<L>FG` 表示该位点固定为 `L`。
如需关闭该约束，可加：`--disable_dash_pos_constraint`。

如果你要提高 `eval gflownet_reward` 的 strict 命中率（`found`/`coverage`），可以开启候选集约束采样：

```bash
python run.py infer gflownet_fasta -f input.fasta -o output.fasta -n 20 -t 1.0 \
  --constrain_to_tsv_candidates --candidate_kobs_min 0.0 --candidate_topk_by_logp 0
```

关闭 `-` 位置约束示例：

```bash
python run.py infer gflownet_fasta -f input.fasta -o output.fasta -n 20 -t 1.0 \
  --disable_dash_pos_constraint
```

如果你想额外导出一个“不带 `-`”的 FASTA，可加：

```bash
python run.py infer gflownet_fasta -f input.fasta -o output.fasta \
  --output_fasta_no_dash output_no_dash.fasta
```

### Eval GFlowNet (reward)

用 reward（`reward = log(Kobs + 1e-8)`，`Kobs` 来自 `MMP9.tsv`）对生成 FASTA 做好坏对比，并与随机采样基线对照：

```bash
python run.py eval gflownet_reward --gen_fasta eval_gflownet/gen_seq.fasta --n_random 1000
```

默认会把图和统计输出到 `plot/` 下（`plot/gflownet_reward_compare.png` 和 `plot/gflownet_reward_summary.csv`）。

### Eval GFlowNet (seq_logp scoring for FASTA)

用 GFlowNet 的 `seq_logp` 给 FASTA 序列逐条打分（更适合看模型偏好，不等价于实验活性）：

```bash
python run.py eval gflownet_seq_logp_fasta \
  --fasta output.fasta \
  --gflownet_checkpoint ./models/test_gflownet.pth \
  --encoder_checkpoint ./models/test_model.pth \
  --out_fasta eval_gflownet/gflownet_seq_logp_scored.fasta \
  --topn 20
```

输出 FASTA 的 header 会写入分数，例如：`>generated_seq_1 seq_logp=-2.314159`。  
如需额外导出 CSV，可再加：`--out_csv plot/gflownet_seq_logp_scores.csv`。

### Eval GFlowNet (reward, auto from condition)

把“条件序列生成 + 推理生成 FASTA + 评估”融合成一条命令：

- 条件序列：自动从 `MMP9.tsv` 中随机抽取 `--cond_n` 条（过滤 `Kobs>=0`，并套用生成侧合法性过滤）
- 生成：用 `infer gflownet_fasta` 的同构采样逻辑生成序列
- 评估：立刻对比 reward vs 随机采样基线，并输出到 `plot/`

```bash
python run.py eval gflownet_reward_from_fasta \
  --cond_n 50 --n_sample 20 --temperature 1.0 \
  --encoder_checkpoint ./models/test_model.pth \
  --model_path ./models/test_gflownet.pth
```

默认输出：
- `eval_gflownet/gen_seq_from_condition.fasta`
- `plot/gflownet_reward_compare_from_condition.png`
- `plot/gflownet_reward_summary_from_condition.csv`

如果也想提高 strict 命中率，可以在融合命令里开启候选集约束采样：

```bash
python run.py eval gflownet_reward_from_fasta \
  --cond_n 50 --n_sample 20 --temperature 1.0 \
  --constrain_to_tsv_candidates --candidate_kobs_min 0.0 --candidate_topk_by_logp 0 \
  --encoder_checkpoint ./models/test_model.pth \
  --model_path ./models/test_gflownet.pth
```

同样支持关闭 `-` 位置约束：追加 `--disable_dash_pos_constraint`。

Evaluation protocol (recommended):

- **Constrained sampling (ranking evaluation):**
  - Enable `--constrain_to_tsv_candidates`.
  - This tests whether the model ranks known TSV candidates better than random.
  - Suitable for comparing reward-oriented ranking ability.
- **Unconstrained sampling (generation evaluation):**
  - Do **not** enable `--constrain_to_tsv_candidates`.
  - This tests real generation behavior outside the strict TSV candidate pool.
  - Report validity/uniqueness/novelty plus strict reward lookup coverage.

### Latest result (candidate-constrained sampling)

Image (preview may not inline local files):
[gflownet_reward_compare_from_condition.png](./plot/gflownet_reward_compare_from_condition.png)

Key metrics:
- Gen: `total=644 valid=644 found=644 coverage=1.000 mean=5.0011 median=8.7212 max=11.8490 top20_mean=11.3041`
- Random: `total=1000 valid=1000 found=1000 coverage=1.000 mean=3.7416 median=7.9780 max=11.5983 top20_mean=11.1688`

CSV: `plot/gflownet_reward_summary_from_condition.csv`

## Notes / limitations

- `infer encoder_template_inference2` is currently **not available** in this refactor because the legacy `inference2.py` was removed from the root directory.
- For reference only, migrated legacy scripts are kept under `legacy_deprecated/`.

