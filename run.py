#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Project entrypoint (Phase B):统一入口 CLI

用法示例：
  python run.py infer gflownet_fasta -f input.fasta -o output.fasta -n 20 -t 1.0
  python run.py train encoder
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent

# key: "<task> <mode>"
# value:
#   - tuple (kind, target)
#   - kind == "module": run `python -m <target> ...`
#   - kind == "file": run `python <ROOT>/<target> ...`
DISPATCH: dict[str, tuple[str, str]] = {
    "train encoder": ("module", "mmp9.training.train_encoder"),
    "train gflownet": ("module", "mmp9.training.train_gflownet"),
    "infer encoder_template_inference2": ("file", "inference2.py"),
    "infer gflownet_fasta": ("module", "mmp9.inference.infer_gflownet_fasta"),
    "eval gflownet_reward": ("module", "mmp9.evaluation.eval_gflownet_reward"),
    "eval gflownet_reward_from_fasta": ("module", "mmp9.evaluation.eval_gflownet_reward_from_fasta"),
    "eval gflownet_seq_logp_fasta": ("module", "mmp9.evaluation.eval_gflownet_seq_logp_fasta"),
    "eval mask_recovery": ("module", "mmp9.evaluation.eval_mask_recovery"),
}


def print_help() -> None:
    msg = """
Usage:
  python run.py <task> <mode> [args...]

Tasks/Modes:
  train encoder
  train gflownet

  infer encoder_template_inference2
  infer gflownet_fasta -f input.fasta -o output.fasta -n 20 -t 1.0

  eval gflownet_reward
  eval gflownet_reward_from_fasta
  eval gflownet_seq_logp_fasta
  eval mask_recovery
"""
    print(msg.strip())


def main() -> int:
    if len(sys.argv) < 3 or sys.argv[1] in {"-h", "--help"} or sys.argv[2] in {"-h", "--help"}:
        print_help()
        return 0

    task = sys.argv[1]
    mode = sys.argv[2]
    rest = sys.argv[3:]

    key = f"{task} {mode}"
    target = DISPATCH.get(key)
    if not target:
        print_help()
        print(f"\nUnknown command: {key}")
        return 2

    kind, target = target
    if kind == "module":
        return subprocess.call([sys.executable, "-m", target, *rest], cwd=str(ROOT))
    if kind == "file":
        script_path = ROOT / target
        return subprocess.call([sys.executable, str(script_path), *rest], cwd=str(ROOT))
    print(f"\nInvalid dispatch kind: {kind}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

