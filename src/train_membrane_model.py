#!/usr/bin/env python3
"""Fine-tune a Cellpose model for membrane segmentation."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def build_command(args: argparse.Namespace) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "cellpose",
        "--train",
        "--dir",
        str(args.train_dir),
        "--pretrained_model",
        args.base_model,
        "--chan",
        "1",
        "--chan2",
        "2",
        "--n_epochs",
        str(args.epochs),
        "--learning_rate",
        str(args.learning_rate),
        "--weight_decay",
        str(args.weight_decay),
        "--mask_filter",
        "_masks",
        "--img_filter",
        "_img",
    ]

    if args.use_gpu:
        command.append("--use_gpu")

    if args.model_name:
        command.extend(["--model_name", args.model_name])

    return command


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune Cellpose on the prepared 2-channel membrane dataset."
    )
    parser.add_argument("--train-dir", type=Path, required=True)
    parser.add_argument("--base-model", default="tissuenet")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--model-name", default="membrane_hint_aware_model")
    parser.add_argument("--use-gpu", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.train_dir.exists():
        raise FileNotFoundError(f"Training directory does not exist: {args.train_dir}")

    command = build_command(args)
    print("Starting Cellpose fine-tuning")
    print(" ".join(command))

    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="")

    return process.wait()


if __name__ == "__main__":
    raise SystemExit(main())
