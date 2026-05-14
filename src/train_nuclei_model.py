#!/usr/bin/env python3
"""Fine-tune a Cellpose nuclei model from annotated 3D volumes."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import psutil
import tifffile
from cellpose import models, train


def get_ram_usage_gb() -> float:
    return psutil.virtual_memory().used / 1e9


def extract_labeled_slices(
    raw_dir: Path,
    masks_dir: Path,
    slice_dir: Path,
) -> tuple[list[Path], list[Path]]:
    slice_dir.mkdir(parents=True, exist_ok=True)
    image_paths: list[Path] = []
    mask_paths: list[Path] = []
    global_index = 0

    for raw_path in sorted(raw_dir.glob("*.tif")):
        mask_path = masks_dir / f"{raw_path.stem}_masks.tif"
        if not mask_path.exists():
            print(f"Skipping {raw_path.name}: no matching mask found")
            continue

        print(f"Processing {raw_path.name}; RAM {get_ram_usage_gb():.2f} GB")
        raw_3d = tifffile.imread(raw_path).astype(np.float32)
        mask_3d = tifffile.imread(mask_path).astype(np.uint16)

        kept = 0
        for z, mask_slice in enumerate(mask_3d):
            if np.max(mask_slice) == 0:
                continue

            image_path = slice_dir / f"img_{global_index:05d}.tif"
            label_path = slice_dir / f"mask_{global_index:05d}.tif"
            tifffile.imwrite(image_path, raw_3d[z])
            tifffile.imwrite(label_path, mask_slice)

            image_paths.append(image_path)
            mask_paths.append(label_path)
            global_index += 1
            kept += 1

        print(f"Extracted {kept} labeled slices")

    return image_paths, mask_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune Cellpose nuclei segmentation from 3D annotated data."
    )
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--masks-dir", type=Path, required=True)
    parser.add_argument("--slice-dir", type=Path, default=Path("data/slices_tmp"))
    parser.add_argument("--model-name", default="finetuned_nuclei_model_3d_sliced")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--use-gpu", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    image_paths, mask_paths = extract_labeled_slices(
        raw_dir=args.raw_dir,
        masks_dir=args.masks_dir,
        slice_dir=args.slice_dir,
    )
    if not image_paths:
        raise RuntimeError("No labeled slices were found.")

    print(f"Loading {len(image_paths)} slices into memory")
    train_images = [tifffile.imread(path) for path in image_paths]
    train_masks = [tifffile.imread(path) for path in mask_paths]

    model = models.CellposeModel(gpu=args.use_gpu, model_type="nuclei")
    model_path, train_losses, test_losses = train.train_seg(
        model.net,
        train_data=train_images,
        train_labels=train_masks,
        channels=[0, 0],
        normalize=True,
        weight_decay=args.weight_decay,
        SGD=True,
        learning_rate=args.learning_rate,
        n_epochs=args.epochs,
        model_name=args.model_name,
    )

    print(f"Training complete: {model_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
