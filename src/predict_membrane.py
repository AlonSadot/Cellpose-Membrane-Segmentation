#!/usr/bin/env python3
"""Run membrane segmentation with a trained Cellpose model."""

from __future__ import annotations

import argparse
import time
from datetime import timedelta
from glob import glob
from pathlib import Path

import cv2
import numpy as np
from cellpose import core, models
from skimage import io
from skimage.restoration import denoise_wavelet
from tqdm import tqdm


def normalize_uint8(image: np.ndarray) -> np.ndarray:
    image_norm = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
    return image_norm.astype(np.uint8)


def to_zhwc(volume: np.ndarray) -> np.ndarray:
    if volume.ndim != 4:
        raise ValueError(f"Expected a 4D volume, got shape {volume.shape}")

    if volume.shape[-1] <= 10 and volume.shape[1] > 10 and volume.shape[2] > 10:
        return volume
    if volume.shape[0] <= 10 and volume.shape[2] > 10 and volume.shape[3] > 10:
        return np.moveaxis(volume, 0, -1)

    raise ValueError(
        f"Ambiguous channel layout for shape {volume.shape}; expected ZHWC or CZHW."
    )


def preprocess_stack(
    membrane_stack: np.ndarray,
    intercell_stack: np.ndarray,
    alpha: float,
) -> np.ndarray:
    membrane_stack = normalize_uint8(membrane_stack)
    intercell_stack = normalize_uint8(intercell_stack)

    if membrane_stack.shape != intercell_stack.shape:
        raise ValueError("Membrane and intercellular stacks must have the same shape.")

    processed_stack = np.empty_like(membrane_stack)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    large_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    small_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    for z in range(membrane_stack.shape[0]):
        subtracted = cv2.addWeighted(membrane_stack[z], 1.0, intercell_stack[z], -alpha, 0)
        subtracted = np.clip(subtracted, 0, 255).astype(np.uint8)

        enhanced = clahe.apply(subtracted)
        background = cv2.morphologyEx(enhanced, cv2.MORPH_OPEN, large_kernel)
        top_hat = cv2.subtract(enhanced, background)
        closed = cv2.morphologyEx(top_hat, cv2.MORPH_CLOSE, small_kernel)
        cleaned = cv2.morphologyEx(closed, cv2.MORPH_OPEN, small_kernel)

        if np.max(cleaned) == 0:
            processed_stack[z] = 0
            continue

        denoised = denoise_wavelet(
            cleaned.astype(np.float32) / 255.0,
            channel_axis=None,
            convert2ycbcr=False,
            rescale_sigma=True,
        )
        processed_stack[z] = np.clip(np.nan_to_num(denoised) * 255, 0, 255).astype(
            np.uint8
        )

    return processed_stack


def segment_volume(
    model: models.CellposeModel,
    volume: np.ndarray,
    args: argparse.Namespace,
) -> np.ndarray:
    membrane_stack = volume[..., args.membrane_channel]
    intercell_stack = volume[..., args.intercellular_channel]
    processed_stack = preprocess_stack(membrane_stack, intercell_stack, alpha=args.alpha)

    masks = []
    iterator = processed_stack if args.no_progress else tqdm(processed_stack, leave=False)
    for image_slice in iterator:
        if np.max(image_slice) == 0:
            masks.append(np.zeros_like(image_slice, dtype=np.int32))
            continue

        mask = model.eval(
            image_slice,
            channels=[0, 0],
            normalize=True,
            flow_threshold=args.flow_threshold,
            cellprob_threshold=args.cellprob_threshold,
            min_size=args.min_size,
            niter=args.niter,
            tile_overlap=args.tile_overlap,
            bsize=args.bsize,
        )[0]
        masks.append(mask.astype(np.int32, copy=False))

    return np.stack(masks, axis=0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Segment membrane microscopy volumes with a trained Cellpose model."
    )
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--membrane-channel", type=int, default=1)
    parser.add_argument("--intercellular-channel", type=int, default=2)
    parser.add_argument("--alpha", type=float, default=0.0)
    parser.add_argument("--flow-threshold", type=float, default=0.35)
    parser.add_argument("--cellprob-threshold", type=float, default=-0.1)
    parser.add_argument("--min-size", type=int, default=100)
    parser.add_argument("--niter", type=int, default=1000)
    parser.add_argument("--tile-overlap", type=float, default=0.45)
    parser.add_argument("--bsize", type=int, default=224)
    parser.add_argument("--extension", default="tif", choices=["tif", "tiff"])
    parser.add_argument("--suffix", default="_pred")
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    tif_files = sorted(glob(str(args.input_dir / f"*.{args.extension}")))
    if not tif_files:
        raise RuntimeError(f"No .{args.extension} files found in {args.input_dir}")

    use_gpu = core.use_gpu()
    model = models.CellposeModel(pretrained_model=args.model_path, gpu=use_gpu)

    total_time = 0.0
    for index, path in enumerate(tif_files, start=1):
        input_path = Path(path)
        print(f"[{index}/{len(tif_files)}] {input_path.name}")
        start_time = time.time()

        volume = to_zhwc(io.imread(input_path))
        mask_volume = segment_volume(model, volume, args)

        output_path = args.output_dir / f"{input_path.stem}{args.suffix}.tif"
        io.imsave(output_path, mask_volume)

        elapsed = time.time() - start_time
        total_time += elapsed
        print(f"Saved {output_path} in {timedelta(seconds=int(elapsed))}")

    print(f"Finished {len(tif_files)} volumes in {timedelta(seconds=int(total_time))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
