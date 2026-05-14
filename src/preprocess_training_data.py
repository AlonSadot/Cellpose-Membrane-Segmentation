#!/usr/bin/env python3
"""Prepare 2D Cellpose training samples from annotated 3D microscopy volumes.

The membrane model was trained on 2-channel 2D slices:
  channel 1: cleaned membrane signal
  channel 2: a blurred nuclei-label hint

Raw input volumes are expected to contain nuclei, membrane, and intercellular
channels. Ground-truth membrane masks are expected as matching 3D label stacks.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2
import numpy as np
import tifffile
from cellpose import models
from scipy.ndimage import gaussian_filter
from skimage.restoration import denoise_wavelet


def normalize_uint8(image: np.ndarray) -> np.ndarray:
    image_norm = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
    return image_norm.astype(np.uint8)


def to_zhwc(volume: np.ndarray) -> np.ndarray:
    """Standardize either (Z, H, W, C) or (C, Z, H, W) input to (Z, H, W, C)."""
    if volume.ndim != 4:
        raise ValueError(f"Expected a 4D volume, got shape {volume.shape}")

    if volume.shape[-1] <= 10 and volume.shape[1] > 10 and volume.shape[2] > 10:
        return volume
    if volume.shape[0] <= 10 and volume.shape[2] > 10 and volume.shape[3] > 10:
        return np.moveaxis(volume, 0, -1)

    raise ValueError(
        f"Ambiguous channel layout for shape {volume.shape}; expected ZHWC or CZHW."
    )


def generate_nuclei_hint(nuclei_labels: np.ndarray, blur_sigma: float) -> np.ndarray:
    binary_mask = (nuclei_labels > 0).astype(np.float32) * 255.0
    hint_stack = np.empty_like(binary_mask, dtype=np.uint8)

    for z, mask_slice in enumerate(binary_mask):
        if np.max(mask_slice) == 0:
            hint_stack[z] = 0
            continue
        smoothed = gaussian_filter(mask_slice, sigma=blur_sigma)
        hint_stack[z] = np.clip(smoothed, 0, 255).astype(np.uint8)

    return hint_stack


def preprocess_membrane_stack(
    membrane_stack: np.ndarray,
    intercell_stack: np.ndarray,
    nuclei_labels: np.ndarray,
    alpha: float,
    nuclei_blur_sigma: float,
) -> np.ndarray:
    membrane_stack = normalize_uint8(membrane_stack)
    intercell_stack = normalize_uint8(intercell_stack)

    if membrane_stack.shape != intercell_stack.shape:
        raise ValueError("Membrane and intercellular stacks must have the same shape.")

    z_count, height, width = membrane_stack.shape
    processed = np.empty((z_count, height, width, 2), dtype=np.uint8)
    nuclei_hint = generate_nuclei_hint(nuclei_labels, blur_sigma=nuclei_blur_sigma)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    large_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    small_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    for z in range(z_count):
        subtracted = cv2.addWeighted(membrane_stack[z], 1.0, intercell_stack[z], -alpha, 0)
        subtracted = np.clip(subtracted, 0, 255).astype(np.uint8)

        enhanced = clahe.apply(subtracted)
        background = cv2.morphologyEx(enhanced, cv2.MORPH_OPEN, large_kernel)
        top_hat = cv2.subtract(enhanced, background)
        closed = cv2.morphologyEx(top_hat, cv2.MORPH_CLOSE, small_kernel)
        cleaned = cv2.morphologyEx(closed, cv2.MORPH_OPEN, small_kernel)

        if np.max(cleaned) == 0:
            membrane_clean = np.zeros_like(cleaned)
        else:
            denoised = denoise_wavelet(
                cleaned.astype(np.float32) / 255.0,
                channel_axis=None,
                convert2ycbcr=False,
                rescale_sigma=True,
            )
            membrane_clean = np.clip(np.nan_to_num(denoised) * 255, 0, 255).astype(
                np.uint8
            )

        processed[z, :, :, 0] = membrane_clean
        processed[z, :, :, 1] = nuclei_hint[z]

    return processed


def apply_annotation_mask(
    image_2ch: np.ndarray,
    ground_truth_mask: np.ndarray,
    dilation_pixels: int,
) -> np.ndarray:
    annotated_area = (ground_truth_mask > 0).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    safe_zone = cv2.dilate(annotated_area, kernel, iterations=dilation_pixels) > 0

    masked = image_2ch.copy()
    masked[~safe_zone, :] = 0
    return masked


def process_volume_pair(
    name: str,
    raw_path: Path,
    mask_path: Path,
    output_dir: Path,
    nuclei_model: models.CellposeModel,
    args: argparse.Namespace,
) -> int:
    print(f"Processing {name}")
    raw_volume = to_zhwc(tifffile.imread(raw_path))
    membrane_masks = tifffile.imread(mask_path)

    nuclei_raw = raw_volume[..., args.nuclei_channel]
    membrane_raw = raw_volume[..., args.membrane_channel]
    intercell_raw = raw_volume[..., args.intercellular_channel]

    nuclei_labels, *_ = nuclei_model.eval(nuclei_raw, channels=[0, 0], z_axis=0)
    processed_2ch = preprocess_membrane_stack(
        membrane_raw,
        intercell_raw,
        nuclei_labels,
        alpha=args.alpha,
        nuclei_blur_sigma=args.nuclei_blur_sigma,
    )

    saved_count = 0
    for z, mask_slice in enumerate(membrane_masks):
        if np.max(mask_slice) == 0:
            continue

        training_image = apply_annotation_mask(
            processed_2ch[z],
            mask_slice,
            dilation_pixels=args.dilation_pixels,
        )

        if random.random() < args.nuclei_dropout_rate:
            training_image[:, :, 1] = 0

        image_path = output_dir / f"{name}_{z:03d}_img.tif"
        mask_out_path = output_dir / f"{name}_{z:03d}_masks.tif"

        tifffile.imwrite(image_path, np.transpose(training_image, (2, 0, 1)), imagej=True)
        tifffile.imwrite(mask_out_path, mask_slice)
        saved_count += 1

    print(f"Saved {saved_count} annotated slices for {name}")
    return saved_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a 2-channel Cellpose training dataset from annotated volumes."
    )
    parser.add_argument(
        "--pair",
        action="append",
        nargs=3,
        metavar=("NAME", "RAW_TIF", "MASK_TIF"),
        required=True,
        help="Training volume triplet. Can be passed multiple times.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--nuclei-model-path", type=str, default="nuclei")
    parser.add_argument("--use-gpu", action="store_true")
    parser.add_argument("--nuclei-channel", type=int, default=0)
    parser.add_argument("--membrane-channel", type=int, default=1)
    parser.add_argument("--intercellular-channel", type=int, default=2)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--nuclei-blur-sigma", type=float, default=2.0)
    parser.add_argument("--dilation-pixels", type=int, default=40)
    parser.add_argument("--nuclei-dropout-rate", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=13)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    random.seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    nuclei_model = models.CellposeModel(
        gpu=args.use_gpu,
        pretrained_model=args.nuclei_model_path,
    )

    total_saved = 0
    for name, raw_path, mask_path in args.pair:
        total_saved += process_volume_pair(
            name=name,
            raw_path=Path(raw_path),
            mask_path=Path(mask_path),
            output_dir=args.output_dir,
            nuclei_model=nuclei_model,
            args=args,
        )

    print(f"Dataset ready: {total_saved} training slices in {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
