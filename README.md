# Cellpose Membrane Segmentation

This repository contains a compact microscopy segmentation workflow for training and applying a custom Cellpose membrane model.

The project focuses on two practical problems:

1. Preparing annotated 3D microscopy volumes as 2D Cellpose training examples.
2. Fine-tuning and applying a membrane segmentation model to new TIFF volumes.

Raw microscopy data and generated prediction stacks are intentionally excluded because they are large and dataset-specific. The trained membrane model is included in `models/final_membrane_model` so the prediction workflow can be run with compatible TIFF volumes.

## Repository Layout

```text
cellpose_membrane_public/
  README.md
  requirements.txt
  src/
    preprocess_training_data.py
    train_membrane_model.py
    predict_membrane.py
    train_nuclei_model.py
  models/
    final_membrane_model
  data/
    .gitkeep
```

## Workflow

### 1. Install Dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Cellpose training is much faster with a CUDA-capable GPU and a matching PyTorch installation.

### 2. Prepare Membrane Training Data

`src/preprocess_training_data.py` converts annotated 3D volumes into Cellpose-compatible 2D image/mask pairs. Each output image has two channels:

- cleaned membrane signal
- blurred nuclei-label hint

Example:

```bash
python src/preprocess_training_data.py \
  --pair frame25 /path/to/frame_025.tif /path/to/frame_025_membrane_masks.tif \
  --pair slice50 /path/to/slice50.tif /path/to/slice50_membrane_masks.tif \
  --output-dir data/membrane_training \
  --nuclei-model-path nuclei \
  --use-gpu
```

The script accepts multiple `--pair NAME RAW_TIF MASK_TIF` entries. It handles either `(Z, H, W, C)` or `(C, Z, H, W)` raw TIFF layouts.

### 3. Fine-Tune Cellpose

`src/train_membrane_model.py` trains from the prepared 2-channel dataset using the Cellpose command-line interface.

```bash
python src/train_membrane_model.py \
  --train-dir data/membrane_training \
  --base-model tissuenet \
  --epochs 200 \
  --learning-rate 0.05 \
  --model-name membrane_hint_aware_model \
  --use-gpu
```

The training data should follow Cellpose naming conventions:

```text
sample_000_img.tif
sample_000_masks.tif
sample_001_img.tif
sample_001_masks.tif
```

### 4. Run Inference

`src/predict_membrane.py` applies a trained Cellpose model slice-by-slice to new TIFF volumes.

```bash
python src/predict_membrane.py \
  --input-dir /path/to/raw_volumes \
  --output-dir data/predictions \
  --model-path models/final_membrane_model \
  --membrane-channel 1 \
  --intercellular-channel 2
```

The output is one labeled TIFF stack per input volume, saved as `<input_name>_pred.tif`.

## Optional Nuclei Model Training

`src/train_nuclei_model.py` is included because the membrane workflow can use nuclei labels as spatial hints. It fine-tunes a Cellpose nuclei model from annotated 3D nuclei volumes by extracting labeled 2D slices.

```bash
python src/train_nuclei_model.py \
  --raw-dir /path/to/nuclei/raw \
  --masks-dir /path/to/nuclei/masks \
  --slice-dir data/nuclei_slices \
  --epochs 100 \
  --use-gpu
```

## Notes

- Large microscopy volumes, intermediate TIFF stacks, generated flow files, and visualizations are excluded from this public version.
- The code is written as scripts rather than a package so the workflow is easy to inspect and run step by step.
- Paths are provided through CLI arguments to keep the repository portable across machines and datasets.
