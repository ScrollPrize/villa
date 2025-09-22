# Preparing Data for Vesuvius Training

The Vesuvius trainer can ingest volumes exported from napari, TIFF collections, or Zarr stores. This guide explains how to organize files and configure targets for common segmentation workloads.

## Supported Ingestion Paths

### Napari Sessions

The `vesuvius.napari_trainer` utility reads directly from an open napari viewer. Name layers so that each label layer shares the image prefix:

```
image_layer_name: "scroll_001"
label_layer_name: "scroll_001_ink"   # becomes target "ink"
mask_layer_name: "scroll_001_mask"    # optional label mask
```

Napari exports are copied in-memory; you do not need to persist `.zarr` files manually unless you want to reuse them outside napari.

### File-Based Datasets (TIFF or Zarr)

Provide a directory with `images/` and `labels/` subfolders. Each label file must end with the target name that it represents:

```
data_path/
├── images/
│   ├── fragment01.tif
│   ├── fragment01_ink.tif      # TIFF with the same voxels as the labels
│   └── fragment02.tif
└── labels/
    ├── fragment01_ink.tif
    ├── fragment01_damage.tif   # second task for the same fragment
    └── fragment02_ink.tif
```

For Zarr input, each entry is a directory ending with `.zarr`. Use the same naming convention:

```
images/
├── fragment01.zarr/
├── fragment01_ink.zarr/
└── fragment01_damage.zarr/
labels/
├── fragment01_ink.zarr/
└── fragment01_damage.zarr/
```

The loader automatically matches `fragment01_damage` labels to either `fragment01_damage.zarr` or the shared `fragment01.zarr` image.

## Configuring Targets

Targets are defined in the training YAML under `dataset_config.targets`. Each key describes a prediction head and its losses.

### Binary Segmentation Example

```yaml
tr_setup:
  model_name: "ink_segmentation"

dataset_config:
  normalization_scheme: "zscore"
  skip_patch_validation: false
  targets:
    ink:
      out_channels: 1
      activation: "sigmoid"
      losses:
        - name: "BCEWithLogitsLoss"
          weight: 0.5
        - name: "SoftDiceLoss"
          weight: 0.5
```

Expect labels with background `0` and any positive value for foreground. Store them either as a single-channel volume with values `{0, 1}` or the raw grayscale mask produced by annotation tools. The loader converts non-zero entries to foreground when it evaluates patch coverage.

### Multi-Class Segmentation

Provide one label volume per voxel target with integer class IDs. Configure the head with the desired number of channels and a softmax-friendly loss:

```yaml
tr_setup:
  model_name: "material_types"

dataset_config:
  targets:
    materials:
      out_channels: 4        # background + three classes
      activation: "softmax"
      losses:
        - name: "nnUNet_DC_and_CE_loss"
          weight: 1.0
```

If your annotations are already one-hot encoded (multiple channels), store them as multi-channel arrays. The loss functions can operate on either one-hot encodings or single-channel class maps.

### Multi-Task Learning

Add one entry per task. Each target receives its own decoder and loss configuration.

```yaml
tr_setup:
  model_name: "ink_and_damage"

dataset_config:
  targets:
    ink:
      out_channels: 1
      activation: "sigmoid"
      losses:
        - name: "SoftDiceLoss"
          weight: 1.0
    damage:
      out_channels: 1
      activation: "sigmoid"
      losses:
        - name: "BCEWithLogitsLoss"
          weight: 1.0
```

Label files must exist for each task (e.g., `fragment01_ink.tif` and `fragment01_damage.tif`). If one task lacks annotations for a volume, leave that file out; the loader can skip unlabeled pairs when `allow_unlabeled_data: true` is set in the config.

### Auxiliary Targets

The trainer can synthesize auxiliary regression targets on the fly. Declare them under `auxiliary_tasks` and reference the source classification task:

```yaml
auxiliary_tasks:
  distance_transform:
    type: distance_transform
    source_target: ink
    loss_weight: 0.1
    losses:
      - name: "SignedDistanceLoss"
        weight: 1.0
```

During training the dataset computes distance transforms from the `ink` labels; you do not need to save additional volumes unless you want complete control over the tensors.

## Dataset Parameters Worth Tuning

- `normalization_scheme`: one of `zscore`, `instance_zscore`, `instance_minmax`, `ct`, or `none`.
- `skip_patch_validation`: skip the expensive valid-patch search when labels are dense.
- `min_labeled_ratio`: minimum fraction of labeled voxels required for a patch to be considered valid (default `0.10`).
- `min_bbox_percent`: bounding-box coverage threshold used during patch validation (default `0.95`).
- `allow_unlabeled_data`: permit volumes without labels so you can mix supervised and unsupervised data.
- `slice_sampling`: optional configuration for extracting 2D slices from 3D inputs.

All fields are documented in `vesuvius/models/configuration/config_manager.py` and in the sample YAML files under `vesuvius/models/configuration/`.

## Label Encoding Guidelines

- Use unsigned integer dtypes (`uint8`, `uint16`, …) whenever possible.
- Background must be zero; any positive value is treated as foreground when computing valid patches.
- For multi-class tasks store consecutive integers starting at zero. There is no automatic remapping, so ensure that the values in the label volume match the configured number of channels.
- When exporting from napari, choose the “labels” layer type to obtain integer masks instead of floating-point overlays.

## Verifying a Dataset

`vesuvius.train` validates the dataset while it warms up the dataloaders. To perform a quick dry run that checks file layout, patch sampling, and normalization without committing to a full training session:

```bash
vesuvius.train \
  --config path/to/config.yaml \
  --input /path/to/data \
  --output ./checkpoints/debug \
  --max-epoch 1 \
  --max-steps-per-epoch 1 \
  --max-val-steps-per-epoch 1 \
  --skip-intensity-sampling
```

Watch the log for messages about missing targets, unlabeled patches, or normalization warnings. The command above produces a single forward/backward pass and exits.

## Quick Reference

| Scenario | Data Layout | Target Config | Label Notes |
|----------|-------------|---------------|-------------|
| Binary segmentation | One label file per volume with `{0, 1}` values | `out_channels: 1`, `activation: sigmoid` | Any positive voxel counts as foreground |
| Multi-class segmentation | Single label map with integer class IDs | `out_channels: N`, `activation: softmax`, CE/Dice losses | Provide sequential integers `[0, N-1]` |
| Multi-task | Separate label file per target | Multiple entries in `dataset_config.targets` | Targets share the same image volume |
| Auxiliary regression | Primary label only | Configure under `auxiliary_tasks` | Auxiliary tensors are derived automatically |
