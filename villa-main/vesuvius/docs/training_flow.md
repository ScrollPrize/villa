# Training Flow Documentation: From Script to Model

This document provides a comprehensive overview of the training pipeline, from initial script execution through model training and checkpoint saving.

Note that significant portions of this pipeline were borrowed/inspired by [dynamic-network-architectures](https://github.com/MIC-DKFZ/dynamic-network-architectures) and [nnUNetv2](https://github.com/MIC-DKFZ/nnUNet), both from the Division of Medical Image Computing, German Cancer Research Center (DKFZ). Their work is very much appreciated and has been the foundation from which i've learned anything at all about machine learning. I'd highly recommend checking out any of their repositories. 

## Table of Contents
1. [High-Level Training Flow](#high-level-training-flow)
2. [Entry Point: Training Script](#entry-point-training-script-initialization)
3. [Configuration Management](#configuration-management)
4. [Model Architecture Building](#model-architecture-building)
5. [Dataset Pipeline](#dataset-pipeline)
6. [Training Loop](#training-loop)
7. [Key Features](#key-features-and-design-patterns)
8. [Error Handling](#error-handling-and-validation)
9. [Output and Results](#output-and-results)
10. [Auxiliary Tasks](#auxiliary-tasks)

## High-Level Training Flow

![Training Pipeline Flow](images/training_flow_diagram.png)


## 1. Entry Point: Training Script Initialization

### 1.1 Command Line Entry (`vesuvius.train`)
Invoke the trainer through the bundled console script:

```bash
vesuvius.train --config path/to/config.yaml --input /path/to/data --output ./checkpoints
```

When running directly from the repository, use `python -m vesuvius.models.training.train` with the same flags.

### 1.2 Argument Processing
- **Required arguments**:
  - `--config` / `--config-path`: Training YAML containing `tr_setup`, `tr_config`, and `dataset_config`.
  - `-i/--input`: Dataset root with `images/` and `labels/` subdirectories (or an active napari session when `--format napari`).

- **Common optional arguments** (see `vesuvius.train --help` for the full list):
  - `-o/--output`: Directory for checkpoints and configs (defaults to `./checkpoints`).
  - `--format`: Override automatic detection (`image`, `zarr`, or `napari`).
  - `--val-dir`: Separate validation dataset mirroring the training layout.
  - `--batch-size`, `--patch-size`, `--loss`: Override values from the YAML.
  - `--train-split`: Split ratio when a dedicated validation set is not provided.
  - `--skip-intensity-sampling` / `--no-skip-intensity-sampling`: Control per-dataset intensity statistics.
  - `--gpus`, `--ddp`, `--nproc-per-node`: Distributed training controls.
  - `--wandb-project`, `--wandb-entity`, `--verbose`: Logging and diagnostics.

## 2. Configuration Management

### 2.1 ConfigManager Initialization

The `ConfigManager` class serves as the central configuration hub:

1. **Load Configuration**:
   - If `--config-path` provided: Load from YAML file
   - Otherwise: Initialize with default values
   
2. **Configuration Structure**:
   ```yaml
   tr_setup:       # Training setup (model name, checkpoints, etc.)
   tr_config:      # Training parameters (lr, epochs, batch size)
   model_config:   # Model architecture settings
   dataset_config: # Dataset parameters
   inference_config: # Inference settings
   ```

3. **Dynamic Configuration Updates**:
   - Command-line arguments override config file values
   - Auto-detection of targets from data directory structure
   - Dimensionality detection (2D vs 3D) based on data

### 2.2 Target Configuration
Targets represent what the model predicts (e.g., ink detection, damage segmentation):

```python
targets = {
    "ink": {
        "out_channels": 1,
        "activation": "sigmoid",
        "losses": [
            {
                "name": "SoftDiceLoss",
                "weight": 1.0,
                "kwargs": {}
            }
        ]
    },
    "damage": {
        "out_channels": 1,
        "activation": "sigmoid",
        "losses": [
            {
                "name": "BCEWithLogitsLoss",
                "weight": 1.0,
                "kwargs": {}
            }
        ]
    }
}
```

## 3. Model Architecture Building

### 3.1 Auto-configuration Process

The `NetworkFromConfig` class builds a flexible U-Net architecture:

1. **Auto-configuration Mode** (if enabled):
   - Analyzes patch size and spacing
   - Calculates optimal pooling operations
   - Determines number of stages and features per stage
   - Adjusts patch size for divisibility requirements

2. **Architecture Components**:
   - **Shared Encoder**: Processes input through downsampling stages
   - **Task-specific Decoders**: One decoder per target
   - **Activation Functions**: Applied per task (sigmoid, softmax, none)

3. **Dimensionality Handling**:
   - Automatically selects 2D or 3D operations based on patch size
   - Conv2D/Conv3D, MaxPool2D/MaxPool3D, etc.

### 3.2 Architecture Decision Flow

The model is built by the NetworkFromConfig class, which determines the network architecture based on input characteristics:

![Architecture Decision Flow](images/architecture_decision_flow.png)

#### Pooling Operation Determination

The pooling decision algorithm follows these steps:

1. **Initialize** with current spacing and patch size
2. **Iteratively check** for each potential pooling operation:
   - **Size constraint**: Can we downsample by 2x and maintain minimum feature map size?
   - **Spacing ratio**: Is the current spacing less than 2x the minimum spacing?
   - **Max operations**: Have we reached the maximum number of pooling operations?
3. **Update parameters** when valid axes are found:
   - Double the spacing for pooled axes
   - Halve the patch size
   - Record pooling kernel sizes
4. **Finalize patch size** to ensure divisibility by 2^(num_pool_per_axis)

#### Stage and Feature Configuration

- **Number of stages** = Number of pooling operations + 1
- **Blocks per stage**:
  - Stage 0: 1 block
  - Stage 1: 3 blocks
  - Stage 2: 4 blocks
  - Stage 3+: 6 blocks each
- **Features per stage**: min(base_features × 2^i, max_features)
  - Default: base_features = 32, max_features = 320

### 3.3 Adaptive Channel Configuration

The model supports flexible input/output channel configurations:

![Adaptive Channel Flow](images/adaptive_channel_flow.png)

#### Input Channel Detection

Input channels are determined from (in order of priority):
1. ConfigManager.in_channels setting
2. Automatic detection from dataset
3. Default: 1 channel

#### Output Channel Configuration

For each task, output channels are determined by:
- If task specifies `channels` or `out_channels`: Use specified value
- Otherwise: Match input channels (adaptive behavior)

This allows for flexible configurations like:
```python
# 3-channel input example
targets = {
    "adaptive_task": {},                    # Outputs 3 channels (matches input)
    "fixed_task": {"channels": 1},         # Outputs 1 channel (specified)
    "multi_class": {"channels": 5}         # Outputs 5 channels (specified)
}
```

## 4. Dataset Pipeline

### 4.1 Dataset Pipeline Flow

![Dataset Pipeline Flow](images/dataset_flow_diagram.png)

### 4.2 Data Organization

Expected directory structure:
```
data_path/
├── images/
│   ├── image1.zarr/      # Shared across all targets
│   └── image2.zarr/
├── labels/
│   ├── image1_ink.zarr/  # Target-specific labels
│   ├── image1_damage.zarr/
│   └── ...
```

### 4.3 Dataset Processing Flow

1. **Volume Initialization** (`_initialize_volumes`):
   - Loads data lazily
   - Groups data by target and image
   - Validates all configured targets have data

2. **Valid Patch Extraction** (`_get_valid_patches`):
   - Finds regions with sufficient labeled voxels in each target
   - Filters by `min_labeled_ratio` and `min_bbox_percent`
   - Stores patch coordinates for training and validation loaders

3. **Patch Retrieval** (`__getitem__`):
   - Get patch coordinates from valid_patches list
   - Extract image patch and normalize to [0,1]
   - Extract label patches for all targets
   - Derive auxiliary tensors (distance transforms, normals, etc.) when requested
   - Apply augmentations (2D: albumentations, 3D: custom)
   - Convert to PyTorch tensors with proper dimensions
   - Return dictionary: {"image": tensor, "target1": tensor, ...}

### 4.4 Data Augmentation
- **2D Data**: Uses albumentations library
  - Synchronized transforms for the image and every target label volume
  - Configured in `compose_augmentations` function
  
- **3D Data**: Custom volume transformations
  - Currently limited to image-only transforms

### 4.5 Slice Sampling Mode (2D slices from 3D volumes)

Set `dataset_config.slice_sampling.enabled: true` when you want to train a 2D network while keeping your input volumes in their native 3D format. The dataset will draw orthogonal planes on-the-fly, producing `[C, H, W]` tensors that can be fed directly into a 2D model.

Key options (all live under `dataset_config.slice_sampling`):

- `enabled`: Turn the mode on. When `true`, the trainer forces 2D ops and disables cached patch indices.
- `sample_planes`: Which axes to sample from. Accepts:
  - A list (`["z", "y", "x"]`) or comma-separated string (`"z,y,x"`).
  - A mapping of plane → weight (`{z: 2, y: 1, x: 1}`) if you want to specify weights inline.
- `sample_rates` (optional): Relative sampling weights per plane when `sample_planes` is a list/string. Provide a dict (`{z: 2, y: 1}`), list (`[2, 1, 1]`), or comma-separated string (`"2,1,1"`). Probabilities are normalised internally so the actual volume of available patches does not affect plane frequency.
- `plane_patch_sizes` (optional): Override the 2D patch size per plane, e.g. `plane_patch_sizes: {y: [64, 256]}` for `(depth, width)` slices. When omitted, the trainer derives sizes from the original 3D patch (z, y, x → [y, x], [z, x], [z, y]).
- `random_rotations` (optional): Enable oblique slicing. Set to a list (`["x","y"]`), boolean (`true` for all planes), or mapping with per-plane limits. Example:

```yaml
slice_sampling:
  random_rotations:
    x: 360      # degrees of freedom around the volume centre (x-plane becomes arbitrary vertical slices)
    y:
      max_degrees: 180
      probability: 0.35   # apply a rotation ~35% of the time for y-planes
```

When enabled for `x` or `y`, each sampled patch draws a fresh rotation angle (uniform in ±`max_degrees`/2) and interpolates the image volume along that plane (labels use nearest-neighbour sampling). This keeps z-planes axis-aligned but lets the vertical planes sweep diagonally through the scroll while staying centred on the same voxel window.
- `random_tilts` (optional): Tilt planes around the global axes to produce fully oblique slices (e.g., tilt a `z` plane in `x` and `y`). Provide per-plane dictionaries of axis → degrees. Example:

```yaml
slice_sampling:
  random_tilts:
    z:
      x: 25
      y: 25
      probability: 0.5    # only tilt the z plane for half the samples
    y:
      x: 15               # lean y-planes by up to ±15° around the x-axis
```

Tilt and rotation angles combine (internally using `grid_sample`), so you can mix yaw + pitch to sample arbitrary planes that still pass through the same patch centre.
- `label_interpolation` (optional): For specific targets, request linear interpolation (matching the image) when sampling oblique slices, then rebinarise with a 0.5 threshold. Accepts per-target values (`linear`/`nearest`) or per-target/per-plane maps:

```yaml
slice_sampling:
  label_interpolation:
    surface: linear           # apply to all planes
    surface_aux:
      x: linear               # only x-planes get linear/binarised
      y: nearest
```

- `save_plane_masks` (optional, default `false`): When set, the dataset emits a binary mask describing the sampled plane. Useful for debugging—see the sampler script's `--save-masks` flag.
- `plane_mask_mode` (optional, default `plane`): Controls the shape of the emitted mask when `save_plane_masks` is enabled. Use `plane` (the default) for a 2D mask matching the slice patch size, or `volume` to pad/crop masks to the largest source volume when you need full-volume context.

Example snippet:

```yaml
dataset_config:
  patch_size: [64, 256, 256]   # original 3D dimensions (z, y, x)
  slice_sampling:
    enabled: true
    sample_planes:
      z: 2        # sample z-plane slices twice as often
      y: 1
    plane_patch_sizes:
      y: [64, 256]  # (depth, width) for y-planes
```

Additional behaviour:

- Patch validation works per plane. A slice is considered valid when the 2D window meets `min_bbox_percent` and `min_labeled_ratio` thresholds using the projected labels.
- When sampling weights are provided the dataloader uses a `WeightedRandomSampler` (single-GPU training) so each epoch reflects the requested plane ratios. In distributed runs the sampler falls back to uniform subset sampling per worker.
- Patch caches are skipped in this mode because additional metadata (plane, slice index) is required for reconstruction.
- The dataset marks slice origin (`plane`, `slice_index`, `patch_size`) in `valid_patches`, and these fields propagate to train/val split summaries.

## 5. Training Loop

### 5.1 Training Loop Flow

![Training Loop Flow](images/training_loop_diagram.png)

### 5.2 Training Components

1. **Optimizer Configuration**:
   - Default: AdamW with configurable learning rate
   - Alternative: SGD or other optimizers via config

2. **Loss Functions**:
   - Per-target loss functions
   - Masked losses (compute only on valid regions)
   - Weighted multi-task loss: `total_loss = Σ(weight_i * loss_i)`

3. **Learning Rate Scheduler**:
   - CosineAnnealingLR by default
   - Adjusts learning rate over epochs

## 6. Auxiliary Tasks

Auxiliary tasks let you add side objectives that are derived from a primary “source” target (e.g., distance transform or surface normals of a binary mask). They use the shared encoder with separate decoders and integrate into the same multi-task loss.

- Configuration: Add an `auxiliary_tasks` block to your YAML. Each entry must specify:
  - `type`: One of `distance_transform`, `surface_normals`.
  - Also supported: `structure_tensor` for supervising tensor components.
  - `source_target`: The primary target name to derive from (e.g., `ink`, `surface`).
  - `losses`: List of losses and weights for the auxiliary head.
  - `loss_weight`: Optional overall weight for the task.
  - `out_channels`: Optional for `distance_transform` (defaults to 1); for `surface_normals` it is set automatically based on dimensionality (2 for 2D, 3 for 3D).

Example (excerpt):

```yaml
dataset_config:
  targets:
    surface:
      losses:
        - name: "BCEWithLogitsLoss"
          weight: 1.0

auxiliary_tasks:
  distance_transform:
    type: distance_transform
    source_target: surface
    loss_weight: 0.10
    # Choose which distance to predict: 'signed' | 'inside' | 'outside'
    distance_type: signed
    losses:
      - name: "SignedDistanceLoss"
        weight: 1.0
        kwargs:
          rho: 6
          beta: 1
          eikonal: true

  surface_normals:
    type: surface_normals
    source_target: surface
    loss_weight: 0.10
    losses:
      - name: "CosineSimilarityLoss"
        weight: 1.0
        kwargs: { dim: 1 }
      - name: "NormalSmoothnessLoss"
        weight: 0.5
        kwargs: { sigma: 2.0, q: 2.0 }
```

Structure tensor example:

```yaml
auxiliary_tasks:
  structure_tensor:
    type: structure_tensor
    source_target: surface
    # Compute from: 'sdt' (signed distance transform) or 'binary' mask
    compute_from: sdt
    grad_sigma: 1.0       # pre-gradient smoothing
    tensor_sigma: 1.5     # post-tensor smoothing
    loss_weight: 0.1
    losses:
      - name: "MaskedMSELoss"
        weight: 1.0
        kwargs:
          ignore_index: -100
```

In-plane direction example:

```yaml
auxiliary_tasks:
  inplane_direction:
    type: inplane_direction
    source_target: surface
    compute_from: sdt    # or 'binary'
    grad_sigma: 1.0
    tensor_sigma: 1.5
    loss_weight: 0.1
    losses:
      - name: "MaskedMSELoss"
        weight: 1.0
        kwargs:
          ignore_index: -100
```

How it works in code:
- Config injection: `ConfigManager` reads `auxiliary_tasks` and calls a small factory to append derived targets into `mgr.targets` with `auxiliary_task: true` and a reference to `source_target`.
- Dataset loading: Datasets only load label volumes for primary targets. Auxiliary targets do not require separate label files and are not looked up on disk.
- Loss computation: During training, losses are computed per key present in the batch label dict. The helper `compute_auxiliary_loss` passes `source_pred=outputs[source_target]` to losses that accept it. This enables:
  - Self-consistency/regularization losses that don’t need explicit GT (e.g., `NormalSmoothnessLoss` using only predicted normals with optional masking from `source_pred`).
  - Losses that need derived GT (e.g., `SignedDistanceLoss`, cosine loss on normals) use the dataset-provided tensors for aux targets.
  - Tip: If you use `SignedDistanceLoss`, set `distance_type: signed` (default). For `inside`/`outside` distances, use regression losses like `MaskedMSELoss` or `WeightedSmoothL1Loss`.
  - For deep supervision, auxiliary targets default to `ds_interpolation: linear`, mapping to bilinear (2D) or trilinear (3D) in downsampling to preserve regression continuity.

Important notes and current behavior:
- The dataset now auto-generates ground-truth tensors for `distance_transform` (signed DT) and `surface_normals` per patch from the `source_target` labels, so related losses work out of the box.
- Auxiliary losses that are self-supervised or use only predictions (e.g., `NormalSmoothnessLoss`) also work and can be masked using `source_pred` passed through `compute_auxiliary_loss`.
- If you prefer, you can still precompute and store auxiliary labels on disk using the same naming convention; the dataset will ignore them unless you wire them explicitly.

This design allows mixing fully supervised targets with auxiliary/self-supervised regularizers and keeps disk I/O simple by deriving common auxiliary targets on the fly.

4. **Mixed Precision Training**:
   - GradScaler for CUDA devices
   - Automatic mixed precision 


### 5.3 Checkpoint Management
- Saves model weights, optimizer state, and configuration
- Keeps the three most recent checkpoints plus the two best validation checkpoints
- Maintains the latest configuration file per run
- Model-specific directory: `checkpoints/model_name/`

## 6. Key Features and Design Patterns

### 6.1 Multi-Task Learning
- Single shared encoder extracts features
- Multiple decoders predict different targets if using multi-task
- Weighted loss combination

### 6.2 Flexible Data Handling
- Supports 2D and 3D data
- Multiple data formats (Zarr, TIFF, Napari)
- Semi-supervised workflows via `allow_unlabeled_data`

### 6.3 Configuration-Driven Design
- YAML-based configuration
- Command-line overrides
- Auto-configuration from data
- Embedded config in checkpoints

## 7. Output and Results

### 7.1 Training Outputs

The training process produces several types of outputs:

**1. Model Checkpoints** (`model_name_epoch.pth`)
   - Complete training state for resuming
   - Saved after each epoch
   - Automatic cleanup retains three recent checkpoints and the two best by validation loss
   - Location: `checkpoints/model_name/`

**2. Configuration Files** (`model_name_config.yaml`)
   - Complete reproducible settings
   - Includes all model, training, and dataset parameters
   - Saved alongside checkpoints
   - Only the latest config is kept (older ones are redundant)

**3. Debug Visualizations** (`model_name_debug.gif/png`)
   - Visual representation of model predictions vs ground truth
   - Generated during validation on the first batch
   - GIFs for 3D data, PNGs for 2D data
   - Useful for monitoring training progress

**4. Training Logs**
   - Console output with loss values and progress
   - Per-task loss tracking
   - Validation metrics

### 7.2 Checkpoint Contents

Each checkpoint file contains:

```python
{
    'model': model.state_dict(),           # Model weights
    'optimizer': optimizer.state_dict(),   # Optimizer state (momentum, etc.)
    'scheduler': scheduler.state_dict(),   # Learning rate scheduler state
    'epoch': epoch,                        # Current epoch number
    'model_config': model.final_config     # Complete model configuration
}
```
