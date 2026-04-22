# Training a 3D UNet to Predict Lasagna Data Channels

## Overview

The lasagna surface optimizer consumes dense volumetric channels at downsampled
(step) resolution. Currently the 2D UNet produces these per-slice, and a 3-axis
fusion step assembles them into 3D volumes. A trained **3D UNet** could learn to
predict these channels directly from the raw scroll CT volume, replacing both the
2D UNet inference and the fusion pipeline.

The representation we train matches the **2D UNet output format** — not the
final fused lasagna format. Specifically:

- **cos**: cosine-encoded winding position (not a binary mask)
- **grad_mag**: sheet density / gradient magnitude
- **3×2 direction channels**: per-axis double-angle direction encoding (6 ch)

The `pred_dt` channel is derived from a separate surface-prediction UNet and is
**not** trained here.

This document describes:

1. The 2D UNet output representation (what we train)
2. The fit-data output that provides training supervision
3. How to derive training targets from fit-data output
4. Post-inference conversion to the lasagna input format
5. A training plan using the vesuvius library


---

## 1. The 2D UNet Output Representation (Training Target)

The existing 2D UNet processes CT slices along each of the three orthogonal axes
(z, y, x) and outputs **4 channels per slice**:

| Channel  | Name       | Range   | Semantics |
|----------|------------|---------|-----------|
| 0        | `cos`      | [0, 1]  | Cosine-encoded winding position |
| 1        | `grad_mag` | [0, 1+] | Sheet density = `|∇(fractional_winding)|` |
| 2        | `dir0`     | [0, 1]  | Direction encoding: `0.5 + 0.5·cos(2θ)` |
| 3        | `dir1`     | [0, 1]  | Direction encoding: `0.5 + 0.5·cos(2θ + π/4)` |

### cos — cosine-encoded winding position

The cos channel encodes the **fractional winding position** as a periodic signal:

```
cos = 0.5 + 0.5 · cos(2π · winding)
```

This gives:
- **1.0** at integer winding values (on a sheet)
- **0.0** at half-integer winding values (between sheets)
- **1.0** again at the next integer (on the next sheet)

The winding number increases monotonically through the scroll layers. The cosine
encoding turns this linear ramp into a periodic signal where sheets appear as
peaks. The 2D UNet `compute_geom_targets()` uses:
```python
cos_gt = 0.5 - 0.5 * cos(π * mono_clamped)  # mono in [0,1] = fractional position
```

### grad_mag — sheet density

The gradient magnitude of the fractional winding position field. Measures how
fast the winding position changes spatially = how densely packed sheets are.
High grad_mag = sheets close together; low = sheets far apart.

Derived via finite differences of the blurred fractional winding field:
```python
gx = f_blur[:,:,:,1:] - f_blur[:,:,:,:-1]
gy = f_blur[:,:,1:,:] - f_blur[:,:,:-1,:]
grad_mag = sqrt(gx² + gy² + ε)
```

### dir0, dir1 — double-angle direction encoding

Encode the local sheet normal direction projected into the slicing plane using
a 180°-symmetric double-angle representation (documented in `modeling.md`):

```
θ = atan2(gy, gx)              # gradient direction in the slice plane
dir0 = 0.5 + 0.5·cos(2θ)
dir1 = 0.5 + 0.5·cos(2θ + π/4)
```

Equivalent without computing θ explicitly:
```
r² = gx² + gy² + ε
cos2θ = (gx² - gy²) / r²
sin2θ = 2·gx·gy / r²
dir0 = 0.5 + 0.5·cos2θ
dir1 = 0.5 + 0.5·(cos2θ - sin2θ) / √2
```

This encoding is 180°-symmetric (direction, not orientation) which is correct
for surface normals that have inherent sign ambiguity.

### The 3×2 representation

When processing all three axes, we get **6 direction channels** total:

| Axis | Slice plane | Gradient components | Channels |
|------|-------------|---------------------|----------|
| z    | XY plane    | gx=nx, gy=ny        | dir0_z, dir1_z |
| y    | XZ plane    | gx=nx, gy=nz        | dir0_y, dir1_y |
| x    | YZ plane    | gx=ny, gy=nz        | dir0_x, dir1_x |

The gradient direction in each slicing plane is the projection of the 3D surface
normal onto that plane. This is why dir encodes the surface orientation — and
why the 3×2 representation is richer than storing just (nx, ny): it preserves
per-axis reliability information that the lasagna model's direction loss uses
(weighting by in-plane projection magnitude).


---

## 2. Fit-Data Output (Training Supervision Source)

`lasagna_fit_data.py` takes a **frozen fitted model** and produces dense 3D
volumes representing the model's surface geometry. These are the source from
which we derive UNet training labels.

### Output volumes

| Volume       | Shape       | dtype   | Range           | Description |
|-------------|-------------|---------|-----------------|-------------|
| **normal**   | (3,Z,Y,X)  | float32 | [-1, 1]         | Per-voxel unit surface normal (nx, ny, nz) |
| **winding**  | (Z,Y,X)    | float32 | [w_min, w_max]  | Continuous winding number (layer index) |
| **validity** | (Z,Y,X)    | float32 | {0, 1}          | Binary: 1 inside model extent, 0 outside |
| **density**  | (Z,Y,X)    | float32 | [0, ∞)          | Sheet density (integral ≈ 1 between layers) |

### Output formats

**Zarr** (`--output path.zarr`):
```
path.zarr/
├── normal      (3, Z, Y, X) float32
├── winding     (Z, Y, X)    float32
├── validity    (Z, Y, X)    float32
├── density     (Z, Y, X)    float32
└── attrs:
    ├── origin_fullres: [x0, y0, z0]
    ├── spacing: [sx, sy, sz]
    └── size: [Z, Y, X]
```


---

## 3. Deriving Training Targets from Fit-Data Output

The 3D UNet should predict **8 channels**: cos (1) + grad_mag (1) + 3×2
direction channels (6). All are derived from the fit-data normal and winding
volumes.

### Channel derivation

```python
import numpy as np

# Load fit-data output (float32)
normal = ...    # (3, Z, Y, X) float32 [-1, 1], unit normals
winding = ...   # (Z, Y, X)   float32, continuous winding number
validity = ...  # (Z, Y, X)   float32 {0, 1}
density = ...   # (Z, Y, X)   float32 [0, ∞)

nx, ny, nz = normal[0], normal[1], normal[2]

# ─── cos: cosine-encoded winding position ───
# Oscillates: 1 on sheet (integer winding), 0 between sheets
cos_target = 0.5 + 0.5 * np.cos(2 * np.pi * winding)

# ─── grad_mag: sheet density ───
# The density field from fit-data IS the sheet density (gradient magnitude
# of the winding field). It can be used directly, or recomputed from winding
# via finite differences if preferred.
grad_mag_target = density
# Alternative: compute from winding field directly
# dwdz = winding[1:] - winding[:-1]; ...
# grad_mag_target = sqrt(dwdx² + dwdy² + dwdz²)

# ─── 3×2 direction channels from 3D normal ───
eps = 1e-8

def _encode_dir(gx, gy):
    """Double-angle direction encoding from 2D gradient vector."""
    r2 = gx * gx + gy * gy + eps
    cos2t = (gx * gx - gy * gy) / r2
    sin2t = (2.0 * gx * gy) / r2
    dir0 = 0.5 + 0.5 * cos2t
    inv_sqrt2 = 1.0 / np.sqrt(2.0)
    dir1 = 0.5 + 0.5 * (cos2t - sin2t) * inv_sqrt2
    return dir0, dir1

# Z-slices (XY plane): gradient direction = (nx, ny)
dir0_z, dir1_z = _encode_dir(nx, ny)

# Y-slices (XZ plane): gradient direction = (nx, nz)
dir0_y, dir1_y = _encode_dir(nx, nz)

# X-slices (YZ plane): gradient direction = (ny, nz)
dir0_x, dir1_x = _encode_dir(ny, nz)

# ─── Validity mask ───
# Use validity to mask supervision: only train where fit-data has data.
# Where validity=0, the UNet output is unsupervised (or use ignore_index).
mask = validity  # (Z, Y, X) float32 {0, 1}
```

### Summary of training targets

| Channel     | Source                       | Formula | Range |
|-------------|------------------------------|---------|-------|
| cos         | winding                      | `0.5 + 0.5·cos(2π·winding)` | [0, 1] |
| grad_mag    | density (or ∇winding)        | direct from fit-data | [0, ∞) |
| dir0_z      | normal (nx, ny)              | `0.5 + 0.5·cos(2·atan2(ny,nx))` | [0, 1] |
| dir1_z      | normal (nx, ny)              | `0.5 + 0.5·cos(2·atan2(ny,nx)+π/4)` | [0, 1] |
| dir0_y      | normal (nx, nz)              | `0.5 + 0.5·cos(2·atan2(nz,nx))` | [0, 1] |
| dir1_y      | normal (nx, nz)              | `0.5 + 0.5·cos(2·atan2(nz,nx)+π/4)` | [0, 1] |
| dir0_x      | normal (ny, nz)              | `0.5 + 0.5·cos(2·atan2(nz,ny))` | [0, 1] |
| dir1_x      | normal (ny, nz)              | `0.5 + 0.5·cos(2·atan2(nz,ny)+π/4)` | [0, 1] |


---

## 4. Post-Inference Conversion to Lasagna Input Format

After the 3D UNet produces its 8-channel output, it must be converted to the
lasagna input format (5-channel uint8 zarr). This mirrors the existing
`preprocess_cos_omezarr.py` fusion pipeline.

### Step 1: Fuse 3×2 directions into 3D normal (nx, ny, nz)

Decode each axis's dir0/dir1 back to an angle, then solve for the 3D normal via
cross products of constraint rows (same algorithm as `preprocess_cos_omezarr.py`):

```python
def _decode_dir_angle(dir0, dir1):
    cos2t = 2.0 * dir0 - 1.0
    sin2t = cos2t - np.sqrt(2.0) * (2.0 * dir1 - 1.0)
    return np.arctan2(sin2t, cos2t) * 0.5

theta_z = _decode_dir_angle(dir0_z, dir1_z)  # XY plane
theta_y = _decode_dir_angle(dir0_y, dir1_y)  # XZ plane
theta_x = _decode_dir_angle(dir0_x, dir1_x)  # YZ plane

# Cross-product normal candidates from pairs of constraint rows
n1 = (cos(θz)·cos(θy), sin(θz)·cos(θy), cos(θz)·sin(θy))  # z×y
n2 = (cos(θz)·cos(θx), sin(θz)·cos(θx), sin(θz)·sin(θx))  # z×x
n3 = (cos(θy)·sin(θx), sin(θy)·cos(θx), sin(θy)·sin(θx))  # y×x

# Weighted average → normalize → hemisphere encode
n_avg = sign_align(n1 + n2 + n3)
nx, ny, nz = normalize(n_avg)
```

### Step 2: Hemisphere encoding

```python
flip = np.where(nz < 0, -1.0, 1.0)
nx_hemi, ny_hemi = nx * flip, ny * flip
```

### Step 3: Fuse cos and grad_mag

The 3D UNet directly outputs cos and grad_mag (no per-axis fusion needed since
the 3D UNet already sees all three axes simultaneously).

### Step 4: Encode to uint8 lasagna zarr

```python
cos_u8 = np.clip(np.round(cos * 255), 0, 255).astype(np.uint8)
grad_mag_u8 = np.clip(np.round(grad_mag * scale), 0, 255).astype(np.uint8)
nx_u8 = np.clip(np.round(nx_hemi * 127 + 128), 0, 255).astype(np.uint8)
ny_u8 = np.clip(np.round(ny_hemi * 127 + 128), 0, 255).astype(np.uint8)
# pred_dt comes from a separate surface prediction model
```


---

## 5. Resolution Strategy: Full-Res Input, Step-Res Output

### The problem

The lasagna model operates at **step resolution** (typically fullres / 4), but
the 3D UNet should see the **full-resolution CT** — papyrus sheet signals are
only a few voxels wide at full res and would be lost if downsampled before the
network.

### The solution: UNet at full res → average-pool output

The vesuvius UNet (`NetworkFromConfig`) does not support asymmetric
encoder/decoder (the decoder always mirrors the encoder). Instead we use the
simplest approach:

1. **Train and run the UNet at full resolution** — input and output are both
   full-res
2. **Average-pool the output** by the step factor to get step-resolution
   channels for the lasagna model

This works well because all 8 output channels (cos, grad_mag, 6× dir) are
**smooth signals** that downsample cleanly with average pooling:

- cos oscillates at the winding period (many voxels wide)
- grad_mag varies slowly (sheet density)
- dir0/dir1 are smooth direction encodings

```python
import torch.nn.functional as F

step = 4  # scaledown factor
unet_output = model(ct_fullres)           # (B, 8, Z, Y, X) at full res
output_step = F.avg_pool3d(unet_output,
                           kernel_size=step,
                           stride=step)   # (B, 8, Z/4, Y/4, X/4) at step res
```

### Scale-space supervision

Rather than supervising at a single resolution, we use **scale-space
(multi-resolution) supervision**: the UNet output is mean-pooled at multiple
scales and compared against appropriately scaled versions of the training
labels. This matches the 2D UNet's `ScaleSpaceLoss` approach.

The fit-data labels are at step resolution (fullres / step). We create a
resolution pyramid by up- and downsampling:

```python
import torch.nn.functional as F

step = 4
unet_out = model(ct_fullres)  # (B, C, Z, Y, X) full res

# Build scale-space pyramid of UNet output via successive mean pooling
scales = [1, 2, 4, 8]  # pool factors relative to full res
pooled = {}
for s in scales:
    if s == 1:
        pooled[s] = unet_out
    else:
        pooled[s] = F.avg_pool3d(unet_out, kernel_size=s, stride=s)

# Labels are at step resolution = pool factor `step`
# For scales finer than step: upsample labels with trilinear
# For scales coarser than step: mean-pool labels
# For scale == step: use labels directly
label_step = ...  # (B, C, Z_s, Y_s, X_s) at step res

label_at = {}
for s in scales:
    if s == step:
        label_at[s] = label_step
    elif s < step:
        factor = step // s
        label_at[s] = F.interpolate(label_step, scale_factor=factor,
                                     mode='trilinear', align_corners=False)
    else:
        factor = s // step
        label_at[s] = F.avg_pool3d(label_step, kernel_size=factor,
                                    stride=factor)

# Weighted loss across scales (finer scales get higher weight)
total_loss = 0
weights = {1: 1.0, 2: 0.5, 4: 0.25, 8: 0.125}
for s in scales:
    total_loss += weights[s] * loss_fn(pooled[s], label_at[s])
```

This gives the UNet gradient signal at multiple spatial frequencies:
fine scales enforce local accuracy, coarse scales enforce global consistency.

### Patch sizes

At full resolution, memory limits the patch size. Typical choices:

| Full-res patch | Step-res equivalent | Notes |
|---------------|--------------------:|-------|
| 64³           | 16³                 | Small, fast, many patches |
| 128³          | 32³                 | Good balance |
| 192³          | 48³                 | Larger context, needs more VRAM |

The UNet's receptive field at full res covers the same physical extent as
at step res, so 128³ full-res ≈ 32³ step-res in terms of context.


---

## 6. Training Plan Using the Vesuvius Library

### 6.1 Model Architecture

A 3D UNet from `vesuvius.models.build.NetworkFromConfig`:

- **Input**: 1 channel — raw CT scroll intensity at **full resolution**
- **Output**: 8 channels at full resolution, average-pooled to step resolution

```yaml
model_config:
  in_channels: 1
  autoconfigure: true
  architecture_type: "unet"
```

### 6.2 Target Channels and Losses

```yaml
targets:
  # Cosine-encoded winding position — periodic regression in [0, 1]
  cos:
    out_channels: 1
    activation: "sigmoid"
    losses:
      - name: "MSELoss"
        weight: 1.0

  # Sheet density / gradient magnitude — positive regression
  grad_mag:
    out_channels: 1
    activation: "none"
    losses:
      - name: "SmoothL1Loss"
        weight: 1.0

  # Direction encoding for Z-slices (XY plane)
  dir_z:
    out_channels: 2
    activation: "sigmoid"     # output in [0, 1]
    losses:
      - name: "MSELoss"
        weight: 1.0

  # Direction encoding for Y-slices (XZ plane)
  dir_y:
    out_channels: 2
    activation: "sigmoid"
    losses:
      - name: "MSELoss"
        weight: 1.0

  # Direction encoding for X-slices (YZ plane)
  dir_x:
    out_channels: 2
    activation: "sigmoid"
    losses:
      - name: "MSELoss"
        weight: 1.0
```

**Total output channels**: 1 + 1 + 2 + 2 + 2 = **8 channels**

**Loss rationale:**
- All channels are in [0, 1] range (after sigmoid) → MSE is appropriate
- Direction channels use MSE (same as the 2D UNet's `compute_geom_losses()`)
- `MaskedMSELoss` can be used to restrict supervision to valid regions (where
  fit-data validity > 0)

### 6.3 Data Preparation

#### Input (CT volume)

The raw scroll CT volume at **full resolution**. Either:

1. Crop the full-resolution scroll zarr to the region of interest
2. Use the vesuvius `Volume` API to load at full resolution (level 0)

#### Labels (derived from fit-data, at step resolution)

The fit-data output is at step resolution. Labels are stored at step res and
the loss is computed after average-pooling the UNet output (see Section 5).

Convert fit-data zarr output into vesuvius-format label zarrs:

```
data/
├── images/
│   └── region01.zarr/            # CT intensity at FULL resolution
│       └── 0/                    # OME-Zarr level 0
├── labels/
│   ├── region01_cos.zarr/        # (1, Z_s, Y_s, X_s) cos-encoded winding (step res)
│   ├── region01_grad_mag.zarr/   # (1, Z_s, Y_s, X_s) sheet density (step res)
│   ├── region01_dir_z.zarr/      # (2, Z_s, Y_s, X_s) dir0_z, dir1_z (step res)
│   ├── region01_dir_y.zarr/      # (2, Z_s, Y_s, X_s) dir0_y, dir1_y (step res)
│   └── region01_dir_x.zarr/      # (2, Z_s, Y_s, X_s) dir0_x, dir1_x (step res)
```

Note: image shape is (1, Z, Y, X) at full res while labels are (C, Z/step,
Y/step, X/step) at step res. The training loop extracts a full-res patch,
runs the UNet, pools the output, and compares to the corresponding step-res
label patch.

#### Conversion script (fit-data → training labels)

```python
"""Convert fit-data zarr to vesuvius training labels."""
import zarr
import numpy as np

def convert_fit_data_to_training_labels(fit_data_path, output_dir):
    root = zarr.open(fit_data_path, mode='r')
    normal = np.array(root['normal'])     # (3, Z, Y, X)
    winding = np.array(root['winding'])   # (Z, Y, X)
    validity = np.array(root['validity']) # (Z, Y, X)
    density = np.array(root['density'])   # (Z, Y, X)

    nx, ny, nz = normal[0], normal[1], normal[2]

    # cos: cosine-encoded winding
    cos_label = 0.5 + 0.5 * np.cos(2 * np.pi * winding)
    cos_label *= validity  # zero outside valid region

    # grad_mag: sheet density
    grad_mag_label = density * validity

    # 3×2 direction encoding
    eps = 1e-8
    def _encode_dir(gx, gy):
        r2 = gx*gx + gy*gy + eps
        cos2t = (gx*gx - gy*gy) / r2
        sin2t = 2.0*gx*gy / r2
        d0 = 0.5 + 0.5 * cos2t
        d1 = 0.5 + 0.5 * (cos2t - sin2t) / np.sqrt(2.0)
        return d0, d1

    dir0_z, dir1_z = _encode_dir(nx, ny)
    dir0_y, dir1_y = _encode_dir(nx, nz)
    dir0_x, dir1_x = _encode_dir(ny, nz)

    # Write zarrs ...
    # cos_label:     (1, Z, Y, X)
    # grad_mag_label:(1, Z, Y, X)
    # dir_z:         (2, Z, Y, X) = stack(dir0_z, dir1_z)
    # dir_y:         (2, Z, Y, X) = stack(dir0_y, dir1_y)
    # dir_x:         (2, Z, Y, X) = stack(dir0_x, dir1_x)
    # validity:      (1, Z, Y, X) as ignore mask
```

### 6.4 Training Configuration

```yaml
tr_setup:
  model_name: "lasagna_3d_field_predictor"
  autoconfigure: true
  ckpt_out_base: "./checkpoints/"
  tr_val_split: 0.85

tr_config:
  patch_size: [128, 128, 128]    # 3D patches at FULL resolution
  batch_size: 2
  max_epoch: 2000
  initial_lr: 0.001
  optimizer: "AdamW"
  weight_decay: 0.01
  max_steps_per_epoch: 250
  max_val_steps_per_epoch: 50
  enable_deep_supervision: false  # we use our own scale-space supervision

dataset_config:
  data_path: "./training_data"
  normalization_scheme: "instance_zscore"
  min_labeled_ratio: 0.05
```

Note: `patch_size` refers to full-resolution voxels. The corresponding label
patch is `[128/step, 128/step, 128/step]` = `[32, 32, 32]` at step=4.

Scale-space supervision (Section 5) replaces the vesuvius built-in deep
supervision: we pool the UNet output at multiple scales (×1, ×2, ×4, ×8) and
compute loss against appropriately scaled labels at each level. This is
implemented in the training loop, not in the vesuvius config.

### 6.5 Training Workflow

```
                    ┌──────────────────┐
                    │  Scroll CT data  │
                    │  (fullres zarr)  │
                    └────────┬─────────┘
                             │
              ┌──────────────┼──────────────┐
              ▼              ▼              ▼
        ┌──────────┐  ┌──────────┐  ┌──────────┐
        │ Region 1 │  │ Region 2 │  │ Region N │
        │  (crop)  │  │  (crop)  │  │  (crop)  │
        └────┬─────┘  └────┬─────┘  └────┬─────┘
             │              │              │
             ▼              ▼              ▼
        labels_to_lasagna_normals.py (preprocess)
             │              │              │
             ▼              ▼              ▼
        fit.py (optimize surface mesh per region)
             │              │              │
             ▼              ▼              ▼
        lasagna_fit_data.py (dense volumes at step res)
             │              │              │
             ▼              ▼              ▼
        derive training labels (step res):
          cos = 0.5+0.5·cos(2π·winding)
          grad_mag = density
          dir_{z,y,x} = encode_dir(normal projections)
             │              │              │
             ▼              ▼              ▼
        ┌──────────────────────────────────────────────┐
        │  Paired training data:                       │
        │  CT patches (full res) ↔ labels (step res)   │
        └──────────────┬───────────────────────────────┘
                       │
                       ▼
              ┌───────────────────────────────────┐
              │  3D UNet (full res)               │
              │  input: (B, 1, Z, Y, X)           │
              │  output: (B, 8, Z, Y, X)          │
              │           ↓ scale-space pooling    │
              │  pool ×1, ×2, ×4, ×8              │
              │           ↓ loss at each scale     │
              │           vs up/down-scaled labels │
              └───────────────────────────────────┘
                       │
                       ▼
              ┌─────────────────────────────────┐
              │  Trained model:                 │
              │  CT(fullres) → 8ch(fullres)     │
              │  → avg_pool(step) → step res    │
              └────────┬────────────────────────┘
                       │
                       ▼
              Fuse 3×2 dirs → 3D normal → hemisphere encode
              Encode cos, grad_mag → uint8
              + pred_dt from surface prediction UNet
                       │
                       ▼
              Write lasagna-format zarr (5-ch uint8, step res)
                       │
                       ▼
              fit.py (surface optimization)
```

### 6.6 Inference — `predict3d` mode

The `predict3d` mode of `preprocess_cos_omezarr.py` runs a trained 3D UNet on
a raw CT zarr volume and writes a preprocessed zarr ready for `fit.py`. It
replaces the 2D pipeline (3 per-axis runs + `integrate` fusion) with a single
pass.

The pipeline: tiled 3D inference with per-tile sigmoid and avg_pool3d
downscaling on GPU, linear-blended into a disk-backed memmap accumulator
→ chunked normal estimation from 3×2 direction channels → uint8 encoding
→ zarr output. Input tiles are read lazily from zarr (no full-crop load).
Uses CUDA by default when available.

#### Basic usage

```bash
python lasagna/preprocess_cos_omezarr.py predict3d \
    --input /path/to/volume.zarr \
    --output /path/to/preprocessed.zarr \
    --unet-checkpoint /path/to/model.pt
```

#### With crop and pred-dt

```bash
python lasagna/preprocess_cos_omezarr.py predict3d \
    --input /path/to/volume.zarr \
    --output /path/to/preprocessed.zarr \
    --unet-checkpoint /path/to/model.pt \
    --crop-xyzwhd 100 200 300 500 400 600 \
    --pred-dt /path/to/prediction.zarr
```

#### Options

| Flag | Description |
|------|-------------|
| `--input` | Input zarr array (3D, ZYX layout). Required. |
| `--output` | Output zarr path. Required. |
| `--unet-checkpoint` | Path to trained 3D UNet `.pt` checkpoint. Required. |
| `--tile-size` | Cube tile size for tiled inference (default 256). Must be compatible with the model architecture. |
| `--overlap` | Overlap between adjacent tiles in voxels (default 64). |
| `--border` | Hard-discard border at tile edges before linear blending (default 16). |
| `--scaledown` | Output downsample power (default 4). OME-Zarr pyramid level; actual factor = 2^scaledown. |
| `--crop-xyzwhd` | Process only a sub-region: `x y z w h d` in fullres input coordinates. |
| `--pred-dt` | Path to a surface prediction zarr. Adds a `pred_dt` distance-to-surface channel to the output. |
| `--device` | Compute device (default: `cuda` if available, otherwise `cpu`). |
| `--chunk-z` | Zarr chunk size along Z in the output (default 32). |
| `--chunk-yx` | Zarr chunk size along Y and X in the output (default 32). |

#### Output format

The output zarr has shape `(C, Z, Y, X)` with dtype uint8, where C is 4
(cos, grad_mag, nx, ny) or 5 if `--pred-dt` is given. The zarr covers the
full scaled volume dimensions; data is written at the crop offset (unprocessed
regions are zero-filled).

Metadata is stored in `preprocess_params` with keys: `scaledown` (OME-Zarr level, power of 2),
`grad_mag_encode_scale` (1000), `channels`, `crop_xyzwhd`, `source`
("predict3d"). This is directly consumed by `fit_data.load_3d()`.

#### Memory considerations

The accumulator is a disk-backed numpy memmap at **downscaled** resolution
(crop dims // scaledown). Input tiles are read lazily from the zarr — the
full crop is never loaded into RAM.

| Component            | RAM usage (2000³ crop, sd=4) |
|----------------------|-----------------------------|
| Input                | ~0 (lazy zarr reads)         |
| Memmap accumulator   | ~4.5 GiB (fits in page cache)|
| Post-processing      | ~0.5 GiB (Z-chunked)         |
| **Total**            | **~5 GiB**                   |

For very large volumes the memmap spills to disk (e.g. ~1.1 TiB temp for a
full 2 TB volume at sd=1), but RSS stays at a few GiB. The process sets
`oom_score_adj=1000` so the OOM killer targets it before the parent session.


---

## 7. Practical Considerations

### Why the 3×2 representation (not direct 3D normal)

Training the 3×2 direction encoding has several advantages over predicting
(nx, ny, nz) directly:

1. **Matches the 2D UNet**: The representation is identical to what the existing
   2D UNet produces. The fusion algorithm is proven and well-understood.
2. **No hemisphere ambiguity**: The double-angle encoding is 180°-symmetric by
   construction — no need for sign-invariant losses or hemisphere conventions
   during training.
3. **Per-axis reliability**: The 3×2 representation preserves which axis
   provides the most reliable direction information. The lasagna dir loss uses
   per-axis weighting (`w_axis = sqrt(1 - n_axis²)`) which requires knowing
   the per-axis direction, not just the fused normal.
4. **Bounded range**: All 6 channels are in [0, 1], making MSE loss well-behaved
   and sigmoid activation natural.

### Why cos is cosine-encoded (not binary)

The cos channel in the 2D UNet is the **periodic cosine signal** that the
lasagna model's data loss compares against:
```
target_plain = 0.5 + 0.5·cos(2π·x_index / periods)
```
The model fits its mesh so that sampled cos values match this pattern. A binary
mask would only indicate "papyrus or not" — the cosine encoding also carries
**fractional position within the winding period**, which is the primary
data-fitting signal.

### Domain gap and generalization

- Fit-data produces smooth, physically consistent labels — ideal for training
- But fitted regions are where the model already works well
- Train on diverse regions (different scroll areas, partially successful fits)
  to improve generalization
- The 3D UNet's advantage: it sees 3D context simultaneously, potentially
  resolving ambiguities that confuse per-slice 2D inference

### Memory considerations

**Full-res patches** are the main memory concern. At full resolution:

| Patch size | Input (1ch) | UNet output (8ch) | Label (8ch, step) | Notes |
|-----------|-------------|--------------------|--------------------|-------|
| 64³       | 1 MB        | 8 MB               | 0.5 MB (16³)       | Fast, many patches |
| 128³      | 8 MB        | 64 MB              | 4 MB (32³)         | Good balance |
| 192³      | 27 MB       | 216 MB             | 14 MB (48³)        | Needs ≥24GB VRAM |

UNet intermediate activations (encoder features) dominate VRAM, not the
input/output tensors. A 6-stage UNet with 128³ patches typically needs 8–16 GB.

### Available losses from vesuvius library

| Loss | Use for | Notes |
|------|---------|-------|
| `MSELoss` | cos, dir channels | Standard for bounded [0,1] regression |
| `SmoothL1Loss` | grad_mag | Robust to outlier densities |
| `MaskedMSELoss` | All (with validity) | Only supervise where fit-data has data |
| `NormalSmoothnessLoss` | Regularizer | Penalizes sharp orientation changes |
| `CosineSimilarityLoss` | Alternative normal loss | If predicting 3D normals directly |
