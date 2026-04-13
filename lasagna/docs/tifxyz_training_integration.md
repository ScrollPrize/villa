# Deriving 3D UNet Training Channels On-the-Fly from Tifxyz Surfaces

## Overview

This document describes how to derive the 8 lasagna training channels (cos,
grad_mag, 6× direction) on-the-fly from tifxyz surfaces during 3D UNet
training, and outlines integration approaches with the vesuvius training
framework.

### Goal

Train a 3D UNet (CT volume → 8-channel field prediction) using tifxyz surfaces
as the source of ground truth, rather than pre-computing label zarrs from
`lasagna_fit_data.py`. This avoids the need for a fitted lasagna model as a
prerequisite and can leverage the large collection of tifxyz surfaces already
used by the neural tracer.

### Current data

- **CT volumes**: zarr arrays at full resolution (ZYX layout)
- **Tifxyz surfaces**: directory-based surface format (`x.tif`, `y.tif`,
  `z.tif`, `meta.json`) representing 2D grids of 3D (x, y, z) coordinates.
  Each surface is a segmented papyrus sheet layer. The vesuvius `Tifxyz` class
  (`vesuvius/src/vesuvius/tifxyz/types.py`) provides:
  - Per-vertex 3D coordinates at stored or full resolution
  - Per-vertex surface normals via `compute_normals()` or tile-based
    `get_normals()` (cross product of central-difference tangent vectors)
  - A validity mask
  - Label loading (e.g. ink labels)
- **Multiple surfaces per region**: when multiple tifxyz surfaces exist in the
  same volume region, they represent distinct sheet layers of the scroll


---

## 1. Available Training Data

### S3 bucket

The neural tracer datasets are available at:

```
s3://philodemos/paul/neural-tracer-datasets/nt_dataset_022026/
```

### Dataset config (6 scrolls)

The following JSON config defines the 6 datasets used for neural tracer
training. Paths shown are local (`/mnt/raid_nvme/...`) — adapt to your
environment (S3, HTTPS `volumes.aws.ash2txt.org`, or local mount).

```json
"datasets": [
    {
        "volume_path": "/mnt/raid_nvme/volpkgs/PHerc0139_ds2.volpkg/volumes/0139_2um_ds2_raw.zarr",
        "volume_scale": 0,
        "segments_path": "/mnt/raid_nvme/datasets/raw/neural_tracer_022826/PHerc0139/tifxyz",
        "z_range": [2000, 7000]
    },
    {
        "volume_path": "/mnt/raid_nvme/volpkgs/Scroll4.volpkg/volumes/s4.zarr",
        "volume_scale": 0,
        "segments_path": "/mnt/raid_nvme/datasets/raw/neural_tracer_022826/PHerc1667/tifxyz",
        "z_range": [3500, 9000]
    },
    {
        "volume_path": "/mnt/raid_nvme/volpkgs/PHercParis4.volpkg/volumes/s1_uint8.zarr",
        "volume_scale": 0,
        "segments_path": "/mnt/raid_nvme/datasets/raw/neural_tracer_022826/PHercParis4/tifxyz",
        "z_range": [1000, 9250]
    },
    {
        "volume_path": "/mnt/raid_nvme/volpkgs/0343P.volpkg/volumes/2.24um_raw_ds2_ome_v2.zarr",
        "volume_scale": 0,
        "segments_path": "/mnt/raid_nvme/datasets/raw/neural_tracer_022826/PHerc0343p/tifxyz",
        "z_range": [1314, 3000]
    },
    {
        "volume_path": "/mnt/raid_nvme/volpkgs/PHercMANBp.volpkg/volumes/PHercMANBp-ct-2um.zarr",
        "volume_scale": 2,
        "segments_path": "/mnt/raid_nvme/datasets/raw/neural_tracer_022826/PHercMANBp/tifxyz",
        "z_range": [1000, 3000]
    },
    {
        "volume_path": "/mnt/raid_nvme/volpkgs/0500P2.volpkg/volumes/111keV_1.2m_scroll-fragment-0500P2_8um.zarr",
        "volume_scale": 0,
        "segments_path": "/mnt/raid_nvme/datasets/raw/neural_tracer_022826/PHerc0500p2/tifxyz",
        "z_range": [1700, 6000]
    }
]
```

### z_range

Each dataset specifies a `z_range: [lo, hi]` defining the safe Z-slice range
for that scroll. Regions outside this range extend beyond the physical scroll
(e.g. above/below the rolled papyrus) and should be excluded from training.
The patch-finding code in `ink-detection/tifxyz_dataset/patch_finding.py`
already filters segments via `_segment_overlaps_z_range()`.

### Volume access

CT volumes are opened via `vesuvius.data.utils.open_zarr()`, which handles
local paths, S3 (`s3://...`), and HTTPS (`https://volumes.aws.ash2txt.org/...`)
natively using fsspec. For authenticated HTTPS access, pass an `auth_json_path`
pointing to a JSON file with `{"username": "...", "password": "..."}`.

The neural tracer's `find_patches()` calls `open_zarr()` per dataset entry,
so switching from local to remote volumes only requires changing `volume_path`.

### Tifxyz surface access

The tifxyz reader (`vesuvius.tifxyz.io.read_tifxyz()`) currently expects a
local filesystem path — it reads `x.tif`, `y.tif`, `z.tif` and `meta.json`
from a directory. For S3-hosted datasets, two options:

1. **Sync locally first** — `aws s3 sync` the tifxyz directories to a local
   path and point `segments_path` there. Simplest approach; surfaces are small
   relative to volumes.
2. **Extend the reader** — add fsspec-backed I/O to `TifxyzReader` so it can
   open TIFF files from S3 directly. Not yet implemented.


---

## 2. What Tifxyz Surfaces Provide

A `Tifxyz` object (`vesuvius.tifxyz.types.Tifxyz`) stores a surface as a 2D
grid where each grid cell has a 3D position `(x, y, z)` in volume coordinates.
Key capabilities:

| Property | Source | Notes |
|----------|--------|-------|
| 3D position (x,y,z) | `_x`, `_y`, `_z` arrays | Stored at reduced resolution; bicubic upsample to full res |
| Surface normals (nx,ny,nz) | `compute_normals()` | Cross product of central differences, cached, NaN at boundaries |
| Validity mask | `_valid_mask` property | `z > 0` and `isfinite(z)`, plus optional `mask.tif` |
| Scale | `_scale` tuple | Grid spacing: stored-res → full-res factor |
| Labels | `load_label(selector)` | Associated label TIFFs (e.g. ink labels) |

The neural tracer's `TifxyzInkDataset` (`ink-detection/tifxyz_dataset/`)
already demonstrates:
- Finding 3D patches that overlap surfaces
  (`ink-detection/tifxyz_dataset/patch_finding.py`)
- Sampling surface grids into volume patches with interpolation
  (`_sample_patch_supervision_grid` in `common.py`)
- Computing normals per grid point (`_estimate_surface_normals_zyx`)
- Voxelizing surface points into 3D volumes (`_voxelize_surface_from_sampled_grid`)
- Reading CT volume crops from zarr (`_read_volume_crop_from_patch_dict`)


---

## 3. Deriving the 8 Training Channels from Tifxyz

The 8 target channels from `3d_unet_training.md` are: **cos** (1ch),
**grad_mag** (1ch), **dir_z** (2ch), **dir_y** (2ch), **dir_x** (2ch).

### 3.1 Direction channels (6ch) — straightforward

The 6 direction channels encode surface normal projections as double-angle
representations. These are directly derivable from tifxyz normals:

```python
nx, ny, nz = segment.compute_normals()  # at stored resolution

eps = 1e-8
def _encode_dir(gx, gy):
    r2 = gx*gx + gy*gy + eps
    cos2t = (gx*gx - gy*gy) / r2
    sin2t = 2.0*gx*gy / r2
    d0 = 0.5 + 0.5 * cos2t
    d1 = 0.5 + 0.5 * (cos2t - sin2t) / np.sqrt(2.0)
    return d0, d1

dir0_z, dir1_z = _encode_dir(nx, ny)   # Z-slices (XY plane)
dir0_y, dir1_y = _encode_dir(nx, nz)   # Y-slices (XZ plane)
dir0_x, dir1_x = _encode_dir(ny, nz)   # X-slices (YZ plane)
```

These can be computed per grid point and then voxelized into the 3D patch
(using the same splatting approach as the neural tracer).

### 3.2 Cos and grad_mag — from per-surface distance transforms (no winding needed)

The cos channel encodes `0.5 + 0.5 * cos(2π * winding)`, which peaks at 1.0 on
each sheet surface and dips to ~0.5 midway between sheets. The grad_mag channel
encodes local sheet density (inverse of inter-sheet spacing). Crucially, both
signals are **locally determined** — they only depend on the distances to nearby
surfaces, not on a global winding number.

We derive both from **per-surface distance transforms**:

1. Voxelize each surface individually into its own binary 3D mask. The neural
   tracer's `_voxelize_surface_from_sampled_grid` already does per-surface
   voxelization.
2. For each surface `k`, compute the **Euclidean distance transform**
   `dt_k = edt(~mask_k)`. This gives, for every voxel, the distance to the
   nearest voxel of surface `k`.
3. At each voxel, sort the per-surface distances to find the **two smallest**:
   `d1` (nearest surface) and `d2` (second-nearest surface). The local
   inter-sheet spacing is `spacing = d1 + d2`.
4. Derive both channels:
   - **cos**: `0.5 + 0.5 * cos(π * d1 / max(d1 + d2, eps))` — peaks at 1.0 on
     each surface (d1=0), dips to a minimum at the midpoint (d1 = d2), and rises
     again approaching the next sheet.
   - **grad_mag**: `1 / (d1 + d2)` — the sheet density, high where sheets are
     close together, low where they are far apart.

```python
from scipy.ndimage import distance_transform_edt

# Compute per-surface distance transforms
# surface_masks: list of (Z, Y, X) bool, one per surface in this patch
dts = [distance_transform_edt(~mask) for mask in surface_masks]

# Stack and sort to find two smallest distances at each voxel
dt_stack = np.stack(dts, axis=0)           # (K, Z, Y, X)
dt_sorted = np.sort(dt_stack, axis=0)       # sorted along surface axis
d1 = dt_sorted[0]                           # distance to nearest surface
d2 = dt_sorted[1] if len(dts) > 1 else d1  # distance to 2nd-nearest

spacing = d1 + d2  # local inter-sheet spacing
half_spacing = spacing / 2.0

cos_channel = 0.5 + 0.5 * np.cos(np.pi * np.clip(d1 / (half_spacing + 1e-6), 0, 1))
grad_mag = 1.0 / (spacing + 1e-6)
```

This approach has key advantages:
- **No winding assignment needed** — no inter-surface ordering, no global layer
  counting. Each surface contributes independently via its own distance
  transform.
- **Grad_mag is directly derived** — the inter-sheet spacing at each voxel is
  simply the sum of the two smallest per-surface distances.
- **Works with any number of surfaces** — with 2+ surfaces the two-smallest
  distances give exact local spacing; with only 1 surface, cos and grad_mag
  cannot be supervised (only direction channels are usable).
- **Naturally handles the midpoint** — the cosine profile smoothly transitions
  between sheets using the local half-spacing.

### 3.3 Validity filtering

Not all voxels produce reliable training signal. The existing 2D label pipeline
(`gen_post_data.py:compute_label_supervision`) uses several layers of filtering
to determine valid supervision regions, and the 3D tifxyz case needs analogous
filtering.

#### Existing 2D filtering (reference)

The 2D code works on label slices `{0=background, 1=surface, 2=ignore}` and
applies these filters in `gen_post_data.py` and `train_unet.py`:

1. **Border removal**: 3px border marked as ignore (label=2) to avoid edge
   artifacts from distance transforms.

2. **Connected-component validation**: Each label-CC (foreground strip) must
   touch ignore regions in at least **two disjoint locations** — this ensures
   the surface is "sandwiched" between background regions on both sides, giving
   a meaningful distance field. CCs that only touch ignore on one side are
   rejected (they represent sheet edges, where distances are unreliable).

3. **Chain validation**: The inner components (alternating fg/bg strips) must
   form a **simple linear chain** (no branching). The chain is found by
   building a 4-connected adjacency graph between fg/bg components and walking
   from a degree-1 endpoint. Branching or incomplete chains are rejected.

4. **Endpoint exclusion**: The first and last components in the chain are
   skipped entirely — they represent the outermost half-layers where there is
   no neighboring surface on one side, making the distance field one-sided and
   unreliable.

5. **Outer-CC erosion**: Valid outer CC masks are eroded by 16px
   (`cv2.erode` with a 33×33 kernel) to stay away from boundaries where the
   distance field is affected by the mask edge rather than actual sheet
   structure. Stored as `outer_cc_idx` (contiguous 1..K indices).

6. **Gradient validity mask**: For grad_mag and direction channels, an
   additional check requires **both neighbors** in the finite-difference
   stencil to lie within the eroded valid mask. This is computed as the
   intersection of shifted masks: `m_grad = m_x & m_y` where
   `m_x[..., :-1] = eroded[..., 1:] & eroded[..., :-1]` (and similarly for y).
   This prevents gradients from straddling valid/invalid boundaries.

7. **frac_pos sentinel**: Areas outside valid regions get `frac_pos = -1.0`,
   which is masked out in the loss via `valid_frac = (outer_idx > 0) &
   (frac_pos > -0.5)`.

#### 3D tifxyz filtering (proposed)

The analogous filtering for voxelized tifxyz surfaces:

1. **No surfaces → no loss**: If a patch contains zero surfaces, there is
   nothing to supervise — skip entirely (zero validity mask on all channels).
   With exactly one surface, only direction channels (normals) can be
   supervised; cos and grad_mag get a zero validity mask since they require
   inter-sheet spacing from d1+d2.

2. **Distance-based validity**: Only supervise voxels where `d2` (distance to
   second-nearest surface) is finite and below a reasonable threshold (e.g.
   half the patch size). Voxels far from any second surface have unreliable
   spacing estimates.

3. **Boundary erosion**: Erode the valid region by a margin (e.g. 8–16 voxels)
   near patch edges where the distance transform is clipped by the patch
   boundary rather than reflecting true sheet structure. This mirrors the
   16px erosion and 3px border removal in the 2D code.

4. **Gradient validity**: For direction channels derived via finite differences
   on the splatted field, require both neighbors in the stencil to be valid
   (same `m_x & m_y` approach as the 2D code).

5. **Surface-edge exclusion**: Near the edges of a tifxyz surface's valid
   region (where `_valid_mask` transitions from True to False), the distance
   transform is truncated. Exclude voxels within a margin of these boundaries
   — analogous to the 2D endpoint exclusion.

```python
# Validity mask for cos and grad_mag
valid = np.ones_like(d1, dtype=bool)
valid &= len(surface_masks) >= 2          # need ≥2 surfaces
valid &= d2 < max_reliable_distance       # d2 not clipped by patch boundary
valid &= erode_3d(valid, margin=16)       # stay away from patch edges

# For direction channels: gradient validity
# (both neighbors in finite-difference stencil must be valid)
```


---

## 4. Voxelization Strategy

Each training sample is a (CT patch, label patch) pair. The CT patch is a 3D
crop at full resolution. The label patch contains the 8 channels at step
resolution (fullres / scaledown).

To produce labels on-the-fly from tifxyz surfaces for a given patch:

```
1. Identify which tifxyz surfaces intersect this patch's world bbox
2. For each surface k:
   a. Bicubic-upsample surface grid from stored to full resolution
   b. Sample the surface grid within the padded bbox
   c. Compute normals at the sampled grid points
   c. Encode direction channels (6 values per grid point)
   d. Voxelize into a per-surface binary mask (mask_k)
3. Voxelize direction channels:
   a. Splat the 6 direction values at each surface point's 3D position
   b. Normalize splatted direction values (divide by weight accumulator)
4. Per-surface distance transforms + cos/grad_mag:
   a. For each surface k: dt_k = edt(~mask_k)
   b. Sort per-voxel to find d1 (nearest) and d2 (second-nearest)
   c. cos = 0.5 + 0.5 * cos(π * d1 / ((d1+d2)/2))
   d. grad_mag = 1 / (d1 + d2)
5. Validity filtering:
   a. Mask out voxels with <2 surfaces or d2 > threshold
   b. Erode valid region near patch/surface boundaries
6. Average-pool to step resolution if needed
```

The splatting infrastructure already exists in the neural tracer:
- `_voxelize_surface_from_sampled_grid()` — splats binary surface presence
- `_build_normal_offset_mask_from_labeled_points()` — splats along normals
- `_splat_points_trilinear_numba()` — numba-accelerated trilinear splatting

We extend this to splat multi-channel values (not just binary presence).


---

## 5. Integration Approaches

### Approach A: New dataset class (like TifxyzInkDataset)

Create a `TifxyzLasagnaDataset` modeled on `TifxyzInkDataset` that:
- Uses the same patch-finding infrastructure (`find_patches`)
- In `__getitem__`, reads a CT crop and derives the 8 label channels on-the-fly
- Plugs into the vesuvius `BaseTrainer` as a custom dataset

**Pros:**
- Full control over label derivation
- Reuses existing patch-finding and voxelization code
- Can handle multi-surface patches natively
- Natural fit — same data sources (volumes + tifxyz folders)

**Cons:**
- New dataset class to maintain
- Doesn't use vesuvius `ZarrDataset`'s pre-built patch validation / caching
- On-the-fly computation adds latency to data loading

**Implementation sketch:**

```python
class TifxyzLasagnaDataset(Dataset):
    """Dataset that derives lasagna training channels from tifxyz surfaces."""

    def __init__(self, config, apply_augmentation=True):
        self.scaledown = config.get("scaledown", 4)
        self.patch_size = config["patch_size"]  # full-res patch size

        # Reuse TifxyzInkDataset's patch finding
        self.patches = find_patches(config, ...)

        # For each patch, we know: volume, segment, world_bbox
        if apply_augmentation:
            self.augmentations = create_training_transforms(...)

    def __getitem__(self, idx):
        patch = self.patches[idx]
        # 1. Read CT crop (full res)
        vol_crop = _read_volume_crop_from_patch_dict(patch, ...)

        # 2. Sample tifxyz surface within patch bbox
        segment = patch["segment"]
        sampled = _sample_patch_supervision_grid(self, segment, ...)
        # sampled contains: local_grid, normals_zyx, in_patch, valid_interp

        # 3. Derive 8 channels from normals + distance transform
        labels = self._compute_lasagna_labels(sampled, ...)

        # 4. Pool labels to step resolution
        labels_step = avg_pool3d(labels, self.scaledown)

        return {"image": vol_crop, "labels": labels_step}

    def _compute_lasagna_labels(self, sampled, ...):
        """Derive cos, grad_mag, dir_{z,y,x} from surface grid."""
        normals = sampled["normals_zyx"]  # (H, W, 3) in ZYX
        nz, ny, nx = normals[..., 0], normals[..., 1], normals[..., 2]

        # Direction encoding
        dir0_z, dir1_z = _encode_dir(nx, ny)
        dir0_y, dir1_y = _encode_dir(nx, nz)
        dir0_x, dir1_x = _encode_dir(ny, nz)

        # Voxelize direction channels + per-surface binary masks
        # ... (splat per-point channel values using trilinear weighting)

        # Per-surface distance transforms → cos and grad_mag (see §3.2)
        # dts = [edt(~mask_k) for mask_k in per_surface_masks]
        # d1, d2 = two smallest distances at each voxel
        # cos = 0.5 + 0.5 * cos(π * d1 / ((d1+d2)/2))
        # grad_mag = 1 / (d1 + d2)
        # ...
```

### Approach B: Preprocessing script + ZarrDataset

Pre-compute label zarrs from tifxyz surfaces, then use the standard
`ZarrDataset` for training.

**Script: `tifxyz_to_lasagna_labels.py`**

```
For each tifxyz surface collection:
  1. Load all surfaces in the region
  2. Voxelize normals + per-surface binary masks into 3D volumes
  3. Per-surface distance transforms → sort → d1, d2 per voxel
  4. Derive cos (from d1, d2) and grad_mag (1/(d1+d2))
  5. Encode direction channels from splatted normals
  6. Compute validity mask (≥2 surfaces, d2 < threshold, boundary erosion)
  7. Write label zarrs matching ZarrDataset's expected layout:
     labels/region_cos.zarr, labels/region_grad_mag.zarr, etc.
```

Then train with the existing vesuvius pipeline:
```yaml
dataset_config:
  data_path: ./training_data
  targets:
    cos: { out_channels: 1, losses: [{name: MSELoss, weight: 1.0}] }
    grad_mag: { out_channels: 1, losses: [{name: SmoothL1Loss, weight: 1.0}] }
    dir_z: { out_channels: 2, losses: [{name: MSELoss, weight: 1.0}] }
    dir_y: { out_channels: 2, losses: [{name: MSELoss, weight: 1.0}] }
    dir_x: { out_channels: 2, losses: [{name: MSELoss, weight: 1.0}] }
```

**Pros:**
- Minimal code — reuses entire vesuvius training stack
- Label computation is a one-time cost
- Easy to inspect/debug labels before training
- ZarrDataset handles patch validation, normalization, augmentation

**Cons:**
- Storage overhead for label zarrs
- Preprocessing step before training
- Less flexible — changing label derivation requires re-running preprocessing

### Approach C: Hybrid — ZarrDataset with on-the-fly transform

Extend `ZarrDataset` with a custom transform that reads sparse tifxyz data
and computes labels on-the-fly. The "labels" stored in zarr would be
lightweight (e.g. just a validity mask), and the real labels are computed in
the transform from tifxyz data.

This is more complex and less natural than Approach A or B.

### Approach D: Extend BaseTrainer with custom dataset hook

The vesuvius `BaseTrainer` builds its dataset via `_configure_dataset()`. We
can subclass `BaseTrainer` to override this method and return a
`TifxyzLasagnaDataset` instead of `ZarrDataset`, while keeping everything else
(loss, optimizer, checkpointing, DDP) from the base trainer.

```python
class LasagnaTrainer(BaseTrainer):
    def _configure_dataset(self):
        train_ds = TifxyzLasagnaDataset(self.mgr, split="train")
        val_ds = TifxyzLasagnaDataset(self.mgr, split="val")
        return train_ds, val_ds
```

**Pros:**
- Reuses all of BaseTrainer (loss building, optimizer, scheduler, DDP,
  checkpointing, metrics, logging)
- Clean separation: dataset logic in TifxyzLasagnaDataset, training logic in
  BaseTrainer

**Cons:**
- BaseTrainer's `_train_step` expects specific dict keys from the dataset;
  need to match that interface


---

## 6. Key Code Entry Points

| Component | File | Key class/function |
|-----------|------|--------------------|
| Tifxyz data type | `vesuvius/src/vesuvius/tifxyz/types.py` | `Tifxyz`, `compute_normals()`, `get_normals()` |
| Tifxyz loading | `vesuvius/src/vesuvius/tifxyz/io.py` | `read_tifxyz()`, `load_folder()` |
| Tifxyz interpolation | `vesuvius/src/vesuvius/tifxyz/upsampling.py` | `interpolate_at_points()` |
| Neural tracer dataset | `ink-detection/tifxyz_dataset/tifxyz_dataset.py` | `TifxyzInkDataset` |
| Patch finding | `ink-detection/tifxyz_dataset/patch_finding.py` | `find_patches()` |
| Surface sampling | `ink-detection/tifxyz_dataset/common.py` | `_sample_patch_supervision_grid()` |
| Normal estimation | `ink-detection/tifxyz_dataset/common.py` | `_estimate_surface_normals_zyx()` |
| Voxelization | `ink-detection/tifxyz_dataset/common.py` | `_voxelize_surface_from_sampled_grid()`, `_splat_points_trilinear_numba()` |
| Volume crop reading | `ink-detection/tifxyz_dataset/common.py` | `_read_volume_crop_from_patch_dict()` |
| Vesuvius ZarrDataset | `vesuvius/src/vesuvius/models/datasets/zarr_dataset.py` | `ZarrDataset` |
| Vesuvius BaseTrainer | `vesuvius/src/vesuvius/models/training/train.py` | `BaseTrainer` |
| Config manager | `vesuvius/src/vesuvius/models/configuration/config_manager.py` | `ConfigManager` |
| Model builder | `vesuvius/src/vesuvius/models/build/build_network_from_config.py` | `NetworkFromConfig` |
| Loss factory | `vesuvius/src/vesuvius/models/training/loss/losses.py` | `_create_loss()` |
| Augmentation | `vesuvius/src/vesuvius/models/augmentation/pipelines/training_transforms.py` | `create_training_transforms()` |
| Auxiliary tasks | `vesuvius/src/vesuvius/models/training/auxiliary_tasks/apply_aux_targets.py` | `create_auxiliary_task()` |


---

## 7. Recommended Approach

**Start with Approach B (preprocessing + ZarrDataset)** for initial
experiments, then migrate to **Approach A/D (on-the-fly dataset)** once the
label derivation is validated.

### Phase 1: Validate label derivation (Approach B)

1. Write `tifxyz_to_lasagna_labels.py`:
   - Load tifxyz surfaces for a region via `tifxyz.load_folder()`
   - Compute normals via `Tifxyz.compute_normals()`
   - Voxelize normals + per-surface binary masks using adapted splatting from
     the neural tracer
   - Per-surface distance transforms → sort → d1, d2 → cos and grad_mag
   - Encode direction channels from splatted normals
   - Compute validity mask (≥2 surfaces, boundary erosion)
   - Write as zarr label volumes

2. Visually verify labels are correct (compare to existing 2D UNet output or
   fit-data output)

3. Train with standard vesuvius pipeline using `ZarrDataset`

### Phase 2: On-the-fly dataset (Approach A + D)

1. Create `TifxyzLasagnaDataset` reusing neural tracer infrastructure

2. Create `LasagnaTrainer(BaseTrainer)` that uses the new dataset

3. This avoids the storage overhead and allows iterating on label derivation
   without re-preprocessing

### Voxelization detail: multi-channel splatting

The neural tracer's existing splatting code (`_splat_points_trilinear_numba`)
only splats binary/scalar values. For the lasagna channels, we need to splat
8-channel vectors. Extend the numba kernel:

```python
@njit(cache=True)
def _splat_multichannel_trilinear(points, values, size_z, size_y, size_x, n_channels):
    """Splat (N, n_channels) values at (N, 3) positions into a volume."""
    vox = np.zeros((n_channels, size_z, size_y, size_x), dtype=np.float32)
    weights = np.zeros((size_z, size_y, size_x), dtype=np.float32)
    for i in range(points.shape[0]):
        pz, py, px = points[i, 0], points[i, 1], points[i, 2]
        z0, y0, x0 = int(np.floor(pz)), int(np.floor(py)), int(np.floor(px))
        dz, dy, dx = pz - z0, py - y0, px - x0
        for oz in range(2):
            zi = z0 + oz
            if zi < 0 or zi >= size_z: continue
            wz = (1.0 - dz) if oz == 0 else dz
            for oy in range(2):
                yi = y0 + oy
                if yi < 0 or yi >= size_y: continue
                wy = (1.0 - dy) if oy == 0 else dy
                for ox in range(2):
                    xi = x0 + ox
                    if xi < 0 or xi >= size_x: continue
                    wx = (1.0 - dx) if ox == 0 else dx
                    w = wz * wy * wx
                    weights[zi, yi, xi] += w
                    for c in range(n_channels):
                        vox[c, zi, yi, xi] += w * values[i, c]
    # Normalize by accumulated weight
    for zi in range(size_z):
        for yi in range(size_y):
            for xi in range(size_x):
                if weights[zi, yi, xi] > 0:
                    for c in range(n_channels):
                        vox[c, zi, yi, xi] /= weights[zi, yi, xi]
    return vox
```

### Handling multiple surfaces in one patch

When a patch contains multiple tifxyz surfaces (multiple sheet layers), each
surface is voxelized into its own binary mask and a shared direction-channel
volume. The direction splatting naturally handles overlap: each surface
contributes values near its location, and the weight normalization averages
overlapping contributions.

The per-surface distance transforms are then computed independently. At each
voxel, sorting the K distance values gives d1 and d2, from which cos and
grad_mag are derived. This naturally handles any number of surfaces — no
inter-surface ordering is required, just the two nearest distances.

For direction channels between surfaces, the normal-offset splatting approach
from `_build_normal_offset_mask_from_labeled_points` can be used to fill the
inter-sheet space (splat along normals).


---

## 8. Design Decisions

- **Training resolution**: Full resolution. Labels are voxelized at full res;
  the UNet operates at full res and its output is pooled to step res downstream.
