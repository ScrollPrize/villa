# Tifxyz Training Guide

Train a 3D UNet to predict lasagna channels (cos, grad_mag, 6x direction) from
CT volumes, using tifxyz surfaces as ground truth. Labels are derived on-the-fly
on GPU -- no pre-computed label zarrs needed.

For architecture details and design rationale, see
[tifxyz_training_integration.md](tifxyz_training_integration.md).


## Prerequisites

- CT volumes as OME-Zarr (local, S3, or HTTPS)
- Tifxyz surface segments (local directories with `x.tif`, `y.tif`, `z.tif`,
  `meta.json`)
- CUDA GPU (EDT + label derivation runs on GPU)
- Python environment with `vesuvius` installed (`pip install -e ".[all]"` from
  `vesuvius/`)


## Data Setup

### Tifxyz segments

The neural tracer dataset segments are on S3:

```
s3://philodemos/paul/neural-tracer-datasets/nt_dataset_022026/
```

Sync them locally (they're small relative to volumes):

```bash
aws s3 sync s3://philodemos/paul/neural-tracer-datasets/nt_dataset_022026/ \
    /path/to/data/nt_dataset_022026/
```

Each scroll has a `tifxyz/` subdirectory containing one directory per segment.

### CT volumes

Volumes can be local paths, S3 URIs, or HTTPS URLs. Remote volumes are
supported via fsspec with mandatory local caching (see below).

| Scroll | Volume scale |
|--------|-------------|
| PHerc0139 | 0 |
| PHerc1667 | 0 |
| PHercParis4 | 0 |
| PHerc0343p | 0 |
| PHercMANBp | 2 |
| PHerc0500p2 | 0 |


## Config File

The training script takes a JSON config. See
[`lasagna/configs/tifxyz_train_s3.json`](../configs/tifxyz_train_s3.json)
for a full 6-scroll example.

### Minimal example

```json
{
    "volume_cache_dir": "/data/zarr_cache",
    "datasets": [
        {
            "volume_path": "s3://bucket/scroll/volume.zarr",
            "volume_scale": 0,
            "segments_path": "/local/path/to/tifxyz",
            "z_range": [2000, 7000]
        }
    ]
}
```

### Config fields

| Field | Scope | Required | Description |
|-------|-------|----------|-------------|
| `volume_cache_dir` | top-level | for remote volumes | Local directory for caching remote zarr chunks via fsspec `filecache` |
| `volume_auth_json` | top-level or per-dataset | for authenticated HTTPS | Path to JSON with `{"username": "...", "password": "..."}` |
| `datasets` | top-level | yes | Array of dataset entries |
| `volume_path` | per-dataset | yes | Path to zarr volume -- local, `s3://`, or `https://` |
| `volume_scale` | per-dataset | yes | Resolution level in the zarr group (0 = full res) |
| `segments_path` | per-dataset | yes | Local path to tifxyz segment directories |
| `z_range` | per-dataset | recommended | `[z_min, z_max]` -- safe Z-slice range; excludes regions outside the scroll |
| `cache_scale` | per-dataset | no | Volume scale used for patch-finding/caching (default = `volume_scale`). Set to match another dataset entry's `volume_scale` to share its patch cache when using `transform`. |
| `transform` | per-dataset | no | Inline 3×4 affine matrix (XYZ, row-major) mapping segment coords → volume level-0 coords (after optional inversion). Enables cross-volume training. |
| `transform_invert` | per-dataset | no | If `true`, invert the `transform` matrix before applying (default `false`). |
| `scale_aug_prob` | top-level | no | Per-sample probability of scale augmentation (default `0.0` = off) |
| `scale_aug_factor` | top-level | no | Scale augmentation downscale factor (default `2`) |

### Volume path formats

| Format | Example | Notes |
|--------|---------|-------|
| Local | `/mnt/nvme/volumes/scroll.zarr` | No caching needed |
| S3 | `s3://bucket/path/volume.zarr` | Requires `volume_cache_dir`; uses AWS credentials from environment |
| HTTPS | `https://volumes.example.com/volume.zarr` | Requires `volume_cache_dir`; optional `volume_auth_json` for basic auth |


## Volume Caching

Remote volumes (S3/HTTPS) require `volume_cache_dir` in config. Without it,
`open_zarr()` raises:

```
ValueError: Remote volume path 's3://...' requires 'volume_cache_dir' in config.
Set it to a local directory for caching, e.g. {"volume_cache_dir": "/data/zarr_cache"}
```

How it works:

- fsspec `filecache` downloads each zarr chunk file on first access
- Subsequent reads (across epochs and training runs) are served from local disk
- Only accessed chunks are cached, not the entire volume
- Cache persists until manually cleared: `rm -rf /data/zarr_cache`
- First epoch with a new volume is slower (downloading); subsequent epochs are
  fast

For best performance, use an NVMe-backed path for `volume_cache_dir`.


## Cross-Volume Training

When a higher-resolution scan of the same scroll becomes available, you can
reuse existing GT surfaces (tifxyz) with the new volume by providing an affine
transform that maps between coordinate systems.

```json
{
    "volume_path": "s3://bucket/new_scan.zarr",
    "volume_scale": 2,
    "cache_scale": 0,
    "segments_path": "/path/to/same/tifxyz",
    "z_range": [1000, 9250],
    "transform": [
        [a00, a01, a02, t0],
        [a10, a11, a12, t1],
        [a20, a21, a22, t2]
    ],
    "transform_invert": true
}
```

- The `transform` is a 3×4 affine matrix in XYZ order (same as `transform.json`
  from volume registration). It maps segment coordinates to volume level-0
  coordinates after optional inversion.
- `transform_invert: true` inverts the matrix before applying — use this when
  the transform.json maps new→old but you need old→new.
- `cache_scale` should match the `volume_scale` of the original dataset entry
  that uses the same `segments_path`. This ensures the patch cache is shared:
  patches are tiled in cache-scale coordinates, and the affine is applied at
  data-loading time to map coordinates to the target volume.
- `z_range` is in cache-scale coordinates (matching the original entry).


## Running Training

```bash
python lasagna/train_tifxyz.py \
    --config lasagna/configs/tifxyz_train_s3.json \
    --patch-size 128 \
    --batch-size 2 \
    --epochs 100
```

### Quick test run

```bash
python lasagna/train_tifxyz.py \
    --config lasagna/configs/tifxyz_train_s3.json \
    --patch-size 64 \
    --batch-size 1 \
    --epochs 2 \
    --num-workers 0
```

### CLI arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--config` | (required) | JSON config path |
| `--patch-size` | 128 | Cubic CT input patch size in voxels |
| `--label-patch-size` | same as patch-size | GT region size (smaller = more CT context around labels) |
| `--model-patch-size` | same as patch-size | Architecture patch size (for loading checkpoints trained at different sizes) |
| `--batch-size` | 2 | Training batch size |
| `--epochs` | 100 | Number of epochs |
| `--lr` | 1e-2 | Learning rate (AdamW) |
| `--num-workers` | 4 | DataLoader workers (0 for debugging) |
| `--val-fraction` | 0.15 | Fraction of patches for validation |
| `--log-dir` | `runs/tifxyz3d` | TensorBoard log directory |
| `--run-name` | `tifxyz` | Run name (used in log subdirectory) |
| `--weights` | None | Checkpoint path to resume from |
| `--reinit-decoder-scales` | 0 | After loading checkpoint, re-initialize the last N decoder stages (highest-res conv blocks, upsample modules, seg heads, encoder stem). Use with `--weights` |
| `--norm-type` | `none` | Normalization: `instance`, `group`, `none` |
| `--upsample-mode` | `trilinear` | Decoder upsampling: `transpconv`, `trilinear`, `pixelshuffle` |
| `--output-sigmoid` | off | Apply sigmoid to model output |
| `--precision` | `bf16` | Training precision: `bf16`, `fp16`, `fp32` |
| `--w-cos` | 1.0 | Loss weight for cos channel |
| `--w-mag` | 1.0 | Loss weight for grad_mag channel |
| `--w-dir` | 1.0 | Loss weight for sparse direction channels |
| `--w-dir-dense` | 0.1 | Loss weight for dense (slerp-filled) direction channels |
| `--w-smooth` | 0.1 | Loss weight for direction smoothness penalty |
| `--device` | auto | `cuda` or `cpu` (auto-detected) |
| `--verbose` / `-v` | off | Per-sample progress bar with running loss |
| `--wandb` | off | Enable Weights & Biases logging (alongside TensorBoard) |
| `--wandb-project` | None | W&B project name |
| `--wandb-entity` | None | W&B entity/user |
| `--wandb-run-name` | None | W&B run name |
| `--wandb-tags` | None | Comma-separated W&B tags |
| `--no-himag-filter` | off | Disable hi-mag sample filtering entirely |
| `--no-deform` | off | Disable per-sample GT deformation refinement |
| `--deform-stride` | 8 | Deformation grid stride (grid = label_patch_size / stride) |
| `--deform-inner-iters` | 100 | Inner optimization iterations per training step |
| `--deform-inner-lr` | 1000 | Start LR for deformation inner loop (ramps to 100x on log scale) |
| `--deform-max-frac` | 0.3 | Max displacement as fraction of inter-surface distance |
| `--refine` | off | Enable multi-scale refinement mode (11ch input, disables scale_aug) |

### Multi-GPU (DDP)

Launch with `torchrun` for data-parallel training:

```bash
torchrun --nproc_per_node=4 lasagna/train_tifxyz.py \
    --config config.json --patch-size 128 --batch-size 2
```

Use `--master-port` to avoid collisions with other users on a shared server:

```bash
torchrun --nproc_per_node=4 --master-port=29501 lasagna/train_tifxyz.py ...
```

Works with `CUDA_VISIBLE_DEVICES` — `local_rank` indexes into the visible set.

### Label patch size

When `--label-patch-size` < `--patch-size`, the GT occupies a smaller region
centered (with random offset) inside a larger CT crop. This gives the model
more spatial context without needing larger label regions. Patch tiling uses
the label size to preserve cache efficiency. A random paste offset is applied
automatically when label < patch.

### Partial decoder reinitialization

When resuming from a checkpoint, `--reinit-decoder-scales N` re-initializes
the N highest-resolution decoder stages with fresh He-normal weights. This is
useful when fine-resolution outputs have degraded during training. It resets:
- Decoder conv blocks (skip-fusion convolutions)
- Upsample modules (transpconv / trilinear / pixelshuffle)
- All segmentation output heads
- The encoder stem (feeds the highest-res skip connection)

### Loss weighting by gradient magnitude

The cos and grad_mag losses are weighted proportionally to the GT gradient
magnitude (= 1/spacing between adjacent surfaces). Normalization: 20 voxel
spacing → weight 1.0. Closer surfaces get higher weight (5 vx → 4×),
farther surfaces get lower weight (100 vx → 0.2×).

### Hi-mag filtering with hysteresis

Samples with high gradient magnitude (surfaces very close together) are
filtered using a sliding-window hysteresis: disabled when ≥75 of the last
100 samples are above threshold, re-enabled when <50 are above. This
prevents training from being dominated by pathological tight-surface regions.

### Scale augmentation

Per-batch augmentation that randomly reads CT data from a coarser zarr
pyramid level, giving the model a wider field of view at lower effective
resolution. GT labels (EDT, cos, grad_mag, direction) are derived at
**full resolution** in the proportionally larger world region, then
valid-pooled to match the CT tensor shape in the train loop. This ensures
clean GT without thin-line artifacts from pooling masks before EDT.

The decision is **per-batch** (all samples in a batch use the same scale).
Implemented via two DataLoaders — one normal, one always scale-aug — with
a per-step coin flip in the train loop.

Enabled via config keys:

```json
{
    "scale_aug_prob": 0.2,
    "scale_aug_factor": 2
}
```

| Config key | Default | Description |
|------------|---------|-------------|
| `scale_aug_prob` | `0.0` | Per-batch probability of applying scale augmentation (0 = off) |
| `scale_aug_factor` | `2` | Downscale factor (2 = read from zarr level `scale+1`, 2× coarser) |

When active on a batch:
- CT is read from zarr level `scale + 1` (same tensor shape, 2× larger world region)
- Surface masks and tensor moments are voxelized at full resolution in the
  larger region — `compute_batch_targets` (EDT, cos, grad_mag) runs at full res
- Final targets and masks are valid-pooled / max-pooled to `crop_size` in the
  train loop after `compute_batch_targets`
- The zarr Group is opened once per dataset at init (only when `scale_aug_prob > 0`)
- Falls back to no augmentation if the target zarr level doesn't exist
- A second dataset instance + DataLoader is created for the scale-aug path

### GT deformation refinement

Per-sample learnable deformation of the cos and grad_mag GT labels. The neural
tracer GT surfaces are not perfectly accurate (smoothed, artifacts). A low-res
volumetric displacement field per sample is optimized during training to warp the
GT closer to what the model predicts, effectively refining the GT.

**How it works:**
- Each sample gets a `(3, G, G, G)` displacement field (default G = patch_size/8 = 24)
- Every training step (non-scale-aug), the deformation is optimized for N inner
  iterations (default 100) to minimize cos + grad_mag loss between the model's
  prediction and the warped GT
- Displacement magnitude is clamped to 0.3 × inter-surface distance (from grad_mag)
- The warp is applied only to cos and grad_mag channels (not direction)
- Deformations are stored in pre-augmentation GT space; spatial augmentation
  (flips/rot90) is applied/reversed automatically
- Skipped for scale-aug batches (different coordinate frame)

**Storage:**
- Disk-backed via numpy memmap: `run_dir/deformations/<key>.bin`
- Keyed by `(segments_path, dataset_idx)` for portability across config changes
- Float16 on disk (~83KB per sample), converted to float32 on GPU
- Automatically persisted — resume loads existing deformations

**Visualization:**
- TensorBoard/W&B shows both `{channel}_gt` (original) and `{channel}_gt_deformed`
  for cos and grad_mag
- `train/deform_mean` and `train/deform_max` track displacement magnitude

**CLI flags:** `--no-deform`, `--deform-stride`, `--deform-inner-iters`,
`--deform-inner-lr`, `--deform-max-frac` (see table above).

### Multi-scale refinement mode

Enable with `--refine`. The model takes 11 input channels (CT + 8ch prior
prediction + validity mask + scale indicator) and learns to refine its own
output across scales.

**Mutually exclusive with scale augmentation** — `--refine` forces
`scale_aug_prob=0`.

**7 training modes** (chosen uniformly per batch):

| Mode | Passes | Batch size | Description |
|------|--------|-----------|-------------|
| 0 | 1 | full | Scale -1 only (coarser, 0.5x) |
| 1 | 1 | full | Scale 0 only (base, 1x) |
| 2 | 1 | full | Scale +1 only (finer, 2x) |
| 3 | 2 | half | Chain -1 -> 0 |
| 4 | 2 | half | Chain 0 -> +1 |
| 5 | 2 | half | Chain -1 -> +1 |
| 6 | 2 | half | Self-refinement N -> N |

Every forward pass in a chain is supervised. The prior from the first pass
is detached before feeding into the second pass.

**Scale channel values:** 0.5 (coarser), 1.0 (base), 2.0 (finer). Always
active, even when no prior is provided.

**GT at different scales:**
- Scale -1: avg_pool from scale-0 GT
- Scale 0: native from dataset
- Scale +1: native fine-resolution voxelization (dataset computes at 2x for
  a random subregion)

**Checkpoint loading:** When loading a 1-channel checkpoint with `--refine`,
the stem conv is expanded from 1 to 11 input channels (channel 0 from
checkpoint, channels 1-10 zero-initialized).

**Test tool:**
```bash
python lasagna/scripts/test_refine_scales.py \
    --config config.json --weights model.pt \
    --num-samples 5 --output-dir tmp/refine_test
```

### GPU pause/resume

Training supports pausing to free the GPU for other applications.
Multiple processes can chain: training → inference A pauses training →
inference B pauses A → B finishes, A resumes → A finishes, training resumes.

Each GPU-using process runs a `GpuPauseServer` on a PID-based Unix socket
(`/tmp/gpu_pause.<pid>.sock`).  A stack file (`~/.gpu_owner_stack`) tracks
the LIFO ownership order.  If a pausing process crashes, the paused process
auto-resumes within 5 seconds (watchdog).

Control from another terminal (auto-discovers current GPU owner):

```bash
python lasagna/gpu_pause.py pause    # finish batch, offload to CPU, reply ok
python lasagna/gpu_pause.py resume   # reload to GPU, reply ok
python lasagna/gpu_pause.py status   # prints "running" or "paused"
```

Protocol: text over Unix socket, version handshake (`gpu_pause v1`), then
one command per connection (`pause`, `resume`, `status`).

Integration into other training scripts:

```python
from gpu_pause import GpuPauseServer

pause_server = GpuPauseServer()

def offload():
    model.cpu()
    for s in optimizer.state.values():
        for k, v in s.items():
            if isinstance(v, torch.Tensor):
                s[k] = v.cpu()
    torch.cuda.empty_cache()

def reload():
    model.to(device)
    for s in optimizer.state.values():
        for k, v in s.items():
            if isinstance(v, torch.Tensor):
                s[k] = v.to(device)

# After each optimizer step:
pause_server.check(offload, reload)

# At exit:
pause_server.close()
```

### Read-error resilience

S3/zarr transient read errors (PermissionError, OSError, ConnectionError,
TimeoutError) are caught in the dataset and the sample is skipped. The
warmup counter only increments for successfully used samples.


## Output

Training outputs go to `<log-dir>/<timestamp>_<run-name>/`:

| File | Description |
|------|-------------|
| `config.json` | Saved run configuration |
| `model_current.pt` | Latest checkpoint |
| `model_best.pt` | Best validation loss checkpoint |
| `events.out.tfevents.*` | TensorBoard logs |

Monitor with:

```bash
tensorboard --logdir runs/tifxyz3d
```

### Checkpoint format

Checkpoints contain:

```python
{
    "state_dict": model.state_dict(),
    "norm_type": "none",
    "upsample_mode": "trilinear",
    "output_sigmoid": False,
    "patch_size": 128,
    "in_channels": 1,
    "out_channels": 8,
}
```

Resume training with `--weights path/to/model_current.pt`. The loader is
flexible by default — it matches keys by name AND shape, skipping mismatched
parameters and randomly initializing missing ones.


## Pipeline Overview

1. **Patch finding** -- `find_world_chunk_patches()` tiles the volume into
   3D chunks and finds surface wraps within each chunk. Results are cached to
   `.patch_cache/world_chunks_*.json` per segments directory.

2. **Chain ordering** -- `build_patch_chains()` in
   `tifxyz_lasagna_dataset.py` groups the wraps of each patch into ordered
   chains (see "Surface chain ordering" below). This replaces the old
   EDT-pairwise greedy ordering; it is geometry + filename-winding driven and
   supports multiple independent chains per patch.

3. **Dataset `__getitem__`** -- reads a CT crop from zarr, voxelizes surface
   masks and direction channels from tifxyz grids, and emits per-retained-mask
   chain metadata (`chain`, `pos`, `has_prev`, `has_next`) aligned with
   `surface_masks`.

4. **GPU label derivation** -- `compute_patch_labels()` computes per-surface
   EDTs once per sample, then uses the externally supplied chains to derive
   cos and grad_mag **only for voxels bracketed between two chain-adjacent
   surfaces**. There is no EDT-based ordering search.

5. **Augmentation** -- random flips, 90-degree rotations, intensity jitter
   applied after label computation.

6. **Loss** -- masked multi-scale MSE (cos, direction) + smooth L1 (grad_mag),
   weighted by validity masks. Cos and grad_mag are supervised only in the
   between-neighbors region (via the chain-derived validity mask).
   Directions are supervised **densely** inside the same between-neighbors
   bracket: the encoded direction channels are filled by a chain-adjacent
   DT blend (see "Direction densification" below). The direction loss is
   split into two terms with independent weights and TB scalars:
   `loss_dir_sparse` (MSE on the original splatted wrap voxels, weight
   `--w-dir` default `1.0`) and `loss_dir_dense` (MSE on the
   DT-blended fill voxels, weight `--w-dir-dense` default `0.1`).
   A third `loss_smooth` term (weight `--w-smooth` default `0.1`)
   applies an L1 spatial-gradient penalty to the predicted direction
   channels inside the validity bracket — it smooths the model's
   output directly, independent of any residual roughness in the
   densified target. Having the three terms separate makes it
   possible to watch the true wrap-voxel error (`loss_dir_sparse`)
   in isolation from the softer densified target error. At
   coarser scales `ScaleSpaceLoss3D` performs **masked-average pooling**
   on prediction and target (averaging only over valid voxels per
   2×2×2 block) and uses an **any-valid** rule for the validity mask
   itself (`tifxyz_labels.scale_space_pool_validity` = `max_pool3d`).
   This keeps coarse supervision wherever a block has at least one
   valid voxel instead of eroding signal away.


## Surface chain ordering

The dataset produces **multi-chain** ordering metadata per patch via
`build_patch_chains(patch, max_wraps)` in
`lasagna/tifxyz_lasagna_dataset.py`. This ports the triplet neighbor logic
from `vesuvius/neural_tracing/datasets/dataset_rowcol_cond._build_triplet_neighbor_lookup`:

1. For each wrap, compute a 2D median `(x, y)` over its stored-resolution
   coordinates (`_compute_wrap_order_stats`).
2. Pick the dominant-spread axis as the local through-the-scroll direction.
3. Sort wraps along that axis. For each wrap, link to the nearest
   **compatible** neighbor on each side. "Compatible"
   (`_triplet_wraps_compatible`) means same segment, OR wrap ids parsed from
   the segment filename via the `w<N>` convention differ by exactly 1.
4. Walk reciprocal next-links from chain heads to form chains; leftover
   wraps become singleton chains.

The returned dict has one entry per wrap:
`{wrap_idx: {"chain", "pos", "has_prev", "has_next", "label"}}`.
The dataset carries this through to `compute_patch_labels` as the
`surface_chain_info` field on each sample (per-retained-mask dicts aligned
with `surface_masks`).

### Validity ("between neighboring surfaces")

For every voxel, `derive_cos_gradmag_validity()`:

- Finds the globally nearest surface across all chains.
- Restricts supervision to the chain the nearest surface belongs to, at its
  chain position.
- A voxel is **valid only when it is strictly between the nearest surface
  and one of its chain-adjacent neighbors**. Between-ness is detected via
  the DT-gradient dot product: `dot(grad(dt_near), grad(dt_neighbor)) < 0`
  means the gradients point toward each other, so the voxel is bracketed.
- Chain endpoints with no neighbor on the "open" side are invalid on that
  side. Chain-of-one wraps contribute no valid voxels.

### Direction densification (raw-normal slerp, last-second encoding)

The dataset does **not** carry a pre-encoded 6-channel
`direction_channels` volume any more. Instead it splats raw ZYX
normals:

```
_splat_multichannel(points_local, normals_zyx, crop_size)
  → raw_normals   (3, Z, Y, X)   # sparse trilinear splat of unit vectors
  → normals_valid (Z, Y, X)      # where the splat landed
```

These are carried through augmentation (`augment_batch_inplace`
re-splats from the already-transformed `surface_geometry`) and
handed to `compute_patch_labels` alongside the surface masks. The
single place that touches the double-angle encoding is
`derive_cos_gradmag_validity` — and it does so *after* slerp-blending
in angle space:

- Per-wrap EDTs are computed jointly with **feature transforms** via
  `edt_torch_with_indices(~mask_i)` (one CuPy call, returning both
  distances and nearest-on-wrap ZYX indices). These are threaded
  down from `compute_batch_targets` via `precomputed_fts` alongside
  `precomputed_dts`.
- Inside the per-chain, per-position loop (the same loop that routes
  voxels to their chain-adjacent `prev` or `next` bracket for cos),
  the raw normals at the two bracketing wraps are gathered at those
  feature-transform indices (so the blend always reads from the
  surface-local raw normals, not from a decoded encoding) and
  combined via `slerp_unit(n_lo, n_hi, frac)` using the **same**
  `frac = d_lo / (d_lo + d_hi)` that drives cos. Slerp is
  sign-invariant (it flips `n2` into `n1`'s hemisphere so the
  shorter arc is taken) and degenerates gracefully into lerp when
  the two normals are near-parallel.
- At `d_lo = 0` (on the `lo` surface) the blend yields 100 % `n_lo`;
  across the gap it rotates smoothly through the cross-product
  geodesic to 100 % `n_hi` at the `hi` surface — correctly in angle
  space, **not** in encoded space.
- `apply_same_surface_merge` recomputes feature transforms from the
  unioned mask for merged groups, so the nearest-on-wrap lookup
  stays exact after a same-surface merge.
- After the chain loop, `compute_patch_labels` sparse-overrides
  `n_dense` with the original `raw_normals` wherever
  `normals_valid` is `True` (splatted wrap voxels keep their hard
  ground-truth values), then calls
  `encode_direction_channels(nx, ny, nz)` **once** to produce the
  final 6-channel `targets[2:8]`. If the encoding ever needs to
  change, only that one function needs to change.
- Per-plane relevance weights fall out of the densified raw-normal
  field: for each voxel, `w_z = √(nx² + ny²)`, `w_y = √(nx² + nz²)`,
  `w_x = √(ny² + nz²)`. The 6-channel `dir_axis_weight` is
  `[w_z, w_z, w_y, w_y, w_x, w_x]`. A voxel whose normal is
  perpendicular to a given slice plane gets near-zero weight on
  that plane's direction pair — the encoding there is degenerate
  noise so the loss correctly ignores it. This restores the
  per-plane weighting the old `train_unet_3d.py` had.
- Two masks are returned separately: `dir_sparse_mask`
  (splatted wrap voxels) and `dir_dense_mask` (bracket fill). The
  train loop runs the direction MSE twice — once per mask, both
  with `weight=dir_axis_weight` — and logs
  `train/loss_dir_sparse` vs `train/loss_dir_dense` so the hard
  wrap-voxel error is visible separately from the softer bracket
  fill.

Why iterative approaches failed:

- An earlier densifier picked the globally two-nearest wraps per
  voxel → visible streaks at Voronoi boundaries and cross-chain
  leaks. Using the chain-adjacent routing fixes both.
- Blending the already-encoded 6-channel `direction_channels`
  linearly → wrong: the double-angle encoding is nonlinear in θ,
  so a 50/50 encoded-space average is not the angular midpoint
  and the magnitudes shrink toward 0.5. Slerping raw normals and
  encoding last-second is the correct operation.

## Inspecting the dataset

See [`lasagna3d_cli.md`](lasagna3d_cli.md) for the
`python -m lasagna3d dataset vis` tool, which renders three-plane JPEGs of
dataset samples with the same chain labels wired into the training loop.
Use it to sanity-check chain ordering before long runs.


## 3D Inference (predict3d)

Run 3D UNet inference on a CT volume and write a `.lasagna.json` manifest with per-group zarr arrays:

```bash
python lasagna/preprocess_cos_omezarr.py predict3d \
    --input vol.zarr --output pred.lasagna.json \
    --unet-checkpoint model_best.pt \
    --cos-scaledown 2 --scaledown 4

# Add pred-dt channel later (updates existing JSON, leaves cos/prediction untouched):
python lasagna/preprocess_cos_omezarr.py predict3d \
    --input vol.zarr --output pred.lasagna.json \
    --unet-checkpoint model_best.pt \
    --pred-dt pred_surface.zarr
```

### predict3d CLI flags

| Flag | Default | Description |
|------|---------|-------------|
| `--input` | required | Input zarr array (3D ZYX) |
| `--output` | required | Output `.lasagna.json` path |
| `--unet-checkpoint` | required | 3D UNet checkpoint (.pt) |
| `--cos-scaledown` | 2 | Downsample power for cos channel (OME-Zarr level, actual factor = 2^value) |
| `--scaledown` | 4 | Downsample power for other channels (OME-Zarr level, actual factor = 2^value) |
| `--source-to-base` | 1.0 | Source volume to VC3D base coordinate factor |
| `--tile-size` | 256 | Inference tile size |
| `--overlap` | 64 | Tile overlap in voxels |
| `--border` | 16 | Hard discard border at tile edges |
| `--crop` | none | Crop region: x y z w h d |
| `--pred-dt` | none | Source zarr for distance-to-surface channel |
| `--device` | auto | Device (cuda/cpu) |
| `--chunk-z` | 32 | Output zarr chunk size along Z |
| `--chunk-yx` | 32 | Output zarr chunk size for Y and X |
| `--calibrate-norm` | off | Calibrate InstanceNorm before inference |

### Lasagna volume format (.lasagna.json)

Output is a JSON manifest describing channel groups, each stored in a separate zarr:

```json
{
  "version": 2,
  "source_to_base": 1.0,
  "crop_xyzwhd": [0, 0, 0, 4000, 4000, 4000],
  "grad_mag_encode_scale": 1000.0,
  "groups": {
    "cos": {
      "zarr": "cos.ome.zarr/3",
      "scaledown": 3,
      "channels": ["cos"]
    },
    "grad_mag": {
      "zarr": "grad_mag.ome.zarr/4",
      "scaledown": 4,
      "channels": ["grad_mag"]
    },
    "nx": {
      "zarr": "nx.ome.zarr/4",
      "scaledown": 4,
      "channels": ["nx"]
    },
    "ny": {
      "zarr": "ny.ome.zarr/4",
      "scaledown": 4,
      "channels": ["ny"]
    },
    "pred_dt": {
      "zarr": "pred_dt.ome.zarr/3",
      "scaledown": 3,
      "channels": ["pred_dt"]
    }
  }
}
```

- **`source_to_base`**: Factor from source volume voxels to base (VC3D) voxels. Default 1.0 (source = base). Set to 4 if source is 4x coarser than the VC3D coordinate system.
- **`scaledown`** per group: OME-Zarr pyramid level (power of 2). The actual scale factor is `2^scaledown`. E.g. `scaledown: 4` → level 4 → 16x downsampled from base. Use `ChannelGroup.sd_fac` in code to get the actual factor.
- **`channels`**: Ordered list — position = channel index in the CZYX zarr (for 3D zarrs, only one channel per group).
- Zarr paths are relative to the JSON file's directory and include the OME-Zarr level suffix.
- Updating a single group (e.g., adding pred_dt) leaves other groups untouched.


## Implementation Files

| File | Purpose |
|------|---------|
| `lasagna/train_tifxyz.py` | Training script and CLI (DDP, W&B, hi-mag filtering, partial reinit) |
| `lasagna/tifxyz_lasagna_dataset.py` | `TifxyzLasagnaDataset` (CT crops, surface voxelization, augmentation) and `build_patch_chains()` (multi-chain ordering) |
| `lasagna/tifxyz_labels.py` | `compute_patch_labels()`, `chains_from_surface_info()`, `derive_cos_gradmag_validity()` (GPU, chain-aware) |
| `lasagna/lasagna_volume.py` | `.lasagna.json` manifest read/write: `LasagnaVolume`, `ChannelGroup` dataclasses |
| `lasagna/preprocess_cos_omezarr.py` | Inference: per-axis 2D preprocessing and tiled 3D `predict3d` mode (writes `.lasagna.json`) |
| `lasagna/fit_data.py` | `load_3d()` reads `.lasagna.json`, `FitData3D` with per-channel spacing and `grid_sample_fullres()` |
| `lasagna/scripts/download_omezarr.py` | Parallel S3 OME-Zarr chunk downloader with progress display |
| `lasagna/lasagna3d/` | `python -m lasagna3d` analysis CLI — see [`lasagna3d_cli.md`](lasagna3d_cli.md) |
| `vesuvius/src/vesuvius/models/build/build_network_from_config.py` | `NetworkFromConfig` — shared encoder + decoder/task-heads UNet builder |
| `vesuvius/src/vesuvius/neural_tracing/datasets/patch_finding.py` | `find_world_chunk_patches()` — world-chunk patch discovery |
| `vesuvius/src/vesuvius/neural_tracing/datasets/common.py` | `open_zarr()`, `voxelize_surface_grid_masked()`, `ChunkPatch`, `_compute_wrap_order_stats()`, `_triplet_wraps_compatible()` |
| `vesuvius/src/vesuvius/image_proc/edt.py` | GPU-accelerated EDT (CuPy > edt > scipy fallback) |
