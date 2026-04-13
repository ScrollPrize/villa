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
| `--patch-size` | 128 | Cubic patch size in voxels |
| `--batch-size` | 2 | Training batch size |
| `--epochs` | 100 | Number of epochs |
| `--lr` | 1e-4 | Learning rate (AdamW) |
| `--num-workers` | 4 | DataLoader workers (0 for debugging) |
| `--val-fraction` | 0.15 | Fraction of patches for validation |
| `--log-dir` | `runs/tifxyz3d` | TensorBoard log directory |
| `--run-name` | `tifxyz` | Run name (used in log subdirectory) |
| `--weights` | None | Checkpoint path to resume from |
| `--norm-type` | `instance` | Normalization: `instance`, `group`, `none` |
| `--upsample-mode` | `trilinear` | Decoder upsampling: `transpconv`, `trilinear`, `pixelshuffle` |
| `--precision` | `bf16` | Training precision: `bf16`, `fp16`, `fp32` |
| `--w-cos` | 1.0 | Loss weight for cos channel |
| `--w-mag` | 1.0 | Loss weight for grad_mag channel |
| `--w-dir` | 1.0 | Loss weight for direction channels |
| `--device` | auto | `cuda` or `cpu` (auto-detected) |


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
    "norm_type": "instance",
    "upsample_mode": "trilinear",
    "precision": "bf16",
}
```

Resume training with `--weights path/to/model_current.pt`.


## Pipeline Overview

1. **Patch finding** -- `find_world_chunk_patches()` tiles the volume into
   3D chunks and finds surface wraps within each chunk. Results are cached to
   `.patch_cache/world_chunks_*.json` per segments directory.

2. **Dataset `__getitem__`** -- reads a CT crop from zarr, voxelizes surface
   masks and direction channels from tifxyz grids.

3. **GPU label derivation** -- `compute_patch_labels()` runs EDT + chain
   ordering + cos/grad_mag computation on GPU per batch.

4. **Augmentation** -- random flips, 90-degree rotations, intensity jitter
   applied after label computation.

5. **Loss** -- masked multi-scale MSE (cos, direction) + smooth L1 (grad_mag),
   weighted by validity masks.


## Implementation Files

| File | Purpose |
|------|---------|
| `lasagna/train_tifxyz.py` | Training script and CLI |
| `lasagna/tifxyz_lasagna_dataset.py` | `TifxyzLasagnaDataset` -- CT crops + surface voxelization |
| `lasagna/tifxyz_labels.py` | `compute_patch_labels()` -- GPU label derivation |
| `vesuvius/src/vesuvius/neural_tracing/datasets/patch_finding.py` | `find_world_chunk_patches()` -- world-chunk patch discovery |
| `vesuvius/src/vesuvius/neural_tracing/datasets/common.py` | `open_zarr()`, `voxelize_surface_grid_masked()`, `ChunkPatch` |
| `vesuvius/src/vesuvius/image_proc/edt.py` | GPU-accelerated EDT (CuPy > edt > scipy fallback) |
