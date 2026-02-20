# 2D TIFF layer UNet trainer

## Design decisions

- Minimal dependencies: PyTorch, tifffile, TensorBoard, single in-repo U-Net implementation.
- Use multi-layer TIFF stacks where each layer is treated as an independent 2D sample.
- Supervision from label TIFFs with three-valued encoding:
  - `0` → target intensity 0, contributes to loss.
  - `1` → target intensity 1, contributes to loss.
  - `2` → ignored (no loss contribution).
- Train on random 256×256 crops (patches) sampled per layer to normalize varying input sizes; the patch size is configurable in [`TiffLayerDataset`](train_unet.py:11).

## Project structure

- [`train_unet.py`](train_unet.py)
  - [`TiffLayerDataset`](train_unet.py:11): iterates over all layers in each multi-layer TIFF pair from `images/` and `labels/`, returning a random square patch (default `256×256`) per sample.
  - [`UNet`](train_unet.py:94): small 2D U-Net for single-channel input and output (values in `[0, 1]`).
  - [`masked_mse_loss`](train_unet.py:143): implements the label semantics (0 → 0, 1 → 1, 2 → ignore).
  - [`train`](train_unet.py:170): basic training loop with Adam, masked MSE, and TensorBoard logging.
  - [`main`](train_unet.py:215): CLI entry point for configuring paths and hyperparameters.

Expected data layout (relative to the project root):

```text
images/
  sample_001.tif
  sample_002.tif
labels/
  sample_001_surface.tif
  sample_002_surface.tif
```

Each image TIFF in `images/` must have a matching label TIFF in `labels/` with identical number of layers and filename pattern:

- image: `sample_XYZ.tif`
- label: `sample_XYZ_surface.tif`

## Supervision utilities (gen_post_data)

- [`gen_post_data.py`](gen_post_data.py) provides:
  - A CLI tool to generate various TIFF visualizations (`vis.tif`, `vis_monotone*.tif`, `vis_labels_cc*.tif`, `vis_frac_pos*.tif`) for a single label layer, useful for debugging geometry and supervision.
  - A planned importable API that computes the same fractional-order supervision used in `vis_frac_pos.tif`, plus connected-component masks, directly from tensors.

Planned module API (for later integration into training):

- A single function (name TBD) that, given a batch of label maps as a PyTorch tensor of shape `(N, H, W)` (values in `{0, 1, 2}` with `2` = ignore), will compute:

  - `frac_pos`: float32 tensor of shape `(N, H, W)`
    - Per-pixel fractional order along the inferred chain inside each valid large CC.
    - Pixels not participating in a valid chain are set to a negative sentinel (e.g. `-1`), matching the current `frac_pos` TIFF semantics.

  - `outer_cc_idx`: integer tensor of shape `(N, H, W)`
    - Encodes the *large outer* connected components that passed the current validity checks.
    - Each such CC is eroded by 16 pixels (in the 2D plane) before being written into `outer_cc_idx`.
    - Outside these eroded outer CCs, `outer_cc_idx` is `0`.
    - Inside them, `outer_cc_idx` takes values `1..K`, where indices are strictly increasing with no gaps: if a candidate CC is skipped by the geometric checks, its index is not used and the next valid CC reuses the next consecutive index.

  - `max_cc_idx`: integer scalar
    - The maximum CC index used across the entire batch, i.e. `max(outer_cc_idx)` over all `N` samples.
    - This allows downstream code to reason about the global number of outer CCs present in the batch.

The existing CLI behavior of [`gen_post_data.py`](gen_post_data.py) (reading a single TIFF, computing all intermediate fields, and writing visualization TIFFs next to the input) will be preserved by calling this function internally when the module is executed as a script.

## Dependencies

- `torch`
- `tifffile`
- `tensorboard` (via `torch.utils.tensorboard`)

## Running training

Example command:

```bash
python train_unet.py \
  --images-dir images \
  --labels-dir labels \
  --log-dir runs/unet \
  --run-name unet_baseline
```

Logs and checkpoints will be written into a timestamped subdirectory of `--log-dir`, for example:

```text
runs/unet/20251124_121207_unet_baseline/
```

## Exporting tifxyz (one per winding)

Export a fitted model snapshot (state_dict) into a directory of tifxyz surfaces:

```bash
python fit2tifxyz.py --input path/to/model_*.pt --output out_tifxyz/
```

This writes `out_tifxyz/winding_XXXX.tifxyz/` directories containing `x.tif`, `y.tif`, `z.tif`, and `meta.json`.

Notes:

- `x/y` are written in **original pixel units** by multiplying by `--downscale` (default 4.0).
- `--offset x y z` adds a global translation in original pixel/voxel units (for crop & z-start alignment).
- `meta.json` contains required `uuid` (dirname) and `type="seg"`.


## Exporting PLY (one per winding)

- Written automatically during visualization to: `vis/ply/winding_XXXX/<postfix>.ply`
- Connected grid mesh along the winding direction for every z slice (no skipping)


## Preprocessing: `preprocess_cos_omezarr.py`

Runs tiled UNet inference on an OME-Zarr volume and writes an 8-bit OME-Zarr with
cos, gradient-magnitude, direction, and validity channels.

### Multi-axis processing

The `--axis` flag controls which dimension is sliced through:

| `--axis` | Slice dim | 2D plane fed to UNet | Sparse output dim |
|----------|-----------|----------------------|-------------------|
| `z` (default) | Z | Y x X | Z |
| `y` | Y | Z x X | Y |
| `x` | X | Z x Y | X |

The step (`--step` / `--z-step`) applies along the chosen slice axis. The
downscale (`--downscale` / `--downscale-xy`) applies to both plane dimensions.
The crop (`--crop-xyzwhd`) is always in absolute input coordinates regardless of
axis.

Each per-axis output zarr has shape `(5, out_Z, out_Y, out_X)` but the
effective resolution differs per dimension:

- **Slice dim**: `full_size // (step * downscale)` — coarse (stepped)
- **Plane dims**: `full_size // downscale` — dense (downscaled only)

Channels (identical for all axes):

| Index | Name | Encoding |
|-------|------|----------|
| 0 | `cos` | `clip(cos * 255, 0, 255)` uint8 |
| 1 | `grad_mag` | `clip(grad_mag * 1000, 0, 255)` uint8 |
| 2 | `dir0` | `clip(dir0 * 255, 0, 255)` uint8 |
| 3 | `dir1` | `clip(dir1 * 255, 0, 255)` uint8 |
| 4 | `valid` | 255 where processed, 0 otherwise |

Note: `dir0`/`dir1` represent gradient directions in the 2D plane perpendicular
to the slice axis. For axis=z these are YX-plane directions; for axis=y, ZX-plane;
for axis=x, ZY-plane.

### Direction integration (`integrate` subcommand)

After running preprocessing for multiple axes (typically all three), the
`integrate` subcommand merges direction channels from axis-y and axis-x volumes
into the axis-z reference volume:

```bash
python preprocess_cos_omezarr.py integrate \
    --z-volume z_cos.zarr \
    --y-volume y_cos.zarr \
    --x-volume x_cos.zarr \
    --output merged.zarr
```

The output keeps the axis-z volume's shape `(N, Z_ref, Y_ref, X_ref)` and appends
direction channels from the other axes:

| Index | Name | Source |
|-------|------|--------|
| 0 | `cos` | z-volume |
| 1 | `grad_mag` | z-volume |
| 2 | `dir0` | z-volume (YX-plane directions) |
| 3 | `dir1` | z-volume (YX-plane directions) |
| 4 | `valid` | z-volume |
| 5 | `dir0_y` | y-volume (ZX-plane directions, resized) |
| 6 | `dir1_y` | y-volume (ZX-plane directions, resized) |
| 7 | `dir0_x` | x-volume (ZY-plane directions, resized) |
| 8 | `dir1_x` | x-volume (ZY-plane directions, resized) |

Channels 5-6 and 7-8 are only present when the corresponding volume is provided.
Either `--y-volume` or `--x-volume` (or both) must be given.

**How the merge works:**

1. Z index mapping: z-volume index `i` maps to source index `i * step` (the
   ratio `z_step_eff / downscale`). This is exact when all runs use the same
   `--step` and `--downscale`.

2. Plane resize: each source Z-slice has one sparse dimension (Y for axis-y,
   X for axis-x) that is `step` times smaller than the z-volume reference.
   Bilinear interpolation (`torch.nn.functional.interpolate`) upsamples it to
   match. The other plane dimension is already the same size.

3. Processing is batched along Z (`--batch-size`, default 32) for efficiency.

### Full pipeline example

```bash
# 1. Run UNet inference along all three axes (same crop & params)
for ax in z y x; do
    python preprocess_cos_omezarr.py \
        --axis $ax \
        --input volume.zarr/0 \
        --output ${ax}_cos.zarr \
        --unet-checkpoint model.pt \
        --step 10 --downscale 4
done

# 2. Merge direction channels into z-volume
python preprocess_cos_omezarr.py integrate \
    --z-volume z_cos.zarr \
    --y-volume y_cos.zarr \
    --x-volume x_cos.zarr \
    --output merged_cos.zarr
```
