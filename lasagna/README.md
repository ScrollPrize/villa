# 2D TIFF layer UNet trainer

## Installation

Lasagna requires [`uv`](https://docs.astral.sh/uv/) and currently targets
Python 3.14. The recommended installer creates an editable virtual environment,
selects a driver-compatible official PyTorch build, and installs the fit
service, downloader, and full 2D/3D preprocessing stack.

```bash
cd /path/to/villa/lasagna
python3 scripts/bootstrap_venv.py --venv .venv
source .venv/bin/activate
```

The bootstrap uses `nvidia-smi` to select a pinned official PyTorch build for
CUDA 12.8, CUDA 13.0, or CPU, then installs Lasagna. Override detection with
`--backend cpu`, `--backend cu128`, or `--backend cu130`, for example:

```bash
python3 scripts/bootstrap_venv.py --venv .venv --backend cu128
```

Verify the selected PyTorch build and GPU access:

```bash
python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())'
```

The install exposes these commands:

```bash
lasagna-fit-service --help
lasagna-download --help
lasagna-download-list --help
lasagna-preprocess --help
lasagna-preprocess integrate --help
lasagna-preprocess predict3d --help
las_manager --help
```

The bootstrap installs Lasagna with `-e`, so changes in this checkout are
immediately visible in the environment without reinstalling.

### Inference manager

`las_manager` provides configuration, discovery, prefetch, and durable tmux
execution for managed Fiber and Lasagna inference. Initialize its XDG configuration, fill
in `cache_dir`, `output_dir`, `venv`, and one or more `snapshot_dirs`, then:

```bash
las_manager config init
${EDITOR:-vi} "${XDG_CONFIG_HOME:-$HOME/.config}/las_manager/config.toml"
las_manager fetch
las_manager volume ls --sample PHerc0332
las_manager snapshot ls
las_manager volume prefetch <volume> 1 --workers 512
las_manager inference run <snapshot> <volume> 1 -- --devices all
las_manager inference ls
```

Current Fiber checkpoints carry their authoritative inference config, so no
separate config argument is needed. Direct inference uses the same rule:

```bash
python -m vesuvius.neural_tracing.fiber_trace_3d.infer \
  --input /path/to/volume.ome.zarr/1 \
  --output /path/to/artifacts/fiber.lasagna.json \
  --checkpoint /path/to/snapshots/best.pt
```

Pass a config JSON as the optional positional argument only for a legacy
checkpoint without embedded config. Every run writes portable
`inference.json` next to the output manifest.

Direct Lasagna checkpoints are listed under `lasagna/...` selectors and use
the same command shape:

```bash
las_manager inference run lasagna/<run>/<checkpoint.pt> <volume> 1 -- --devices all
```

The manager dispatches these to `preprocess_cos_omezarr predict3d`; both direct
and managed Lasagna inference write portable provenance and remain compatible
with the existing Atlas `lasagna` artifact type.

Commands and subcommands accept unique prefixes, so `las_manager sn l` is the
same as `las_manager snapshot ls`. Install path-aware Bash completion once with
`las_manager completion install`; it follows whichever registered venv's
`las_manager` is currently selected by `PATH`. Configuration, cache semantics,
and selector details are documented in [`docs/manager.md`](docs/manager.md).
Cached volumes, snapshots, runs, option values, and locally known OME scale
indices are completed contextually. A final `help` token shows help for the
longest recognized command prefix, for example `las_manager vol pre help`.

This installation currently expects the `villa` monorepo layout: Lasagna uses
the sibling `vesuvius/src` model implementation and installs its declared model
dependencies. It deliberately does not build Volume Cartographer, which is not
needed for preprocessing. Copying only the `lasagna/` directory is therefore
not yet a supported standalone installation.

### Batch-download the PHerc scale-0 volumes

Pass a one-S3-URI-per-line list from the directory that should contain the
scroll directories:

```bash
cd /path/to/scrolls
lasagna-download-list /ephemeral/hendrik/las_volumes/pherc_volumes.txt
```

This processes one volume at a time and uses 512 parallel chunk-transfer
workers inside each volume. The resulting layout is:

```text
scrolls/
  PHerc0125/
    info.json
    volumes/
      20250821151825-9.362um-1.2m-113keV-masked.zarr/
```

Use `--dry-run` to inspect a run without writing anything:

```bash
lasagna-download-list /path/to/other-volumes.txt
lasagna-download-list /ephemeral/hendrik/las_volumes/pherc_volumes.txt --dry-run
```

CuPy remains an optional acceleration path and is not required for the
preprocessing commands above.

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

The `predict3d` subcommand is the Lasagna cos/normal wrapper around the shared
3D tiled inference code in `tiled_predict3d.py`. It keeps the existing CLI,
manifest, fixed-depth circular Z accumulation, chunk-resume, and OME-Zarr pyramid
behavior; product-specific Lasagna logic remains in
`preprocess_cos_omezarr.py`.

Whole-volume predict3d can use every visible CUDA device through the shared
runner:

```bash
lasagna-preprocess predict3d ... --devices all
```

New Fiber and Lasagna `predict3d` OME-Zarr arrays use the same Zarr-v2
Blosc/Zstd compressor (`clevel=3`, byte shuffle) at every pyramid level.
Existing arrays keep their compressor when inference resumes; a mismatch is
reported. Use `--ome-compressor none` only when uncompressed compatibility
output is required.

Use a subset with `--devices cuda:0,cuda:2`. Input tiles default to asynchronous
TensorStore bounding-box reads outside GPU workers. The tile Cartesian product
is generated lazily and read-ahead is bounded independently from reusable GPU
shared-memory slots. `--slots-per-gpu` (default 2) controls GPU input/result
buffers; `--prefetch-tiles-per-gpu` (default 4) controls outstanding/ready input
tiles. TensorStore defaults to a 4 GiB cache, 16 file-I/O threads, and 4
decode/copy threads, configurable with `--input-cache-gib`,
`--input-io-threads`, and `--input-copy-threads`. Use
`--input-reader python-zarr` for the old backend; `--prefetch-workers` controls
only its reader threads. Existing `--device` selects single-device inference,
which also maintains bounded TensorStore read-ahead during GPU forwards.

Add `--profile-pipeline` to a multi-device run for bounded loader/worker
diagnostics. It reports backend read service, active wall span, throughput and
effective outstanding-request concurrency, completion polling and ready-queue delay, shared-memory copy, CPU
conversion, compact integer H2D, CUDA conversion, adapter preprocessing, model inference, output/D2H, result
receipt, and commit time. Stage sums overlap across tiles/workers and are not
elapsed wall time. CUDA events add diagnostic overhead, so disable the flag for
final throughput measurements.

CUDA inference transfers source `uint8`/`uint16` tiles without CPU float
expansion and performs the historical normalization on-device in FP32. The CPU
fallback retains the NumPy conversion path. Model-specific AMP remains owned by
the model adapter; shared weighting, pyramid filtering, D2H results, and
accumulation remain FP32.

The same shared runner overlaps output flushing with subsequent inference using
one enlarged circular mmap ring and persistent spawned writer processes. Each
worker reopens the frozen mmap read-only and handles one output chunk, so no
band-sized RAM snapshot crosses process boundaries and each worker has an
independent Python GIL. `--flush-workers` defaults to the available CPU count
capped at 64; use
`--flush-workers 0` for the synchronous, immediate-release A/B baseline. Only
one flush batch may be outstanding, so the next frontier waits if writers fall
behind. The final `flush stats workers=... chunks=... work_sum=... wait=...`
line reports that backpressure.

Raw product rings default to float16, intentionally permitting small rounding
differences while halving product-ring backing. The shared geometric weight
ring and flush normalization remain float32; pass
`--product-accumulator-dtype float32` when float32 accumulation is required.

On multi-device runs, chunk accumulation is itself process-parallel. Persistent
workers add directly from retained result shared-memory slots into the rolling
mmap, with deterministic per-chunk ownership and FIFO update order.
`--accumulator-workers` defaults to the CPU count capped at 32; zero restores
the synchronous A/B baseline. The native `accumulator_add` module runtime-
dispatches an AVX-512F+F16C row kernel where available and uses a portable
fallback elsewhere; the package is never globally compiled for AVX-512.

Automatic S3 chunk fetching defaults to 64 transfer threads. Set it separately
from inference prefetch with, for example, `--download-workers 256`. Interrupted
or malformed `.dl_cache/*.noremote.json` files are advisory: they are ignored
with a warning and rewritten atomically rather than aborting inference.

### Multi-axis processing

The `--axis` flag controls which dimension is sliced through:

| `--axis` | Slice dim | 2D plane fed to UNet |
|----------|-----------|----------------------|
| `z` (default) | Z | Y x X |
| `y` | Y | Z x X |
| `x` | X | Z x Y |

`--scaledown` (default 4) applies **uniformly in all three dimensions** — both
the slice stepping and the plane downscale use the same factor. This means
1 output voxel = scaledown fullres voxels in every direction.

The crop (`--crop-xyzwhd`) is always in absolute input coordinates regardless of
axis.

Each per-axis output zarr has shape `(5, out_Z, out_Y, out_X)` with uniform
resolution: `full_size // scaledown` in every dimension.

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

### Fusion and integration (`integrate` subcommand)

After running preprocessing for all three axes, the `integrate` subcommand fuses
cos and grad_mag using estimated 3D surface normal weights, and copies per-axis
dir channels into a single output volume:

```bash
python preprocess_cos_omezarr.py integrate \
    --z-volume <Z_PREPROC>.zarr \
    --y-volume <Y_PREPROC>.zarr \
    --x-volume <X_PREPROC>.zarr \
    --output <FUSED>.zarr \
    --pred-dt <PRED_SURFACE>.zarr    # optional: distance-to-skeleton channel
```

Output channels:

| Index | Name | Source |
|-------|------|--------|
| 0 | `cos` | **fused** — normal-weighted average of z/y/x cos |
| 1 | `grad_mag` | **fused** — sum of z/y/x grad_mag / weight_sum |
| 2 | `dir0` | z-volume (YX-plane directions) |
| 3 | `dir1` | z-volume (YX-plane directions) |
| 4 | `valid` | z-volume |
| 5 | `dir0_y` | y-volume (ZX-plane directions, resized) |
| 6 | `dir1_y` | y-volume (ZX-plane directions, resized) |
| 7 | `dir0_x` | x-volume (ZY-plane directions, resized) |
| 8 | `dir1_x` | x-volume (ZY-plane directions, resized) |
| 9 | `pred_dt` | distance to skeleton (only if `--pred-dt` given) |

If only two axis volumes are provided the fusion falls back to z-only
cos/grad_mag (no normal estimation).

### Full pipeline example

```bash
VOLUME=<INPUT_VOLUME>.zarr/0
CKPT=<UNET_CHECKPOINT>.pt
CROP="<X> <Y> <Z> <W> <H> <D>"   # fullres coordinates

# 1. Preprocess each axis (same crop, same scaledown)
for ax in z y x; do
    python preprocess_cos_omezarr.py \
        --axis $ax \
        --input $VOLUME \
        --output ${ax}_cos.zarr \
        --unet-checkpoint $CKPT \
        --crop $CROP
done

# 2. Fuse into single volume (optionally with pred-dt)
python preprocess_cos_omezarr.py integrate \
    --z-volume z_cos.zarr \
    --y-volume y_cos.zarr \
    --x-volume x_cos.zarr \
    --output fused.zarr \
    --pred-dt <PRED_SURFACE>.zarr

# 3. Convert to VC3D OME-Zarr for visualization
python convert_fit_zarr_to_vc3d_omezarr.py \
    --input fused.zarr \
    --output-prefix vc3d
```

Step 3 produces one OME-Zarr pyramid per channel: `vc3d_cos.ome.zarr`,
`vc3d_grad_mag.ome.zarr`, etc. Each pyramid has 5 levels (configurable via
`--levels`) and reads `scaledown` from the input zarr metadata to place the
crop at the correct absolute position in the pyramid.
