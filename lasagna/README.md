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

The install exposes these commands, including the native Fiberlet processor
packaged by the sibling Volume Cartographer Python distribution:

```bash
lasagna-fit-service --help
lasagna-download --help
lasagna-download-list --help
lasagna-preprocess --help
lasagna-preprocess integrate --help
lasagna-preprocess predict3d --help
las_manager --help
vc_fiberlets --help
```

The bootstrap installs Lasagna with `-e`, so changes in this checkout are
immediately visible in the environment without reinstalling. The distribution
also exposes the sibling Vesuvius Fiber packages and canonical `lasagna.*`
modules; direct and managed inference do not require `PYTHONPATH`.

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
las_manager snapshot import /path/to/RUN/snapshots/best.pt --model-identifier OWNER/REPOSITORY
las_manager volume prefetch <volume> 1 --workers 512
las_manager inference run <snapshot> <volume> [scale]
las_manager las-inference <snapshot> <volume> [scale]
las_manager fiber-inference <snapshot> <volume> [scale]
las_manager inference ls
las_manager fiberlet run <completed-fiber-inference>
```

Fresh configs use `params` for Fiber inference defaults. Regular Lasagna uses
its checkpoint and CLI defaults, with `--devices all` supplied by the separate
`lasagna_params` array. The snapshot and volume remain required, while the
input scale defaults to OME-Zarr group 2. `las-inference` and
`fiber-inference` are backend-forcing aliases for `inference run`; all three
share the same launch path. Arguments after `--` remain available for
deliberate per-run overrides.

`inference run` returns as soon as its run directory and detached tmux session
are created. The tmux workflow performs automatic prefetch first and inference
second; use `las_manager run ls`, `las_manager tmux attach <run>`, or follow the
printed run directory's `run.log`. Passing `--no-prefetch` writes only the
source descriptor and lets the inference backend download missing chunks on
demand for its crop; this download belongs to the inference lifecycle.

### Cooperative full-volume GPUs

`las_manager` inference jobs automatically use the local cooperative GPU manager
(`incremental_release` capability required). Prefetch does not reserve GPUs.
The GPU phase uses one tile per forward pass and varies the **number of GPUs**,
not batch size or tile geometry. `--devices` limits the eligible devices; it is
not itself a reservation. CUDA ordinals are resolved to UUIDs before requesting
resources. An older manager fails capability negotiation; it is never bypassed
or restarted behind existing clients' backs.

CPU readers, caches, bounded queues, accumulators and output state persist when
the allocation changes. Removed GPU workers finish submitted tiles, return their
results and exit before release is acknowledged. Retained workers stay loaded;
only newly granted GPUs load a model. Full pause unloads all GPU workers; work
continues automatically after a new grant. Pyramid generation runs after GPU
workers have exited and release has been reported. A manager restart still
requires a GPU-only drain; the CPU pipeline remains intact. Communication errors
are logged and stop new tile admission, drain GPU workers and retry every five
seconds. No full-volume rescan is needed for these transitions.

Until a one-tile peak is measured, the conservative default requests a full
selected GPU's memory (the largest selected capacity on heterogeneous hosts).
To share GPUs, pass a **measured budget including CUDA overhead and reserve**
through `LASAGNA_GPU_MEMORY_GIB`, for example `46` only if that budget has been
verified for the chosen model/tile size. `LASAGNA_GPU_PRIORITY` defaults to `0`;
`VILLA_GPU_MANAGER_SOCKET` optionally selects a socket. These variables are
inherited by the inference runner. The worker limits Torch's allocator to the
grant minus 1 GiB reserved for non-allocator CUDA use; this is not a memory
calibration and an undersized user budget may still fail with OOM. Normalization
calibration must be performed separately, not inside cooperative inference.
Direct API/CLI callers can opt into the same behavior with `LASAGNA_COOPERATIVE=1`;
the manager launcher always sets it for volume inference, not Fiberlet jobs.

Current Fiber checkpoints carry their authoritative inference config, so no
separate config argument is needed. Direct inference uses the same rule:

```bash
python -m vesuvius.neural_tracing.fiber_trace_3d.infer \
  --input /path/to/volume.ome.zarr/1 \
  --output /path/to/artifacts/fiber.zarr \
  --checkpoint /path/to/snapshots/best.pt
```

Pass a config JSON as the optional positional argument only for a legacy
checkpoint without embedded config. Every run writes portable
`inference.json` next to the output manifest, including the exact Villa Git
commit in `inference.code_commit`.

Direct Lasagna checkpoints are listed under `lasagna/...` selectors and use
the same command shape:

```bash
las_manager inference run lasagna/<run>/<checkpoint.pt> <volume> 1 -- --devices all
```

The manager dispatches these to `preprocess_cos_omezarr predict3d`; both direct
and managed Lasagna inference write portable provenance.

New Fiber and Lasagna outputs default to the [compact schema v3](docs/volume_schema_v3.md):
one Zarr store, an embedded manifest, and resolution-based full-channel chunks.
Existing sidecar outputs remain resumable. Fiber additionally defaults to
blanking auxiliary channels at zero presence and rounding/clamping width to
uint8 base-volume voxels; `--no-blank-invalid` and `--no-width-uint8-base`
disable those changes independently.

Atlas/catalogue publication and manager Fiberlet discovery still require
legacy output until their v3 integration is available. For those workflows,
append `-- --output-format legacy` to the manager command. Direct inference
uses `--output-format legacy --output /path/to/fiber.lasagna.json`.

Commands and subcommands accept unique prefixes, so `las_manager sn l` is the
same as `las_manager snapshot ls`. Install path-aware Bash completion once with
`las_manager completion install`; it follows whichever registered venv's
`las_manager` is currently selected by `PATH`. Configuration, cache semantics,
and selector details are documented in [`docs/manager.md`](docs/manager.md).
Cached volumes, snapshots, runs, option values, and locally known OME scale
indices are completed contextually. A final `help` token shows help for the
longest recognized command prefix, for example `las_manager vol pre help`.

Managed Fiberlet preprocessing automatically resolves compatible published
regular Lasagna normals from the open-data catalogue and caches their manifest
and on-demand chunks under the configured manager cache. See
[`docs/manager.md`](docs/manager.md#whole-volume-fiberlet-jobs) for explicit
Atlas/local normal overrides and compatibility rules.

This installation currently expects the `villa` monorepo layout: Lasagna
packages the sibling `vesuvius/src` implementation, installs its declared
model dependencies, and installs the sibling Volume Cartographer Python
distribution with its `vc_fiberlets` command and the sibling `vc_object_store`
Python package, through which surfaces are exchanged. Copying only the `lasagna/`
directory is therefore not yet a supported standalone installation. On Linux,
the bootstrap selects an available GCC 13 or newer for that C++23 build unless
`CC` or `CXX` is already set.

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
- `meta.json` contains required `uuid` (a fresh random UUID per output surface, never derived from the dirname) and `type="seg"`.


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

Use a subset with `--devices cuda:0,cuda:2`. Input tiles are read in the
background through `vc_reader` outside GPU workers. The tile Cartesian product
is generated lazily and read-ahead is bounded independently from reusable GPU
shared-memory slots. `--slots-per-gpu` (default 2) controls GPU input/result
buffers; `--prefetch-tiles-per-gpu` (default 4) controls outstanding/ready input
tiles. The reader defaults to a 4 GiB memory cache and 16 threads, configurable
with `--input-cache-gib` and `--input-io-threads`. `--prefetch-workers` overrides
how many tile reads run at once. Existing `--device` selects single-device
inference, which also maintains bounded read-ahead during GPU forwards.

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

For full-volume inference when one selected input scale is too large to keep,
enable the shared rolling disk cache:

```bash
lasagna-preprocess predict3d ... --input /cache/volume.zarr/0 --live-fetch
python -m vesuvius.neural_tracing.fiber_trace_3d.infer ... \
  --input /cache/volume.zarr/0 --live-fetch
```

The local OME root must contain `_download` S3 metadata. The defaults are a
10 TiB selected-scale target (`--live-cache-gib 10240`) and 10,000 lazy tile
descriptors (`--live-fetch-ahead-tiles 10000`); this disk window is separate
from the much smaller `--prefetch-tiles-per-gpu` array window. Only
whole input Z-chunk planes safely behind a fully committed model row are
removed. Cache pressure may temporarily exceed the target when no obsolete
plane is safe. Live mode is intentionally full-volume only, conflicts with
`--no-download`, and currently rejects a separately remote Lasagna `--pred-dt`.
Interrupted runs retain atomically completed chunks and can be rerun directly.

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
