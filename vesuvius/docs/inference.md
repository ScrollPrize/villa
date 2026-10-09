# Inference Pipeline Overview

The Vesuvius tooling exposes three command-line stages plus a convenience orchestrator:

1. `vesuvius.predict` — run a trained model and write patch logits.
2. `vesuvius.blend_logits` — merge overlapping patches with Gaussian weighting.
3. `vesuvius.finalize_outputs` — convert logits into probabilities or masks and build a multiscale Zarr.

All commands honour local paths and remote storage backed by `fsspec` (for example S3). Run `vesuvius.accept_terms --yes` before accessing remote scroll volumes.

## Stage 1 — `vesuvius.predict`

`vesuvius.predict` loads a checkpoint (nnU-Net v2 compatible or a `vesuvius.train` checkpoint) and produces tiled logits. It supports distributed execution by splitting the volume into `num_parts` and assigning each process a unique `part_id`.

```bash
vesuvius.predict \
  --model_path /path/to/model \
  --input_dir /path/to/input.zarr \
  --output_dir /tmp/logits \
  --num_parts 4 \
  --part_id 0 \
  --device cuda:0
```

### Key Arguments

| Argument | Description |
|----------|-------------|
| `--model_path` (required) | Path to a model directory, a `.pth` checkpoint, or `hf://` repository.
| `--input_dir` (required) | Volume input: Zarr root, TIFF stack, or a directory understood by the `Volume` helper. For open data use the `s3://` access root the catalog publishes, together with `--input_anon`.
| `--input_anon` | Read the input bucket anonymously. Required for `s3://vesuvius-challenge-open-data/...`, which is public but unsigned.
| `--output_dir` (required) | Destination folder for logits (`logits_part_{id}.zarr`) and coordinates.
| `--input_format` | Force `zarr`, `tiff`, or `volume` detection. Usually optional.
| `--tta_type` / `--disable_tta` | Choose `rotation` (default) or `mirroring`, or disable test-time augmentation.
| `--num_parts` / `--part_id` | Partition inference so multiple machines can process different chunks.
| `--bbox` | Restrict inference to a region of interest: `"z0:z1,y0:y1,x0:x1"` in global voxel coordinates, half-open. Omit a bound to reach the volume edge (`"1000:1400,:,2000:"`). See [Region-of-interest inference](#region-of-interest-inference).
| `--chunk_cache_mb` | Keep up to this many MB of fetched input chunks in an in-memory LRU cache (zarr 3), so overlapping patches do not download the same chunks again. Each DataLoader worker keeps its own cache. Default `0` (off). See [Streaming a remote volume](#streaming-a-remote-volume-chunk-cache-patch-order-and-workers).
| `--patch_order` | `auto` (default) is `chunk` when `--chunk_cache_mb > 0` and `zyx` otherwise. `chunk` reads patches along a Morton curve over the input's chunk grid and gives each DataLoader worker one contiguous run of it, so patches that share chunks are read back to back by the same process; `zyx` is the row-major order with plain batching, the pipeline as it was before this flag. The patches, their coordinates, the per-patch logits and the blended output are the same either way; only the index order inside `logits_part_*`/`coordinates_part_*` differs. See [Streaming a remote volume](#streaming-a-remote-volume-chunk-cache-patch-order-and-workers).
| `--overlap` | Fractional patch overlap (0–1, default `0.5`).
| `--batch_size` | Inference batch size (default `1`).
| `--patch_size` | Override the model patch size using a comma-separated list (e.g. `192,192,192`).
| `--mode` | `binary`/`multiclass` (default segmentation) or `surface_frame` to keep 9-channel tangent frames.
| `--tif-activation` | When writing TIFF outputs, pick `softmax`, `argmax`, or `none`.
| `--save_softmax` | Legacy flag for saving softmax logits (consider `--tif-activation`).
| `--normalization` | Runtime normalization (`instance_zscore`, `global_zscore`, `instance_minmax`, `ct`, `none`).
| `--intensity-properties-json` | nnU-Net style JSON with intensity stats for CT normalization.
| `--device` | Device string such as `cuda`, `cuda:1`, or `cpu`.
| `--skip_empty_patches` / `--no_skip_empty_patches` | Toggle automatic removal of homogeneous patches.
| `--zarr_compressor` / `--zarr_compression_level` | Configure output compression (`zstd` with level `3` by default).
| `--scroll_id`, `--segment_id`, `--energy`, `--resolution` | Metadata when reading remote scrolls via the `Volume` helper.
| `--hf_token` | Hugging Face token for private repositories.
| `--config-yaml` | Training YAML to resolve model architecture when the checkpoint lacks embedded metadata.
| `--verbose` | Print detailed progress information.

Distributed execution simply repeats the command with different `part_id` values. All workers must share the same `output_dir`:

```bash
# machine 1
vesuvius.predict --model_path ... --num_parts 4 --part_id 0 --device cuda:0
# machine 2
vesuvius.predict --model_path ... --num_parts 4 --part_id 1 --device cuda:0
```

Each worker writes `logits_part_<id>.zarr` and `coordinates_part_<id>.zarr` into the output directory.

### Region-of-interest inference

`--bbox` restricts the sliding window to one region of the volume, given in **global voxel
coordinates** as `"z0:z1,y0:y1,x0:x1"`. Ranges are half-open (`z0` included, `z1` excluded),
and either bound may be omitted to reach the volume edge:

```bash
vesuvius.predict --model_path hf://scrollprize/surface_recto \
    --input_dir s3://vesuvius-challenge-open-data/PHercParis4/volumes/<volume>.zarr --input_anon \
    --output_dir /tmp/logits \
    --bbox "2000:2200,1000:1200,1000:1200"

# open bounds: from z=1000 to the end, all of y, up to x=2000
vesuvius.predict ... --bbox "1000:,:,:2000"
```

Patch coordinates stay in the global frame, so `blend_logits` and `finalize_outputs` need no
extra flags — the merged output lands at the same absolute position it would have had in a
full-volume run. The requested ROI is recorded in the logits store's `bbox` attribute.

This matters most when streaming a remote volume: only the chunks intersecting the ROI are
ever fetched, so a small region of a multi-terabyte scroll costs seconds of network instead
of hours. It is also the practical way to iterate on a model, inspect one column of text, or
reproduce a result on a laptop.

Notes:
- `--num_parts`/`--part_id` split the ROI's Z extent (not the whole volume), so distributed
  runs stay balanced when a bbox is active.
- An ROI smaller than the model patch size along an axis is grown to the patch size and
  shifted back inside the volume, so the model always sees a full patch.
- Bounds are clamped to the volume; a bbox entirely outside it is an error.

### Streaming a remote volume: chunk cache, patch order and workers

Patches overlap by 50% and are not aligned to the input's chunk grid, so every chunk is needed
by many patches: a 192³ patch over 128³ chunks touches up to 27 of them, and with no cache each
one is downloaded again for every patch that needs it (11.5× the region's size in the run below:
3539 MB fetched for 308 MB of distinct chunks, with 192³ patches at overlap 0.5; more for smaller
patches). `--chunk_cache_mb` keeps fetched chunks in memory, but a cache only helps when the
patches that share a chunk are read close together *and by the same process*:

- With `--patch_order chunk` the patch list follows a Morton curve over the chunk grid, so
  neighbouring patches are consecutive. The row-major order (`zyx`) walks a whole row of x before
  returning to the chunks it just used, which have been evicted by then unless the cache holds an
  entire row band of the sweep.
- With `--num_workers N` the DataLoader hands batch *i* to worker *i mod N*, which would scatter
  neighbouring patches across processes. In `chunk` order `vesuvius.predict` instead emits the
  batches so that each worker gets one contiguous run of the patch list, and each worker's cache
  sees its neighbours. That batch *i* goes to worker *i mod N* is a PyTorch implementation detail
  (its `in_order=True` default); if it ever changed, only this cache locality would be lost, never
  correctness, since every patch is still read and written exactly once by whichever worker gets it.

`--patch_order auto` (the default) is `chunk` when `--chunk_cache_mb > 0` and `zyx` otherwise, so a
run without the cache is the pipeline as it was before this option, sequence for sequence: with the
cache off the order makes no difference to what is fetched (4600 requests either way below).
`--max_patches` picks its evenly spaced subset along whichever order is in effect.

Both change only the order in which patches are read: which patches are read, their coordinates
and the per-patch logits are identical, and only the index each patch gets in
`logits_part_*.zarr`/`coordinates_part_*.zarr` differs. `blend_logits` accumulates the patches
overlapping a voxel in coordinate order rather than index order, so the blended output is
identical too (checked on the run below: the merged logits of a `zyx` run and of a `chunk` run
are byte-identical over the region). Measured on
`PHercParis4.volpkg/volumes_zarr_standardized/54keV_7.91um_Scroll1A.zarr` (128³ zstd chunks),
`--bbox 6400:6592,5120:5312,2048:3584`, 192³ patches, overlap 0.5, batch 1: 288 patches over 400
distinct chunks (308 MB), counting every chunk request made by every process (the logits digest
was the same `dc39226603a3afe1` in every run):

| `--num_workers` | `--chunk_cache_mb` | row-major, round-robin workers (`zyx`) | row-major, contiguous dispatch | chunk order, contiguous dispatch (`chunk`) |
|---|---|---|---|---|
| 4 | 0 | 4600 requests, 3539 MB | – | 4600 requests, 3539 MB |
| 4 | 64 | 2919 requests, 2241 MB | 1292 requests, 996 MB | **590 requests, 455 MB** |
| 4 | 512 | 1360 requests, 1047 MB | 800 requests, 614 MB | **572 requests, 441 MB** |
| 0 | 64 | 1289 requests, 994 MB | – | **506 requests, 392 MB** |
| 0 | 512 | 400 requests, 308 MB | – | 400 requests, 308 MB |

The middle column was measured with an intermediate build that dispatched contiguous runs of the
row-major list; it is not a CLI option and only separates the effect of the dispatch from that of
the order. With round-robin dispatch every worker fetched 340 of the 400 chunks itself; with one
contiguous run each, a worker fetches its own quarter plus the chunks on the boundary with its
neighbours, which is the remaining 1.4×. When the cache cannot hold a whole row band of the sweep
(64 MB here, ~75 chunks against a 135-chunk band) the order decides the count: row-major re-fetches
the band for every row (3.2× in one process, 7.3× with four), the chunk order stays near the floor.
A chunk cache of a few hundred MB per worker is therefore enough once the order is chunk-local,
whereas in row-major order it must hold a whole band (chunks along x × patch depth in y and z),
which for a full-width scroll is several GB per worker. With `--chunk_cache_mb 0` the order makes
no difference to what is fetched, which is why `auto` leaves it row-major then.

## Stage 2 — `vesuvius.blend_logits`

Combine the partial logits by weighting overlaps with a Gaussian window. The command scans the `parent_dir` for matching `logits_part_*.zarr` and `coordinates_part_*.zarr` pairs. The patches overlapping a voxel are accumulated in coordinate order, so the result does not depend on the index order `vesuvius.predict` wrote them in (nor on the blend's internal grid size).

```bash
vesuvius.blend_logits /tmp/logits /tmp/merged_logits.zarr \
  --num_workers 16 \
  --chunk_size 256,256,256
```

### Options

| Argument | Description |
|----------|-------------|
| `parent_dir` | Folder containing the per-part logits and coordinates Zarr stores.
| `output_path` | Destination Zarr for the merged logits.
| `--sigma_scale` | Controls Gaussian falloff (`patch_size / sigma_scale`, default `8.0`).
| `--chunk_size` | Spatial chunk size (`Z,Y,X`) for the merged Zarr. Leave unset to auto-pick.
| `--num_workers` | Number of worker processes. Defaults to `CPU_COUNT - 1`.
| `--compression_level` | Zarr compression level (0–9, default `1`).
| `--quiet` | Suppress verbose logging.

The merged logits retain the same class/channel dimension as the individual parts.

## Stage 3 — `vesuvius.finalize_outputs`

Finalize logits into probabilities or masks and optionally build a multiscale pyramid. The command writes OME-NGFF metadata and (when requested) deletes the intermediate logits directory.

```bash
vesuvius.finalize_outputs /tmp/merged_logits.zarr /tmp/final_output.zarr \
  --mode binary \
  --threshold --threshold_value 0.3 \
  --delete_intermediates
```

`--threshold` toggles binarization on (default cutoff `0.5`). `--threshold_value T` overrides the cutoff with any value in `(0, 1)` and requires `--threshold`. The right cutoff is model-dependent and should come from validation — `0.3` above is illustrative, not a recommendation.

### Options

| Argument | Description |
|----------|-------------|
| `input_path` | Path to the blended logits Zarr (level `0` is the logits array).
| `output_path` | Destination multiscale Zarr root.
| `--mode` | `binary` (default), `multiclass`, or `surface_frame` (keeps 9-channel frame outputs; no thresholding).
| `--threshold` | Binarize the probability map. In `binary` mode cuts at the probability given by `--threshold_value` (default `0.5`). In `multiclass` mode emits the argmax channel. Ignored in `surface_frame` mode.
| `--threshold_value T` | Override the `--threshold` cutoff with `T` in `(0, 1)`. Requires `--threshold`. Binary mode only — rejected in `multiclass` since argmax ignores the cutoff.
| `--delete_intermediates` | Remove the source logits after a successful run.
| `--chunk_size` | Spatial chunk size for the output store (`Z,Y,X`). Defaults to the logits chunking.
| `--num_workers` | Worker processes for finalization (defaults to half of CPU cores).
| `--quiet` | Suppress verbose logging.

Without `--threshold`, binary mode outputs a single softmax foreground channel; multiclass mode writes one channel per class plus an argmax channel. `surface_frame` mode bypasses thresholding entirely and stores orthonormal 9-channel frames in float32.

## Full Remote Workflow Example

```bash
# 1. Run prediction on four machines (part IDs 0–3)
vesuvius.predict --model_path hf://scrollprize/surface_recto \
    --input_dir s3://vesuvius/input/Scroll1.zarr \
    --output_dir s3://vesuvius/tmp/logits \
    --num_parts 4 \
    --part_id 0 \
    --device cuda:0 \
    --zarr_compressor zstd \
    --zarr_compression_level 3 \
    --skip_empty_patches

# ...repeat for part_id 1,2,3 on other hosts...

# 2. Blend logits once all parts finish
vesuvius.blend_logits s3://vesuvius/tmp/logits s3://vesuvius/tmp/merged_logits.zarr \
    --num_workers 32 \
    --chunk_size 256,256,256

# 3. Finalize outputs (bare --threshold cuts at 0.5; add --threshold_value 0.3 for a different cutoff)
vesuvius.finalize_outputs s3://vesuvius/tmp/merged_logits.zarr s3://vesuvius/output/final.zarr \
    --mode binary \
    --threshold \
    --delete_intermediates
```

After finalization the destination Zarr contains a multiscale hierarchy (`0/`, `1/`, …) and a `metadata.json` file describing the inference run. Rechunk the output if you plan to serve it through a viewer that expects different chunk sizes.
