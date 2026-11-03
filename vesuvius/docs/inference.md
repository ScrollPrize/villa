# Inference Pipeline Overview

The Vesuvius tooling exposes three command-line stages:

1. `vesuvius.predict` — run a trained model and write patch logits.
2. `vesuvius.blend_logits` — merge overlapping patches with Gaussian weighting.
3. `vesuvius.finalize_outputs` — convert logits into a channel-first Zarr array of probabilities or labels.

All commands honour local paths and remote storage backed by `fsspec` (for example S3). Remote scroll volumes are covered by the [data licence](../README.md#data-licence).

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
| `--input_dir` (required) | Input Zarr path or a catalog scroll/segment identifier understood by the `Volume` helper. TIFF stacks are not supported by this command. For open data use the `s3://` access root the catalog publishes, together with `--input_anon`.
| `--input_anon` | Read the input bucket anonymously. Required for `s3://vesuvius-challenge-open-data/...`, which is public but unsigned.
| `--output_dir` (required) | Destination folder for logits (`logits_part_{id}.zarr`) and coordinates.
| `--input_format` | Format hint (`zarr` by default, or `volume` for explicit catalog identifiers). Paths are opened as Zarr; this does not enable TIFF input.
| `--tta_type` / `--disable_tta` | Choose `mirroring` (default) or `rotation`, or disable test-time augmentation.
| `--num_parts` / `--part_id` | Partition inference so multiple machines can process different chunks.
| `--bbox` | Select full global-grid patches intersecting a region of interest: `"z0:z1,y0:y1,x0:x1"` in global voxel coordinates, half-open. Omit a bound to reach the volume edge (`"1000:1400,:,2000:"`). See [Region-of-interest inference](#region-of-interest-inference).
| `--chunk_cache_mb` | Keep up to this many MB of fetched input chunks in an in-memory LRU cache (zarr 3), so overlapping patches do not download the same chunks again. Each DataLoader worker keeps its own cache. Default `0` (off).
| `--overlap` | Patch stride as a fraction of patch size (0–1, default `0.5`, giving 50% overlap).
| `--batch_size` | Inference batch size (default `1`).
| `--patch_size` | Override the model patch size using a comma-separated list (e.g. `192,192,192`).
| `--normalization` | Fallback input normalization when the model supplies no setting (default `instance_zscore`). Model-declared normalization takes precedence; required statistics are read from the checkpoint or nnU-Net plans. |
| `--device` | Device string such as `cuda`, `cuda:1`, or `cpu`.
| `--skip_empty_patches` / `--no_skip_empty_patches` | Toggle automatic removal of homogeneous patches.
| `--zarr_compressor` / `--zarr_compression_level` | Configure output compression (`zstd` with level `3` by default).
| `--scroll_id`, `--segment_id`, `--energy`, `--resolution` | Metadata when reading remote scrolls via the `Volume` helper.
| `--hf_token` | Hugging Face token for private repositories.
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

`--bbox` selects patches intersecting a region of the volume, given in **global voxel
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

The bounding box filters the full-volume patch grid; it does not move the grid origin to
the box or trim every written patch to its bounds. For a 256-cube, 128-cube patches and
the default half-patch stride, each axis starts at 0, 64 and 128. The box `32:160` on all
three axes overlaps 27 patches. A one-patch limit selects the middle patch, starting at
`(64, 64, 64)` and reading `64:192` on each axis. Inspect the written coordinate store
when interpreting prediction and blended output positions.


Selecting a smaller region can reduce the number of patches and input reads. Reads cover
the selected full patches and their storage chunks, which can extend beyond the requested
box. Actual transfer and compute costs depend on patch size, chunking, grid coverage and
the input volume.

Notes:
- `--num_parts`/`--part_id` partition the Z span occupied by the selected patch starts.
  Each patch is assigned by its starting Z coordinate; this span can begin outside the box.
- An ROI smaller than the model patch size along an axis is grown to the patch size and
  shifted back inside the volume, so the model always sees a full patch.
- Bounds are clamped to the volume; a bbox entirely outside it is an error.

## Stage 2 — `vesuvius.blend_logits`

Combine the partial logits by weighting overlaps with a Gaussian window. The command scans the `parent_dir` for matching `logits_part_*.zarr` and `coordinates_part_*.zarr` pairs.

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

Finalize a channel-first logits Zarr array into uint8 probabilities or labels. Processing metadata is stored in array attributes; this command does not build a multiscale pyramid. When requested, it deletes the intermediate logits directory after processing.

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
| `input_path` | Path to the channel-first blended logits Zarr array.
| `output_path` | Destination channel-first uint8 Zarr array.
| `--mode` | `binary` (default) or `multiclass`.
| `--threshold` | Binarize the probability map. In `binary` mode cuts at the probability given by `--threshold_value` (default `0.5`). In `multiclass` mode emits the argmax channel.
| `--threshold_value T` | Override the `--threshold` cutoff with `T` in `(0, 1)`. Requires `--threshold`. Binary mode only — rejected in `multiclass` since argmax ignores the cutoff.
| `--delete_intermediates` | Remove the source logits after a successful run.
| `--chunk_size` | Spatial chunk size for the output store (`Z,Y,X`). Defaults to the logits chunking.
| `--num_workers` | Worker processes for finalization (defaults to half of CPU cores).
| `--quiet` | Suppress verbose logging.

Without `--threshold`, binary mode writes foreground probabilities for each target (sigmoid for a one-channel head, foreground softmax for a two-channel head); multiclass mode writes one probability channel per class plus an argmax channel.

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

After finalization the destination is a Zarr array with inherited and processing attributes. This command creates neither a multiscale hierarchy nor a separate run `metadata.json`. Build any viewing pyramid in a separate postprocessing step, and rechunk if your viewer needs different chunk sizes.
