# Inference Pipeline Overview

The Vesuvius tooling exposes three command-line stages plus a convenience orchestrator:

1. `vesuvius.predict` — run a trained model and write patch logits.
2. `vesuvius.blend_logits` — merge overlapping patches with Gaussian weighting.
3. `vesuvius.finalize_outputs` — convert logits into probabilities or masks and build a multiscale Zarr.
4. `vesuvius.inference_pipeline` — execute all steps sequentially on a single machine (optionally using multiple GPUs).

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
| `--input_dir` (required) | Volume input: Zarr root, TIFF stack, or a directory understood by the `Volume` helper.
| `--output_dir` (required) | Destination folder for logits (`logits_part_{id}.zarr`) and coordinates.
| `--input_format` | Force `zarr`, `tiff`, or `volume` detection. Usually optional.
| `--tta_type` / `--disable_tta` | Choose `rotation` (default) or `mirroring`, or disable test-time augmentation.
| `--num_parts` / `--part_id` | Partition inference so multiple machines can process different chunks.
| `--overlap` | Fractional patch overlap (0–1, default `0.5`).
| `--batch_size` | Inference batch size (default `1`).
| `--patch_size` | Override the model patch size using a comma-separated list (e.g. `192,192,192`).
| `--tif-activation` | When writing TIFF outputs, pick `softmax`, `argmax`, or `none`.
| `--save_softmax` | Legacy flag for saving softmax logits (consider `--tif-activation`).
| `--normalization` | Runtime normalization (`instance_zscore`, `global_zscore`, `instance_minmax`, `ct`, `none`).
| `--intensity-properties-json` | nnU-Net style JSON with intensity stats for CT normalization.
| `--device` | Device string such as `cuda`, `cuda:1`, or `cpu`.
| `--skip-empty-patches` / `--no-skip-empty-patches` | Toggle automatic removal of homogeneous patches.
| `--zarr-compressor` / `--zarr-compression-level` | Configure output compression (`zstd` with level `3` by default).
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
| `--weights_path` | Optional path for the temporary weight accumulator.
| `--sigma_scale` | Controls Gaussian falloff (`patch_size / sigma_scale`, default `8.0`).
| `--chunk_size` | Spatial chunk size (`Z,Y,X`) for the merged Zarr. Leave unset to auto-pick.
| `--num_workers` | Number of worker processes. Defaults to `CPU_COUNT - 1`.
| `--compression_level` | Zarr compression level (0–9, default `1`).
| `--keep_weights` | Preserve the weight accumulator instead of deleting it after blending.
| `--quiet` | Suppress verbose logging.

The merged logits retain the same class/channel dimension as the individual parts.

## Stage 3 — `vesuvius.finalize_outputs`

Finalize logits into probabilities or masks and optionally build a multiscale pyramid. The command writes OME-NGFF metadata and (when requested) deletes the intermediate logits directory.

```bash
vesuvius.finalize_outputs /tmp/merged_logits.zarr /tmp/final_output.zarr \
  --mode binary \
  --threshold \
  --delete-intermediates
```

### Options

| Argument | Description |
|----------|-------------|
| `input_path` | Path to the blended logits Zarr (level `0` is the logits array).
| `output_path` | Destination multiscale Zarr root.
| `--mode` | `binary` (default) or `multiclass`.
| `--threshold` | For binary: emit a single-channel argmax mask. For multiclass: emit just the argmax channel.
| `--delete-intermediates` | Remove the source logits after a successful run.
| `--chunk-size` | Spatial chunk size for the output store (`Z,Y,X`). Defaults to the logits chunking.
| `--num-workers` | Worker processes for finalization (defaults to half of CPU cores).
| `--quiet` | Suppress verbose logging.

Without `--threshold`, binary mode outputs a single softmax foreground channel; multiclass mode writes one channel per class plus an argmax channel.

## Single-Machine Convenience — `vesuvius.inference_pipeline`

`vesuvius.inference_pipeline` automates the three stages on one machine. It can route different parts to different GPUs and manage intermediate directories.

```bash
vesuvius.inference_pipeline \
  --input /data/Scroll1.zarr \
  --output /results/ink.zarr \
  --model hf://scrollprize/surface_recto \
  --mode binary \
  --threshold \
  --gpus 0,1 \
  --parts-per-gpu 2 \
  --batch-size 2
```

Important flags:

- `--workdir`: where to place intermediate logits (defaults to `<output>_work`).
- `--skip-predict`, `--skip-blend`, `--skip-finalize`: rerun only specific stages.
- `--parts-per-gpu`: how many logical parts each GPU should process.
- `--keep-intermediates`: retain the intermediate logits/chunks for debugging.

The pipeline command internally invokes the individual CLIs, so pass model- and inference-specific flags exactly as you would to `vesuvius.predict`.

## Full Remote Workflow Example

```bash
# 1. Run prediction on four machines (part IDs 0–3)
vesuvius.predict --model_path hf://scrollprize/surface_recto \
    --input_dir s3://vesuvius/input/Scroll1.zarr \
    --output_dir s3://vesuvius/tmp/logits \
    --num_parts 4 \
    --part_id 0 \
    --device cuda:0 \
    --zarr-compressor zstd \
    --zarr-compression-level 3 \
    --skip-empty-patches

# ...repeat for part_id 1,2,3 on other hosts...

# 2. Blend logits once all parts finish
vesuvius.blend_logits s3://vesuvius/tmp/logits s3://vesuvius/tmp/merged_logits.zarr \
    --num_workers 32 \
    --chunk_size 256,256,256

# 3. Finalize outputs
vesuvius.finalize_outputs s3://vesuvius/tmp/merged_logits.zarr s3://vesuvius/output/final.zarr \
    --mode binary \
    --threshold \
    --delete-intermediates
```

After finalization the destination Zarr contains a multiscale hierarchy (`0/`, `1/`, …) and a `metadata.json` file describing the inference run. Rechunk the output if you plan to serve it through a viewer that expects different chunk sizes.
