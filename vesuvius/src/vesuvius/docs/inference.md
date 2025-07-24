# Overview of the Inference Pipeline

**Running on a single machine**
If you want to run the entire pipeline in one go on a single machine (even with multiple gpus), you can use `vesuvius.inference_pipeline`. This handles all three stages (prediction, blending, and finalization) in a single command without having to run each step separately.


The inference pipeline consists of three main stages:

1. __Prediction (`vesuvius.predict`)__: Runs model inference on input data, creating logits (model outputs before softmax/argmax).
2. __Blending (`vesuvius.blend_logits`)__: Combines overlapping predictions using Gaussian weighting.
3. __Finalization (`vesuvius.finalize_outputs`)__: Processes logits into final segmentation outputs (probabilities or class predictions).

*Note:*
While the patch size of the model is potentially not the best chunk size for _viewing_ a zarr , we use it throughout this pipeline for reading/writing them. This is due to the performance penalty that you pay when writing partial chunks. 

Because inference outputs are saved as exactly patch size to speed up writing, we keep this same chunk size throughout. It is a good idea to rechunk the final output before using it for downstream tasks.


## 1. `vesuvius.predict` - Model Inference

This component runs the model on input data, creating raw logits.

### Purpose

- Takes input volumetric data and applies a trained nnUNetv2 model to produce prediction logits
- Supports data partitioning for distributed processing
- Handles Test-Time Augmentation (TTA) y

### Distributed Capability: ✅ Fully Distributed

`vesuvius.predict` is designed to run in a distributed manner by dividing the input volume into parts that can be processed independently. Each part can run on a separate machine with no coordination required between parts during inference.

### Key Parameters

| Parameter | Type | Description | Default | Recommendation |
|-----------|------|-------------|---------|---------------|
| `--model_path` | str | Path to the trained nnUNet model folder or HF model (can start with "hf://") | Required | |
| `--input_dir` | str | Path to the input volume (Zarr store) | Required | |
| `--output_dir` | str | Directory to store output predictions | Required | |
| `--input_format` | str | Format of input data (`zarr`, `volume`) | `zarr` | Empty, Defaults |
| `--tta_type` | str | Test-time augmentation type (`mirroring`, `rotation`) | `rotation` | Empty, Defaults |
| `--disable_tta` | flag | Disable test-time augmentation | `False` | Empty, Defaults |
| `--num_parts` | int | Number of parts to split processing into | `1` | |
| `--part_id` | int | Part ID to process (0-indexed) | `0` | |
| `--overlap` | float | Overlap between patches (0-1) | `0.5` | Empty, Defaults |
| `--batch_size` | int | Batch size for inference | `1` | |
| `--patch_size` | str | Override patch size (comma-separated, e.g. "192,192,192") | Model default | Empty, Defaults |
| `--save_softmax` | flag | Save softmax outputs | `False` | Empty, Defaults |
| `--normalization` | str | Normalization scheme (`instance_zscore`, `global_zscore`, `instance_minmax`, `none`) | `instance_zscore` | Same as trained model |
| `--device` | str | Device to use (`cuda`, `cpu`) | `cuda` | Empty, Defaults |
| `--verbose` | flag | Enable verbose output | `False` | Empty, Defaults |
| `--skip-empty-patches` | flag | Skip patches that are empty (all values the same) | `True` | Empty, Defaults |
| `--zarr-compressor` | str | Zarr compression algorithm (`zstd`, `lz4`, `zlib`, `none`) | `zstd` | Empty, Defaults |
| `--zarr-compression-level` | int | Compression level (1-9) | `3` | Empty, Defaults |
| `--scroll_id` | str | Scroll ID to use (if input_format is volume) | `None` | |
| `--segment_id` | str | Segment ID to use (if input_format is volume) | `None` | |
| `--energy` | int | Energy level to use (if input_format is volume) | `None` | |
| `--resolution` | float | Resolution to use (if input_format is volume) | `None` | |
| `--hf_token` | str | Hugging Face token for accessing private repositories | `None` | |


### How to Run Distributed

To run distributed inference across multiple machines:

1. __Split processing into parts__:

   ```bash
   # Machine 1: Process part 0
   vesuvius.predict --model_path /path/to/model --input_dir /path/to/input \
     --output_dir /path/to/output --num_parts 4 --part_id 0

   # Machine 2: Process part 1
   vesuvius.predict --model_path /path/to/model --input_dir /path/to/input \
     --output_dir /path/to/output --num_parts 4 --part_id 1

   # Machine 3: Process part 2
   vesuvius.predict --model_path /path/to/model --input_dir /path/to/input \
     --output_dir /path/to/output --num_parts 4 --part_id 2

   # Machine 4: Process part 3
   vesuvius.predict --model_path /path/to/model --input_dir /path/to/input \
     --output_dir /path/to/output --num_parts 4 --part_id 3
   ```

2. __Output for each part__:

   - Each machine produces two files:

     - `logits_part_X.zarr`: Contains prediction logits
     - `coordinates_part_X.zarr`: Contains coordinates of each patch

### Performance Tips

- Use `--skip-empty-patches` to accelerate processing by skipping empty (completely homogonous) patches. note that these are still fed to inference.py from vc_dataset.py
    but are only "written" in the sense that the index : location mapping is preserved in the coordinate array. we default to using `write_empty_chunks=False` , which 
    will prevent this from truly being written in the zarr. 
- Adjust `--batch_size` based on available GPU memory
- Use `--compression-level` to balance storage space and processing speed
    - compression is only used due to the giant size of the intermediate arrays. if you have enough tmp storage, feel free to omit for tmp arrays. 
    compression of the final array is still a good idea, as its mostly blank. 
- Set an appropriate `--overlap` value (0.5 is a good default)

## 2. `vesuvius.blend_logits` - Merging Predictions

This component combines prediction outputs from multiple parts into a single coherent volume.

### Purpose

- Merges partial inference results from different parts
- Applies Gaussian weighting to handle overlapping regions
- Creates a single, unified prediction volume

### Distributed Capability: ❌ Single Machine

`vesuvius.blend_logits` must run on a single machine as it needs to coordinate the merging of all part files. It processes the data in chunks to manage memory usage.

By default use the chunk size of the input volume (from logits) which is the patch size of the model. 

### Key Parameters

| Parameter | Type | Description | Default | Recommendation |
|-----------|------|-------------|---------|---------------|
| `parent_dir` | str | Directory containing the partial inference results (logits_part_X.zarr, coordinates_part_X.zarr) | Required | |
| `output_path` | str | Path for the final merged Zarr output file | Required | |
| `--weights_path` | str | Path for the temporary weight accumulator Zarr | `<output_path>_weights.zarr` | Empty, Defaults |
| `--sigma_scale` | float | Sigma scale for Gaussian blending (patch_size / sigma_scale) | `8.0` | Empty, Defaults |
| `--chunk_size` | str | Spatial chunk size (Z,Y,X) for output Zarr | Based on patch size | Empty, Defaults |
| `--num_workers` | int | Number of worker processes | CPU_COUNT / 2 | Empty, Defaults |
| `--compression_level` | int | Compression level (0-9, 0=none) | `1` | Empty, Defaults |
| `--keep_weights` | flag | Do not delete the weight accumulator Zarr after merging | `False` | Empty, Defaults |
| `--quiet` | flag | Disable verbose progress messages | `False` | Empty, Defaults |


### How to Run

After all predict jobs have completed:

```bash
vesuvius.blend_logits /path/to/prediction_parts /path/to/output/merged_logits.zarr \
            --num_workers 16 
```

### Memory Considerations

- The blending process is memory-intensive but handles large volumes by:

  - Processing in chunkwise to limit memory usage
  - Using a shared weight accumulator for normalization

## 3. `vesuvius.finalize_outputs` - Post-processing Outputs

This component processes merged logits into final segmentation outputs (probabilities or class predictions).

### Purpose

- Apply softmax/argmax to produce final segmentation outputs
- Convert to common uint8 format for visualization and further analysis
- Optionally apply argmax to create binary or multiclass masks

### Distributed Capability: ❌ Single Machine

`vesuvius.finalize_outputs` runs on a single machine. It processes data in chunks to manage memory usage.

### Key Parameters

| Parameter | Type | Description | Default | Recommendation |
|-----------|------|-------------|---------|---------------|
| `input_path` | str | Path to the merged logits Zarr store | Required | |
| `output_path` | str | Path for the finalized output Zarr store | Required | |
| `--mode` | str | Processing mode (`binary`, `multiclass`) | `binary` | Empty, Defaults |
| `--threshold` | flag | Apply argmax and only save class predictions | `False` | |
| `--delete-intermediates` | flag | Delete intermediate logits after processing | `False` | Empty, Defaults |
| `--chunk-size` | str | Spatial chunk size (Z,Y,X) for output Zarr | Use input chunks | Empty, Defaults |
| `--num-workers` | int | Number of worker processes | CPU_COUNT / 2 | Empty, Defaults |
| `--quiet` | flag | Suppress verbose output | `False` | Empty, Defaults |


### How to Run

After blending is complete:

```bash
vesuvius.finalize_outputs /path/to/merged_logits.zarr /path/to/final_output.zarr \
  --mode binary --threshold --delete-intermediates
```

### Output Format

- __Without `--threshold`__:

  - __Binary mode__: 1 channel [softmax_fg]
  - __Multiclass mode__: N+1 channels [softmax_c0...softmax_cN, argmax]

- __With `--threshold`__:

  - __Binary mode__: 1 channel [binary_mask]
  - __Multiclass mode__: 1 channel [argmax]

## Full Distributed Workflow Example

The following example shows how to run the entire workflow in a distributed setting with 4 machines:

### 1. Run Prediction on Multiple Machines

```bash
# Machine 1: Process part 0
vesuvius.predict --model_path hf://scrollprize/surface_recto \
    --input_dir s3://input-bucket/volume.zarr \
    --output_dir s3://output-bucket/logits.zarr \
    --num_parts 4 \
    --part_id 0 \
    --zarr-compressor zstd \
    --zarr-compression-level 3 \
    --skip-empty-patches \
    --batch_size 2

# Machine 2: Process part 1
vesuvius.predict --model_path /path/to/model \
    --input_dir s3://input-bucket/volume.zarr \
    --output_dir s3://output-bucket/logits.zarr \
    --num_parts 4 \
    --part_id 1 \
    --zarr-compressor zstd \
    --zarr-compression-level 3 \
    --skip-empty-patches \
    --batch_size 2

# Machine 3: Process part 2
vesuvius.predict --model_path /path/to/model \
    --input_dir s3://input-bucket/volume.zarr \
    --output_dir s3://output-bucket/predictions \
    --num_parts 4 \
    --part_id 2 \
    --zarr-compressor zstd \
    --zarr-compression-level 3 \
    --skip-empty-patches \
    --batch_size 2


# Repeat for as many machines as you'd like to schedule this on
# num_parts should be divisible by  z height /  patch size * overlap%
```

### 2. Blend Logits on a Single Machine

```bash
vesuvius.blend_logits s3://output-bucket/predictions \
            s3://output-bucket/merged_logits.zarr \
            --num_workers 16 
```

### 3. Finalize Output on a Single Machine

```bash
# After blending has completed
vesuvius.finalize_outputs s3://output-bucket/merged_logits.zarr \
            s3://output-bucket/final_segmentation.zarr \
            --delete-intermediates \
            --num-workers 16

# if you do not want the softmax probabilities, you can use the flag --threshold to take the argmax over the channels and extract only the fg channel
```
