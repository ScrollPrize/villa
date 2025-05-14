# Vesuvius Inference Tools

This directory contains scripts for running inference on volumes using the Vesuvius toolkit. These scripts can be used both from the command line and as Python modules.

## Components

The inference pipeline consists of four main components:

1. `inference.py` - Performs model inference on input data in multiple parts
2. `blending.py` - Merges model prediction parts with Gaussian weighting
3. `finalize_outputs.py` - Processes blended logits into final outputs
4. `vesuvius_pipeline.py` - Orchestrates all steps with multi-GPU support

## Command Line Usage

### 1. Running Inference (vesuvius.predict)

This script runs model inference on an input volume, supporting partitioning for multi-GPU execution.

```bash
vesuvius.predict \
  --model_path /path/to/model \
  --input_dir /path/to/input/volume.zarr \
  --output_dir /path/to/output/directory \
  --part_id 0 \
  --num_parts 1 \
  --overlap 0.5 \
  --batch_size 2 \
  --device cuda:0 \
  --skip_empty_patches
```

Key options:
- `--model_path`: Path to nnUNet model folder (also supports "hf://org/model" format)
- `--input_dir`: Path to input Zarr volume
- `--output_dir`: Where to store prediction results
- `--num_parts`: Number of parts to split data into (for multi-GPU)
- `--part_id`: Which part to process (0-indexed)
- `--overlap`: Overlap factor between patches (0-1)
- `--skip_empty_patches`: Skip processing homogeneous patches (faster)
- `--disable_tta`: Disable test time augmentation (faster but less accurate)

### 2. Blending Outputs (vesuvius.blend_logits)

This script merges inference results from multiple parts with Gaussian weighting.

```bash
vesuvius.blend_logits \
  /path/to/parts/directory \
  /path/to/output/blended.zarr \
  --cache_gb 8 \
  --sigma_scale 8.0
```

Key options:
- First argument: Directory containing part outputs
- Second argument: Path for blended output Zarr
- `--sigma_scale`: Controls the Gaussian weighting falloff
- `--cache_gb`: Memory cache size in gigabytes
- `--chunk_size`: Output chunk size (comma-separated Z,Y,X)

### 3. Finalizing Outputs (vesuvius.finalize_outputs)

This script processes blended logits into final prediction maps.

```bash
vesuvius.finalize_outputs \
  /path/to/blended.zarr \
  /path/to/final_output.zarr \
  --mode binary \
  --threshold \
  --cache-gb 8
```

Key options:
- First argument: Path to blended logits Zarr
- Second argument: Path for final output Zarr
- `--mode`: "binary" (2-class) or "multiclass" (>2 classes)
- `--threshold`: Output thresholded class masks instead of probabilities
- `--delete-intermediates`: Remove intermediate blended logits

### 4. Complete Pipeline (vesuvius.inference_pipeline)

This script runs all steps in sequence with multi-GPU support.

```bash
vesuvius.inference_pipeline \
  --input /path/to/input/volume.zarr \
  --output /path/to/final_output.zarr \
  --model /path/to/model \
  --gpus 0,1 \
  --mode binary \
  --threshold \
  --batch-size 2
```

Key options:
- `--input`: Path to input volume
- `--output`: Path for final output
- `--model`: Path to model directory
- `--gpus`: GPU IDs to use (comma-separated or "all")
- `--parts-per-gpu`: Number of parts per GPU
- `--workdir`: Directory for intermediate files
- Control flags: `--skip-predict`, `--skip-blend`, `--skip-finalize`

## Python Module Usage

### 1. Using Inference

```python
from models.run.inference import Inferer

inferer = Inferer(
    model_path="/path/to/model", 
    input_dir="/path/to/input.zarr",
    output_dir="/path/to/output",
    device="cuda:0",
    skip_empty_patches=True,
    batch_size=2,
    do_tta=True
)

# Run inference
logits_path, coords_path = inferer.infer()
```

### 2. Using Blending

```python
import asyncio
from models.run.blending import merge_inference_outputs

async def blend_my_results():
    await merge_inference_outputs(
        parent_dir="/path/to/parts/directory",
        output_path="/path/to/blended.zarr",
        cache_pool_gb=8.0,
        sigma_scale=8.0
    )

# Run the async function
asyncio.run(blend_my_results())
```

### 3. Using Finalization

```python
import asyncio
from models.run.finalize_outputs import finalize_logits

async def finalize_my_results():
    await finalize_logits(
        input_path="/path/to/blended.zarr",
        output_path="/path/to/final_output.zarr",
        mode="binary",
        threshold=True,
        cache_pool_gb=8.0
    )

# Run the async function
asyncio.run(finalize_my_results())
```

### 4. Using Complete Pipeline

```python
from models.run.vesuvius_pipeline import run_pipeline
import sys

# Arguments would normally come from command line
sys.argv = [
    "vesuvius.inference_pipeline",
    "--input", "/path/to/input.zarr",
    "--output", "/path/to/output.zarr",
    "--model", "/path/to/model",
    "--gpus", "0,1",
    "--threshold"
]

# Run the pipeline
exit_code = run_pipeline()
```

## Pipeline Process Flow

1. **Inference**: The volume is divided into overlapping patches. The model runs inference on each patch to generate logits.

2. **Blending**: Overlapping logits are merged using Gaussian weights (higher in patch centers, lower at edges).

3. **Finalization**: Blended logits are converted to:
   - For binary mode: Foreground probability maps or binary masks
   - For multiclass mode: Per-class probability maps or class labels

## Performance Tips

- Use `--skip_empty_patches` for faster inference by skipping homogeneous regions
- Adjust batch size based on your GPU memory
- Choose parts_per_gpu to optimize memory usage
- Use multiple GPUs with the vesuvius.inference_pipeline command for parallelization
- For binary segmentation, use `--threshold` to get masks directly