#!/usr/bin/env python3
"""
Coherence Enhancing Diffusion Filter - Batched and Multi-GPU Version
Based on J. Weickert, "Coherence-Enhancing Diffusion Filtering", 
International Journal of Computer Vision, 1999, vol.31, p.111-127.

PyTorch implementation that matches the Java/Scala implementation that is available in the ImageJ plugin
"CoherenceEnhancingDiffusionFilter.java" and "CoherenceEnhancingDiffusionFilter.scala".
This script implements the coherence-enhancing diffusion filter using PyTorch.
"""

import argparse
import multiprocessing as mp
import os
from pathlib import Path
from typing import Dict, List

import numpy as np
import tifffile
import torch
import zarr
from tqdm import tqdm

from vesuvius.image_proc.geometry.diffusion import coherence_enhancing_diffusion as run_coherence_diffusion
try:
    from skimage.filters import threshold_otsu
    _HAVE_SKIMAGE = True
except Exception:
    _HAVE_SKIMAGE = False


"""
CONFIGURATION PARAMETERS GUIDE

The coherence-enhancing diffusion filter uses several key parameters that control its behavior:

LAMBDA (λ) - Edge threshold parameter (default: 1.0, typical range: 0.1-5.0)
    Purpose: Controls the sensitivity to edge strength. Determines the threshold for 
             distinguishing between coherent structures and noise.
    Effects:
    - Increase (e.g., 2.0-5.0): More aggressive smoothing, only very strong edges preserved.
      Results in fewer detected structures, more homogeneous regions.
    - Decrease (e.g., 0.1-0.5): More sensitive to weak edges and fine structures.
      Preserves more detail but may also preserve noise.
    Mathematical role: Normalizes the structure tensor's eigenvalue difference (alpha)
                      in the diffusivity function c2 = γ + (1-γ)exp(-CM/(alpha/λ)^m)

SIGMA (σ) - Gaussian smoothing for gradient computation (default: 3.0, typical range: 1.0-10.0)
    Purpose: Pre-smooths the image before computing gradients to reduce noise influence.
             Essential for stable gradient computation in noisy images.
    Effects:
    - Increase (e.g., 5.0-10.0): Stronger pre-smoothing, more robust to noise.
      Gradients computed from larger-scale structures, fine details lost.
    - Decrease (e.g., 0.5-2.0): Less pre-smoothing, preserves fine-scale gradients.
      More susceptible to noise, may cause unstable diffusion.
    Note: Should generally be smaller than RHO for proper scale separation.

RHO (ρ) - Gaussian smoothing for structure tensor (default: 5.0, typical range: 3.0-20.0)
    Purpose: Determines the integration scale for local orientation analysis.
             Controls the neighborhood size for computing coherent flow direction.
    Effects:
    - Increase (e.g., 10.0-20.0): Larger integration scale, detects more global structures.
      Better for images with large-scale coherent patterns.
    - Decrease (e.g., 2.0-4.0): Smaller integration scale, more local orientation analysis.
      Better for images with fine-scale or rapidly changing structures.
    Relationship: Should be larger than SIGMA (typically ρ ≈ 1.5-3 × σ) to ensure
                 proper multi-scale analysis.

STEP_SIZE - Time step for diffusion iteration (default: 0.24, typical range: 0.05-0.25)
    Purpose: Controls the speed of diffusion evolution. Larger steps mean faster diffusion
             but risk numerical instability.
    Effects:
    - Increase (e.g., 0.3-0.5): Faster diffusion, fewer iterations needed.
      Risk of overshooting and numerical instability.
    - Decrease (e.g., 0.05-0.15): Slower, more stable diffusion.
      Requires more iterations but gives finer control.
    Stability: Must satisfy step_size ≤ 0.25 for explicit scheme stability.

M - Exponent for diffusivity function (default: 1.0, typical range: 0.5-4.0)
    Purpose: Controls the sharpness of the transition between isotropic and anisotropic
             diffusion regions. Affects how diffusivity changes with edge strength.
    Effects:
    - Increase (e.g., 2.0-4.0): Sharper transition, more binary-like behavior.
      Stronger distinction between edges and homogeneous regions.
    - Decrease (e.g., 0.5-0.8): Smoother transition, more gradual diffusivity changes.
      More subtle enhancement of structures.
    Special case: M = 1.0 gives exponential diffusivity function (most common choice).

NUM_STEPS - Number of diffusion iterations (default: 25, typical range: 10-100)
    Purpose: Total number of diffusion steps to perform. More steps mean more smoothing
             and stronger enhancement of coherent structures.
    Effects:
    - Increase (e.g., 50-100): Stronger smoothing and enhancement effect.
      Risk of over-smoothing and loss of important details.
    - Decrease (e.g., 5-15): Milder effect, preserves more original detail.
      May not fully enhance coherent structures.
    Relationship: Total diffusion time = NUM_STEPS × STEP_SIZE

ALGORITHM CONSTANTS (typically not modified):

EPS - Machine epsilon for numerical stability (default: 2^-52)
    Purpose: Prevents division by zero in eigenvalue and diffusivity calculations.
    
GAMMA (γ) - Minimum diffusivity (default: 0.01, typical range: 0.001-0.1)
    Purpose: Ensures minimum diffusion even in strong edge regions to maintain
             numerical stability and prevent complete diffusion blocking.
    Effects:
    - Increase: More diffusion across edges, less sharp boundaries.
    - Decrease: Sharper edge preservation, risk of creating artifacts.

CM - Exponential constant (default: 7.2848)
    Purpose: Controls the steepness of the exponential diffusivity function.
             Derived from Weickert's formulation for optimal coherence enhancement.
    Note: This value is specifically chosen to achieve c2(λ) ≈ 0.01 for proper
          diffusivity scaling and should rarely be modified.
"""

# Configuration parameters (constants)
LAMBDA = 1.0          # Edge threshold parameter
SIGMA = 3.0           # Gaussian smoothing for gradient computation
RHO = 5.0             # Gaussian smoothing for structure tensor
STEP_SIZE = 0.24      # Time step size for diffusion
M = 1.0               # Exponent for diffusivity function
NUM_STEPS = 100        # Number of diffusion iterations

def process_zarr_chunk(
    input_path: str,
    output_path: str,
    z_start: int,
    z_end: int,
    gpu_id: int,
    config: Dict,
    batch_size: int = 1,
) -> None:
    """Process a chunk of z-slices on a specific GPU."""
    # Set GPU device - explicitly specify which GPU to use
    if torch.cuda.is_available() and torch.cuda.device_count() > gpu_id:
        device = torch.device(f'cuda:{gpu_id}')
        torch.cuda.set_device(device)
    else:
        device = torch.device('cpu')
    
    print(f"GPU {gpu_id}: Using device: {device}")
    print(f"GPU {gpu_id}: Processing slices {z_start} to {z_end-1}")
    
    # Print additional debug info
    if device.type == 'cuda':
        print(f"GPU {gpu_id}: CUDA device name: {torch.cuda.get_device_name(gpu_id)}")
    
    # Open zarr arrays
    zarr_in = zarr.open(input_path, mode='r')
    zarr_out = zarr.open(output_path, mode='r+')  # r+ for read/write
    
    shape = zarr_in.shape
    dtype = zarr_in.dtype
    
    # Process slices in batches
    total_slices = z_end - z_start
    num_batches = (total_slices + batch_size - 1) // batch_size
    
    pbar = tqdm(total=total_slices, desc=f"GPU {gpu_id}", position=gpu_id)
    
    for batch_idx in range(num_batches):
        # Calculate batch boundaries
        batch_start = z_start + batch_idx * batch_size
        batch_end = min(batch_start + batch_size, z_end)
        actual_batch_size = batch_end - batch_start
        
        # Load batch of slices
        batch_data = []
        batch_indices = []
        
        for z in range(batch_start, batch_end):
            slice_data = zarr_in[z, :, :]
            
            # Check if slice is all zeros
            if np.all(slice_data == 0):
                # Write zeros directly to output
                zarr_out[z, :, :] = np.zeros_like(slice_data, dtype=dtype)
                pbar.update(1)
                continue
            
            batch_data.append(slice_data)
            batch_indices.append(z)
        
        # Skip if no non-zero slices in this batch
        if len(batch_data) == 0:
            continue
        
        # Convert batch to tensor
        batch_array = np.stack(batch_data, axis=0).astype(np.float32)
        batch_tensor = torch.from_numpy(batch_array).unsqueeze(1).to(device)  # Add channel dimension
        
        # Perform diffusion on batch
        with torch.no_grad():
            result_batch = run_coherence_diffusion(
                batch_tensor,
                config,
                show_progress=False,
            )
        
        # Convert back and write results
        result_array = result_batch.squeeze(1).cpu().numpy()  # Remove channel dimension
        
        for i, z in enumerate(batch_indices):
            # Clip values based on dtype
            if dtype == np.uint8:
                result_slice = np.clip(result_array[i], 0, 255).astype(np.uint8)
                # Set values below 1 to 0 for uint8
                result_slice[result_slice < 1] = 0
            elif dtype == np.uint16:
                result_slice = np.clip(result_array[i], 0, 65535).astype(np.uint16)
            else:
                result_slice = result_array[i].astype(dtype)
            
            # Write to output zarr
            zarr_out[z, :, :] = result_slice
            pbar.update(1)
        
        # Free GPU memory
        del batch_tensor, result_batch
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    pbar.close()
    print(f"GPU {gpu_id}: Completed processing slices {z_start} to {z_end-1}")


def launch_multi_gpu_processing(
    input_path: str,
    output_path: str,
    config: Dict,
    num_gpus: int,
    batch_size: int = 1,
) -> None:
    """Launch multiple processes for multi-GPU processing."""
    # Use spawn context for CUDA compatibility
    ctx = mp.get_context('spawn')
    
    # Open input zarr to get shape
    zarr_in = zarr.open(input_path, mode='r')
    shape = zarr_in.shape
    dtype = zarr_in.dtype
    num_slices = shape[0]
    
    print(f"Total slices to process: {num_slices}")
    print(f"Using {num_gpus} GPUs")
    
    # Create output zarr with chunks=(1, height, width)
    print(f"Creating output zarr array: {output_path}")
    zarr_out = zarr.open(
        output_path, 
        mode='w', 
        shape=shape, 
        chunks=(1, shape[1], shape[2]),  # One chunk per z-slice
        dtype=dtype,
        compressor=zarr_in.compressor if hasattr(zarr_in, 'compressor') else None,
        write_empty_chunks=False
    )
    
    # Calculate chunk boundaries for each GPU
    slices_per_gpu = num_slices // num_gpus
    remainder = num_slices % num_gpus
    
    chunk_boundaries = []
    start = 0
    for i in range(num_gpus):
        # Distribute remainder slices across first GPUs
        chunk_size = slices_per_gpu + (1 if i < remainder else 0)
        end = start + chunk_size
        chunk_boundaries.append((start, end))
        start = end
    
    print("GPU assignments:")
    for i, (start, end) in enumerate(chunk_boundaries):
        print(f"  GPU {i}: slices {start}-{end-1} ({end-start} slices)")
    
    # Launch processes using spawn context
    processes = []
    for gpu_id, (z_start, z_end) in enumerate(chunk_boundaries):
        p = ctx.Process(
            target=process_zarr_chunk,
            args=(input_path, output_path, z_start, z_end, gpu_id, config, batch_size),
        )
        p.start()
        processes.append(p)
    
    # Wait for all processes to complete
    for p in processes:
        p.join()
    
    print("\nAll GPU processes completed!")


def process_zarr_array(input_path, output_path, config, batch_size=1):
    """Process a zarr array slice by slice with coherence enhancing diffusion (single GPU)."""
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Open input zarr array
    print(f"Opening zarr array: {input_path}")
    zarr_in = zarr.open(input_path, mode='r')
    
    # Get array shape and dtype
    shape = zarr_in.shape
    dtype = zarr_in.dtype
    print(f"Zarr shape: {shape}, dtype: {dtype}")
    
    # Handle different dimensionality
    if len(shape) == 2:
        # 2D array - process as single image
        num_slices = 1
        z_axis = None
    elif len(shape) == 3:
        # 3D array - process slice by slice
        num_slices = shape[0]
        z_axis = 0
    else:
        raise ValueError(f"Unsupported array dimensionality: {len(shape)}. Expected 2D or 3D array.")
    
    # Create output zarr array with chunks=(1, height, width) for 3D
    print(f"Creating output zarr array: {output_path}")
    if len(shape) == 3:
        chunks = (1, shape[1], shape[2])
    else:
        chunks = zarr_in.chunks if hasattr(zarr_in, 'chunks') else None
    
    zarr_out = zarr.open(
        output_path, 
        mode='w', 
        shape=shape, 
        chunks=chunks,
        dtype=dtype,
        compressor=zarr_in.compressor if hasattr(zarr_in, 'compressor') else None,
        write_empty_chunks=False
    )
    
    # Process each slice in batches
    print(f"Processing {num_slices} slices with batch size {batch_size}...")
    skipped_slices = 0
    processed_slices = 0
    
    num_batches = (num_slices + batch_size - 1) // batch_size
    
    for batch_idx in tqdm(range(num_batches), desc="Processing batches"):
        # Calculate batch boundaries
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, num_slices)
        actual_batch_size = batch_end - batch_start
        
        # Load batch of slices
        batch_data = []
        batch_indices = []
        
        for z in range(batch_start, batch_end):
            # Read slice
            if z_axis is None:
                slice_data = zarr_in[:]
            else:
                slice_data = zarr_in[z, :, :]
            
            # Check if slice is all zeros
            if np.all(slice_data == 0):
                # Skip processing - just write zeros to output
                if z_axis is None:
                    zarr_out[:] = np.zeros_like(slice_data, dtype=dtype)
                else:
                    zarr_out[z, :, :] = np.zeros_like(slice_data, dtype=dtype)
                skipped_slices += 1
                continue
            
            batch_data.append(slice_data)
            batch_indices.append(z)
        
        # Skip if no non-zero slices in this batch
        if len(batch_data) == 0:
            continue
        
        # Convert batch to tensor
        batch_array = np.stack(batch_data, axis=0).astype(np.float32)
        batch_tensor = torch.from_numpy(batch_array).unsqueeze(1).to(device)  # Add channel dimension
        
        # Perform diffusion on batch
        with torch.no_grad():
            result_batch = run_coherence_diffusion(
                batch_tensor,
                config,
                show_progress=False,
            )
        
        # Convert back and write results
        result_array = result_batch.squeeze(1).cpu().numpy()  # Remove channel dimension
        
        for i, z in enumerate(batch_indices):
            # Clip values based on dtype
            if dtype == np.uint8:
                result_slice = np.clip(result_array[i], 0, 255).astype(np.uint8)
                # Set values below 1 to 0 for uint8
                result_slice[result_slice < 1] = 0
            elif dtype == np.uint16:
                result_slice = np.clip(result_array[i], 0, 65535).astype(np.uint16)
            else:
                result_slice = result_array[i].astype(dtype)
            
            # Write to output zarr
            if z_axis is None:
                zarr_out[:] = result_slice
            else:
                zarr_out[z, :, :] = result_slice
            
            processed_slices += 1
        
        # Free GPU memory
        del batch_tensor, result_batch
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    print(f"\nProcessing complete!")
    print(f"Total slices: {num_slices}")
    print(f"Processed slices: {processed_slices}")
    print(f"Skipped slices (all zeros): {skipped_slices}")
    print(f"Output saved to: {output_path}")


def load_tiff(filepath):
    """Load a TIFF image and convert to torch tensor."""
    img = tifffile.imread(filepath)
    img_array = np.asarray(img)
    original_shape = img_array.shape
    original_dtype = img_array.dtype

    # DO NOT normalize - keep original values! Cast to float32 for processing.
    img_array = img_array.astype(np.float32, copy=False)

    tensor = torch.from_numpy(img_array)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if tensor.ndim == 2:
        # Single 2D slice -> (1, 1, H, W)
        tensor = tensor.unsqueeze(0).unsqueeze(0)
    elif tensor.ndim == 3:
        if original_shape[-1] <= 4:
            # Channel-last layout (H, W, C) -> (1, C, H, W)
            tensor = tensor.permute(2, 0, 1).contiguous().unsqueeze(0)
        elif original_shape[0] <= 4:
            # Channel-first layout (C, H, W) -> (1, C, H, W)
            tensor = tensor.unsqueeze(0)
        else:
            # Stack of 2D slices along the first dimension -> (Z, 1, H, W)
            tensor = tensor.unsqueeze(1)
    elif tensor.ndim >= 4:
        # Flatten all leading dimensions into the batch dimension and keep a single channel
        spatial_h, spatial_w = original_shape[-2], original_shape[-1]
        leading = int(np.prod(original_shape[:-2]))
        tensor = tensor.reshape(leading, spatial_h, spatial_w).unsqueeze(1)
    else:
        raise ValueError(f"Unsupported TIFF dimensionality ({tensor.ndim}) for file: {filepath}")

    tensor = tensor.to(device)
    
    return tensor, original_shape, original_dtype


def save_tiff(tensor, filepath, original_shape, original_dtype):
    """Save torch tensor as TIFF image."""
    tensor_cpu = tensor.detach().cpu()

    if tensor_cpu.ndim < 2:
        raise ValueError("Expected at least two dimensions in tensor output.")

    if len(original_shape) == 2:
        # Original data was a single 2D slice
        img_array = tensor_cpu.squeeze().numpy().reshape(original_shape)
    elif len(original_shape) == 3:
        if original_shape[-1] <= 4 and tensor_cpu.shape[0] == 1 and tensor_cpu.shape[1] == original_shape[-1]:
            # Channel-last layout (H, W, C)
            img_array = tensor_cpu.squeeze(0).permute(1, 2, 0).contiguous().numpy()
        elif tensor_cpu.shape[0] == 1 and tensor_cpu.shape[1] == original_shape[0]:
            # Channel-first layout (C, H, W)
            img_array = tensor_cpu.squeeze(0).numpy().reshape(original_shape)
        elif tensor_cpu.shape[1] == 1 and tensor_cpu.shape[0] == original_shape[0]:
            # Stack of slices (Z, H, W)
            img_array = tensor_cpu.squeeze(1).numpy().reshape(original_shape)
        else:
            img_array = tensor_cpu.squeeze().numpy().reshape(original_shape)
    else:
        leading = int(np.prod(original_shape[:-2]))
        if tensor_cpu.shape[0] == leading and tensor_cpu.shape[1] == 1:
            img_array = tensor_cpu.squeeze(1).numpy().reshape(original_shape)
        else:
            img_array = tensor_cpu.squeeze().numpy().reshape(original_shape)

    # Save with original data type; do not rescale.
    if original_dtype == np.uint8:
        img_array = np.clip(img_array, 0, 255).astype(np.uint8)
    elif original_dtype == np.uint16:
        img_array = np.clip(img_array, 0, 65535).astype(np.uint16)
    else:
        img_array = img_array.astype(original_dtype, copy=False)

    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(filepath)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    tifffile.imwrite(filepath, img_array, photometric='minisblack', compression='lzw')


def generate_sample_parameters():
    """Generate a set of sample parameter combinations for demonstration."""
    samples = [
        # Low values - minimal smoothing, preserves fine details
        {'lambda': 0.2, 'sigma': 1.0, 'rho': 2.0, 'desc': 'minimal_smoothing'},
        
        # Low-medium values - gentle smoothing
        {'lambda': 0.5, 'sigma': 2.0, 'rho': 3.0, 'desc': 'gentle_smoothing'},
        
        # Medium-low values - moderate smoothing
        {'lambda': 0.8, 'sigma': 2.5, 'rho': 4.0, 'desc': 'moderate_low_smoothing'},
        
        # Default values - balanced smoothing
        {'lambda': 1.0, 'sigma': 3.0, 'rho': 5.0, 'desc': 'default_balanced'},
        
        # Medium-high values - stronger smoothing
        {'lambda': 1.5, 'sigma': 4.0, 'rho': 7.0, 'desc': 'moderate_high_smoothing'},
        
        # High values - strong smoothing
        {'lambda': 2.0, 'sigma': 5.0, 'rho': 10.0, 'desc': 'strong_smoothing'},
        
        # Very high values - very strong smoothing
        {'lambda': 3.0, 'sigma': 7.0, 'rho': 15.0, 'desc': 'very_strong_smoothing'},
        
        # Maximum values - extreme smoothing
        {'lambda': 5.0, 'sigma': 10.0, 'rho': 20.0, 'desc': 'extreme_smoothing'},
    ]
    
    # Add fixed parameters to each sample
    for sample in samples:
        sample['step_size'] = STEP_SIZE
        sample['m'] = M
        sample['num_steps'] = NUM_STEPS
    
    return samples


def get_available_gpus():
    """Get the number of available GPUs."""
    if torch.cuda.is_available():
        return torch.cuda.device_count()
    return 0

def _is_zarr_dir(path: Path) -> bool:
    """Heuristically determine if a directory is a Zarr store."""
    if path.suffix == '.zarr':
        return True
    if not path.is_dir():
        return False
    # Common Zarr markers
    if (path / ".zarray").exists() or (path / ".zattrs").exists():
        return True
    # Also consider nested group stores
    try:
        for p in path.iterdir():
            if p.is_dir() and ((p / ".zarray").exists() or (p / ".zattrs").exists()):
                return True
    except Exception:
        pass
    return False

def _list_tiff_files(folder: Path) -> List[Path]:
    exts = {".tif", ".tiff", ".TIF", ".TIFF"}
    return sorted([p for p in folder.iterdir() if p.is_file() and p.suffix in exts])

def process_tiff_folder(
    input_dir: str,
    output_dir: str,
    config: Dict,
    *,
    threshold: bool = False,
) -> None:
    """Process a folder of 2D TIFFs, saving outputs with same filenames."""
    in_path = Path(input_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    files = _list_tiff_files(in_path)
    if not files:
        print(f"No TIFF files found in folder: {in_path}")
        return

    print(f"Found {len(files)} TIFF files in {in_path}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    for idx, f in enumerate(files, 1):
        print(f"[{idx}/{len(files)}] Processing {f.name}")
        img_tensor, original_shape, original_dtype = load_tiff(str(f))
        with torch.no_grad():
            result = run_coherence_diffusion(
                img_tensor,
                config,
                show_progress=False,
            )
        out_fp = str(out_path / f.name)
        if threshold:
            if not _HAVE_SKIMAGE:
                raise RuntimeError("--threshold requires scikit-image to be installed.")
            arr = result.squeeze().cpu().numpy()
            thr = threshold_otsu(arr)
            bin8 = (arr >= thr).astype(np.uint8) * 255
            tifffile.imwrite(out_fp, bin8, photometric='minisblack', compression='lzw')
        else:
            save_tiff(result, out_fp, original_shape, original_dtype)
    print(f"\nProcessing complete. Outputs saved to: {out_path}")


def main():
    parser = argparse.ArgumentParser(description='Coherence Enhancing Diffusion Filter - Batched and Multi-GPU Version')
    parser.add_argument('input', help='Input: TIFF file, folder of TIFFs, or Zarr store path')
    parser.add_argument('output', help='Output: TIFF file, folder (for TIFFs), or Zarr store path')
    parser.add_argument('--lambda', type=float, default=LAMBDA, help='Edge threshold parameter')
    parser.add_argument('--sigma', type=float, default=SIGMA, help='Gaussian smoothing for gradients')
    parser.add_argument('--rho', type=float, default=RHO, help='Gaussian smoothing for structure tensor')
    parser.add_argument('--step-size', type=float, default=STEP_SIZE, help='Diffusion step size')
    parser.add_argument('--m', type=float, default=M, help='Exponent for diffusivity function')
    parser.add_argument('--num-steps', type=int, default=NUM_STEPS, help='Number of diffusion steps')
    parser.add_argument('--batch-size', type=int, default=1, help='Number of slices to process at once on GPU')
    parser.add_argument('--num-gpus', type=int, default=None, help='Number of GPUs to use (default: auto-detect)')
    parser.add_argument('--sample-results', action='store_true', help='Generate sample results with various parameter combinations')
    parser.add_argument('--threshold', action='store_true', help='Apply Otsu threshold to TIFF outputs and save as uint8 (0/255)')
    
    args = parser.parse_args()
    
    # Create config dictionary
    config = {
        'lambda': getattr(args, 'lambda'),
        'sigma': args.sigma,
        'rho': args.rho,
        'step_size': args.step_size,
        'm': args.m,
        'num_steps': args.num_steps
    }
    
    # Determine number of GPUs to use
    available_gpus = get_available_gpus()
    if args.num_gpus is None:
        num_gpus = available_gpus
    else:
        num_gpus = min(args.num_gpus, available_gpus)
    
    if available_gpus == 0:
        print("No GPUs available. Processing will use CPU.")
        num_gpus = 0
    else:
        print(f"Available GPUs: {available_gpus}")
        
    # Resolve input/output paths
    input_path = Path(args.input)
    output_path = Path(args.output)

    # Determine input mode
    if input_path.is_dir() and not _is_zarr_dir(input_path):
        # Folder of 2D TIFFs
        if args.sample_results:
            print("Sample results generation is only supported for single TIFF inputs.")
            return

        if output_path.exists() and not output_path.is_dir():
            raise ValueError("When input is a folder of TIFFs, output must be a directory.")
        print("Using folder-of-TIFFs processing")
        process_tiff_folder(
            str(input_path),
            str(output_path),
            config,
            threshold=args.threshold,
        )
    elif input_path.suffix == '.zarr' or _is_zarr_dir(input_path):
        # Process zarr array/store
        if args.sample_results:
            print("Sample results generation is not supported for zarr arrays")
            return
        
        # Decide whether to use multi-GPU or single GPU processing
        if num_gpus > 1:
            print(f"Using multi-GPU processing with {num_gpus} GPUs")
            launch_multi_gpu_processing(
                str(input_path),
                str(output_path),
                config,
                num_gpus=num_gpus,
                batch_size=args.batch_size,
            )
        else:
            print("Using single GPU processing")
            process_zarr_array(
                str(input_path),
                str(output_path),
                config,
                batch_size=args.batch_size,
            )
    else:
        # Process TIFF file
        if num_gpus > 1:
            print("Multi-GPU processing is only supported for zarr arrays. Using single GPU for TIFF processing.")
        
        # Load input image
        print(f"Loading image: {args.input}")
        img_tensor, original_shape, original_dtype = load_tiff(args.input)
        print(f"Image shape: {original_shape}, dtype: {original_dtype}")
        print(f"Value range: [{img_tensor.min().item():.2f}, {img_tensor.max().item():.2f}]")
        
        if args.sample_results:
            # Generate sample results with various parameter combinations
            print("\nGenerating sample results with various parameter combinations...")
            if args.threshold:
                print("Note: --threshold is ignored when --sample-results is used.")
            
            # Create output directory
            output_dir = args.output
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            print(f"Output directory: {output_dir}")
            
            # Get sample parameters
            samples = generate_sample_parameters()
            
            # Process each sample
            for i, sample in enumerate(samples, 1):
                print(f"\n[{i}/{len(samples)}] Processing {sample['desc']}...")
                print(f"  Parameters: λ={sample['lambda']}, σ={sample['sigma']}, ρ={sample['rho']}")
                
                # Create config for this sample
                sample_config = {
                    'lambda': sample['lambda'],
                    'sigma': sample['sigma'],
                    'rho': sample['rho'],
                    'step_size': sample['step_size'],
                    'm': sample['m'],
                    'num_steps': sample['num_steps']
                }
                
                # Perform diffusion
                with torch.no_grad():
                    result = run_coherence_diffusion(
                        img_tensor,
                        sample_config,
                        show_progress=False,
                    )
                
                # Create output filename
                output_filename = f"{i:02d}_{sample['desc']}_l{sample['lambda']}_s{sample['sigma']}_r{sample['rho']}.tif"
                output_path = os.path.join(output_dir, output_filename)
                
                # Save result
                print(f"  Saving to: {output_filename}")
                save_tiff(result, output_path, original_shape, original_dtype)
                print(f"  Result value range: [{result.min().item():.2f}, {result.max().item():.2f}]")
            
            # Also save a reference file with parameter information
            reference_path = os.path.join(output_dir, "parameters_reference.txt")
            with open(reference_path, 'w') as f:
                f.write("Coherence Enhancing Diffusion - Sample Results\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"Input file: {args.input}\n")
                f.write(f"Fixed parameters: step_size={STEP_SIZE}, m={M}, num_steps={NUM_STEPS}\n\n")
                f.write("Variable parameters for each sample:\n\n")
                
                for i, sample in enumerate(samples, 1):
                    f.write(f"{i:02d}. {sample['desc']}:\n")
                    f.write(f"    λ (lambda) = {sample['lambda']}\n")
                    f.write(f"    σ (sigma)  = {sample['sigma']}\n")
                    f.write(f"    ρ (rho)    = {sample['rho']}\n")
                    f.write("\n")
            
            print(f"\nAll samples completed! Results saved to: {output_dir}")
            print(f"Parameter reference saved to: {reference_path}")
            
        else:
            # Single run with specified parameters
            print(f"Configuration: {config}")
            
            # Perform diffusion
            with torch.no_grad():  # Disable gradient computation for inference
                result = run_coherence_diffusion(
                    img_tensor,
                    config,
                    show_progress=False,
                )
            
            print(f"Result value range: [{result.min().item():.2f}, {result.max().item():.2f}]")
            
            # Save result
            print(f"Saving result to: {args.output}")
            if args.threshold:
                if not _HAVE_SKIMAGE:
                    raise RuntimeError("--threshold requires scikit-image to be installed.")
                arr = result.squeeze().cpu().numpy()
                thr = threshold_otsu(arr)
                bin8 = (arr >= thr).astype(np.uint8) * 255
                tifffile.imwrite(args.output, bin8, photometric='minisblack', compression='lzw')
            else:
                save_tiff(result, args.output, original_shape, original_dtype)
            print("Done!")


if __name__ == '__main__':
    main()
