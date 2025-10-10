#!/usr/bin/env python3
"""
Coherence Enhancing Diffusion Filter - Batched and Multi-GPU Version
Based on J. Weickert, "Coherence-Enhancing Diffusion Filtering", 
International Journal of Computer Vision, 1999, vol.31, p.111-127.

PyTorch implementation that matches the Java/Scala implementation that is available in the ImageJ plugin
"CoherenceEnhancingDiffusionFilter.java" and "CoherenceEnhancingDiffusionFilter.scala".
This script implements the coherence-enhancing diffusion filter using PyTorch.
"""

import torch
import torch.nn.functional as F
import numpy as np
import tifffile
import zarr
from tqdm import tqdm
import argparse
from pathlib import Path
import os
from functools import lru_cache
import multiprocessing as mp
from typing import Dict, Tuple, Optional, List
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

# Algorithm constants
EPS = 2**-52          # Machine epsilon for numerical stability
GAMMA = 0.01          # Minimum diffusivity
CM = 7.2848           # Constant for exponential diffusivity function

# Check if torch.compile is available (PyTorch 2.0+)
TORCH_COMPILE_AVAILABLE = hasattr(torch, 'compile')


@lru_cache(maxsize=8)
def _create_gaussian_kernel_1d(sigma, device_str):
    """Create and cache 1D Gaussian kernel."""
    device = torch.device(device_str)
    size = int(2 * np.ceil(3 * sigma) + 1)
    x = torch.arange(size, dtype=torch.float32, device=device) - size // 2
    kernel_1d = torch.exp(-x**2 / (2 * sigma**2))
    kernel_1d = kernel_1d / kernel_1d.sum()
    return kernel_1d, size


def create_gaussian_kernel(sigma, device):
    """Create 2D Gaussian kernel for convolution."""
    kernel_1d, size = _create_gaussian_kernel_1d(sigma, str(device))
    gauss_2d = kernel_1d[:, None] @ kernel_1d[None, :]
    return gauss_2d[None, None, :, :]


def gaussian_blur(img, sigma):
    """Apply Gaussian blur using separable convolution for efficiency."""
    if sigma <= 0:
        return img
    
    # Get cached kernel
    kernel_1d, size = _create_gaussian_kernel_1d(sigma, str(img.device))
    
    batch_size = img.shape[0]
    
    # For batch processing, we need to handle each image in the batch
    # Reshape input from (B, C, H, W) to (B*C, 1, H, W) for grouped convolution
    B, C, H, W = img.shape
    img_reshaped = img.reshape(B * C, 1, H, W)
    
    # Create kernels for horizontal and vertical passes
    kernel_1d_h = kernel_1d.view(1, 1, -1, 1)
    kernel_1d_v = kernel_1d.view(1, 1, 1, -1)
    
    # Apply horizontal convolution
    img_reshaped = F.conv2d(img_reshaped, kernel_1d_h, padding=(size//2, 0))
    
    # Apply vertical convolution
    img_reshaped = F.conv2d(img_reshaped, kernel_1d_v, padding=(0, size//2))
    
    # Reshape back to original batch shape
    img = img_reshaped.reshape(B, C, H, W)
    
    return img


def compute_gradients(img):
    """Compute image gradients using central differences."""
    # Gradient kernels matching Java implementation
    grad_x_kernel = torch.tensor([[-0.5, 0.0, 0.5]], dtype=torch.float32, device=img.device)
    grad_y_kernel = grad_x_kernel.T
    
    grad_x_kernel = grad_x_kernel.view(1, 1, 1, 3)
    grad_y_kernel = grad_y_kernel.view(1, 1, 3, 1)
    
    # Compute gradients with 'replicate' padding for boundary handling
    grad_x = F.conv2d(img, grad_x_kernel, padding=(0, 1))
    grad_y = F.conv2d(img, grad_y_kernel, padding=(1, 0))
    
    return grad_x, grad_y


def compute_structure_tensor(grad_x, grad_y, rho):
    """Compute smoothed structure tensor components."""
    s11 = gaussian_blur(grad_x * grad_x, rho)
    s12 = gaussian_blur(grad_x * grad_y, rho)
    s22 = gaussian_blur(grad_y * grad_y, rho)
    
    return s11, s12, s22


def compute_alpha(s11, s12, s22):
    """Compute eigenvalue measure alpha."""
    a = s11 - s22
    b = s12
    alpha = torch.sqrt(a**2 + 4 * b**2)
    return alpha


def compute_c2(alpha, lambda_param, m):
    """Compute diffusivity function c2."""
    h1 = (alpha + EPS) / lambda_param
    
    if abs(m - 1.0) < 1e-10:
        h2 = h1
    else:
        h2 = torch.pow(h1, m)
    
    h3 = torch.exp(-CM / h2)
    c2 = GAMMA + (1 - GAMMA) * h3
    
    return c2


def compute_diffusion_tensor(s11, s12, s22, alpha, c2, c1=GAMMA):
    """Compute diffusion tensor components D11, D12, D22."""
    dd = (c2 - c1) * (s11 - s22) / (alpha + EPS)
    
    d11 = 0.5 * (c1 + c2 + dd)
    d12 = (c1 - c2) * s12 / (alpha + EPS)
    d22 = 0.5 * (c1 + c2 - dd)
    
    return d11, d12, d22


def _diffusion_step_vectorized_impl(img, d11, d12, d22, step_size):
    """
    Implementation of vectorized diffusion step.
    Separated for torch.compile optimization.
    """
    # Pad all tensors
    img_pad = F.pad(img, (1, 1, 1, 1), mode='replicate')
    d11_pad = F.pad(d11, (1, 1, 1, 1), mode='replicate')
    d12_pad = F.pad(d12, (1, 1, 1, 1), mode='replicate')
    d22_pad = F.pad(d22, (1, 1, 1, 1), mode='replicate')
    
    # Extract shifted versions
    # Center is [1:-1, 1:-1]
    img_c = img_pad[:, :, 1:-1, 1:-1]   # current (i, j)
    img_n = img_pad[:, :, 0:-2, 1:-1]   # north (i-1, j) - xpo in Java
    img_s = img_pad[:, :, 2:, 1:-1]     # south (i+1, j) - xmo in Java
    img_w = img_pad[:, :, 1:-1, 0:-2]   # west (i, j-1) - xop in Java
    img_e = img_pad[:, :, 1:-1, 2:]     # east (i, j+1) - xom in Java
    img_nw = img_pad[:, :, 0:-2, 0:-2]  # northwest (i-1, j-1) - xpp
    img_ne = img_pad[:, :, 0:-2, 2:]    # northeast (i-1, j+1) - xpm
    img_sw = img_pad[:, :, 2:, 0:-2]    # southwest (i+1, j-1) - xmp
    img_se = img_pad[:, :, 2:, 2:]      # southeast (i+1, j+1) - xmm
    
    # Diffusion coefficients
    # Java: c = d22, a = d11, b = d12
    d11_c = d11_pad[:, :, 1:-1, 1:-1]
    d11_n = d11_pad[:, :, 0:-2, 1:-1]
    d11_s = d11_pad[:, :, 2:, 1:-1]
    
    d22_c = d22_pad[:, :, 1:-1, 1:-1]
    d22_w = d22_pad[:, :, 1:-1, 0:-2]
    d22_e = d22_pad[:, :, 1:-1, 2:]
    
    d12_c = d12_pad[:, :, 1:-1, 1:-1]
    d12_n = d12_pad[:, :, 0:-2, 1:-1]
    d12_s = d12_pad[:, :, 2:, 1:-1]
    d12_w = d12_pad[:, :, 1:-1, 0:-2]
    d12_e = d12_pad[:, :, 1:-1, 2:]
    
    # First derivative terms (matching Java formula)
    c_cop = d22_c + d22_w  # (i,j) + (i,j-1)
    a_amo = d11_s + d11_c  # (i+1,j) + (i,j)
    a_apo = d11_n + d11_c  # (i-1,j) + (i,j) 
    c_com = d22_c + d22_e  # (i,j) + (i,j+1)
    
    first_deriv = (
        c_cop * img_w +                                         # c_cop * xop
        a_amo * img_s -                                         # a_amo * xmo
        (a_amo + a_apo + c_com + c_cop) * img_c +             # -(sum) * x
        a_apo * img_n +                                         # a_apo * xpo
        c_com * img_e                                           # c_com * xom
    )
    
    # Second derivative (cross) terms
    bmo = d12_s  # (i+1, j)
    bop = d12_w  # (i, j-1)
    bpo = d12_n  # (i-1, j)
    bom = d12_e  # (i, j+1)
    
    second_deriv = (
        -1 * ((bmo + bop) * img_sw + (bpo + bom) * img_ne) +  # -((bmo+bop)*xmp + (bpo+bom)*xpm)
        (bpo + bop) * img_nw +                                  # (bpo+bop)*xpp
        (bmo + bom) * img_se                                    # (bmo+bom)*xmm
    )
    
    # Update
    img_new = img + step_size * (0.5 * first_deriv + 0.25 * second_deriv)
    
    return img_new


# Create compiled versions of key functions if torch.compile is available
if TORCH_COMPILE_AVAILABLE:
    # Compile the core computational functions
    # Disable CUDA graphs to avoid tensor reuse issues when processing multiple slices
    compile_options = {"triton.cudagraphs": False}
    gaussian_blur_compiled = torch.compile(gaussian_blur, options=compile_options)
    compute_gradients_compiled = torch.compile(compute_gradients, options=compile_options)
    compute_structure_tensor_compiled = torch.compile(compute_structure_tensor, options=compile_options)
    compute_alpha_compiled = torch.compile(compute_alpha, options=compile_options)
    compute_c2_compiled = torch.compile(compute_c2, options=compile_options)
    compute_diffusion_tensor_compiled = torch.compile(compute_diffusion_tensor, options=compile_options)
    _diffusion_step_vectorized_impl_compiled = torch.compile(_diffusion_step_vectorized_impl, options=compile_options)
else:
    # Use uncompiled versions
    gaussian_blur_compiled = gaussian_blur
    compute_gradients_compiled = compute_gradients
    compute_structure_tensor_compiled = compute_structure_tensor
    compute_alpha_compiled = compute_alpha
    compute_c2_compiled = compute_c2
    compute_diffusion_tensor_compiled = compute_diffusion_tensor
    _diffusion_step_vectorized_impl_compiled = _diffusion_step_vectorized_impl


def diffusion_step_vectorized(img, d11, d12, d22, step_size):
    """
    Vectorized version of diffusion step (faster but may have slight differences).
    Matches the Java implementation's formula.
    """
    return _diffusion_step_vectorized_impl_compiled(img, d11, d12, d22, step_size)


def coherence_enhancing_diffusion(img_tensor, config, use_vectorized=True, show_progress=False, use_compiled=True):
    """Main function to perform coherence enhancing diffusion."""
    img = img_tensor.clone()
    
    # Choose compiled or uncompiled functions based on parameter and availability
    if use_compiled and TORCH_COMPILE_AVAILABLE:
        blur_fn = gaussian_blur_compiled
        grad_fn = compute_gradients_compiled
        struct_fn = compute_structure_tensor_compiled
        alpha_fn = compute_alpha_compiled
        c2_fn = compute_c2_compiled
        diff_tensor_fn = compute_diffusion_tensor_compiled
    else:
        blur_fn = gaussian_blur
        grad_fn = compute_gradients
        struct_fn = compute_structure_tensor
        alpha_fn = compute_alpha
        c2_fn = compute_c2
        diff_tensor_fn = compute_diffusion_tensor
    
    # Use tqdm if show_progress is True, otherwise use a simple range
    iterator = tqdm(range(config['num_steps']), desc="Diffusion steps", leave=False) if show_progress else range(config['num_steps'])
    
    for step in iterator:
        if not show_progress:
            print(f"Step {step + 1}/{config['num_steps']}", end='\r')
        
        # Gaussian smoothing
        img_smooth = blur_fn(img, config['sigma'])
        
        # Compute gradients
        grad_x, grad_y = grad_fn(img_smooth)
        
        # Compute structure tensor
        s11, s12, s22 = struct_fn(grad_x, grad_y, config['rho'])
        
        # Compute eigenvalue measure
        alpha = alpha_fn(s11, s12, s22)
        
        # Compute diffusivity
        c2 = c2_fn(alpha, config['lambda'], config['m'])
        
        # Compute diffusion tensor
        d11, d12, d22 = diff_tensor_fn(s11, s12, s22, alpha, c2)
        
        # Perform diffusion step
        if use_vectorized:
            img = diffusion_step_vectorized(img, d11, d12, d22, config['step_size'])
        else:
            raise NotImplementedError("Pixel-by-pixel implementation not supported for batch processing")
    
    if not show_progress:
        print("\nDiffusion complete!")
    
    return img


def process_zarr_chunk(input_path: str, output_path: str, z_start: int, z_end: int, 
                      gpu_id: int, config: Dict, batch_size: int = 1,
                      use_vectorized: bool = True, use_compiled: bool = True) -> None:
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
            result_batch = coherence_enhancing_diffusion(
                batch_tensor, config, 
                use_vectorized=use_vectorized, 
                show_progress=False, 
                use_compiled=use_compiled
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


def launch_multi_gpu_processing(input_path: str, output_path: str, config: Dict, 
                               num_gpus: int, batch_size: int = 1,
                               use_vectorized: bool = True, use_compiled: bool = True) -> None:
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
            args=(input_path, output_path, z_start, z_end, gpu_id, config, batch_size,
                  use_vectorized, use_compiled)
        )
        p.start()
        processes.append(p)
    
    # Wait for all processes to complete
    for p in processes:
        p.join()
    
    print("\nAll GPU processes completed!")


def process_zarr_array(input_path, output_path, config, batch_size=1, use_vectorized=True, use_compiled=True):
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
            result_batch = coherence_enhancing_diffusion(
                batch_tensor, config, 
                use_vectorized=use_vectorized, 
                show_progress=False, 
                use_compiled=use_compiled
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
    img_array = np.array(img, dtype=np.float32)
    
    # DO NOT normalize - keep original values!
    # The Java implementation works with raw float values
    
    # Convert to torch tensor and add batch and channel dimensions
    img_tensor = torch.from_numpy(img_array).unsqueeze(0).unsqueeze(0).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    
    return img_tensor, img_array.shape, img_array.dtype


def save_tiff(tensor, filepath, original_shape, original_dtype):
    """Save torch tensor as TIFF image."""
    # Remove batch and channel dimensions
    img_array = tensor.squeeze().cpu().numpy()
    
    # Save with original data type
    # The Java implementation doesn't rescale values
    if original_dtype == np.uint8:
        img_array = np.clip(img_array, 0, 255).astype(np.uint8)
    elif original_dtype == np.uint16:
        img_array = np.clip(img_array, 0, 65535).astype(np.uint16)
    else:
        img_array = img_array.astype(original_dtype)
    
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

def process_tiff_folder(input_dir: str, output_dir: str, config: Dict, *, use_vectorized: bool = True, use_compiled: bool = True, threshold: bool = False) -> None:
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
            result = coherence_enhancing_diffusion(
                img_tensor,
                config,
                use_vectorized=use_vectorized,
                use_compiled=use_compiled,
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
    parser.add_argument('--no-vectorized', action='store_true', help='Use pixel-by-pixel implementation (slower but exact match to Java)')
    parser.add_argument('--no-compile', action='store_true', help='Disable torch.compile optimization (useful for debugging)')
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
            str(input_path), str(output_path), config,
            use_vectorized=not args.no_vectorized,
            use_compiled=not args.no_compile,
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
                str(input_path), str(output_path), config, 
                num_gpus=num_gpus, batch_size=args.batch_size,
                use_vectorized=not args.no_vectorized, 
                use_compiled=not args.no_compile
            )
        else:
            print("Using single GPU processing")
            process_zarr_array(
                str(input_path), str(output_path), config, 
                batch_size=args.batch_size,
                use_vectorized=not args.no_vectorized, 
                use_compiled=not args.no_compile
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
                    result = coherence_enhancing_diffusion(
                        img_tensor, sample_config, 
                        use_vectorized=not args.no_vectorized, 
                        use_compiled=not args.no_compile
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
                result = coherence_enhancing_diffusion(
                    img_tensor, config, 
                    use_vectorized=not args.no_vectorized, 
                    use_compiled=not args.no_compile
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
