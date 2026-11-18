import functools
import os
import socket
import torch.distributed as dist
import torch.multiprocessing as mp


def get_input_shape(self):
    """
    Return the spatial dimensions (Z,Y,X) of the input array.
    For 4D arrays (C,Z,Y,X), returns just the spatial part (Z,Y,X).
    """
    # Handle both 3D array (Z,Y,X) and 4D array with channels (C,Z,Y,X)
    return self.input_shape if len(self.input_shape) == 3 else self.input_shape[1:]


import numpy as np


def generate_positions(min_val, max_val, patch_size, step_size):
    """
    Generate positions for patch extraction with step size.

    Args:
        min_val: Minimum value of the range
        max_val: Maximum value of the range
        patch_size: Size of the patch in this dimension
        step_size: Step size for moving the patch window

    Returns:
        List of positions (starts of patches)
    """
    # Calculate the total range
    range_size = max_val - min_val

    # If the range is smaller than a patch, return just the minimum value
    if range_size <= patch_size:
        return [min_val]

    # Generate positions with the specified step size
    positions = list(range(min_val, max_val - patch_size + 1, step_size))

    # Always include the last position to ensure we cover the full range
    if positions[-1] + patch_size < max_val:
        positions.append(max_val - patch_size)

    return positions


def compute_steps_for_sliding_window(image_size, patch_size, step_size_factor):
    """
    Compute the positions for a single dimension.
    This matches nnUNet's per-dimension computation in their sliding window function.

    Args:
        image_size: size of this dimension
        patch_size: patch size for this dimension
        step_size_factor: step size as a fraction (0 <= step_size_factor <= 1)
                         0 means no overlap (full stride)

    Returns:
        List of step positions for this dimension
    """
    assert image_size >= patch_size, "image size must be larger than patch_size"
    assert 0 <= step_size_factor <= 1, 'step_size must be between 0 and 1 (inclusive)'

    # Handle special case: step_size_factor = 0 means no overlap (full stride)
    if step_size_factor == 0:
        target_step_size = patch_size
    else:
        # Calculate step size in voxels - this is key
        target_step_size = int(patch_size * step_size_factor)
    
    # Ensure target_step_size is at least 1
    target_step_size = max(1, target_step_size)

    # Calculate number of steps
    num_steps = int(np.ceil((image_size - patch_size) / target_step_size)) + 1

    # Calculate actual steps for uniform spacing
    max_step_value = image_size - patch_size
    if num_steps > 1:
        # When we have multiple steps, distribute them uniformly
        actual_step_size = max_step_value / (num_steps - 1)
    else:
        # When we have just one step, place it at position 0
        actual_step_size = 99999999999  # Only one step at position 0

    # Generate all step positions
    steps = [int(np.round(actual_step_size * i)) for i in range(num_steps)]

    steps = [min(step, max_step_value) for step in steps]

    return steps


def compute_steps_for_sliding_window_tuple(image_size_tuple, patch_size_tuple, step_size_factor):
    """
    Compute steps for sliding window for all dimensions at once.
    This is an exact reimplementation of nnUNet's compute_steps_for_sliding_window function.

    Args:
        image_size_tuple: Tuple with sizes for each dimension (e.g., (Z, Y, X))
        patch_size_tuple: Tuple with patch sizes for each dimension
        step_size_factor: Step size as a fraction (0 <= step_size_factor <= 1)
                         0 means no overlap (full stride)

    Returns:
        List of lists of step positions, one list per dimension
    """
    assert all(i >= j for i, j in zip(image_size_tuple, patch_size_tuple)), \
        "image size must be as large or larger than patch_size"
    assert 0 <= step_size_factor <= 1, 'step_size must be between 0 and 1 (inclusive)'

    # Handle special case: step_size_factor = 0 means no overlap (full stride)
    if step_size_factor == 0:
        target_step_sizes_in_voxels = list(patch_size_tuple)
    else:
        # Target step sizes in voxels for all dimensions
        target_step_sizes_in_voxels = [int(i * step_size_factor) for i in patch_size_tuple]
    
    # Ensure all target step sizes are at least 1
    target_step_sizes_in_voxels = [max(1, step_size) for step_size in target_step_sizes_in_voxels]

    # Calculate number of steps for each dimension
    num_steps = [int(np.ceil((i - k) / j)) + 1 for i, j, k in
                 zip(image_size_tuple, target_step_sizes_in_voxels, patch_size_tuple)]

    # Calculate steps for each dimension
    steps = []
    for dim in range(len(patch_size_tuple)):
        # The highest step value for this dimension
        max_step_value = image_size_tuple[dim] - patch_size_tuple[dim]
        if num_steps[dim] > 1:
            actual_step_size = max_step_value / (num_steps[dim] - 1)
        else:
            actual_step_size = 99999999999  # Only one step at position 0

        steps_here = [int(np.round(actual_step_size * i)) for i in range(num_steps[dim])]
        steps_here = [min(step, max_step_value) for step in steps_here]
        steps.append(steps_here)

    return steps


import torch
from typing import List, Dict, Any, Tuple


def merge_tensors(tensors: List[Tuple[torch.Tensor, Dict[str, Any]]]) -> torch.Tensor:
    """
    Merge multiple tensors using transformation metadata and weights.

    Args:
        tensors: List of tuples, each containing:
            - A tensor to be merged
            - Dictionary with metadata including:
                - 'inverse_func': Function to transform the tensor before merging
                - 'weight': Weight for this tensor in the final result

    Returns:
        Merged tensor
    """
    # Initialize with zeros like the first tensor
    result = torch.zeros_like(tensors[0][0])

    # Apply inverse transform to each tensor and add to result with appropriate weight
    for tensor, metadata in tensors:
        # Apply transformation
        transformed = metadata['inverse_func'](tensor)
        # Add to result with weight
        result += metadata['weight'] * transformed

    return result


def setup_ddp(rank, world_size):
    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup_ddp():
    dist.destroy_process_group()


def find_free_network_port() -> int:
    """Finds a free port on localhost.

    It is useful in single-node training when we don't want to connect to a real main node but have to set the
    `MASTER_PORT` environment variable.
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("", 0))
    port = s.getsockname()[1]
    s.close()
    return port

def softmax_helper_dim0(x: torch.Tensor) -> torch.Tensor:
    return torch.softmax(x, 0)


def softmax_helper_dim1(x: torch.Tensor) -> torch.Tensor:
    return torch.softmax(x, 1)


def empty_cache(device: torch.device):
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    elif device.type == 'mps':
        from torch import mps
        mps.empty_cache()
    else:
        pass


class dummy_context(object):
    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


# oneDNN ISA caps that exclude AMX, so BF16 would not use AMX tiles.
_ONEDNN_CAPS_WITHOUT_AMX = frozenset({
    "SSE41", "AVX", "AVX2", "AVX2_VNNI", "AVX2_VNNI_2", "AVX512_CORE", "AVX512_CORE_VNNI",
    "AVX512_CORE_BF16", "AVX512_CORE_FP16", "AVX10_1_512", "AVX10_2_512",
})


def _onednn_cap_allows_amx():
    """False when ONEDNN_MAX_CPU_ISA (or DNNL_MAX_CPU_ISA) caps oneDNN below AMX."""
    cap = os.getenv("ONEDNN_MAX_CPU_ISA") or os.getenv("DNNL_MAX_CPU_ISA") or ""
    return cap.strip().upper() not in _ONEDNN_CAPS_WITHOUT_AMX


@functools.lru_cache(maxsize=1)
def _cpu_has_amx_bf16():
    """True when the CPU has Intel AMX BF16 tiles (Sapphire Rapids and later).

    Torch 2.11+ has torch.cpu.get_capabilities(); older torch only has the
    private AMX_TILE probe. A missing or failing probe counts as no AMX.
    """
    try:
        get_capabilities = getattr(torch.cpu, "get_capabilities", None)
        if get_capabilities is not None:
            return bool(get_capabilities().get("amx_bf16", False))
        probe = getattr(torch.cpu, "_is_amx_tile_supported", None)
        return bool(probe()) if probe is not None else False
    except Exception:
        return False


def inference_autocast_enabled(device):
    """Whether automatic inference should use BF16 autocast on *device*.

    Accelerators keep AMP. On CPU, BF16 3D convolution measured slower than
    FP32 on CPUs without AMX (including ones with AVX512_BF16) and far slower
    on CPUs without native BF16, so CPU uses FP32 unless the CPU has AMX
    (and no ONEDNN_MAX_CPU_ISA cap excludes it), where the previous BF16
    behaviour is kept.
    """
    if device.type != "cpu":
        return True
    return _onednn_cap_allows_amx() and _cpu_has_amx_bf16()
