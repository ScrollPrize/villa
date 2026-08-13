from functools import lru_cache
import numpy as np
import torch
from scipy.ndimage import gaussian_filter



def compute_gaussian_3d(
        tile_size: tuple,   # either (D,H,W) or (C,D,H,W)
        sigma_scale: float = 1. / 8,
        value_scaling_factor: float = 1.,
        device=torch.device('cpu'),
        dtype=torch.float32
):
    """
    Returns a Gaussian weighting map for either:
        - a 3D patch of shape (D, H, W), or
        - a 4D patch of shape (C, D, H, W).

    If tile_size is (C, D, H, W), we first compute a (D, H, W) Gaussian,
    then replicate it across C channels.

    Args:
        tile_size (tuple): (D,H,W) or (C,D,H,W)
        sigma_scale (float): scale factor for the Gaussian sigma
        value_scaling_factor (float): scale the peak value
        device (torch.device): target device
        dtype (torch.dtype): dtype of the output tensor

    Returns:
        torch.Tensor, shape (D,H,W) or (C,D,H,W).
    """
    if len(tile_size) not in (3, 4):
        raise ValueError(
            f"Tile size must be either (D,H,W) or (C,D,H,W). Got: {tile_size}"
        )

    if len(tile_size) == 4:
        c, d, h, w = tile_size
        tile_size_3d = (d, h, w)
    else:  # length == 3
        c = None
        d, h, w = tile_size
        tile_size_3d = tile_size

    # Create a volume with a single 1 in the center, then Gaussian-filter it
    tmp = np.zeros(tile_size_3d, dtype=np.float32)
    center_coords = [dim // 2 for dim in tile_size_3d]
    tmp[tuple(center_coords)] = 1

    sigmas = [dim * sigma_scale for dim in tile_size_3d]
    gaussian_map = gaussian_filter(tmp, sigma=sigmas, mode='constant', cval=0)

    # Rescale so the peak is value_scaling_factor
    gaussian_map /= (gaussian_map.max() / value_scaling_factor)

    # Avoid zero (for numerical safety if dividing by weights later)
    minval = gaussian_map[gaussian_map > 0].min()
    gaussian_map[gaussian_map == 0] = minval

    # Convert to Torch
    gaussian_map = torch.from_numpy(gaussian_map).to(device=device, dtype=dtype)

    # If we had a channel dimension, replicate across channels
    if c is not None:
        gaussian_map = gaussian_map.unsqueeze(0).expand(c, d, h, w)

    return gaussian_map


def get_gaussian_map(channels: int, patch_size_3d: tuple, sigma_scale=1./8, device=torch.device('cpu')):
    """
    Retrieve or compute a Gaussian weight map of shape (channels, D, H, W).
    If channels=1, shape is (1, D, H, W); if channels=3, shape is (3, D, H, W), etc.
    """
    # We store the patch_size as (channels, D, H, W) to unify the logic
    # internally (some tasks might have 1 channel, some 3, etc.)
    tile_size = (channels,) + patch_size_3d

    # We'll cache based on (channels, patch_size_3d)
    cache_key = (channels, patch_size_3d)
    if cache_key not in _gauss_map_cache:
        gauss_t = compute_gaussian_3d(
            tile_size=tile_size,
            sigma_scale=sigma_scale,
            device=device,
            dtype=torch.float32
        )
        _gauss_map_cache[cache_key] = gauss_t.cpu().numpy()  # store as numpy for easy Zarr usage

    return _gauss_map_cache[cache_key]

