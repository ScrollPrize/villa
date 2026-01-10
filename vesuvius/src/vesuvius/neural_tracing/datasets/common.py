import torch
from dataclasses import dataclass
from vesuvius.tifxyz import Tifxyz
import zarr
from typing import Tuple

@dataclass
class Patch:
    seg: Tifxyz                           # Reference to the segment
    volume: zarr.Array                    # zarr volume
    scale: float                          # volume_scale from config
    grid_bbox: Tuple[int, int, int, int]  # (row_min, row_max, col_min, col_max) in the tifxyz grid
    world_bbox: Tuple[float, ...]         # (z_min, z_max, y_min, y_max, x_min, x_max) in world coordinates (volume coordinates)



def make_gaussian_heatmap(coords, crop_size, sigma: float = 2.0, axis_1d=None):
    """
    Create a 3D gaussian heatmap centered at one or more coords.

    Args:
        coords: (N, 3) or (3,) tensor, or list of (3,) tensors - position(s) in crop-local coordinates (0 to crop_size-1)
        crop_size: int - size of the output volume
        sigma: float - gaussian standard deviation (default 2.0)

    Returns:
        (crop_size, crop_size, crop_size) tensor with gaussian(s) centered at coords.
        If multiple coords provided, heatmaps are combined using max.
    """
    # Convert list to tensor
    if isinstance(coords, list):
        if len(coords) == 0:
            return torch.zeros(crop_size, crop_size, crop_size)
        coords = torch.stack(coords)

    device = coords.device if isinstance(coords, torch.Tensor) else 'cpu'
    dtype = coords.dtype if isinstance(coords, torch.Tensor) else torch.float32

    # Ensure coords is 2D: (N, 3)
    if coords.dim() == 1:
        coords = coords.unsqueeze(0)

    if axis_1d is None:
        axis_1d = torch.arange(crop_size, device=device, dtype=dtype)
    else:
        axis_1d = axis_1d.to(device=device, dtype=dtype)

    # coords: (N, 3), axis_1d: (crop_size,)
    # Compute squared distances: (N, crop_size) for each axis
    dz = (axis_1d[None, :] - coords[:, 0:1]) ** 2  # (N, crop_size)
    dy = (axis_1d[None, :] - coords[:, 1:2]) ** 2  # (N, crop_size)
    dx = (axis_1d[None, :] - coords[:, 2:3]) ** 2  # (N, crop_size)

    # Broadcast to (N, crop_size, crop_size, crop_size)
    dist_sq = dz[:, :, None, None] + dy[:, None, :, None] + dx[:, None, None, :]
    heatmaps = torch.exp(-dist_sq / (2 * sigma ** 2))

    # Take max over N points
    return heatmaps.max(dim=0).values

