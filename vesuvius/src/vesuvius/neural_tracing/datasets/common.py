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
    Create a 3D gaussian heatmap centered at coords.

    Args:
        coords: (3,) tensor - position in crop-local coordinates (0 to crop_size-1)
        crop_size: int - size of the output volume
        sigma: float - gaussian standard deviation (default 2.0)

    Returns:
        (crop_size, crop_size, crop_size) tensor with gaussian centered at coords
    """
    device = coords.device if isinstance(coords, torch.Tensor) else 'cpu'
    dtype = coords.dtype if isinstance(coords, torch.Tensor) else torch.float32

    if axis_1d is None:
        axis_1d = torch.arange(crop_size, device=device, dtype=dtype)
    else:
        axis_1d = axis_1d.to(device=device, dtype=dtype)

    dz = (axis_1d - coords[0]) ** 2
    dy = (axis_1d - coords[1]) ** 2
    dx = (axis_1d - coords[2]) ** 2
    heatmap = torch.exp(-(dz[:, None, None] + dy[None, :, None] + dx[None, None, :]) / (2 * sigma ** 2))
    return heatmap

