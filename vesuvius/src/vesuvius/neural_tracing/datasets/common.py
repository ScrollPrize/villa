import torch
import numpy as np
from numba import njit
from dataclasses import dataclass
from vesuvius.tifxyz import Tifxyz
import zarr
from typing import Any, Dict, List, Tuple


@njit
def _draw_line_3d(volume: np.ndarray, z0: int, y0: int, x0: int, z1: int, y1: int, x1: int) -> None:
    """Draw a 3D line using Bresenham's algorithm. Modifies volume in-place."""
    dz = abs(z1 - z0)
    dy = abs(y1 - y0)
    dx = abs(x1 - x0)

    sz = 1 if z0 < z1 else -1
    sy = 1 if y0 < y1 else -1
    sx = 1 if x0 < x1 else -1

    # Determine the dominant axis
    if dx >= dy and dx >= dz:
        # X is dominant
        err_y = 2 * dy - dx
        err_z = 2 * dz - dx
        while x0 != x1:
            if 0 <= z0 < volume.shape[0] and 0 <= y0 < volume.shape[1] and 0 <= x0 < volume.shape[2]:
                volume[z0, y0, x0] = 1.0
            if err_y > 0:
                y0 += sy
                err_y -= 2 * dx
            if err_z > 0:
                z0 += sz
                err_z -= 2 * dx
            err_y += 2 * dy
            err_z += 2 * dz
            x0 += sx
    elif dy >= dx and dy >= dz:
        # Y is dominant
        err_x = 2 * dx - dy
        err_z = 2 * dz - dy
        while y0 != y1:
            if 0 <= z0 < volume.shape[0] and 0 <= y0 < volume.shape[1] and 0 <= x0 < volume.shape[2]:
                volume[z0, y0, x0] = 1.0
            if err_x > 0:
                x0 += sx
                err_x -= 2 * dy
            if err_z > 0:
                z0 += sz
                err_z -= 2 * dy
            err_x += 2 * dx
            err_z += 2 * dz
            y0 += sy
    else:
        # Z is dominant
        err_x = 2 * dx - dz
        err_y = 2 * dy - dz
        while z0 != z1:
            if 0 <= z0 < volume.shape[0] and 0 <= y0 < volume.shape[1] and 0 <= x0 < volume.shape[2]:
                volume[z0, y0, x0] = 1.0
            if err_x > 0:
                x0 += sx
                err_x -= 2 * dz
            if err_y > 0:
                y0 += sy
                err_y -= 2 * dz
            err_x += 2 * dx
            err_y += 2 * dy
            z0 += sz

    # Set the final point
    if 0 <= z1 < volume.shape[0] and 0 <= y1 < volume.shape[1] and 0 <= x1 < volume.shape[2]:
        volume[z1, y1, x1] = 1.0


@njit
def voxelize_surface_grid(
    zyx_grid: np.ndarray,
    crop_size: tuple,
) -> np.ndarray:
    """
    Voxelize a 2D grid of 3D points by drawing lines between adjacent points.

    Args:
        zyx_grid: (H, W, 3) array of ZYX coordinates in local crop space
        crop_size: (D, H, W) shape of output volume

    Returns:
        (D, H, W) binary volume with lines connecting adjacent grid points
    """
    volume = np.zeros(crop_size, dtype=np.float32)
    n_rows, n_cols = zyx_grid.shape[0], zyx_grid.shape[1]

    # Draw horizontal lines (between adjacent columns)
    for r in range(n_rows):
        for c in range(n_cols - 1):
            z0 = int(round(zyx_grid[r, c, 0]))
            y0 = int(round(zyx_grid[r, c, 1]))
            x0 = int(round(zyx_grid[r, c, 2]))
            z1 = int(round(zyx_grid[r, c + 1, 0]))
            y1 = int(round(zyx_grid[r, c + 1, 1]))
            x1 = int(round(zyx_grid[r, c + 1, 2]))
            _draw_line_3d(volume, z0, y0, x0, z1, y1, x1)

    # Draw vertical lines (between adjacent rows)
    for r in range(n_rows - 1):
        for c in range(n_cols):
            z0 = int(round(zyx_grid[r, c, 0]))
            y0 = int(round(zyx_grid[r, c, 1]))
            x0 = int(round(zyx_grid[r, c, 2]))
            z1 = int(round(zyx_grid[r + 1, c, 0]))
            y1 = int(round(zyx_grid[r + 1, c, 1]))
            x1 = int(round(zyx_grid[r + 1, c, 2]))
            _draw_line_3d(volume, z0, y0, x0, z1, y1, x1)

    return volume

@dataclass
class Patch:
    """A single patch from the hierarchical tiling method."""
    seg: Tifxyz                           # Reference to the segment
    volume: zarr.Array                    # zarr volume
    scale: float                          # volume_scale from config
    grid_bbox: Tuple[int, int, int, int]  # (row_min, row_max, col_min, col_max) in the tifxyz grid
    world_bbox: Tuple[float, ...]         # (z_min, z_max, y_min, y_max, x_min, x_max) in world coordinates (volume coordinates)


@dataclass
class ChunkPatch:
    """A world-space chunk containing one or more surface wraps.

    This is the output of the world-chunk tiling method. Each chunk can contain
    multiple wraps from potentially multiple segments.
    """
    chunk_id: Tuple[int, int, int]         # (cz, cy, cx) index in chunk grid
    volume: Any                            # zarr.Array or zarr.Group
    scale: int                             # volume_scale from config
    world_bbox: Tuple[float, ...]          # (z_min, z_max, y_min, y_max, x_min, x_max)
    wraps: List[Dict]                      # [{"segment": Tifxyz, "bbox_2d": tuple, "wrap_id": int, "segment_idx": int}, ...]
    segments: List[Tifxyz]                 # All segments (for lookup by segment_idx)

    @property
    def wrap_count(self) -> int:
        """Number of wraps in this chunk."""
        return len(self.wraps)

    @property
    def has_multiple_wraps(self) -> bool:
        """Whether this chunk has more than one wrap."""
        return len(self.wraps) > 1

    @property
    def segment_ids(self) -> List[str]:
        """List of unique segment UUIDs in this chunk."""
        return list(set(w["segment"].uuid for w in self.wraps))



def make_gaussian_heatmap(coords, crop_size, sigma: float = 2.0, axis_1d=None):
    """
    Create a 3D gaussian heatmap centered at one or more coords.

    Uses sparse/scattered placement to avoid massive memory allocation.
    Only computes gaussian values within 3*sigma of each point.

    Args:
        coords: (N, 3) or (3,) tensor, or list of (3,) tensors - position(s) in crop-local coordinates (0 to crop_size-1)
        crop_size: int or tuple - size of the output volume
        sigma: float - gaussian standard deviation (default 2.0)
        axis_1d: ignored (kept for API compatibility)

    Returns:
        (D, H, W) tensor with gaussian(s) centered at coords.
        If multiple coords provided, heatmaps are combined using max.
    """
    # Handle inputs
    if isinstance(coords, list):
        if len(coords) == 0:
            if isinstance(crop_size, int):
                return torch.zeros(crop_size, crop_size, crop_size)
            else:
                return torch.zeros(*crop_size)
        coords = torch.stack(coords)

    if coords.dim() == 1:
        coords = coords.unsqueeze(0)

    # Determine output shape
    if isinstance(crop_size, int):
        shape = (crop_size, crop_size, crop_size)
    else:
        shape = tuple(crop_size)

    # Initialize output
    heatmap = torch.zeros(shape, dtype=torch.float32)

    # Radius to compute (3*sigma captures 99.7% of gaussian)
    radius = int(np.ceil(3 * sigma))

    # Precompute 1D gaussian values for efficiency
    r = torch.arange(-radius, radius + 1, dtype=torch.float32)
    gauss_1d = torch.exp(-r**2 / (2 * sigma**2))

    # Place gaussian at each point location
    for i in range(len(coords)):
        cz, cy, cx = coords[i]
        cz, cy, cx = int(round(cz.item())), int(round(cy.item())), int(round(cx.item()))

        # Compute bounds (clipped to volume)
        z0, z1 = max(0, cz - radius), min(shape[0], cz + radius + 1)
        y0, y1 = max(0, cy - radius), min(shape[1], cy + radius + 1)
        x0, x1 = max(0, cx - radius), min(shape[2], cx + radius + 1)

        if z0 >= z1 or y0 >= y1 or x0 >= x1:
            continue  # Point is outside the volume

        # Corresponding indices into gaussian kernel
        kz0, kz1 = z0 - (cz - radius), z1 - (cz - radius)
        ky0, ky1 = y0 - (cy - radius), y1 - (cy - radius)
        kx0, kx1 = x0 - (cx - radius), x1 - (cx - radius)

        # Compute local 3D gaussian via outer product of 1D gaussians
        local_gauss = gauss_1d[kz0:kz1, None, None] * gauss_1d[None, ky0:ky1, None] * gauss_1d[None, None, kx0:kx1]

        # Update with max (for overlapping gaussians)
        heatmap[z0:z1, y0:y1, x0:x1] = torch.maximum(
            heatmap[z0:z1, y0:y1, x0:x1],
            local_gauss
        )

    return heatmap


def compute_heatmap_targets(
    cond_direction: str,
    r_split: int, c_split: int,
    r_min_full: int, r_max_full: int,
    c_min_full: int, c_max_full: int,
    patch_seg,
    min_corner: np.ndarray,
    crop_size: tuple,
    step_size: int,
    step_count: int,
    sigma: float = 2.0,
    axis_1d: torch.Tensor = None,
) -> torch.Tensor:
    """
    Generate heatmap with gaussians at expected positions in the masked region.

    Samples a sparse grid of points with step_size spacing in both row and col.

    Args:
        cond_direction: One of "left", "right", "up", "down"
        r_split, c_split: Split boundary in UV grid
        r_min_full, r_max_full, c_min_full, c_max_full: Patch bounds in UV grid
        patch_seg: Tifxyz segment for indexing world coords
        min_corner: Crop origin in world coords (ZYX)
        crop_size: Output crop size tuple (D, H, W)
        step_size: Spacing between gaussians in UV grid units (both row and col)
        step_count: Number of steps to sample in the extrapolation direction
        sigma: Gaussian standard deviation
        axis_1d: Pre-computed axis tensor for efficiency (ignored, kept for API compat)

    Returns:
        (D, H, W) tensor with gaussians at expected positions
    """
    all_local_coords = []

    # Generate row indices with step_size spacing
    row_indices = list(range(r_min_full, r_max_full, step_size))

    if cond_direction == "left":
        for k in range(1, step_count + 1):
            col = c_split + k * step_size
            if col >= c_max_full:
                continue
            for row in row_indices:
                x, y, z, valid = patch_seg[row:row+1, col:col+1]
                if not valid.all():
                    continue
                world_zyx = np.array([z.item(), y.item(), x.item()])
                all_local_coords.append(world_zyx - min_corner)

    elif cond_direction == "right":
        for k in range(1, step_count + 1):
            col = c_split - k * step_size
            if col < c_min_full:
                continue
            for row in row_indices:
                x, y, z, valid = patch_seg[row:row+1, col:col+1]
                if not valid.all():
                    continue
                world_zyx = np.array([z.item(), y.item(), x.item()])
                all_local_coords.append(world_zyx - min_corner)

    elif cond_direction == "up":
        col_indices = list(range(c_min_full, c_max_full, step_size))
        for k in range(1, step_count + 1):
            row = r_split + k * step_size
            if row >= r_max_full:
                continue
            for col in col_indices:
                x, y, z, valid = patch_seg[row:row+1, col:col+1]
                if not valid.all():
                    continue
                world_zyx = np.array([z.item(), y.item(), x.item()])
                all_local_coords.append(world_zyx - min_corner)

    elif cond_direction == "down":
        col_indices = list(range(c_min_full, c_max_full, step_size))
        for k in range(1, step_count + 1):
            row = r_split - k * step_size
            if row < r_min_full:
                continue
            for col in col_indices:
                x, y, z, valid = patch_seg[row:row+1, col:col+1]
                if not valid.all():
                    continue
                world_zyx = np.array([z.item(), y.item(), x.item()])
                all_local_coords.append(world_zyx - min_corner)

    if not all_local_coords:
        print(f"[compute_heatmap_targets] No valid coords found for direction={cond_direction}")
        return None

    all_coords = np.stack(all_local_coords, axis=0)
    # Filter to in-bounds
    in_bounds = (
        (all_coords[:, 0] >= 0) & (all_coords[:, 0] < crop_size[0]) &
        (all_coords[:, 1] >= 0) & (all_coords[:, 1] < crop_size[1]) &
        (all_coords[:, 2] >= 0) & (all_coords[:, 2] < crop_size[2])
    )
    all_coords = all_coords[in_bounds]

    if len(all_coords) == 0:
        print(f"[compute_heatmap_targets] All coords out of bounds for direction={cond_direction}")
        return None

    coords_tensor = torch.from_numpy(all_coords).float()
    return make_gaussian_heatmap(coords_tensor, crop_size[0], sigma=sigma, axis_1d=axis_1d)
