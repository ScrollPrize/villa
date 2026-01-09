"""TifXYZ format I/O for the Vesuvius project.

The tifxyz format stores 3D surface point clouds in a directory structure::

    segment_name/
        x.tif      - X coordinates (32-bit float)
        y.tif      - Y coordinates (32-bit float)
        z.tif      - Z coordinates (32-bit float)
        meta.json  - Metadata (scale, bbox, uuid, etc.)
        mask.tif   - Optional validity mask

Example
-------
>>> from vesuvius.tifxyz import read_tifxyz, write_tifxyz
>>>
>>> # Read a tifxyz surface
>>> surface = read_tifxyz("/path/to/segment")
>>>
>>> # Get full resolution dimensions
>>> print(f"Shape: {surface.shape}")  # e.g., (84300, 87460)
>>>
>>> # Access coordinates at any position (interpolated)
>>> x, y, z, valid = surface[2000, 4000]           # Single point
>>> x, y, z, valid = surface[1000:1100, 2000:2100] # 100x100 tile
>>>
>>> # Write a surface
>>> write_tifxyz("/path/to/output", surface, overwrite=True)
"""

from .reader import TifxyzInfo, TifxyzReader, list_tifxyz, load_folder, read_tifxyz
from .types import Tifxyz
from .upsampling import (
    catmull_rom_smooth_1d,
    compute_grid_bounds,
    interpolate_at_points,
    upsample_coordinates,
)
from .writer import TifxyzWriter, write_tifxyz

__all__ = [
    # Main class
    "Tifxyz",
    # Reader
    "read_tifxyz",
    "TifxyzReader",
    # Discovery
    "list_tifxyz",
    "load_folder",
    "TifxyzInfo",
    # Writer
    "write_tifxyz",
    "TifxyzWriter",
    # Upsampling utilities
    "upsample_coordinates",
    "interpolate_at_points",
    "compute_grid_bounds",
    "catmull_rom_smooth_1d",
]
