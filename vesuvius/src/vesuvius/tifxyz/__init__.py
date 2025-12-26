"""TifXYZ format I/O for the Vesuvius project.

The tifxyz format stores 3D surface point clouds in a directory structure::

    segment_name/
        x.tif      - X coordinates (32-bit float)
        y.tif      - Y coordinates (32-bit float)
        z.tif      - Z coordinates (32-bit float)
        meta.json  - Metadata (scale, bbox, uuid, etc.)
        mask.tif   - Optional validity mask

Data is typically stored at 20x downsampling (scale=[20.0, 20.0]).
The scale indicates how many voxels each grid point represents.

Example
-------
>>> from vesuvius.tifxyz import Tifxyz, read_tifxyz, write_tifxyz
>>>
>>> # Read a tifxyz surface
>>> surface = read_tifxyz("/path/to/segment")
>>>
>>> # Get stored grid coordinates
>>> print(f"Grid shape: {surface.shape}")  # e.g., (500, 500)
>>> print(f"Scale: {surface.scale}")  # (20.0, 20.0)
>>>
>>> # Upsample to full resolution
>>> full_res = surface.upsample(target_scale=1.0)
>>> print(f"Full resolution shape: {full_res.shape}")  # (10000, 10000)
>>>
>>> # Query points at specific nominal locations
>>> x, y, z, valid = surface.get_points_at_nominal(
...     nominal_y=np.array([100, 200, 300]),
...     nominal_x=np.array([150, 250, 350])
... )
>>>
>>> # Write a surface
>>> write_tifxyz("/path/to/output", surface, overwrite=True)
"""

from .reader import TifxyzReader, read_tifxyz
from .types import Tifxyz
from .upsampling import compute_grid_bounds, interpolate_at_points, upsample_coordinates
from .writer import TifxyzWriter, write_tifxyz

__all__ = [
    # Main class
    "Tifxyz",
    # Reader
    "read_tifxyz",
    "TifxyzReader",
    # Writer
    "write_tifxyz",
    "TifxyzWriter",
    # Upsampling utilities
    "upsample_coordinates",
    "interpolate_at_points",
    "compute_grid_bounds",
]
