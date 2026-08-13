"""Zarr task implementations.

This package contains all registered zarr task implementations.
Tasks are automatically registered when their modules are imported.
"""

# Import all task modules to trigger registration
from . import edt_dilate
from . import export_tifs
from . import merge
from . import recompress
from . import remap
from . import resize
from . import scale
from . import threshold
from . import transpose
from . import zero_range

__all__ = [
    "edt_dilate",
    "export_tifs",
    "merge",
    "recompress",
    "remap",
    "resize",
    "scale",
    "threshold",
    "transpose",
    "zero_range",
]
