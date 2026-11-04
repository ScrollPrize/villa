"""Public entry point for the Vesuvius package."""

from . import data
from .data import Cube
from .utils import list_cubes

# Protect VCDataset because it depends on PyTorch.
try:
    from .data import VCDataset  # requires the 'models' extra (torch)
except Exception:
    VCDataset = None  # type: ignore

# Guard optional heavy modules.  They will be None unless their extras are installed.
try:
    from . import models  # heavy ML extras
except Exception:
    models = None  # type: ignore
try:
    from . import structure_tensor  # heavy segmentation extras
except Exception:
    structure_tensor = None  # type: ignore
# Keep the utilities namespace available alongside its public cube listing.
try:
    from . import utils  # type: ignore
except Exception:
    utils = None  # type: ignore

__all__ = [
    "Cube",
    "list_cubes",
    "VCDataset",
    "data",
    "utils",
    "models",
    "structure_tensor",
]
