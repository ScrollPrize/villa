"""Public entry point for the Vesuvius package."""

from . import data, install, utils

# Always expose Volume; protect VCDataset because it depends on PyTorch.
from .data import Volume
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

# Re-export helper functions
from .utils import is_aws_ec2_instance, list_cubes, list_files, update_list

__all__ = [
    "Volume",
    "VCDataset",
    "data",
    "install",
    "utils",
    "models",
    "structure_tensor",
    "is_aws_ec2_instance",
    "list_cubes",
    "list_files",
    "update_list",
]
