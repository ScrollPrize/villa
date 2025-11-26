"""Public entry point for the Vesuvius package."""

from . import data, install

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

# Attempt to import utils.  utils requires aiohttp and nest_asyncio; if they aren't
# installed (e.g. when you install with [volume-only] or without extras), we silently
# fall back to None.  Users who need catalog utilities should install the appropriate
# optional extra.
try:
    from . import utils  # type: ignore
    from .utils import is_aws_ec2_instance, list_cubes, list_files, update_list  # type: ignore
except Exception:
    utils = None  # type: ignore
    is_aws_ec2_instance = None  # type: ignore
    list_cubes = None  # type: ignore
    list_files = None  # type: ignore
    update_list = None  # type: ignore

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