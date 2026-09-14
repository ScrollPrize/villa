"""Public entry point for the Vesuvius package."""

from . import data, install
from ._optional import requires_extra as _requires_extra

# Volume needs only the core dependencies. VCDataset needs torch, and becomes
# a placeholder that names the missing extra when it is absent.
from .data import Volume, VCDataset

# Guard optional heavy modules.  They will be None unless their extras are installed.
try:
    from . import models  # heavy ML extras
except Exception:
    models = None  # type: ignore
try:
    from . import structure_tensor  # heavy segmentation extras
except Exception:
    structure_tensor = None  # type: ignore
try:
    from . import tifxyz  # tifxyz format I/O (requires tifffile, scipy)
except Exception:
    tifxyz = None  # type: ignore

# utils needs aiohttp and nest_asyncio, which a bare `pip install vesuvius` and
# the volume-only extra do not pull in.  Keep the module itself None so
# `if vesuvius.utils is None` still works, but give the callables a placeholder
# that explains what to install.
try:
    from . import utils  # type: ignore
    from .utils import is_aws_ec2_instance, list_cubes, list_files, update_list  # type: ignore
except Exception as _utils_exc:  # pragma: no cover - depends on install extras
    utils = None  # type: ignore
    is_aws_ec2_instance = _requires_extra("is_aws_ec2_instance", "all", _utils_exc)  # type: ignore
    list_cubes = _requires_extra("list_cubes", "all", _utils_exc)  # type: ignore
    list_files = _requires_extra("list_files", "all", _utils_exc)  # type: ignore
    update_list = _requires_extra("update_list", "all", _utils_exc)  # type: ignore

__all__ = [
    "Volume",
    "VCDataset",
    "data",
    "install",
    "utils",
    "models",
    "structure_tensor",
    "tifxyz",
    "is_aws_ec2_instance",
    "list_cubes",
    "list_files",
    "update_list",
]