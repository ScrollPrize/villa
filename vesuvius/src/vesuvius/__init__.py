"""Public entry point for the Vesuvius package."""

from . import data, install
from ._missing_extra import MissingExtra as _MissingExtra

# Always expose Volume; protect VCDataset because it depends on PyTorch.
from .data import Volume


try:
    from .data import VCDataset  # requires the 'models' extra (torch)
except Exception as exc:  # pragma: no cover - exercised only without the extra
    VCDataset = _MissingExtra("VCDataset", "models", exc)  # type: ignore

# Guard optional heavy modules.  They report which extra is missing when used.
try:
    from . import models  # heavy ML extras
except Exception as exc:  # pragma: no cover - exercised only without the extra
    models = _MissingExtra("models", "models", exc)  # type: ignore
try:
    from . import structure_tensor  # heavy segmentation extras
except Exception as exc:  # pragma: no cover - exercised only without the extra
    structure_tensor = _MissingExtra("structure_tensor", "models", exc)  # type: ignore
try:
    from . import tifxyz  # tifxyz format I/O (requires tifffile, scipy)
except Exception as exc:  # pragma: no cover - exercised only without the extra
    tifxyz = _MissingExtra("tifxyz", "render", exc)  # type: ignore

# utils reads packaged catalog YAML, which needs nothing beyond the base install.
# Its network paths import aiohttp and nest-asyncio lazily, so this import no longer
# fails on a volume-only install. The guard stays for anything else that could break.
try:
    from . import utils  # type: ignore
    from .utils import is_aws_ec2_instance, list_cubes, list_files, update_list  # type: ignore
except Exception as exc:  # pragma: no cover - defensive
    utils = _MissingExtra("utils", "all", exc)  # type: ignore
    is_aws_ec2_instance = _MissingExtra("is_aws_ec2_instance", "all", exc)  # type: ignore
    list_cubes = _MissingExtra("list_cubes", "all", exc)  # type: ignore
    list_files = _MissingExtra("list_files", "all", exc)  # type: ignore
    update_list = _MissingExtra("update_list", "all", exc)  # type: ignore

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