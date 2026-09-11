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
try:
    from . import tifxyz  # tifxyz format I/O (requires tifffile, scipy)
except Exception:
    tifxyz = None  # type: ignore

# Optional dependencies of ``vesuvius.utils``. They are not core dependencies, so
# the plain ``uv sync`` that CONTRIBUTING.md tells contributors to run leaves them
# out, and the catalog helpers below cannot be imported.
_CATALOG_REQUIREMENTS = ("aiohttp", "nest_asyncio")

# Package name on PyPI, where it differs from the import name.
_CATALOG_DISTRIBUTIONS = {"aiohttp": "aiohttp", "nest_asyncio": "nest-asyncio"}


def _needs_catalog_extra(name, missing, cause):
    """Build a stand-in for a catalog helper whose dependencies are missing.

    Binding the helpers to ``None`` made the failure surface at the call site as
    ``TypeError: 'NoneType' object is not callable``, which says nothing about the
    real cause.  Keeping them callable lets the failure name the missing package
    and a command that actually installs it.
    """

    distribution = _CATALOG_DISTRIBUTIONS.get(missing, missing)

    def _stub(*_args, **_kwargs):
        raise ImportError(
            f"vesuvius.{name}() needs the optional dependency {missing!r}, "
            f"which is not installed. Install it with "
            f"`uv pip install {distribution}` (or `pip install {distribution}`). "
            f"Original import error: {cause}"
        ) from cause

    _stub.__name__ = name
    _stub.__qualname__ = name
    _stub.__doc__ = (
        f"Unavailable: vesuvius.{name}() requires the optional dependency "
        f"{missing!r}. Calling it raises ImportError."
    )
    return _stub


# ``utils`` pulls in the packages above.  When one of them is genuinely absent the
# catalog helpers become stubs that name it.  Any other import failure - a syntax
# error, a version clash, a bug inside utils - is re-raised untouched, so real
# breakage is never reported as a missing dependency.
try:
    from . import utils  # type: ignore
    from .utils import is_aws_ec2_instance, list_cubes, list_files, update_list  # type: ignore
except ModuleNotFoundError as _catalog_import_error:
    _missing = _catalog_import_error.name
    if _missing not in _CATALOG_REQUIREMENTS:
        raise
    utils = None  # type: ignore
    is_aws_ec2_instance = _needs_catalog_extra("is_aws_ec2_instance", _missing, _catalog_import_error)  # type: ignore
    list_cubes = _needs_catalog_extra("list_cubes", _missing, _catalog_import_error)  # type: ignore
    list_files = _needs_catalog_extra("list_files", _missing, _catalog_import_error)  # type: ignore
    update_list = _needs_catalog_extra("update_list", _missing, _catalog_import_error)  # type: ignore

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