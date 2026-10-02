"""Expose the primary data classes for the vesuvius.data package."""

# Always import Volume; it only relies on the minimal dependencies.
from .volume import Volume

# VCDataset requires torch and other heavy ML packages, so guard its import.
try:
    from .vc_dataset import VCDataset  # type: ignore
except Exception as _vc_dataset_exc:  # pragma: no cover - depends on install extras
    from .._optional import requires_extra as _requires_extra

    VCDataset = _requires_extra("VCDataset", "models", _vc_dataset_exc)  # type: ignore

__all__ = ["Volume", "VCDataset"]