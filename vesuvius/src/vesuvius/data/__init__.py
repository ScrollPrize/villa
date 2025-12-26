"""Expose the primary data classes for the vesuvius.data package."""

# Always import Volume; it only relies on the minimal dependencies.
from .volume import Volume

# VCDataset requires torch and other heavy ML packages, so guard its import.
try:
    from .vc_dataset import VCDataset  # type: ignore
except Exception:
    VCDataset = None  # type: ignore

__all__ = ["Volume", "VCDataset"]