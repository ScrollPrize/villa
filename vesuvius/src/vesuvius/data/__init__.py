"""Expose the primary data classes for the vesuvius.data package."""

# Always import Volume; it only relies on the minimal dependencies.
from .volume import Volume

# VCDataset requires torch and other heavy ML packages, so guard its import.
# The placeholder reports the missing module and the extra that provides it when
# used, rather than surfacing as a bare None at the caller's call site.
try:
    from .vc_dataset import VCDataset  # type: ignore
except Exception as exc:  # pragma: no cover - exercised only without the extra
    from .._missing_extra import MissingExtra

    VCDataset = MissingExtra("data.VCDataset", "models", exc)  # type: ignore

__all__ = ["Volume", "VCDataset"]
