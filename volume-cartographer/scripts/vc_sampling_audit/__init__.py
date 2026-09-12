"""Report-only final-ray audit. Requires NumPy; no renderer or CT import."""
from .capture import read_capture, report_capture
from .slab import CONTRACT, report_grid

__all__ = ["CONTRACT", "read_capture", "report_capture", "report_grid"]
