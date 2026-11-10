"""helirec: compact helical (extended-FOV) tomography reconstruction.

C++ core with CPU and CUDA kernels; this package exposes them through nanobind.
"""
from . import _core
from ._core import (cuda_available, built_with_cuda, release_pinned_memory, flatfield, backproject, FourierFilter, ghbp, ghbp_plan_info,
                    filter_types, filter_margin, filter_describe, apply_filters, roundfft, minus_log, rebalance, ramp_filter, ramp_geometry,
                    RawReader, ScanData, make_scan_data, SliceReconstructor, span_info, ProgressBar, OutputMask, ReferenceMask,
                    BlockCorrelator, RegisteredVolume, resample_affine, zarr_levels, zarr_read)
from .omezarr import to_uint8

__all__ = ["cuda_available", "built_with_cuda", "release_pinned_memory", "flatfield", "backproject", "FourierFilter", "ghbp", "ghbp_plan_info", "filter_types", "filter_margin", "filter_describe", "apply_filters",
           "roundfft", "minus_log", "rebalance", "ramp_filter", "ramp_geometry", "RawReader", "ScanData", "make_scan_data",
           "SliceReconstructor", "ProgressBar", "OutputMask", "ReferenceMask", "BlockCorrelator", "RegisteredVolume",
           "resample_affine", "zarr_levels", "zarr_read", "to_uint8", "devices"]
__version__ = "0.1.0"


def devices():
    """Devices usable in this build: ['cpu'] or ['cpu', 'cuda']."""
    return ["cpu", "cuda"] if cuda_available() else ["cpu"]
