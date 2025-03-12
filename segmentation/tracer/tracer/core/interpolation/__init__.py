"""
Interpolation utilities for the tracer library.

This module provides classes for interpolating values from 3D volumes.
The TrilinearInterpolator is an autodiff-compatible implementation that
preserves gradient chains for use with Theseus optimization.
"""

from .trilinear_interpolator_autodiff import TrilinearInterpolatorAutoDiff

# Use the autodiff implementation for all usage
TrilinearInterpolator = TrilinearInterpolatorAutoDiff

__all__ = ["TrilinearInterpolator", "TrilinearInterpolatorAutoDiff"]
