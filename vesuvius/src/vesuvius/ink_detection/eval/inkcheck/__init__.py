"""inkcheck: does an ink-detection model see *known* ink at a given resolution?

Compare a predicted ink map against a published reference ink map of the same
segment (possibly rendered at a different voxel size), with registration,
band-pass correlation, pixel/stroke metrics and search-matched null tests.
"""
from .core import CheckConfig, run_check  # noqa: F401

__version__ = "0.1.0"
