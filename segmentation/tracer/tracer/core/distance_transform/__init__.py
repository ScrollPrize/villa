"""
Distance transform implementation for binary volumes.

This module provides functions for computing distance transforms of binary volumes,
which are used to guide surface optimization during tracing.

The distance transform maps each voxel to its distance from the nearest voxel that
meets a specified intensity threshold, typically marking object boundaries.
"""

from tracer.core.distance_transform.transform import (
    distance_transform,
    thresholded_distance,
    create_distance_field,
)

from tracer.core.distance_transform.chunked import (
    ChunkCache,
    ChunkedTensor,
    CachedChunked3dInterpolator,
)