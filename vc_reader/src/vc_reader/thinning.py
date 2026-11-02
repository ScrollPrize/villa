"""Binary skeletons and ordered distance-ridge traces on CPU or CUDA.

``skeletonize`` accepts a contiguous uint8 array with two or more dimensions;
leading dimensions batch independent YX masks (this is not 3D skeletonization).
Any nonzero is foreground and output is 0/255. ``method`` selects ``zhang_suen``,
``guo_hall``, or ``zhang_suen_guo_hall``. ``preserve_border=False`` treats pixels
outside each mask as background; the default preserves outer pixels.
``skeletonize_into`` also accepts CUDA arrays with a matching preallocated output
and ``backend="cuda"``. Finish external device writes before calling.

``trace_ridges`` accepts one host uint8 YX mask and returns visit-index pixels.
With ``return_traces=True`` it returns (image, list of XY int32 polylines).
``min_length`` prunes short polylines and rasterizes the survivors as 255.
A ``TraceWorkspace`` reuses CPU scratch buffers; use one per thread. CUDA performs
distance, seeds, and tracing on device, parallel across disconnected foreground
components. ``trace_ridges_into`` accepts resident CUDA input/output.

``manhattan_distance`` computes exact 2D/3D distance to an in-grid zero, clipped
at 255 (also the result if no zero exists), on either backend. For Euclidean,
signed, and normal-field distances use ``transforms.DistanceTransform``.
"""
from ._thinning import (TraceWorkspace, skeletonize, skeletonize_into,
                        trace_ridges, trace_ridges_into, manhattan_distance, connected_components)

__all__ = ["TraceWorkspace", "skeletonize", "skeletonize_into", "trace_ridges",
           "manhattan_distance", "trace_ridges_into", "connected_components"]
