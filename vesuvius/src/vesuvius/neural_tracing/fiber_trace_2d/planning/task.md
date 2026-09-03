# Task: Staged filtering for crop tracing

Allow `vc_fiber_trace_chunk trace` to apply the existing ordered, transient
Fiberlet reduction stages before materializing the crop graph. The requested
1024-base-voxel crop must support the 256-aligned, 256-half-offset, and final
512-half-offset schedule so the final 512 boxes straddle the crop boundary.
Stage planning must include all preceding-stage and endpoint-reach support,
must read sparse input chunks as empty, and must not modify the source Zarr.
