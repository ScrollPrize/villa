# VC3D BBox Dependency Metadata For 3D Prefetch

Add a VC3D Python binding that returns the same per-chunk dependency metadata
as `collect_coords_dependencies`, but for a selected-level ZYX bbox directly.

The 3D fiber prefetcher should then pass its augmentation-envelope bbox to that
API instead of materializing representative coordinates or converting bbox
chunks in Python. Chunk conversion and persistent-cache path/remote metadata
must remain VC3D-owned.

Because the direct bbox API removes coordinate generation from 3D prefetch, the
intermediate `prefetch_sampler_device` knob is no longer meaningful and should
be removed rather than carried as dead configuration.
