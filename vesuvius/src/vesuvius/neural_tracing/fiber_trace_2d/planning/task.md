Trim `fiber_trace_3d/configs/metric_sd2_s1.json` to contain only fields needed
to run the native 3D Trace2CP metric.

This metric config only needs to support JSON fibers supplied via the
`--fiber-json` CLI argument. The config should not carry its own fiber glob or
fiber path list, and should not carry NML training datasets, NML affine
transform settings, augmentation/training loss settings, prefetch settings, or
duplicate train/test dataset blocks.
