Change the native 3D whole-fiber Trace2CP restart metric from restart fraction
per segment to restarts per reference-fiber length.

The primary stdout/summary metric should be restarts per 1000 selected-level
voxels (`kvx`), measured along the original loaded fiber line. If explicit
physical voxel-size metadata is available from the configured data volume or
dataset config, also report restarts per meter using the same original-line
reference.

Whole-fiber progress output should report the same physical-unit metric while
the trace is running, including an explicit `physical_unit=m` token whenever
the per-meter metric is available.
