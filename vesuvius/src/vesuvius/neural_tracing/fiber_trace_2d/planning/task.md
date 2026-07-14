# NML Fiber Loading With Affine Volume Transforms

Extend the fiber trace 2D dataset loader so dataset entries can load Knossos /
WebKnossos `.nml` fiber annotations such as
`fibers_s1a_00497z_01497y_03997x_256_v00.nml`.

The NML coordinates may be in an older source scan frame. Dataset entries must
be able to specify the same affine volume transform convention already used by
Lasagna cross-volume data loading so the parsed fiber coordinates are mapped
into the current Lasagna/base-volume coordinate frame before normal sampling,
strip construction, prefetch, training, and Trace2CP tooling use them.

Keep the existing VC3D JSON fiber path working unchanged.
