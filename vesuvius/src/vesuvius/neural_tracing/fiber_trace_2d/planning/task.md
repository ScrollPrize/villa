# Native Fiber Trace VC3D Corner-Batch Sampling

Replace the native C++ fiber tracer's contended per-point Zarr/cache sampling
with VC3D's batched, threaded chunk reader. Keep one decoded cache per physical
prediction/Lasagna volume, fetch the eight nearest-neighbor voxel corners for
every candidate as a batch, and perform scalar and orientation-aware
interpolation in the tracer.

Convert the native fiber tracer's internal geometry, direction, loss, and beam
math to float. Numeric parity with the previous double implementation is no
longer required, but deterministic candidate ordering remains required and the
representative whole-fiber restart metric must be checked for quality
regressions.
