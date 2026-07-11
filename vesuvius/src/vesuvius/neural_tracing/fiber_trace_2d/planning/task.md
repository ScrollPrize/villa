# Whole-Batch Loader Parallelization

Fix fiber-trace 2D loader parallelism so `--load-only --profile` can exercise
multiple independent training batches in flight, not just CP workers inside one
batch.

Requirements:

- Use a bounded whole-batch queue/future pipeline for load-only benchmarks.
- Keep deterministic sample order: batches may load out of order internally but
  must be consumed by step number.
- Use the existing `FiberStrip2DLoader.load_batch` path; do not introduce a
  second sampling/cache implementation.
- Do not make worker loaders fully independent by default: share the base
  loader's parsed fibers, Lasagna channels, zarr handles, deterministic
  sample-order cache, and VC3D sampler/cache unless isolated samplers are
  explicitly requested.
- Keep optional isolated VC3D sampler support, but expose a per-sampler VC3D
  cache budget so duplicated samplers do not multiply the default cache
  footprint uncontrollably.
- Default intended whole-batch parallelism should allow eight concurrent batch
  loads with a deeper queue.
- Profile output must make the actual process CPU factor and wall throughput
  visible so the speedup can be checked against system CPU/GPU monitors.
- Reuse the established load-only benchmark command for validation.
