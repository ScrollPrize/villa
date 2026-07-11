# Remove Loader Serialization Bottlenecks

The current load-only benchmark is only a slight improvement and still leaves
CPU resources idle. Find what is serializing fiber-strip batch loading and fix
it.

Requirements:

- Reuse the established `--benchmark --load-only --profile` command for
  measurements.
- Preserve deterministic sample ordering.
- Keep using the shared `FiberStrip2DLoader.load_batch` path and VC3D
  coordinate sampler.
- Identify whether the bottleneck is Python-side queueing, strip-coordinate
  cache reads, line-coordinate transforms, VC3D sampling, or C++ cache locking.
- Remove any avoidable serial tail in `load_batch`.
- Report before/after throughput and CPU/thread factors.
