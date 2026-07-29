# Require Lasagna Normals For Native Trace2CP

Native 3D Trace2CP must not silently run normal-aware tangent/normal smoothing
without Lasagna normals.

Desired behavior:

- Python native Trace2CP should fail if a normal-aware smoothing configuration
  reaches tracing with `normal_sampler=None`.
- Python lower-level scoring may keep generic isotropic code paths, but they
  must not be reachable from normal-aware native Trace2CP without a Lasagna
  sampler.
- `vc_fiber_trace_metric` must require an explicit `--normal-manifest` Lasagna
  manifest and fail before tracing if it is omitted.
- `vc_fiber_trace_metric` must not try to use normals from the fiber prediction
  manifest. We do not create those manifests with normal channels.
- The C++ fiber tracer core should reject normal-aware smoothing requests when
  no normal sampler is passed, so the CLI cannot accidentally fall back to
  isotropic smoothing through the library.

Out of scope for this task:

- do not remove lower-level isotropic/no-normal code that is useful for explicit
  non-normal-aware callers;
- do not change scale semantics or the default trace-control values.
