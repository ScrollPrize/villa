# Native C++ Trace2CP Parity Fix Log

2026-07-29:

- Implemented the non-output parity fixes in `vc_fiber_tracer`: circular
  default angle-step cone candidates, legacy square-to-disk fallback for
  disabled angle steps, presence-ignored start branch selection,
  presence-weighted current branch selection, angle-squared normal-aware
  smoothness, cumulative tangent smoothness, interpolated target-plane
  crossing points, and spatial beam pruning after lookahead.
- Exposed C++ metric CLI flags for beam pruning, smoothness free angle,
  cumulative tangent smoothness, and legacy cone grid size.
- Added focused public-regression tests in `test_fiber_trace3d.cpp` for the
  default candidate count, crossing interpolation, start/current branch
  selection behavior, required normals, beam diversity pruning, invalid start
  samples, and the Python-style max-step guard calculation.
- Updated the native Trace2CP spec/docs summaries and changelog.
- Independent review found no blocking parity mismatch. It noted that the plan
  wording for degenerate tangent projections should match Python's current
  behavior; the plan was corrected to say valid-normal degenerate projections
  use the isotropic tangent angle inside split smoothness while retaining the
  normal/elevation component.
- Verified with:
  `cmake --build volume-cartographer/build --target test_fiber_trace3d vc_fiber_trace_metric -j 4`,
  `volume-cartographer/build/bin/test_fiber_trace3d`, and
  `ctest --test-dir volume-cartographer/build -R test_fiber_trace3d --output-on-failure`.
- Ran the provided whole-fiber C++ metric workload with absolute local paths:
  final default result was `native_trace2cp_fiber err/kvx=2.3 restarts=9 segments=87`,
  `trace_wall_s=135.020`, `trace_cpu_s=98.235`.
- Diagnostic rerun with `--cumulative-smoothness-tangent-weight 0` gave
  `err/kvx=2.0 restarts=8 segments=87`, `trace_wall_s=95.777`,
  `trace_cpu_s=95.397`. The default remains `2.0` because that is the
  Python/native spec default.
