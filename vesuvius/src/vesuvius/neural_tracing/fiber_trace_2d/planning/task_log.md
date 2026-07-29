# Native VC3D 3D Fiber Metric Runner Task Log

## Implemented

- Extended `vc_fiber_tracer` with shared `vc3d_fiber` JSON loading,
  `traceFiberOneWay`, and `traceWholeFiberMetric`.
- Kept CP matching strict: control points must be exact `line_points` entries,
  and the loader fails instead of guessing nearest points.
- Added `vc_fiber_trace_metric`, a Qt-free C++ CLI runner that opens a
  precomputed fiber inference `.lasagna.json`, loads one fiber JSON, runs
  one-sided full-fiber CP-to-CP tracing, and prints restart-rate metrics.
- Ported `beamLookaheadSteps` in the native core: each frontier expands the
  configured number of trace steps before pruning to `beamWidth`.
- Added focused C++ test coverage for the whole-fiber restart metric.
- Updated specs, code-structure docs, changelog, and current status.

## Validation

- `cmake --build volume-cartographer/build/ci-tests-clang-systemdeps --target test_fiber_trace3d`
  passed.
- `volume-cartographer/build/ci-tests-clang-systemdeps/bin/test_fiber_trace3d`
  passed: `2 test case(s) passed`.
- `cmake --build volume-cartographer/build/ci-tests-clang-systemdeps --target vc_fiber_trace_metric`
  passed.
- `cmake --build volume-cartographer/build/ci-tests-clang-systemdeps --target VC3D`
  passed. Existing unrelated Qt deprecation warnings were emitted in
  `InkDetectionOverlayController.cpp` and `CPointCollectionWidget.cpp`.
- `volume-cartographer/build/ci-tests-clang-systemdeps/bin/vc_fiber_trace_metric --help`
  passed.
- `git diff --check` passed.

## Deviations / Remaining Work

- The new CLI has not yet been run on the real Python reference benchmark
  fiber/inference data, so real-data parity is still unvalidated.
- Persisted tracer-optimized segment metadata/invalidation, protection from
  regular Lasagna reoptimization, and numeric GUI progress remain native GUI
  gaps from the earlier integration pass.
