# Per-Segment Interpolation Goals And Cubic-Spline Fallback Task Log

## Implementation

- Added strict version-3 CP-owned descriptors with `interp_goal`, actual
  `interp_mode`, optional mode-dependent `metric`, compact `msg`, and retained
  trace/Lasagna diagnostics. VC3D, Atlas, the native readers/probe, Python, and
  merge validation accept v1/v2/v3; VC3D saves v3.
- Added the shared `vc::lasagna::interpolateLineControlPoints` core helper.
  Connected cubic spans use exact CPs, shared internal tangents, optional hard
  boundary directions, bounded handles, deterministic shape checks, and base
  spacing resampling without consulting normals or predictions.
- Reworked the fiber coordinator around per-span goal resolution. Global spans
  below 100 base voxels select cubic spline; trace falls through to Lasagna and
  Lasagna candidate failure demotes only that span to cubic spline. Protected
  trace/cubic/manual spans feed hard endpoint directions to the existing Ceres
  solve.
- CP edits now dirty adjacent spans, goal changes dirty the selected span, and
  global-mode changes dirty global goals. Dirty cubic spans expand through the
  connected run; unrelated explicit spans stay protected.
- Replaced the old one-span trace/revert GUI path with a checked interpolation
  goal submenu. Removed its unreachable controller worker and task-result
  plumbing.
- Persisted labels now show actual mode (`C`, `L`, or `T`), metric, and message.
  Partially visible spans remain labeled; viewport-space packing clamps and
  pushes labels and uses a deterministic second row when needed.

## Plan Decisions And Deviations

- The plan proposed exporting the private Lasagna span initializer. No copy was
  made. The coordinator instead calls the already public shared full
  reinitializer, consumes its precise failed-span index, demotes that one span,
  and retries. This preserves the single established rollout implementation
  and its final joint solve with a smaller API change.
- The cubic geometry helper is normal-independent. VC3D line annotation still
  opens its selected regular Lasagna dataset for surrounding line models,
  Lasagna fallbacks, and extrapolation; this task does not create a separate
  manifest-free line-annotation workflow.
- Core/generated-view tests cover schema, goal resolution, trace fallback,
  spline geometry, ownership mutation, and display descriptor generation. The
  Qt viewport collision layout is build-verified but does not have a dedicated
  event-driven GUI test harness in the current suite.

## Validation

- `cmake --build volume-cartographer/build/ci-tests-clang-systemdeps --target test_line_annotation_generated_views -j32`
- `volume-cartographer/build/ci-tests-clang-systemdeps/bin/test_line_annotation_generated_views`
  passed 55 cases.
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace.py volume-cartographer/scripts/tests/test_fiber_merge.py`
  passed 109 cases.
- `cmake --build volume-cartographer/build --target VC3D -j32` completed.
