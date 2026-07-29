# Plan: Native 3D Trace2CP Target Plane Normals

## Context

The current Python and C++ native 3D Trace2CP whole-fiber paths use the
straight CP-to-CP chord as the target-plane normal. That is not the intended
fiber-local stopping geometry. The trace should instead stop only after
crossing all three target-local planes and score the best CP in-plane error
among those crossings.

## Implementation

1. Shared plane model
   - Add a small target-plane representation in the Python tracer containing:
     name, target point, unit normal, crossed flag, and crossing point.
   - Add the equivalent representation to `vc_fiber_tracer` in C++.
   - Ensure every normal is normalized, finite, and rejected loudly if
     degenerate.

2. Line-neighbor target normals
   - For a target CP, derive line-neighbor normals from the loaded fiber line
     point indices, not from CP-to-CP chords.
   - `next` normal: vector from target CP line point to the next line point when
     available.
   - `prev` normal: vector from target CP line point to the previous line point
     when available.
   - If the target CP is at a line endpoint, only the available neighbor normal
     is used; if neither exists, fail because the target-plane geometry is
     invalid.
   - Preserve coordinate-scale conventions: normals are in selected-scale ZYX
     for Python live tracing and in the C++ tracer working coordinate system
     for VC3D.

3. Inference-direction target normal
   - Sample the fiber prediction direction at the target CP using the same field
     sampling and ambiguous-direction alignment used by trace scoring.
   - Align the sampled direction sign with the local reference fiber tangent at
     the target CP so the plane normal orientation is stable. Plane crossing
     itself should still work regardless of sign.
   - If the sampled target direction is invalid, fail loudly for this tracing
     run rather than silently falling back to the chord.

4. One-way trace termination
   - Replace single-plane crossing state with three-plane crossing state.
   - At each accepted trace step, test the segment from current point to next
     point against every not-yet-crossed target plane and store the interpolated
     crossing point when crossed.
   - Continue stepping until all configured target planes are crossed.
   - If max steps, invalid current point, invalid candidates, or other existing
     failures occur before all planes are crossed, return a visible failure
     reason that names the missing planes.

5. Endpoint error and success
   - Compute in-plane error for each crossed target plane using that plane's
     own normal and crossing point.
   - Select the smallest finite error as the segment endpoint error.
   - Store/report the selected plane name and crossing point in Python segment
     summaries and C++ result structs.
   - Whole-fiber restart logic uses the selected/best error with the existing
     threshold.

6. Visualization
   - Keep failed overlay trimming behavior unchanged for now so long strips do
     not overlap restart CP regions.
   - Add enough summary/debug metadata to explain which plane produced the
     accepted/restart error and where the selected crossing was.
   - Do not introduce new rows or visual clutter in this task unless needed for
     debugging after implementation.

7. Python/C++ parity
   - Update both:
     - `vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/trace2cp_tool.py`
     - `volume-cartographer/core/src/fiber_tracer/FiberTrace.cpp`
   - Keep CLI defaults and existing benchmark commands unchanged.
   - Update native metric output only if necessary to expose selected-plane
     debug fields; the public `err/kvx` and `err/m` labels remain unchanged.

## Spec Update

- Replace the current chord-normal target-plane spec with the three-plane
  target-local rule.
- State explicitly that CP-to-CP chord normals are not valid Trace2CP
  termination planes.
- Document that success/error uses the smallest in-plane error after all
  configured target planes have been crossed.
- Document failure behavior when one or more required planes are not crossed.

## Docs Updates

- Update `docs/code_structure.md` in the native 3D Trace2CP section to describe
  the three target planes and the selected best-error reporting.
- Update the C++ native metric section if public summary/debug fields change.

## Tests

1. Python unit tests
   - Add a synthetic one-way trace test where chord-plane crossing would stop
     early but one of the target-local planes is not crossed yet.
   - Add a test that all target planes are crossed and the selected error is
     the minimum across plane-specific crossing points.
   - Add endpoint-line-point cases for CPs at the beginning/end of a fiber line.

2. C++ unit tests
   - Add equivalent target-plane tests in `test_fiber_trace3d`.
   - Verify whole-fiber restart success uses the selected best plane error.

3. Regression validation
   - Run the Python 3D neural tracing tests:
     `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_3d.py`
   - Build and run the C++ fiber tracer tests:
     `cmake --build volume-cartographer/build/ci-tests-clang-systemdeps --target test_fiber_trace3d`
     `volume-cartographer/build/ci-tests-clang-systemdeps/bin/test_fiber_trace3d`
   - Run `git diff --check`.

## Changelog

- Add a 2026-07-29 entry noting that native 3D Trace2CP target-plane
  termination no longer uses CP chords and now uses target-local line-neighbor
  and inferred-direction planes.

## Review Notes

- This plan keeps trace scoring/search unchanged; only target-plane
  termination/error selection changes.
- No fallback to chord normals is allowed.
- No visualization redesign is planned beyond summary/debug metadata.
