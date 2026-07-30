# Task Log: Native Fiber Trace Meeting Search And Persisted Diagnostics

## 2026-07-30 - Discovery

- CP-pair tracing currently supplies target-local planes but does not pass the
  configured 20-base-voxel threshold into either one-way request. Each trace
  can therefore stop as soon as all planes have been crossed even when its
  selected in-plane crossing is too far from the target CP; pair acceptance
  then rejects it without using the remaining step budget.
- The one-way result can snap its returned final sample to a selected crossing.
  Retaining complete exhausted paths is required for the proposed post-trace
  moving-plane search.
- Current C++ fusion selects discrete forward/reverse points using
  `gap_factor * gap + traveled_arcs` and inserts a midpoint. The Python
  reference additionally cuts at the selected meeting, warps both partial
  traces to their midpoint by arc-length fraction, concatenates, and
  arc-length-resamples.
- `segment_to_next` currently means accepted/protected native geometry whenever
  present. Persisting failed native attempts in the same CP-owned object
  requires an explicit outcome and replacement of all presence-only protection
  checks.
- Generated strip labels currently contain only asynchronous Lasagna
  normal-alignment state. Native diagnostics must be sourced directly from the
  session's CP metadata so they remain visible after reload.

## Planned Contract

- Search endpoint-plane and symmetric moving-plane meetings, choose the
  smallest raw in-plane/3D gap, and accept at a default maximum ratio of 0.10
  against the selected combined partial trace length.
- Store accepted and fallback native outcomes in the owning CP's single
  `segment_to_next` record; only accepted outcomes protect geometry.

## Deviations

- None at plan creation.

## Independent Plan Review

- Added a deterministic old-versus-new fallback corpus because the requested
  reduction cannot be established by builds and isolated geometry tests alone.
- Defined explicit outcome lifecycle transitions so persisted fallback
  diagnostics cannot silently outlive changed geometry.
- Added stable combined-result codes and precedence separate from verbose
  one-way and exception detail, which is necessary for strict persistence and
  compact GUI labels.
- Defined previous-schema handling: accepted-only records remain accepted,
  their obsolete gap factor is ignored rather than converted, and new writers
  persist the validated 10% ratio setting.
- Added a non-unit scale and JSON round-trip regression for raw base-voxel
  error and the scale-independent ratio.

## Implementation

- Passed the base-to-trace converted endpoint threshold into both one-way
  CP-pair requests and disabled selected-crossing endpoint snapping so exhausted
  paths remain available to fusion.
- Replaced gap-weighted discrete pair selection in the shared C++ tracer with
  deterministic half-step resampling, symmetric local tangent-plane
  intersections, qualifying endpoint-plane candidates, minimum raw-error
  selection, and a default 0.10 meeting-error ratio limit.
- Ported the Python arc-length fusion geometry: cut at the interpolated meeting,
  warp both partial traces to their midpoint by cumulative arc fraction,
  concatenate, resample at the configured step, and restore exact CP endpoints.
- Added trace/base meeting error, ratio, combined partial traced length, source,
  stable result code, and detail to the segment result.
- Versioned `segment_to_next` outcomes as `accepted_native` or
  `lasagna_fallback`. Only accepted outcomes protect geometry or expose the
  revert action; fallback outcomes retain diagnostics and are retried by later
  native optimization.
- Persisted direct-action and mixed-mode failures, updated strict C++, Python,
  and merge readers, and made generated-strip labels prefer persisted native
  meeting/failure diagnostics over Lasagna normal alignment.
- Updated the contract, code-structure notes, VC3D fiber documentation, and
  changelog.

## Validation

- `cmake --build volume-cartographer/build/ci-tests-clang-systemdeps --target test_fiber_trace3d test_line_annotation_generated_views test_lasagna_line_optimizer -j32`
- `test_fiber_trace3d`: 41 passed.
- `test_line_annotation_generated_views`: 49 passed.
- `test_lasagna_line_optimizer`: 35 passed.
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace.py`: 52 passed.
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q volume-cartographer/scripts/tests/test_fiber_merge.py`: 57 passed.
- `cmake --build volume-cartographer/build --target VC3D vc_fiber_trace_metric -j32`: both production targets built successfully.
- The broader `test_atlas` binary still reports three unrelated fixture
  failures because legacy fixtures reference flat CZYX Lasagna channels that
  VC3D intentionally no longer accepts. Its target builds successfully; no
  Atlas source or fixture was changed for this task.
- `git diff --check` passed. The final schema audit confirmed that current
  writers contain no obsolete gap-factor field and aligned accepted/fallback
  consistency checks across VC3D, core C++, Python, and merge readers.
