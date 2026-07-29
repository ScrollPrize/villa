# Manifest-Scale Native Fiber Metric Task Log

## Implementation Notes

- Started from the current native metric CLI, which still exposed
  `--working-to-base-scale`.
- Added `inferFiberPredictionWorkingToBaseScale(...)` in `vc_fiber_tracer`.
  It reuses the same single-output and prefixed multi-output channel discovery
  convention as `FiberPredictionField`.
- The inferred scale is the common effective scale across all tracer-used
  prediction channels: `manifest.sourceToBase * group.scaleFactor()`.
- `vc_fiber_trace_metric` no longer exposes `--working-to-base-scale`; it opens
  the fiber inference manifest, infers the scale, sets the prediction dataset
  runtime scale to that value, and opens an optional `--normal-manifest` with
  the same value.
- Added unit coverage for single-output scale inference, prefixed multi-output
  scale inference, missing prediction channels, and mixed channel scales.

## Deviations / Deferred Items

- Independent agent review of `task_plan.md` was skipped because this session
  is proceeding directly in default execution mode; the plan was checked
  locally against the existing spec/docs before implementation.
- Fiber JSON coordinate conversion into manifest base coordinates remains out
  of scope. The JSON fiber is assumed to already be in the manifest base
  coordinate system.
- GUI native segment tracing scale/session handling was left unchanged because
  this task targets the command-line whole-fiber metric path.

## Validation

- `cmake -S volume-cartographer -B volume-cartographer/build`
  - passed; needed because the existing generated build tree did not yet expose
    the newer `vc_fiber_trace_metric` and `test_fiber_trace3d` targets.
- `cmake --build volume-cartographer/build --target test_fiber_trace3d`
  - first overlapping attempt raced with another Make build in the same tree;
    rerun serially passed.
- `cmake --build volume-cartographer/build --target vc_fiber_trace_metric`
  - passed.
- `volume-cartographer/build/bin/test_fiber_trace3d`
  - passed: 6 test cases.
- `volume-cartographer/build/bin/vc_fiber_trace_metric --help`
  - passed and no longer shows `--working-to-base-scale`; `--step-voxels` is
    documented as manifest prediction voxels.
- `volume-cartographer/build/bin/vc_fiber_trace_metric dummy.lasagna.json dummy_fiber.json --working-to-base-scale 4`
  - failed early as expected with `unknown option: --working-to-base-scale`.
- `cmake --build volume-cartographer/build --target test_lasagna_manifest`
  - passed.
- `volume-cartographer/build/bin/test_lasagna_manifest`
  - passed: 11 test cases.
- `git diff --check`
  - passed.
