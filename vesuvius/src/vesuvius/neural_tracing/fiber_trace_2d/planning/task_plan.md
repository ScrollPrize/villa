# Plan: Manifest-Scale Native Fiber Metric

## Scope

- Change only the native VC C++ fiber metric/tracer command path needed for
  manifest-derived scale.
- Keep generic Lasagna dataset runtime scale support for existing callers.
- Keep JSON fiber coordinates interpreted as manifest-base coordinates.
- Keep remote manifest/cache support unchanged.
- Do not add a fiber coordinate scale argument yet.

## Implementation

1. Add a reusable manifest scale helper
   - Add a `vc_fiber_tracer` helper that discovers prediction options with the
     same channel naming rules as `FiberPredictionField`.
   - Accept both single-output `presence/nx/ny` manifests and prefixed
     multi-output manifests such as
     `option_000_presence/option_000_nx/option_000_ny`.
   - Compute each required channel's effective base scale as
     `manifest.sourceToBase * group.scaleFactor()`.
   - Require all persisted prediction channels used by the tracer to have the
     same finite positive effective scale; fail loudly on missing or mismatched
     channels.

2. Wire `vc_fiber_trace_metric`
   - Remove the `--working-to-base-scale` command-line option.
   - Open the fiber inference manifest normally, infer the working scale, then
     construct the prediction dataset with `manifest.workingToBaseScale` set to
     that inferred scale.
   - Open an optional `--normal-manifest` with the same inferred working scale.
   - Log the inferred scale and make the whole-fiber metric request use it.

3. Preserve fiber coordinate semantics
   - Keep `loadFiberJson(...)` unchanged: `line_points` and `control_points`
     are base-coordinate points.
   - Keep `traceWholeFiberMetric(...)` scaling base fiber points into working
     voxels by dividing by `request.workingToBaseScale`.
   - Do not add a fiber-to-manifest-base conversion argument in this task.

4. Tests
   - Add focused `test_fiber_trace3d` cases for inferred effective scale,
     missing required prediction channels, and mismatched channel scales.
   - Keep existing whole-fiber metric tests passing.

## Spec Update

- Update the native metric spec to say `vc_fiber_trace_metric` derives the
  tracer working scale from persisted fiber prediction channel scales, not from
  a `--working-to-base-scale` argument.
- Explicitly state that JSON fibers are assumed to already be in the manifest
  base coordinate system.
- Document the fail-fast behavior for missing or scale-mismatched prediction
  channels.

## Docs Update

- Update `docs/code_structure.md` for the manifest-derived tracer scale and
  remove `--working-to-base-scale` guidance from the metric command.

## Changelog

- Add a short 2026-07-29 entry for manifest-derived native fiber metric scale.

## Validation

- `cmake --build volume-cartographer/build --target test_fiber_trace3d`
- `cmake --build volume-cartographer/build --target vc_fiber_trace_metric`
- `volume-cartographer/build/bin/vc_fiber_trace_metric --help`
- `volume-cartographer/build/bin/test_fiber_trace3d`
- `git diff --check`

## Deferred Explicitly

- Fiber JSON coordinate conversion into the manifest base coordinate system.
- Changing GUI/native segment tracing scale/session semantics.
