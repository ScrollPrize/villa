# Native VC3D 3D Fiber Metric Runner Plan

## Scope

- Continue the native VC3D 3D fiber tracer implementation.
- Add a Qt-free C++ whole-fiber metric runner for preprocessed fiber inference
  datasets.
- Keep the CLI thin: parsing arguments, opening manifests, calling
  `vc_fiber_tracer`, and printing metrics only.
- Do not implement visualization, PyTorch inference, or whole-fiber GUI
  tracing in this task.

## Implementation

1. Extend the reusable tracer library
   - Add shared `FiberInput` / `loadFiberJson` helpers for `vc3d_fiber` JSON
     files.
   - Require exact CP-to-line-point matches so corrupt or transformed inputs
     fail loudly.
   - Add a public one-way trace API that accepts explicit start point, target
     point, initial direction, and target-plane normal.
   - Reuse the existing candidate scoring, branch handling, presence, and
     optional Lasagna normal smoothness path.

2. Add whole-fiber metric API
   - Add `traceWholeFiberMetric` to `vc_fiber_tracer`.
   - Convert JSON base coordinates into caller working coordinates by dividing
     by `working_to_base_scale`; this matches the Python selected-scale metric
     when the caller passes the dataset scale.
   - Iterate consecutive CP pairs in order.
   - For each segment, trace one-sided from the current traced point to the
     next CP target plane.
   - On success, continue from the plane crossing and terminal trace direction.
   - On failure, increment restart count and resume from the failed target CP
     using the reference line tangent toward the next CP.
   - Compute metric length from the original reference fiber line in working
     voxels and report restarts per 1000 working voxels.
   - If the caller passes a positive `voxel_size_um`, also report restarts per
     meter and the successfully traced reference length in millimeters.

3. Add CLI runner
   - Add `volume-cartographer/apps/src/vc_fiber_trace_metric.cpp`.
   - Command shape:
     `vc_fiber_trace_metric <fiber.lasagna.json> <fiber.json> [options]`
   - Options cover the Python benchmark defaults relevant to persisted
     inference:
     `--working-to-base-scale`, `--normal-manifest`, `--voxel-size-um`,
     `--step-voxels`, `--cone-angle-degrees`, `--cone-angle-step-degrees`,
     `--beam-width`, `--beam-lookahead-steps`,
     `--smoothness-weight`, `--smoothness-normal-weight`,
     `--smoothness-tangent-weight`, `--max-step-factor`,
     `--error-threshold-voxels`, `--cache-gib`, and `--quiet`.
   - Use the fiber manifest itself for normals only when it has
     `grad_mag`/`nx`/`ny`; otherwise allow an explicit `--normal-manifest`.
   - Print progress as an updating single carriage-return line and final
     metric lines similar to the Python tool.

4. Tests
   - Extend `test_fiber_trace3d` with a fake prediction-source whole-fiber
     metric test that proves zero restarts on a straight fiber and restart
     accounting on an invalid segment.
   - Keep the existing segment smoke test.

5. Build integration
   - Register the CLI in `volume-cartographer/apps/CMakeLists.txt`.
   - Rebuild `vc_fiber_tracer`, `test_fiber_trace3d`, the new CLI target, and
     `VC3D`.

## Spec update

- Add a native C++ metric runner spec describing inputs, no-inference behavior,
  one-sided whole-fiber metric semantics, coordinate scale handling, optional
  physical units, and progress output.
- Record that configured multi-step beam lookahead is part of the native core
  metric behavior.

## Docs updates

- Update `docs/code_structure.md` with the new CLI name, inputs, key options,
  and output metrics.

## Changelog

- Add a 2026-07-29 entry for the native C++ whole-fiber metric runner.

## Validation

- `cmake --build volume-cartographer/build/ci-tests-clang-systemdeps --target test_fiber_trace3d`
- `volume-cartographer/build/ci-tests-clang-systemdeps/bin/test_fiber_trace3d`
- `cmake --build volume-cartographer/build/ci-tests-clang-systemdeps --target vc_fiber_trace_metric`
- `cmake --build volume-cartographer/build/ci-tests-clang-systemdeps --target VC3D`
- `git diff --check`

## Known Gaps To Keep Visible

- Persisted tracer-optimized segment metadata/invalidation and regular
  optimizer protection remain unimplemented.
- GUI progress is still the existing busy state/log path, not a numeric
  progress overlay.
- Real-data parity against the Python benchmark command still needs to be run
  after the CLI is available.
