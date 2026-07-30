# Path-Based Lasagna Volumes And Base-Space Fiber Tracing

## Planning

- [x] Read the user request and repository instructions.
- [x] Re-read the fiber planning process and current active specifications.
- [x] Trace the encoded project identities to their cache-layout origin.
- [x] Inspect project attachment, reconciliation, detach, and serialization.
- [x] Inspect GUI line, trace, prediction, and normal-sampler coordinate paths.
- [x] Replace `planning/task.md` with the current task.
- [x] Replace `planning/task_plan.md` with implementation, removal, tests,
  spec, docs, changelog, and risks.
- [x] Perform a direct consistency review against current code/specs.
- [ ] Obtain independent agent review (not permitted by the active
  no-delegation runtime policy unless explicitly requested by the user).
- [x] Obtain user approval to begin implementation.

## Implementation

- [x] Delete the encoded URL identity helper, layout, sidecar field, tests,
  and documentation without backward compatibility.
- [x] Implement the readable remote source-path cache layout.
- [x] Add authoritative human-readable group source-location resolution.
- [x] Replace synthetic derived-volume locations and encoded provenance with
  actual local/remote paths.
- [x] Remove previous-format handling without a decoder or migration path.
- [x] Update actual-location deduplication, ownership, reconciliation, and
  detach behavior.
- [x] Add a shared base/trace coordinate adapter.
- [x] Prepare distinct trace-scale prediction and normal samplers.
- [x] Convert GUI segment inputs to trace space and accepted results to base
  space.
- [x] Correct base/trace coordinate units and preserve exact endpoints.

## Validation And Documentation

- [x] Add and run focused local/remote path and cache-layout tests.
- [x] Add and run scale-conversion, sampler-space, physical-unit, and splice
  tests.
- [x] Build focused core, CLI, and VC3D targets.
- [x] Run broader applicable regression tests.
- [ ] Perform the manual VC3D base-volume segment smoke test (requires an
  interactive GUI session and the user's local project).
- [x] Update `planning/specs.md`.
- [x] Update project/cache/code-structure documentation.
- [x] Update `planning/changelog.md`.
- [x] Replace `planning/task_log.md` with final commands, results, findings,
  and deviations.
- [x] Report limitations or skipped requirements explicitly.

## Base-Voxel Threshold Follow-Up

- [x] Confirm the C++ CLI, Python CLI, and VC3D working coordinate spaces.
- [x] Update the task and implementation plan to supersede the physical
  acceptance threshold.
- [x] Use a `20` base-voxel threshold in the shared C++ tracer and metric CLI.
- [x] Make VC3D physical voxel metadata optional and report-only.
- [x] Use a `20` base-voxel threshold in the Python native CLI.
- [x] Add focused scale and missing-physical-metadata regression tests.
- [x] Build with `-j32`, run focused C++/Python tests, and audit stale names.
- [x] Update specs, docs, changelog, and the current task log.
