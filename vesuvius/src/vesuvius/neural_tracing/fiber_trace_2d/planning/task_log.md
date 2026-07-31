# CI Repairs For 3D Lasagna And Fiber Inference

## Findings

- The Atlas pred-snap test fixture still wrote the removed packed CZYX Lasagna
  representation, so strict 3D channel binding failed before Atlas assertions.
- Fiber inference used floor-sized tensor geometry as the bound for ceil-sized
  OME pyramid arrays. Odd edge planes could consequently remain unwritten.
- Independently owned project volume entries skipped both prepared-volume
  installation and compatibility checks, allowing manifest spacing to be
  silently replaced by incompatible ordinary-volume metadata.

## Plan Review

- The plan preserves strict independent 3D channel arrays and does not restore
  legacy packed-array handling.
- Tensor downsampling remains unchanged; only persisted storage bounds receive
  ceil endpoint semantics.
- Runtime UUID equality is intentionally excluded because prepared Lasagna
  wrappers and ordinary attachments use different identities for one source.
- Independent review was not used because delegation is prohibited unless the
  user explicitly requests subagents; the plan was reviewed locally instead.

## Implementation

- Replaced the Atlas pred-snap fixture's packed four-channel CZYX array with
  separate 3D `grad_mag`, `nx`, `ny`, and `pred_dt` arrays/groups.
- Added Fiber inference storage-bound ceil division for full level shapes and
  absolute exclusive region endpoints without changing shared floor-sized
  tensor downsampling.
- Added independently owned volume lookup and compatibility validation before
  reuse. Validation covers geometry, dtype, fill, scale levels/chunks, and
  manifest spacing while intentionally allowing distinct UUIDs.
- Added regressions for a 257-voxel edge crossing a 64-voxel output chunk,
  compatible distinct-UUID reuse, incompatible-spacing rollback, and the
  repaired Atlas fixture.

## Validation

- `cmake --build volume-cartographer/build/ci-tests-clang-systemdeps -j32
  --target test_atlas test_volume_pkg`: passed.
- `ctest --test-dir volume-cartographer/build/ci-tests-clang-systemdeps -R
  '^(test_atlas|test_volume_pkg|test_lasagna_project_volumes)$'
  --output-on-failure`: 3/3 passed.
- Focused Fiber 3D pytest with third-party plugin autoload disabled: 2 passed,
  180 deselected.
- Full `test_fiber_trace_3d.py` with third-party plugin autoload disabled: 180
  passed, 2 skipped.
- `git diff --check`: passed.
