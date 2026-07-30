# Task Log: VC3D Persistent Fiber-Traced Segments

## 2026-07-30 - Planning

- Replaced the active task and plan with the requested persistent per-segment
  native fiber-trace state, ordinary-optimizer protection, scoped CP
  invalidation, and Ctrl-right-click Lasagna revert workflow.
- Confirmed the current failure path: native trace returns generic
  `Incremental` state, auto-save finalizes it through a full Lasagna pass, and
  the task result carries no typed segment provenance.
- Confirmed the context menu already resolves a Ctrl-right-click to the
  adjacent CP pair, so trace and revert can share one span-selection helper.
- Confirmed existing-line optimization supports fixed point indices, while
  full reinitialization rebuilds CP spans and therefore needs explicit shared
  protected-span support.
- Initial plan used a separate endpoint-signature segment registry. User review
  correctly identified that this adds unnecessary CP-to-segment tracking.
  Revised the design so each VC3D CP owns optional `segmentToNext` metadata for
  its span to the immediate successor.
- A first CP-owned revision kept numeric `control_points` plus a parallel
  metadata array for compatibility. User review rejected that because old
  readers could silently ignore important segment semantics.
- Final planned schema packs `position` and optional `segment_to_next` into
  each version-2 control-point object. Updated readers also accept version-1
  point arrays as ordinary CPs; writers emit version 2, and old binaries fail
  loudly on the version check. There are no parallel arrays or duplicated
  endpoint signatures.
- Expanded scope to audit and update the shared Python parser, native metric
  parser, C++ probes, VC3D import/export, and sync/merge tooling. Version-2
  unknown or malformed segment metadata is a hard error.
- Chosen validity rule: moving CP `i` clears metadata on `i-1` and `i`;
  inserting/deleting clears only the previous adjacency plus metadata erased
  with a deleted CP; unrelated CP state moves unchanged.
- Chosen revert rule: exclude only the selected record from protection, run
  endpoint-bounded Lasagna optimization, and clear the starting CP's metadata
  only after success.

## Deviations And Open Workflow Items

- The nested workflow requests independent agent review. The active runtime
  policy prohibits sub-agent delegation unless the user explicitly requests
  it, so no independent review was performed. A direct consistency review was
  completed and the independent-review status remains open.
- User approved implementation on 2026-07-30. Implementation began from the
  agreed version-2 CP-owned schema.

## 2026-07-30 - Implementation

- Added CP-owned optional `segmentToNext` state to the VC3D live session and
  stored-fiber model. Version-2 writers emit object-valued control points;
  version-1 numeric control points remain readable as ordinary, unprotected
  annotations.
- Added strict shared C++ and Python parsing for the version-2 segment schema.
  Updated the native tracer metric input, Lasagna line probe, Atlas source
  fiber loader, and merge/sync tooling. Synthetic merge geometry deliberately
  clears provenance instead of claiming that newly joined spans were traced.
- Native trace completion now applies geometry and typed provenance atomically
  and leaves the line finalized. Failed trace or revert tasks restore the
  pre-task optimization state.
- Added protected dense-point ranges to existing-line Lasagna optimization and
  protected original CP spans to full reinitialization. Protected spans retain
  their exact stored geometry while unrelated spans may be optimized.
- Centralized CP mutation rules: moves invalidate incoming and outgoing spans,
  insertion invalidates the split predecessor span, and deletion invalidates
  the newly adjacent predecessor span while removing the deleted CP's outgoing
  record. Unrelated records move with their owning CP.
- Added Ctrl-right-click reversion for traced spans. It optimizes a task-local
  copy with only the selected span unprotected and commits the geometry plus
  metadata removal only after successful Lasagna optimization.
- Updated the persistent-segment spec, code-structure notes, shared fiber JSON
  documentation, and changelog.

## 2026-07-30 - Verification

- Built `test_lasagna_line_optimizer`,
  `test_line_annotation_generated_views`, `test_atlas`,
  `test_fiber_trace3d`, `vc_lasagna_line_probe`, and `VC3D` with:
  `cmake --build volume-cartographer/build/ci-tests-clang-systemdeps --target ... -j32`.
- `test_lasagna_line_optimizer`: 29 cases passed, including bit-exact
  existing-line and full-reinitialization protection tests.
- `test_line_annotation_generated_views`: 45 cases passed, including strict
  v2 round-trip/final-CP validation and CP mutation invalidation tests.
- `test_fiber_trace3d`: 27 cases passed.
- Python command:
  `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src python -m pytest vesuvius/tests/neural_tracing/test_fiber_trace.py volume-cartographer/scripts/tests/test_fiber_merge.py volume-cartographer/scripts/tests/test_vc_sync_helpers.py`.
  Result: 185 passed.
- The complete Atlas test binary has three unrelated pred-snap fixture failures
  because its test manifest resolves `nx` to a non-3D zarr. The Atlas v2
  source-fiber round-trip fixture passes in that run; the repository's minimal
  test runner does not implement per-case filtering.

## Remaining Workflow Items

- A display- and dataset-dependent manual VC3D trace/edit/save/reload/revert
  smoke test was not run in this agent session. The user had already indicated
  that VC3D and line-annotation usage testing would follow implementation.
- The live GUI task lifecycle is covered through pure segment-state,
  serialization, optimizer, and compiled controller/menu paths rather than an
  automated Qt interaction test; controller-private task machinery makes a
  faithful headless end-to-end test disproportionate for this change.
