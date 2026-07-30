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
- No implementation, build, or tests were run during this planning-only step.
