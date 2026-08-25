# Task log: apply staged Fiberlet filtering before replay

## Discovery

- Existing staged reduction is currently wired only to `chunk-route-stats`.
  Its analysis, simplification, and sparse overlay writes already live in
  reusable core functions, while orchestration and temporary-layer lifetime
  remain in `vc_fiberlets.cpp`.
- Current stage offsets are relative to the selected diagnostic region. Replay
  requires a globally anchored lattice so focused intervals and repeated runs
  address the same boxes.
- The on-demand preprocessor currently uses one tube selector for source
  generation and cached replay traversal. Filtering requires source data beyond
  the original tube, so those responsibilities must be separated.
- Chunk-route materialization reads incident Fiberlets and endpoint anchors
  outside each box. Backward stage closure must therefore include the metadata
  maximum endpoint reach, not only geometric overlap caused by stage offsets.
- Persistent source caches already use globally anchored storage keys. The
  transient overlays may retain that exact storage grid while filter boxes use
  unrelated globally anchored sizes.

## Deviations

- The focused `test_fiber_replay` executable retains an unrelated existing
  line-strip dimension failure. The touched planner/storage tests and
  `test_fiberlet_paths` pass; no replay-strip implementation or expectation is
  changed by this task.

## Implementation

- Added a pure core planner for globally anchored complete filter boxes,
  backward endpoint-reach closure, and source-generation support.
- Made both replay and the regional staged diagnostic normalize stage offsets
  modulo their side and use global base-volume anchoring.
- Extracted evaluated-anchor view creation and transient stage application into
  shared orchestration used by both diagnostic and replay callers.
- Extended cached replay with an explicit traversal predicate so expanded
  source generation does not expand seed or traversal eligibility.
- Added optional replay filter CLI wiring, eager-graph rejection, transient
  overlay lifetime management, and replay-bundle provenance.

## Validation

- Built `vc_fiberlets`, `test_fiberlet_storage`, `test_fiberlet_paths`, and
  `test_fiber_replay` in the regular build tree using repository-local compiler
  temporary storage.
- `test_fiberlet_storage`: passed, including global anchoring, normalized
  negative offsets, deterministic ordering, complete-boundary exclusion,
  backward expansion, and storage/filter-size independence.
- `test_fiberlet_paths`: passed.
- `test_fiber_replay`: existing strip geometry assertions fail in untouched
  replay visualization code (`replay strip component dimensions are invalid`
  and stale overview dimensions).
