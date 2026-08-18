# VC3D control-point collapse rollback task log

## Findings

- After `collapseControlPointsAtClick()`, all click-edit cases have exactly one
  changed replacement control and a valid replacement index. Removing multiple
  controls does not require a multi-control line-update API.
- The existing local updater rebuilds both spans adjacent to its changed
  control and places that control directly into the updated line before fiber
  optimization.
- The PR 1484 multi-collapse branch bypasses this updater and passes the
  replacement plus unchanged old line to the fiber optimizer, whose independent
  nearest-3-D control lookup can select another winding.
- Multi-collapse already carries a complete asynchronous rollback object. The
  ordinary synchronous update catch captures but fails to restore the previous
  optimization state.
- Independent review found that reusing the existing local-update controller
  block verbatim would mutate reciprocal branches and schedule saves before the
  asynchronous result is known. Multi-collapse must keep only its session-local
  branch remap before success and defer reciprocal synchronization as it does
  today.
- Independent review also found that all-control collapse leaves one replacement
  with no adjacent span. The local updater returns the old line unchanged in
  that case, and the one-control optimizer currently derives its tangent by
  nearest 3-D lookup. The targeted fix will use authoritative line position for
  that tangent.
- To cover the production regression rather than manually composing two helpers
  only in tests, automatic collapse plus local reconstruction will be extracted
  into one reusable preparation helper used by controller and tests. Session
  mutation will occur only after that helper succeeds.
- Independent implementation review found that segment metadata was still
  merged back by equal 3-D position. The merge now follows stable ordered
  control identity, with duplicate-position coverage for exact crossings.
- The same review found that multi-collapse invalidated cached metrics before
  asynchronous commit and that generated-view failure could leave the new
  optimization report/flags behind. Metric invalidation is now deferred until
  successful materialization, and the rollback snapshot includes the previous
  report, manifest, optimization flag, and metric-match flag.
- Legacy fiber loading and other optimizer entry points still reconstruct some
  control locations with nearest-3-D matching. That broader persisted topology
  issue is intentionally deferred because this task is limited to the two PR
  1484 regressions and does not change the fiber format.

## Deviations

- The private Qt `LineAnnotationSession` still has no isolated controller test
  seam for forcing asynchronous optimization or generated-view failure.
  Existing multi-collapse rollback is retained and reviewed through its shared
  rollback object and full VC3D compile, while focused tests cover the newly
  extracted production preparation path and synchronous failure atomicity.
- The installed `clang-format` configuration did not match repository style and
  reformatted thousands of unrelated lines. That mechanical rewrite was fully
  removed; only focused semantic edits remain.

## Validation

- Built with all 32 cores:
  `cmake --build volume-cartographer/build --parallel 32 --target test_lasagna_line_optimizer test_line_annotation_generated_views VC3D`
- `test_lasagna_line_optimizer`: 35 test cases passed.
- `test_line_annotation_generated_views`: 79 test cases passed.
- Full `VC3D` target compiled successfully.
