# VC3D control-point collapse rollback plan

## Current failure

`handleGeneratedControlPoint()` first reduces every matched control to one
replacement with a known fractional line position. For zero or one matched
control it then calls `updateExistingLineControlPoint()`, which rebuilds the two
adjacent spans and makes the replacement an exact sample of the updated line.
For more than one matched control, however, it bypasses that update and starts
fiber optimization against the unchanged old line. The fiber optimizer then
projects controls independently by nearest 3-D distance, so a replacement can
bind to another winding of a self-approaching line.

The ordinary local-update exception path separately restores controls,
branches, seed, and focus after already marking the session unoptimized, but
does not restore the captured previous optimization state.

## Implementation

1. Keep `collapseControlPointsAtClick()` as the single pure operation for
   insertion, one-control replacement, and multi-control collapse. Continue to
   use its old-to-new branch mapping when the prepared edit is committed.
2. Extract a reusable automatic-click preparation helper beside the existing
   line-annotation fiber helpers. It accepts the old line/controls, matched
   indices, clicked line position/point, sampler, and local-update config; it
   performs collapse on copies and, when at least two controls survive, invokes
   `updateExistingLineControlPoint()` for the replacement. It returns the
   updated line, rich controls with segment metadata merged back, replacement
   index, requested adjacent dirty spans, and old-to-new branch mapping.
   `handleGeneratedControlPoint()` and focused tests must both call this helper;
   do not duplicate its composition in the controller or tests.
3. Remove the automatic multi-control branch that calls
   `startFiberModeOptimization()` against `session.optimizedLine` without first
   updating that line.
4. Route every automatic click edit with at least two surviving controls through
   the existing `updateExistingLineControlPoint()` call using the collapsed
   replacement index. After compaction there is one changed control regardless
   of how many old controls were removed, so the existing updater can rebuild
   the complete left-neighbor -> replacement and replacement -> right-neighbor
   spans from the replacement's authoritative line position.
5. Handle an all-control collapse explicitly. With one surviving replacement
   there are no adjacent spans to rebuild, so retain the old line only as a
   tangent reference and run the established one-control reinitialization from
   the clicked point. Change that one-control tangent lookup to use the
   replacement's authoritative clamped `linePosition`, never nearest 3-D
   position, so a nearby winding cannot choose the initial direction.
6. Prepare the entire automatic edit before mutating the session. If preparation
   throws, report the error while controls, line, branches, seed/focus,
   optimization state, and alignment-metric state are still untouched. Only
   after preparation succeeds, commit its controls/line, remap session-local
   branch indices, update seed/focus, mark metrics stale, and mark the session
   unoptimized.
7. Derive requested dirty spans from the updater's post-sort replacement index;
   these are exactly the replacement's surviving adjacent spans. The fiber
   optimizer may expand a requested dirty span across a connected C-spline run
   by existing policy.
8. Preserve the original pre-edit line and session fields in the existing
   multi-collapse rollback object until asynchronous fiber optimization and
   generated-view materialization succeed. Both failure-to-start and
   asynchronous failure restore the original pre-edit controls, line, branches,
   seed, focus, and optimization state and clear the rollback object.
9. Keep branch updates transactional for multi-collapse. Apply only the
   old-to-new mapping to session-local branch indices before optimization; do
   not mutate reciprocal fibers or schedule saves. On successful optimization,
   use the existing deferred synchronization in `applyOptimizationTaskResult()`.
   Preserve the current insertion/single-replacement synchronization behavior.
10. Keep no-reoptimization mode unchanged: it records the collapsed controls on
   the current line without running local reconstruction or fiber optimization.
   Do not change persisted control representation or legacy load-time matching.

## Testing and validation

1. Test the extracted production preparation helper with a multi-control
   fixture. Verify its updated line contains the clicked replacement as an exact
   support between surviving neighbors, its controls remain ordered, and its
   requested dirty spans are the replacement's adjacent spans.
2. Add a self-approaching/two-winding fixture where the clicked replacement is
   spatially nearer to an unrelated winding than to the old sample at its line
   position. Invoke the production preparation helper and verify local
   reconstruction anchors it in the intended ordered span and produces strictly
   ordered control indices before full optimization.
3. Retain and run the pure collapse tests for metadata ownership, seed transfer,
   old-to-new branch indices, and dirty spans, ensuring unification does not
   alter collapse semantics.
4. Exercise endpoint/all-control collapse behavior. Verify preparation retains
   the clicked replacement and that the one-control optimizer selects its
   tangent from authoritative line position on a self-approaching line.
5. Add a preparation-failure test using a throwing sampler/update fixture and
   verify caller-owned inputs are unchanged. Review the controller ordering to
   ensure no session field, optimization state, metric state, reciprocal branch,
   or save is mutated before that helper returns successfully.
6. Add focused branch-remap coverage for two linked controls collapsing into
   one: local indices may be remapped during commit, while reciprocal mutation
   is represented only by the later successful-commit synchronization input.
   Verify failure rollback retains the pre-edit branch snapshot. If the private
   Qt session still prevents an isolated asynchronous failure test, record that
   narrow limitation explicitly.
7. Build with all 32 cores and run:
   `cmake --build volume-cartographer/build --parallel 32 --target test_lasagna_line_optimizer test_line_annotation_generated_views VC3D`.
   Run both focused test binaries and `git diff --check`.

## Specification updates

- Create `specs/line_annotation.md` as the durable home for line-annotation
  editing requirements currently represented only in `planning/spec.md`.
- Amend both the durable specification and the existing line-annotation ribbon
  invariant in `planning/spec.md`: every automatically reoptimized
  generated-view click, including a multi-control collapse, must first
  reconstruct the replacement's adjacent spans from its authoritative line
  position; it must not project the replacement onto the unchanged line solely
  by nearest 3-D distance.
- Require transactional failure behavior: failed synchronous preparation leaves
  geometry and optimization state unchanged, while failed asynchronous
  multi-collapse restores the original pre-edit snapshot.
- Do not change the persisted-control format or claim that legacy independent
  nearest-3-D reconstruction is solved.

## Documentation updates

- Update `docs/line_annotation_fibers.md` to state that automatic control-point
  edits locally reconstruct the neighboring spans before full fiber
  optimization, including when several nearby controls collapse to one.
- Document that failed automatic preparation retains the prior line and
  optimization status.

## Changelog update

- Add one dated line to `planning/changelog.md` noting that multi-control
  collapse now anchors its replacement through local span reconstruction and
  synchronous preparation failure leaves the prior edit state unchanged.
