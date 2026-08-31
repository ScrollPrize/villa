# Plan

## Implementation

1. Replace the single anti-correlated phase candidate search in
   `FiberTraceConstraints.cpp` with an independent 2D grid over the two next
   arc advances.
2. Advance from the previously accepted correspondence in each walk direction,
   so bounded per-step corrections can accumulate along curved fibers.
3. Rank candidates by the two squared, target-step-normalized advance residuals
   plus `(unit_connector dot tangent_a)^2 + (unit_connector dot tangent_b)^2`.
   Reject zero-length connectors and do not include connector length in the
   score. Prefer smaller residuals, then smaller offsets, then lexicographic
   offsets for deterministic ties.
4. Keep the closest sampled pair unchanged as the constraint seed, reported
   closest connector, and first parallel-winding sample.
5. Rename the public configuration fields from phase refinement to
   correspondence-grid terminology; this intentional source API break is
   acceptable because this feature is not shipped and compatibility is not
   required. Use a 5%-step, ±25%-limit grid and require a limit below one target
   step to guarantee progress.
6. Preserve the original distance-minimizing phase walk as the default and add
   `--parallel-correspondence perpendicular-grid` as the explicit experimental
   selector. Keep independent configuration for the two algorithms.

## Testing

- Add a curved, parallel-fiber regression where correct correspondence requires
  accumulating independent arc corrections and verify that sampled connectors
  remain approximately perpendicular, both advances remain within the grid
  bound, and independent corrections are actually selected.
- Retain serial/parallel/batched constraint equivalence coverage.
- Build and run the focused fiberlet crop/constraint test target.
- Run `git diff --check`.

## Spec Update

Update `planning/specs.md` to require incremental 2D correspondence search for
parallel constraint walks and to prohibit connector-distance scoring.

## Docs Update

Update `volume-cartographer/docs/fiber_chunk_tracing.md` to describe the new
correspondence objective and bounded grid.

## Changelog

Add one concise entry for the changed parallel correspondence estimator.
