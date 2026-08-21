# Task log: persistent equal-arc fiberlet beam search

## Findings before implementation

- The unshipped fixed-edge replay prunes to 16 after each graph-edge depth, then
  selects one route and commits only its first fiberlet. It therefore compares
  unequal physical lengths and does not retain 16 histories across iterations.
- The experimental fixed-distance replay compares local routes at a common
  192-base-voxel horizon, but selects one first fiberlet and rebuilds the route
  tree at the next anchor. Its width setting only truncates diagnostics after
  selection; search itself is exhaustive up to a one-million-state cap.
- The required algorithm instead retains up to 16 histories from the original
  seed of an uninterrupted segment. Fixed absolute beam-front checkpoints mean
  a beam can cross zero, one, or multiple fiberlets during one iteration.
  Checkpoint and lookahead boundaries must support positions inside fiberlets.
- The first whole-fiber float/Q1 fixed-distance experiment was stopped during
  Q1 at 46.6% after identifying the wrong search semantics. It was restarted at
  user request against the same partially populated caches; its scientific
  result is diagnostic for the discarded local algorithm only.

## Deviations

- No implementation has started. The current code and its in-progress cached
  experiment still implement the superseded local receding-horizon algorithm.

## Independent plan review

- Added distinct inside-edge and exact-at-anchor checkpoint states so boundary
  joins are neither selected nor charged early.
- Defined provisional-history replacement, first-failure freezing, normal end,
  and all-routes-exhausted behavior without using the reference for ranking.
- Made active decoded compact cost authoritative and required an explicit
  diagnostic residual, exact prefix-plus-suffix arithmetic, visited-state-aware
  deduplication, and deterministic total ordering.
- Added bounded route-payload leases, explicit state-cap accounting, strict CLI
  validation, cache-eviction/worker determinism, and adversarial partial-edge
  tests.
- The review proposed a different combined quantization scenario for focused
  validation. This plan keeps `position_q1`, matching the user's current
  baseline/Q1 request; the whole-fiber rerun remains deferred until that focused
  result is understood.
