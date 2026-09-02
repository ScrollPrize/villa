# Plan: oracle winding inlier search

## Semantics

1. Add an explicit `oracle-inliers` reference-pruning policy; preserve every
   existing policy and artifact family.
2. Start from the sign-consistent `conditioned-inliers` working set. Fixed
   reference pieces remain non-removable.
3. Freeze the initially supported reference set. Evaluate a candidate retained set with the existing reference observation,
   gauge calibration, and `est_w` implementation. A reference is exact only
   when its calibrated estimate equals its virtual half-winding label. A
   reference with no estimate is missing, not exact. Initially unsupported
   references remain reported but do not make success impossible.
4. Optimize the frozen supported set lexicographically: maximize exact
   references, then minimize wrong references, then minimize missing
   references. A missing estimate is preferable to preserving a known false
   winding only after no exact estimate is lost. Report missing references
   explicitly.
5. At each oracle round, restrict candidates to ordinary pieces incident to a
   currently wrong reference. Evaluate every single-piece counterfactual by
   removing its observations and recalibrating with the shared benchmark path.
   If no single improves the lexicographic score, evaluate deterministic pairs
   drawn from the highest wrong-reference evidence candidates. Magnitude-class
   weights rank candidate evidence; sign contradictions are mandatory.
6. Treat counterfactual observation deletion only as a proposal. Apply each
   proposed batch to an original-piece-ID mask, rebuild exact ordinary and
   reference-cross subsets, and rerun the fixed-reference conditioned BP. Run
   sign-consistent inlier closure again. Accept a converged realized solve when
   it improves or preserves the exact/wrong/missing tuple; neutral peeling can
   expose collective offenders. Reject regressions and try bounded ordinary
   graph-neighbor alternatives. Repeat until no wrong estimate remains, no
   candidate remains, or a configurable round guard is reached.
7. Retain the direct labels from the final conditioned solve as the oracle
   working set. Run the existing fresh reference-free solve afterward only as
   a stability diagnostic.
8. Publish oracle direct, fresh, and removed-by-round/reason artifacts without
   overwriting the existing `_inliers` family.

## Configuration and reporting

- Add an oracle round limit and bounded pair-candidate limit.
- Add a nonnegative magnitude-ranking weight. Sign evidence remains
  authoritative and cannot be disabled in oracle mode.
- Add a minimum active-observation threshold for identifying a reference.
- Print one compact row per round: input/removed/retained pieces and arc,
  exact/wrong/missing references, constraint agreement, solve status, and
  elapsed time.
- Report a clear terminal reason: zero errors, no improving removal, empty
  evidence, message limit, or round limit.

## Implementation

- Put reusable counterfactual reference scoring and deterministic candidate
  selection in the core winding library. Reuse
  `calibrateFiberTraceReferenceWindings`; do not duplicate estimator logic in
  the CLI.
- Keep graph/subset rebuilding in the CLI orchestration where the existing
  reference and geometry objects live. Every round subsets from original IDs,
  never recursively from stale local IDs.
- Preserve original-piece mappings across all rounds and stable tie-breaking by
  removed arc then original piece index.

## Tests

- Unit-test exact/wrong/missing score ordering, no exact-to-wrong trade,
  single-piece improvement, pair-only improvement, deterministic ties,
  magnitude ranking at zero/nonzero weight, and no-improvement termination.
- Run focused GCC Release and Clang tests plus the Napari artifact tests.
- Run the validated 1024 crop through the oracle policy. The required success
  criterion is zero wrong identifiable reference estimates; report missing
  references separately and compare retained arc with `_inliers`.

## Spec update

Document that oracle pruning is supervised, uses the canonical benchmark as
its objective, re-solves after each accepted batch, and cannot be treated as a
production reference-free filter.

## Documentation updates

Document CLI options, optimization priority, round table, terminal statuses,
and oracle artifact suffixes in `volume-cartographer/docs/fiber_chunk_tracing.md`.

## Changelog

Add concise entries after implementation and evaluation.

## Deferred follow-up

Use the oracle removed/retained cohorts to train or validate an intrinsic
reference-free pruning score.
