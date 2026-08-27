# Plan: Per-fiber BP constraint-consistency diagnostics

## Contract

- Analyze exactly the unique pair factors used by binary BP after duplicate
  measurements are merged. A factor's evidence strength is the absolute
  difference between its same-label and different-label costs; one
  perpendicular measurement with parallel score `p` therefore has strength
  `1 - 2p`.
- Classify horizontalness values at fixed diagnostic thresholds: V at or below
  `0.25`, H at or above `0.75`, and unresolved between them. For every fiber,
  report unique-factor degree, total factor strength, resolved and unresolved
  degree/strength, hard same-label mismatches, resolved hard mismatch rate, and
  resolved strength-weighted mismatch rate.
- Also report smooth diagnostics over all incident factors:
  - soft same-label independence proxy
    `h_i*h_j + (1-h_i)*(1-h_j)`, strength weighted; this is not a calibrated
    BP edge probability;
  - neighbor support balance
    `2*min(H_support,V_support)/(H_support+V_support)`, where
    `H_support=sum(w*(1-h_j))` and `V_support=sum(w*h_j)`.
- Report strength-weighted neighbor certainty alongside support balance because
  unresolved neighbors can otherwise look maximally balanced.
- Every undefined denominator is represented as `NA`, not zero, and excluded
  from summaries. Include valid counts and a tie-aware AUROC for Mixed versus
  trusted fibers for every metric, with prediction direction stated.
- Persist one CSV row per represented fiber, including cohort-local and
  original trace IDs, BP status, thresholds, horizontalness, initial geometric
  direction group, unique degree, and incident measurement count. Print
  count/min/mean/median/p90/max
  summaries separately for trusted dir1, trusted dir2, and Mixed fibers.
- Extract the labeling constraint selector into a shared core helper used by
  both HiGHS and BP-only. Add a BP-only direction-ablation execution path. It builds only the requested
  final cohort, extracts/prunes the same constraints, runs BP and diagnostics,
  and does not call either HiGHS solve. The ordinary ablation path remains
  unchanged.
- This task uses unbalanced/natural BP for the diagnostic experiment. The
  previously drafted population-balance implementation remains experimental;
  adapting it from a fixed target to the user's clarified minimum-population
  prior is explicitly outside this focused diagnostic run.

## Implementation

1. Extract and regression-test shared constraint selection, then expose the
   merged BP factor graph's per-fiber consistency calculation as a
   reusable core function and include the report in BP output ownership.
2. Add deterministic CSV serialization and compact grouped console summaries.
3. Add `--bp-only` to direction-ablation, make its default BP mode natural
   (no population field), and bypass MILP/LP construction and output.
4. Run the centered-384 full-Mixed cohort and compare diagnostic distributions
   for dir1, dir2, and Mixed references.

## Spec Update

Add the consistency metric definitions, thresholds, merged-factor ownership,
and BP-only diagnostic execution contract to `specs.md`.

## Docs Updates

Document `--bp-only`, the per-fiber CSV fields, and the grouped summaries in
`volume-cartographer/docs/fiber_chunk_tracing.md`.

## Testing

- Unit-test inclusive thresholds, hard resolved mismatch, unresolved accounting,
  strength partitions, weighted mismatch, soft proxy, support balance and
  certainty, heterogeneous duplicate-factor merging, order/gauge invariance,
  invalid evidence, and isolated fibers on a small deterministic graph.
- Test CSV ordering/header and validation of horizontalness/report sizes.
- Build `vc_fiber_trace_chunk` and `test_fiberlet_crop_trace` with `-j32`, run
  the focused test binary, and run `git diff --check`.
- Run the centered-384 full-Mixed BP-only command and record exact cohort,
  timing, and grouped quantiles.

## Changelog

Record per-fiber BP consistency diagnostics and the HiGHS-free BP-only
direction-ablation path.
