# Task Plan

## Scope

Keep production rendering, scheduling, dependency replay, attribution,
benchmark coordinates, and native renderer measurements unchanged. Compare
three zero-overlap feature-basis candidates against a matched six-feature
soft-L1 baseline using synthetic data only:

1. Add a base `data_reads` cost.
2. Split L1 and last-level data misses into read and write terms.
3. Combine the data-read cost and split miss terms.

The rejected serialization feature and pointer records remain excluded.

## Synthetic Calibration

1. Extend passive feature extraction with explicit named schemas while
   preserving byte-for-byte calculations for existing six- and seven-feature
   models. Zero-overlap candidates remain a nonnegative linear sum; legacy
   overlap behavior remains restricted to its existing schema.
   Freeze these equations: `data_reads = Dr`; split misses are `D1mr`, `D1mw`,
   `DLmr`, and `DLmw`; and the interaction remains
   `(D1mr + D1mw) * (Bcm + Bim) / max(Ir, 1)`. Calculate the interaction per
   Callgrind thread before summing. Data reads are non-stall work, misses are
   cache-stall work, and every new candidate uses zero overlap.
2. The opened matrix identifies split miss terms but not a base read cost:
   `data_reads` has 0.987 correlation with non-data instructions. Add one
   deterministic generic four-read kernel and one eight-read kernel that vary
   `Dr / Ir` independently, plus a multi-store kernel independent of the
   existing stream writer. The old random `cache-write` kernel remains excluded
   because opened native observations are multimodal. Collect and inspect the
   three new calibration families before freezing the holdout; do not weaken
   the 0.98 correlation gate.
   A follow-up diagnostic adds one/eight read and one/eight write cache-line
   traversals and crossed `r1w1`, `r8w1`, `r1w8`, and `r8w8` subsets. This
   holds line traversal fixed while independently varying access density and
   creates mixed read/write miss profiles instead of single-feature evidence.
3. Reuse the current family weighting, soft-L1 objective, ridge, nonnegative
   bounds, native medians, family labels, and training-only scaling for all four
   matched fits over identical records. Report rank, data-only
   condition/correlation, coefficient bounds, fit errors, and leave-one-family-
   out coefficient movement. Movement is
   `abs(reduced-full) / max(abs(full), 1e-6 ns)`; rank loss fails stability, and
   maximum movement must be at most 20%.
4. After the new calibration matrix and all candidate definitions pass
   identifiability checks, predeclare 40 fresh holdout cases covering all ten
   accepted synthetic work kinds at four unused working-set sizes. Complete
   tuples and sizes must be
   disjoint from every opened fit, holdout, diagnostic, serialization, and
   minimax case. Freeze and hash the case manifest, benchmark binary,
   extraction/fitting code, model schemas, and gates before collection.
5. Run five native trials per case sequentially under fixed-frequency
   monitoring, collect one matched Callgrind profile per case, and hash the raw
   samples/profiles afterward.

## Selection And Reporting

1. Synthetic screening reports full rank, maximum parameter correlation
   below 0.98, no selected coefficient bound hit, leave-one-family-out movement
   at most 20%, fresh median at most 20%, and maximum-error improvement over
   the matched baseline larger than twice the largest per-case native-sample
   MAD. Median/RMS may regress by at most two points and no family maximum may
   regress by more than five points. At the user's direction, failed screening
   does not prevent a post-fit renderer diagnostic and does not itself decide
   whether a basis is useful.
2. Freeze the current pipeline, matched baseline, and all three candidate
   pipelines before renderer evaluation. Refit the same handoff-only
   synchronization structure with identical observations, constraints, and
   fitting procedure.
3. Evaluate all frozen pipelines in one renderer trace pass. Report maximum
   absolute runtime error for worker 1 and pooled workers 2--7, per-worker
   maxima, all 35 rows, and maximum speedup error as monitor-only.
4. Report the renderer result directly for decision-making. Do not refit any
   event or synchronization coefficient after opening it.

## Specification Updates

- Document named feature-schema compatibility, candidate selection gates, and
  synthetic-only basis expansion requirements.

## Documentation Updates

- Record exact case matrices, coefficients, synthetic and renderer errors,
  commands, frequency evidence, failures/deviations, and artifact hashes.

## Testing And Validation

- Unit-test legacy schema compatibility, exact named feature extraction,
  zero-overlap linear sums, unsupported schemas, candidate matrix
  identifiability, read/write-density determinism, leave-one-family behavior,
  fresh-case disjointness, selection gates, and golden predictions for existing
  serialized six- and seven-feature models.
- Build with 32 jobs; run focused Python tests, registered CTests, byte
  compilation, deterministic checksums, frequency validation, provenance hash
  checks, and whitespace checks.

## Changelog Update

- Record whether each feature-basis candidate improves fresh synthetic and
  separate one-worker/many-worker renderer maximum errors.
