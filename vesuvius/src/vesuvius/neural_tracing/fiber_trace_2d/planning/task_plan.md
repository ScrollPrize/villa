# Plan: weighted reference winding diagnostics

## Implementation

- Extend the shared reference observation with its canonical absolute winding
  distance, raw finite-L1 coefficient, cutoff-admitted coefficient, and
  coordinate residual scale. Compute these in the existing observation
  constructor using the dominant hypothesis score and the same power-of-two
  distance multiplier as joint winding BP. Pass the active parallel-distance
  cutoff explicitly; retain raw evidence even when the cutoff excludes it.
- Add a shared core summarizer that groups active reference observations by
  source and by `perp_0.5`, `perp_1.5+`, `parallel_0`, `parallel_1`, and
  `parallel_2+`.
- Make the prepared winding factor mutually exclusive: a measured constraint
  retains only its dominant parallel or perpendicular hypothesis. Evaluate
  that admitted term with its score, distance decay, cutoff suppression,
  measurement scale, and signed ordering. Hard continuity remains parallel.
- Rank candidate reference windings lexicographically by hard signed-order
  violations and then finite winding energy. This mirrors BP's impossible-state
  rule when a zero-violation active state exists; when hard constraints conflict,
  it provides the requested forced-active fallback instead of selecting Defect.
- Map every inferred candidate from its integer gauge into the globally
  calibrated reference coordinate as
  `globalSign * (candidate - gaugeOffset)`. This supports one reference source
  observed in multiple gauges. Evaluate truth at the source's virtual winding.
- Infer each group's preferred winding over the half-integer lattice near all
  finite inferred candidates. Minimize admitted weighted L1, then prefer smaller
  absolute winding and lower signed winding without consulting truth. A group
  with no positive raw coefficient reports `NA`.
- For each group, report observation count, raw coefficient sum, coefficient
  admitted by the current cutoff, hard violations and admitted total and
  coefficient-normalized energy at truth, preferred winding, and its hard
  violations plus total and normalized energy. Add an `all` row and use its
  preferred winding as `est_w`, eliminating the previous support-count/squared
  residual estimator.
- Insert a compact row-oriented table immediately before the existing
  per-reference right/wrong table. The table, `est_w`, and prepared BP factors
  must all use the same dominant-hypothesis semantics.

## Tests

- Extend focused winding-BP tests for all five effective-canonical bucket
  boundaries (including both signs and raw values on quantization boundaries),
  exact score/distance and cutoff-admitted coefficients, non-unit perpendicular
  measurement scale, weighted disagreement at truth, bucket-only inference,
  deterministic flat-optimum ties, contradictory hard signs, same-source multi-gauge conversion with a
  reversed global sign, empty/zero-weight groups, and invalid inputs.
- Validate the rendered table through the established CLI smoke workload,
  including bucket order, placement, precision, and `NA` output.
- Build the production CLI and focused winding-BP test, run the test, and run
  `git diff --check`.

## Spec update

- Specify the new pre-benchmark weighted disagreement table, bucket boundaries,
  effective coefficient, loss, calibration, and empty-value semantics.

## Docs updates

- Document how to interpret the weighted group diagnostic and why it may
  disagree with the existing unweighted right/wrong counts.

## Changelog

- Record the weighted per-reference constraint-group winding diagnostic.
