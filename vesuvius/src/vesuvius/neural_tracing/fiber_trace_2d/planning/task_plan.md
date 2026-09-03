# Plan: baseline quality-threshold benchmark

## Evaluation

1. Load the existing complete ordinary crop artifact without no-overtracing and
   apply an absolute quality threshold at BP input. Freeze the crop, inputs,
   build, and every non-threshold pruning setting across candidates. Keep the
   completed online-threshold run only as a separate diagnostic because online
   rejection changes seed coverage and is not the intended baseline comparison.
2. Run the existing fixed oracle-pruning diagnostic on every candidate and
   capture round-zero exact/wrong/missing references, final reference result,
   retained geometry, problematic constraints, and Release timing.
3. Refine the threshold upward or downward from the first result. Prefer higher
   round-zero exact percentage, then fewer wrong references, then greater
   evaluable-reference and geometry support. Keep the selected result explicitly
   reference-tuned and preserve all historical points.
4. Write a permanent run record with exact commands, source and input identity,
   threshold candidates, and the selection rationale.

## Plotting

1. Extend the checked benchmark plot schema with a pre-pruning reference metric
   computed from raw exact and wrong counts.
2. Plot exact / (exact + wrong) as a percentage. Missing references are reported
   in records but excluded from this fraction because they have no estimate.
3. Add historical fixed-quarter, no-overtrace threshold, and selected ordinary
   threshold points; regenerate deterministic benchmark SVGs.

## Tests

- Run the plot-data validation path and render all benchmark plots.
- Run `git diff --check`; source tests need not be repeated unless source code
  changes during this benchmark-only task.

## Spec update

- Extend the benchmark-progress specification with the pre-pruning reference
  accuracy metric `exact / (exact + wrong)`, its exclusion of missing
  references, derivation from raw counts, required run-record/revision
  provenance for measured points, and the existing marker plus cumulative-best
  display behavior.

## Docs updates

- Add the run record, result-index rows, metric definition, and plot.

## Changelog update

- Record the baseline threshold benchmark and new reference-accuracy plot.
