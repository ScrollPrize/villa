# Plan

1. Define mean distance per failure as total tested directed reference length
   divided by `max(total failures, 1)`.
2. Define its percentage as `100 * distance per failure / total tested directed
   reference length`; therefore a zero-failure run reports the entire tested
   length and 100 percent.
3. Make these the only headline reliability values in CLI and benchmark result
   tables. Preserve failure counts, locations, and seeded-coverage fields in
   versioned JSON as supporting diagnostics.
4. Update unit tests, specifications, documentation, changelog, and benchmark
   records, then rebuild and run focused tests.

All replay failure events count, regardless of reason. Full directed reference
length is used only when every directional case completed evaluation. The
percentage deliberately reduces to `100 / max(total failures, 1)`, so both zero
and one failure report 100 percent; the zero-failure value is a censored lower
bound and remains identifiable from the failure count in JSON.

## Spec Update

Specify the exact distance-per-failure formula and zero-failure convention.

## Docs Update

Replace ambiguous mean failed-span and seeded-coverage headline reporting with
distance per failure and its percentage of tested length.

## Changelog

Record the corrected whole-run reliability metric.

## Validation

Test runs with zero, one, and multiple failures and verify physical conversion,
percentage calculation, JSON fields, and CLI output.
