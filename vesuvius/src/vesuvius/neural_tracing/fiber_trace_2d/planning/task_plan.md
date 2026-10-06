# Raw loss reporting plan

1. Commit existing native baking/data preparation and annotated training changes separately.
2. Retain weighted optimization statistics. Accumulate separate raw annotation and embedding numerators for reporting (including zero weights); use the same counts and collective. No additional forwards/backwards or changed gradients.
3. Report refinement components raw; preserve weighted combined smoothing/total values. Append raw values after the existing weighted statistic prefix; upgrade old pending-test vectors. Omit unmeasured disabled direction or unrecoverable zero-weight historical terms rather than inventing zero losses.
4. Regression tests with non-unit/zero weights verify raw logs, weighted totals and gradient parity; run existing CPU suites through approved pytest. No production restart and no GPU work necessary for scalar-only reporting.

## Spec update
All leaf loss metrics are raw. Combined metrics apply constituent weights internally but not any outer coefficient. Optimization is unchanged.

## Docs updates
Document reporting semantics and test results in annotated-head docs, status and task log.

## Changelog update
Record consistent raw loss reporting and unchanged optimizer semantics.
