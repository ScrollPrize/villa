# Plan

Stop inefficient current run safely; preserve inference. Bound coroutine stacks to a small multiple of workers and reuse allocations. Global native trials stay compact until admitted. Order by trial distance ascending; once two neighboring distances reproduce with existing audit semantics, skip larger pending trials (in-flight trials may finish safely). Keep Python authoritative geometry/audit and retain artifacts for successful trials. Add explicit cancelled count/compact trial summary and live progress. Preserve ordinary evaluation ordering. Remove inference overrides; set extraction CPU grant to192 (leaves32 logical cores for IO/manager), same as fullres stage. Existing inference128 workers unchanged.

## Spec update

No spec.md. No scientific parameters changed. Native pruning requires two adjacent valid audits, preserves every smaller trial. Partial interrupted trial config remains archived/revisioned.

## Tests

Native bounded queue tests including 300k tasks, wait/wakeup, stack reuse, exceptions. Trial pruning tests. Existing replay tests and Python extraction regression. Approved build/run only. Measure progress and utilization after resume.

## Docs updates

Log findings, tests, timings, operational changes and any limitations; update benchmark doc and changelog.
