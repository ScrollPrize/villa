# Task log

## Findings

- Callgrind and DRD are necessarily collected in independent Valgrind runs.
- `estimate_score()` currently passes raw Callgrind thread costs directly to a
  DRD graph keyed by raw thread ID.
- CI failed because DRD thread 2 had no same-numbered Callgrind cost. The local
  `parallel/fallback_3` artifact happened to contain threads 1 through 5 in
  both runs and therefore passed.
- `_validate_metadata()` only requires more than one observed parallel worker;
  it neither guarantees identical worker participation nor makes such a
  requirement valid.
- Existing native attribution already weights each thread's compute windows by
  complete scheduler quanta plus the configured residual fraction. This can be
  shared for pooled worker attribution without duplicating replay logic.
- The event-cost model contains nonlinear interaction features. Raw event
  profiles must therefore continue to be modeled independently and their
  resulting costs summed; merging raw counters first would change total cost.

## Constraints

- Preserve the exact sum of independently modeled Callgrind thread costs.
- Preserve DRD event order, dependencies, and observed scheduling.
- Do not require or invent stable worker identities across executions.
- Keep existing generic per-thread replay functionality available.
- Do not change renderer numerics or benchmark workload.
