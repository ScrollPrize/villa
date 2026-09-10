# Task plan

## Current failure

The render gate runs Callgrind and DRD in separate processes because Valgrind
cannot run both tools together. Evaluation currently indexes modeled Callgrind
cost and DRD events by each run's raw Valgrind thread number. Dynamic worker
participation can differ, so a valid DRD worker can have no same-numbered
Callgrind profile. Even when both IDs exist, the worker may have executed a
different portion of the render, making the attribution semantically invalid.

## Implementation

1. Keep existing raw-profile scoring and serial-fixture behavior unchanged.
2. For parallel render evaluation, split modeled Callgrind cost into two roles:
   thread 1 is the process main/control role; all positive-numbered non-main
   thread costs are summed into one renderer worker-pool total.
3. Add a native replay pooled-attribution operation which:
   - attributes main cost through the existing per-thread window logic;
   - discovers all non-main DRD threads from the loaded graph;
   - builds the existing work-quantum/residual windows for all workers;
   - distributes the worker-pool total globally by window weight;
   - permits idle DRD workers to receive zero cost;
   - fails when positive role cost has no eligible event;
   - verifies that assigned event durations preserve main, worker, and total
     costs within the existing floating-point tolerance.
4. Extend the native JSON protocol and Python client with an explicit pooled
   attribution request. Retain per-thread attribution for existing callers and
   tests; do not emulate pooled attribution in a second Python implementation.
5. Record role costs and observed Callgrind/DRD thread sets in evaluation
   diagnostics without requiring those sets to match.

## Tests

- Native unit tests for pooled main/worker cost conservation, idle workers,
  missing eligible role events, and worker-ID relabeling.
- Driver tests proving that parallel evaluation aggregates all non-main
  Callgrind costs and sends one pooled attribution despite mismatched worker
  sets.
- A regression reproducing the CI case where DRD contains a worker absent from
  Callgrind.
- Existing Python/native replay parity and render-driver tests.
- Local `render_valgrind_ci` target to compare all modeled scores with the
  frozen references. If attribution intentionally changes scores, inspect the
  complete case table before considering any reference update.

## Spec update

Add the synthetic render gate contract to `planning/spec.md`: Callgrind
provides main and aggregate worker-role cost, DRD provides the observed worker
schedule, raw worker IDs never cross run boundaries, and attribution preserves
the complete modeled cost.

## Documentation updates

Update `docs/thread_sync_replay.md` with pooled attribution protocol semantics,
the main/worker role split, work-quantum distribution, and the reason raw
Valgrind worker IDs cannot be correlated across runs.

## Changelog

Add one concise entry describing the corrected cross-run attribution model and
the removal of worker scheduling sensitivity from the render gate.

## Independent plan review

- The plan preserves serial scoring and the existing generic per-thread replay
  API, limiting behavior change to parallel render-gate attribution.
- Aggregation happens after applying the nonlinear event-cost model per
  Callgrind thread, so the old complete modeled cost is preserved exactly.
- Stable logical worker IDs are intentionally not introduced: dynamic
  scheduling means equal logical IDs would still not imply equal work across
  independent runs.
- DRD event order and dependencies remain untouched, so synchronization and
  parallelism continue to come from the observed DRD execution.
- No rendered values, renderer scheduling, benchmark workload, or numerical
  rendering path changes.
