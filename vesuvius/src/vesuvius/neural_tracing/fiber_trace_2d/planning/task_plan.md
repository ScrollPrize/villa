# Plan: raw best checkpoint by reviewed-trace restarts

## Implementation

1. Before either reviewed-trace path starts GPU work, atomically freeze the
   exact raw checkpoint into its attempt directory and bind its SHA256 into the
   job. The benchmark writes a completion marker only after validated metrics
   and cleanup have both completed.
2. Add one shared reviewed-trace checkpoint-selection/finalization helper used
   by managed cooperative training and the unmanaged/test execution path. Bind
   results to a benchmark identity covering the effective workload and hashes
   of the locked split manifest, native binary, and runner; reject comparison
   against a prior best from a different benchmark identity.
3. After a reviewed trace completes successfully, compare its nonnegative
   integer `restarts_combined` against the prior trace-selected raw best.
   Strictly fewer restarts wins; a tie retains the existing checkpoint.
4. Materialize the exact raw checkpoint evaluated as
   `snapshots/best_reviewed_trace.pt`. Make its checkpoint metadata identify
   `reviewed_trace/restarts_combined`, the evaluated checkpoint digest, step,
   sample count, benchmark identity, and complete reviewed-trace metrics. The
   checkpoint itself is authoritative and atomically replaced; mmap its small
   metadata for comparisons rather than maintaining fallible duplicate global
   state. Each result records its selection outcome.
5. Make successful result finalization idempotent. On startup, reconcile a
   completed attempt that was interrupted between checkpoint selection and
   canonical result promotion, while never selecting aborted/incomplete work.
   Delete non-needed frozen attempt checkpoints only after durable selection
   outcome and canonical promotion.
6. Preserve `best.pt`, `current.pt`, and all EMA snapshot selection behavior.
   Expose `best_reviewed_trace.pt` as the `best_trace` variant in the existing
   trace-series evaluation workflow.

## Validation

7. Add CPU-only tests for first selection, strict improvement, tie/regression,
   restart recovery at transaction boundaries, benchmark-identity mismatch,
   checkpoint mutation/digest mismatch, malformed metrics/state, raw-only
   enforcement, incomplete/aborted work, and both reviewed-trace paths.
8. Test `best_trace` selection metadata and confirm existing raw/EMA best files
   are untouched. Run the focused reviewed-trace and experiment-selection tests, Python
   compilation, and `git diff --check`. No GPU run is required because the
   change occurs after already-validated trace output is produced.

## Spec update

- Specify the independent raw trace-selected checkpoint, its exact metric,
  strict tie policy, startup eligibility, benchmark identity, persistence and
  recovery behavior, and separation from dense test-loss and EMA selection.

## Documentation updates

- Document the new snapshot filename and selection semantics in the cooperative
  training validation/runbook.

## Changelog

- Record the additional restart-selected raw checkpoint.
