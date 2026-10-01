# Plan: training-integrated reviewed failure benchmark

## Dataset

1. Define the canonical source evaluations as `lowres`, `fullres`, `gaussian`,
   `halfsigma_raw`, and `halfsigma_ema99` from reviewed cohort v3.
2. Select a whole truncated piece when either direction has at least one saved
   failure in any canonical source evaluation; deduplicate by scroll and source
   index. The expected frozen membership is 32 pieces.
3. Materialize those exact JSON files under a new versioned local test root and
   write a manifest containing source evaluations, source paths, and SHA-256.
   Validation must fail on missing, extra, or changed files.

## Dense training tests

4. Add a derived training configuration whose `test_datasets` point only to the
   frozen failure-union root. Retain the current deterministic dense loss,
   refinement, EMA, cross-section, and TensorBoard implementations unchanged.
5. Make the expected test membership explicit so accidental glob growth fails.

## Native trace test

6. Extract a reusable training-trace benchmark adapter around the existing
   `run_classic_fiber_evaluation.py` preparation/live inference/native replay
   implementation; do not copy its inference or tracing logic.
7. At a configured interval, rank zero freezes the current raw model, prepares
   a per-step workload over the same frozen 32 pieces, and invokes the existing
   optimized runner as a child of the already managed training task. Other DDP
   ranks remain responsive through periodic distributed completion polling.
8. Parse only complete summaries with all 64 directed cases and no
   infrastructure failures. Persist the workload, bindings, summary, inference
   summary, and compact metrics under the training run.
9. Log forward/reverse/combined restarts, reference lengths, and implied mean
   lengths to stdout and TensorBoard. The combined implied length is total
   directed reference millimetres divided by `(combined_restarts + 1)`.
10. Bound transient prediction storage and fail loudly rather than silently
    skipping a trace test.

## Validation

11. Unit-test failure-union selection, manifest validation, directional restart
    aggregation, implied-length arithmetic, subprocess completion/failure, and
    distributed polling without requiring CUDA.
12. Run focused Python tests and existing native replay tests. Then stop the
    currently running old-test configuration and launch a resume using GPUs
    0-5. Confirm one dense test and one 64-direction trace test complete.

## Spec update

- Add the immutable 32-piece reviewed-failure test membership, selection bias,
  canonical source evaluations, trace metrics, and no-fallback requirements.

## Documentation updates

- Add the training-integrated benchmark to `docs/benchmark_runbook.md`, clearly
  identifying it as a failure-enriched regression metric rather than an
  unbiased generalization estimate.

## Changelog

- Record the locked failure-union dataset and training-integrated native trace
  metrics.
