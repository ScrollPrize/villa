# Plan: preload and parallelize fiberlet DP

## Baseline and invariants

1. Retain the measured reference crop and its path counts/scores as the
   baseline: 676 searched and accepted paths, score min/mean/max
   3.18031/11.0126/20.8916; 11.65 s with default threading and 9.38 s with
   `--threads 1` in the current Release build.
2. Preserve candidate ordering, endpoint filters, corridor membership, DP
   traversal/tie behavior, full cost breakdowns, paths, and serialized output.
   Operational timing and worker scheduling are not artifact fields.

## One-time dense preload

3. Separate deterministic candidate generation from candidate solving. Record
   every axis-compatible candidate index without solving it yet.
4. Extract the existing Hermite corridor bound calculation into one shared
   helper used by both preload-bound collection and the solver. Union the
   inclusive clipped bounds of every searchable candidate and every eligible
   virtual endpoint attachment voxel so narrow corridors below `sqrt(3)` keep
   their current search domain.
5. Allocate a checked dense ZYX scoring volume for the rectangular enclosing
   box of that union. Build stored
   prediction indices and matching normal sample points in deterministic ZYX
   order, then invoke the fiber sampler and Lasagna normal batch sampler once
   each with the configured thread count.
6. Store immutable `FiberStoredPredictionSample` values plus `cv::Vec3d` normal
   vector/validity without any narrowing, renormalization, or reordered
   calculation, and without
   retaining per-voxel diagnostic strings. Make `solveCandidate` use direct
   checked dense-volume lookup for every corridor node; remove all sampling
   and sampler thread creation from it.
7. Add a thread-count-aware virtual normal batch overload with a compatibility
   default delegating to the existing batch method. Override it in
   `LasagnaNormalSampler` using its existing implementation, avoiding a type
   test or duplicated Lasagna sampling logic.

## Candidate parallelism

8. Pre-size the candidate result array, then solve searchable candidate indices
   through a fixed `min(--threads, searches)` worker pool. Each worker writes
   only its assigned existing result slot. Capture per-task exceptions and
   rethrow the lowest candidate-index failure after joining so failure choice
   is deterministic.
9. Derive success/failure diagnostics in a deterministic serial pass after all
   workers finish. Do not nest OpenMP or asynchronous sampling inside the path
   workers.
10. Expose preload voxel count and preload/search timing in CLI runtime output
    to make subsequent optimization measurable; do not serialize timings.
11. Add an optional core progress callback driven by completed candidate tasks.
    Serialize reporting separately from result writes, emit monotonic updates no
    more than about once per second, and always emit the final completion. The
    CLI prints completed/total, percentage, elapsed seconds, candidates per
    second, and ETA seconds to stderr; progress remains outside artifacts.
    Search timing begins after preload. Workers count attempted tasks even when
    solving fails, suppress stale callback observations under the reporting
    mutex, and capture callback exceptions rather than letting them escape.
    The coordinator emits the sole terminal update before deterministically
    rethrowing candidate then callback errors. A zero-search run emits `0/0`,
    100 percent, zero rate, and zero ETA.

## Tests and validation

12. Add focused tests proving prediction and normal sampling each occur once,
    all sampled indices are unique and cover every searched corridor bound,
    one-thread and multi-thread reports have identical candidates/costs/paths,
    zero-search input does not sample, and a corridor extending beyond the
    selected anchor crop remains valid. Cover narrow-corridor attachment voxels,
    nonzero dense origins and both grid edges, unique ZYX enumeration, the
    generic normal-batch compatibility path, worker overlap/bounds, and
    deterministic exception selection where practical without production test
    hooks.
    Also verify progress is monotonic and ends at the exact search total.
13. Build with 32 jobs and run the anchor, fiberlet-path, and native tracer
    suites. Re-run the Release reference crop with one and default worker counts,
    compare artifacts excluding runtime output byte-for-byte, and report three
    runs of wall/CPU/memory with min/median/max, path counts, score summaries,
    Release flags, and the source-level hotspot analysis (system `perf` is not
    installed).

## Spec update

- Specify one-time dense scoring-volume materialization over the union of
  candidate corridor bounds, immutable lookup during DP, fixed candidate-level
  parallelism, and deterministic result placement/failure selection.

## Docs update

- Document `--threads` as preload plus candidate-worker parallelism, the current
  dense-memory test-crop assumption, corridor-bound padding behavior, and the
  runtime preload/search diagnostics.

## Changelog and workflow

- Add the exact-output preload/parallel-search acceleration to the current
  fiberlet path changelog entry and record measurements and deviations in the
  current task log.
