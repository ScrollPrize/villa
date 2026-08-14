# Plan: stage-parallel fiberlet extraction

## Implementation

1. Replace per-batch domain construction with a canonical prepared-candidate
   representation containing one Hermite domain, a corridor derived from that
   domain, every local key and mapped prediction-space point, and immutable
   interpolation stencils. The DP stage must not recreate any of them.
2. Prepare all searchable candidates with a fixed worker pool. Each job writes
   only its canonical slot and accumulates native corners into its worker-local
   set. Join all workers and rethrow the lowest candidate-index error.
3. Sort each worker-local corner set and merge those vectors with a
   deterministic pairwise merge tree. The final stored-ZYX ordered vector is
   the complete global unique request set. Its contents and size are invariant
   under worker and sampler batch settings, including non-power-of-two worker
   counts.
4. Rename the internal field to `samplingBatchCoordinates` while keeping the
   CLI spelling `--batch`. It limits only consecutive ranges of the global
   unique coordinate vector. Complete every prediction range first, then every
   normal range, then materialize prepared node scores from the retained
   stencils. Release coordinate, sampled-value, and stencil storage before DP
   where lifetimes allow. No compatibility handling is required for this
   experimental CLI/config field.
5. Run DP across all candidates with a fixed worker pool. Pass each candidate's
   retained domain, nodes, and materialized scores to the solver; the corridor
   is constructed once from that domain for node enumeration and then released.
   Keep canonical result slots, phase-labelled progress, and deterministic
   lowest-candidate-index error behavior.
6. Replace misleading sampling metrics with explicit stage metrics: candidate
   generation, parallel preparation, global corner merge, prediction reads,
   normal reads, lookup materialization, and DP. Report wall seconds, process
   CPU seconds/effective cores, unique requests, sampler calls, and estimated
   persistent/peak temporary bytes owned by the extraction structures.

## Tests

1. Prove output JSON, OBJ, and graph JSON are byte-identical across worker
   counts and very small/large coordinate batch limits.
2. Independently concatenate prediction calls and normal calls and prove each
   equals the same globally sorted unique coordinate vector exactly once.
   Prove total requests are invariant and call counts are
   `ceil(unique_coordinates / batch_limit)` for limits one, a non-divisor, and
   at least the global count, across worker counts.
3. Prove domains/local nodes are prepared once per searched candidate and reused
   by DP through an operational preparation count.
4. Preserve shared interpolation corners, selection-boundary interpolation
   corners, graph transitions across coordinate ranges, phase progress, error
   ordering, and invalid-config coverage.
5. Build with `-j32`, run all C++ fiber tests and focused viewer tests, then run
   the fixed 512-base-voxel Paris4 benchmark with thread and coordinate-batch
   comparisons. Use a Release build, three measured repetitions after one warm
   run, and report wall time plus actual process CPU/effective cores.

## Spec Update

- Replace candidate-batched preparation/sampling semantics with the global
  prepare, unique-corner merge, coordinate-sampling, and DP stages.
- State that batch size cannot affect unique request count or artifacts.
- Document stage CPU/wall metrics and checked owned-allocation estimates.

## Documentation Updates

- Update `volume-cartographer/docs/fiberlets.md` with the staged pipeline,
  revised `--batch` meaning, metrics, and benchmark result.
- Update the planning changelog after validation.
