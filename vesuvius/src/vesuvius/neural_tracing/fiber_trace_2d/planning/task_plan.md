# Plan: parallel constraint scoring and OBJ diagnostics

## Performance implementation

1. Preserve the existing tangent and phase-refinement arithmetic. Use the
   existing float-coordinate grouped Lasagna sampler for winding, as explicitly
   accepted by the user; quantify the resulting difference from scalar-double
   winding in tests and the representative run.
2. Materialize each connector's integration coordinates after spatial candidate
   selection and process them in bounded batches.
3. Add a batch API to `LasagnaNormalSampler` that calls the existing grouped
   corner sampler once per batch for `grad_mag`, `nx`, and `ny`, materializes
   aligned winding density in parallel, and integrates all connectors. Do not
   perform per-point channel-cache lookups.
4. Retain deterministic per-candidate output slots and parallelize independent
   scoring with host CPU count by default. Use coarser guided OpenMP scheduling
   to reduce scheduling overhead while leaving each candidate's arithmetic and
   result order unchanged.
5. Report prefetch and parallel-score time separately and benchmark the same
   500-trace artifact used for the baseline commit.

## OBJ outputs

6. Add a reusable core classifier/writer that converts accepted measured links
   to named two-point polylines. Exclude hard continuity links.
7. Treat `--output PATH` as a basename: remove its final extension, if any, and
   append three literal suffixes:
   `_perpendicular.obj`, `_parallel_same_winding.obj`, and
   `_parallel_separate_winding.obj`. When omitted for `TRACE.zarr`, use
   `TRACE_constraints` beside the input dataset.
8. Use strict score `>0.5`; use strict winding `>0.3` for perpendicular links,
   `<0.5` for parallel same-winding, and `>=0.5` for parallel separate-winding.
   Every parallel link therefore enters at most one parallel file.
9. Name every OBJ object `constraint_piece_A_B` using ascending stable global
   piece IDs. Print all paths and line counts.

## Tests

10. Verify connector classification at threshold boundaries, hard-link
    exclusion, filenames, OBJ line counts, and empty valid outputs.
11. Compare batched winding distances to the scalar implementation within a
    documented float-coordinate tolerance and verify exact one-thread versus
    multi-thread batch output parity.
12. Build and run focused tests with GCC and Clang, run `git diff --check`, and
    benchmark the representative crop with the Release binary.

## Spec update

Extend the stored crop-trace constraint specification with grouped float-point
sampling, deterministic parallel scoring, timing fields, the three diagnostic
selection predicates, output naming, and hard-link exclusion.

## Documentation updates

Document `--output`, default naming, connector semantics, all three thresholds,
and the performance behavior in `docs/fiber_chunk_tracing.md`.

## Changelog

Extend the 2026-08-26 entry with grouped batched scoring and the three
constraint OBJ diagnostics.
