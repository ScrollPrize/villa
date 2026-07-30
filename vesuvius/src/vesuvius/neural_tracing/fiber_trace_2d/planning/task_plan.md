# Native Fiber Trace Lookahead And Pipeline Optimization Plan

## Baseline And Acceptance

- Use commit `2bf48dea0` as the baseline: 21.155s wall / 619.366s CPU,
  105,810,462 candidates, 4,170 generations, and 8 restarts over 87 segments.
- Use only the approved remote-manifest/local-fiber benchmark command and its
  existing remote cache.
- Retain deterministic candidate generation, loss/tie ordering, and target
  selection unless an approximate intermediate-cap experiment is explicitly
  being measured.
- Reject any retained implementation above 8 restarts.
- Measure each option separately. Record wall/CPU time, stage timing, candidate
  count, expanded lookahead parents, and restart count.

## Phase 1: Exact Lazy Lookahead

1. Add result-neutral instrumentation to the exhaustive lookahead path.
   Candidate incremental losses are nonnegative, so an intermediate parent's
   cumulative loss is a lower bound for every descendant.
2. From each exhaustive final frontier, compute the conservative exact parent
   prefix required to reproduce the result:
   - for reached-target generations, use the best reached loss as the bound;
   - otherwise use the worst spatially accepted final beam loss;
   - include every parent whose lower bound is less than or equal to the result
     bound so equal-loss candidate order remains observable.
3. Report total/mean/p50/p95/max intermediate parents and the predicted child
   candidate reduction. Benchmark this instrumentation without changing the
   trace result.
4. If the reduction is substantial, implement batched lazy parent expansion.
   Preserve original global child indices for deterministic ties, expand
   parents by `(lower_bound, original_parent_index)`, and stop only when the
   next lower bound is strictly greater than the established exact threshold.
5. Add focused tests comparing lazy and exhaustive results, including equal
   bounds, spatially rejected candidates, reached-target selection, and cases
   where all parents must be expanded.

## Phase 2: Fused Sample And Score

1. Profile remaining memory traffic after lazy expansion.
2. Extract a shared pinned-corner batch context from the existing VC3D sampler;
   do not duplicate Zarr/cache fetching behavior.
3. For persisted fiber tracing, fuse per-candidate coordinate flooring,
   fraction calculation, pinned corner gathering, compact direction/normal
   decode, and loss calculation in one parallel pass.
4. Avoid materializing full candidate-sized voxel-cube, dependency, six-volume
   corner, decoded-sample, and score arrays where downstream selection does not
   require them.
5. Preserve ordered eight-corner semantics, missing/error handling, compact
   orientation-tensor interpolation, and generic sampler fallbacks.
6. Test boundary, clamped-edge, missing-chunk, malformed-chunk, and fused versus
   existing score parity before benchmarking.

## Phase 3: Approximate Intermediate Cap

- Run only if exact lazy expansion and fusion do not approach the requested
  speedup.
- Add an opt-in deterministic intermediate parent cap; do not silently change
  default search semantics.
- Test caps `64`, `32`, `16`, and `8` independently in that order.
- Compare runtime, candidates, and restarts. Retain a default behavior change
  only if it stays at or below 8 restarts and the spec is updated explicitly.

## Testing

- Build `vc_fiber_trace_metric`, `test_fiber_trace3d`,
  `test_chunked_plane_sampler_fallback`, and `test_lasagna_normal_sampler`.
- Run all three focused test binaries after each retained implementation.
- Run the approved benchmark after each isolated option.
- Run `git diff --check` and review the final diff for deterministic ordering,
  portability, and accidental changes outside the active task.

## Spec Update

- Exact lazy expansion requires no search-semantics change: document that
  configured lookahead may be evaluated lazily using a proven nonnegative lower
  bound, but must return the same reached state and spatially pruned beam set as
  exhaustive expansion.
- Document fused sampling/scoring only if retained, requiring the same ordered
  corners, interpolation, error handling, and deterministic scores.
- Do not change the intermediate-pruning spec for an experiment. Update it only
  if an approximate cap is retained as intentional behavior.

## Docs Updates

- Replace the active task log with this task's measurements and deviations.
- Keep prior optimization history only in `planning/changelog.md`.
- Update `planning/status.md` incrementally.
- Add a concise changelog entry for retained improvements, not each rejected
  experiment.

## Review

- Review this plan directly against `planning/specs.md`, `planning/plan.md`, and
  the task before implementation.
- Independent-agent review is unavailable in the current tool context; record
  that workflow deviation and perform a direct consistency review.
