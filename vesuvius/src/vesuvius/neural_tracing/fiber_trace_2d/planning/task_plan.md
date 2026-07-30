# Native Fiber Trace Locality And Scheduling Optimization Plan

## Baseline And Acceptance

- Use the final optimized cap-32 result as the retained baseline: 0.986s median
  wall / 5.134s median CPU, 6,910,839 candidates, 4,318 generations, and 7
  restarts over 87 segments. The task started from 1.869s wall / 8.222s CPU at
  the same workload and quality.
- Retain only results that preserve the 7-restart baseline.
- For result-neutral phases, require identical trace output, candidate count,
  generation count, and restart count.
- Preserve candidate generation order, original global candidate indices,
  reached-state ordering, spatial pruning semantics, corner order,
  interpolation, missing/error handling, and float math.
- Reuse the exact approved representative command and existing remote cache.
- Before every representative benchmark, sample host CPU utilization and the
  runnable queue. Run directly when the host is quiet under the user's current
  continuing permission; if a compile or other significant CPU workload is
  active, defer the run. Builds, unit tests, and synthetic microbenchmarks do
  not imply that a contaminated representative result is usable.

## Phase 1: Result-Neutral Measurement

Add low-overhead profile counters before choosing implementation details:

1. Record candidate batch sizes by depth and submitted worker-task counts.
2. Record time spent ordering capped parents, allocating/clearing full frontier
   storage, and mapping selected children back to parents.
3. Record per-depth unique chunk keys, unique integer voxel cubes, candidates
   per cube, and p50/p95/max cube reuse.
4. Record dependency overlap between depth one and depth two, plus overlap with
   the following trace step.
5. Keep instrumentation result-neutral and removable or profile-gated.
6. Run focused tests, apply the load gate, then run one instrumented
   representative benchmark. Use those measurements to order Phases 2-4.

## Phase 2: Scheduling And Frontier Overhead

Test each change separately and retain only measured improvements:

1. **Worker granularity**
   - Choose worker count from a minimum candidates-per-worker threshold instead
     of always using every available corner-batch worker.
   - Keep static point ranges and one callback per original candidate.
   - Sweep a small set of thresholds with synthetic tests first; apply the load
     gate before benchmarking only the best candidate.
2. **Top-K parent selection**
   - Replace full parent sorting with deterministic partial selection for the
     configured cap.
   - Sort the retained prefix by `(lower_bound, original_parent_index)` after
     selection so behavior matches the current first 32 exactly.
   - Keep full sorting for uncapped exact lazy mode if it remains simpler.
3. **Compact capped frontier**
   - Store only evaluated child tasks, scores, frontier records, and their
     original global indices.
   - Compare original global indices for equal loss/depth ties.
   - Reconstruct selected states through the compact-to-parent mapping.
   - Keep the current full-index path for exhaustive mode if necessary.
4. **Fused final-frontier construction**
   - Build final frontier records in the existing parallel corner-score
     callback instead of scanning every evaluated score in a serial follow-up
     pass.
   - Preserve compact task/score/frontier alignment and original global child
     indices exactly.

After each retained implementation, run focused tests and apply the load gate
before the unchanged representative benchmark.

## Phase 3: Spatial Sampling Locality

1. Build a stable spatial permutation for each candidate batch using
   `(layout/chunk key, integer voxel cube, original candidate index)`.
2. Gather in spatial order and scatter scores to original candidate indices;
   beam/frontier code must never observe the spatial permutation.
3. If measured cube reuse is material, build one record per unique integer
   voxel cube:
   - gather the ordered eight corners once per physical scalar volume;
   - retain each candidate's interpolation fraction and original index;
   - decode and score every candidate sharing the cube from those corners.
4. Reuse the existing `ChunkedPlaneSampler` layout/cache machinery. Extract a
   shared helper or session; do not copy its private dependency or pin logic
   into the fiber tracer.
5. Cover chunk boundaries, volume edges with clamped corners, mixed physical
   chunk grids, missing chunks/fill values, malformed chunk extents, and
   multiple fractions sharing one cube.
6. Require score and selected-trace parity before benchmarking.

Measurements may justify unique-cube reuse before a separate spatial
permutation. If corner records are already gathered once per unique cube,
retain a permutation only if it produces an additional measured improvement.

## Phase 4: Persistent Two-Depth Sampling Session

The second-depth coordinates depend on first-depth sampled directions, scores,
and selected parents, so retain one explicit decision barrier. Optimize around
that barrier rather than speculatively evaluating all second-depth parents.

1. Introduce a bounded pinned-corner session owned by one lookahead step.
2. Process depth one spatially, select the capped parents, then append only new
   depth-two dependencies to the same session and process depth two.
3. Measure optional prefetch of the conservative depth-two coordinate envelope
   while depth one is being scored. Reject it if overfetch outweighs overlap.
4. Measure a small budgeted rolling pin window across consecutive trace steps.
   It must integrate with the existing decoded-cache budget and release pins
   deterministically when the window advances.
5. Keep generic sampler fallbacks unchanged.
6. Test session lifetime, budget release, error propagation, and output parity.

## Phase 5: Search Experiments

Run only after result-neutral work is measured:

1. Test fixed caps 28, 24, and 20 independently, in that order.
2. Stop descending after a clear quality failure unless the user explicitly
   requests further trials.
3. If useful, test adaptive escalation:
   - start with the best smaller cap;
   - retry at cap 32 or 64 on no target, restart-worthy endpoint error, weak
     selected-loss margin, or selection at the retained-prefix boundary;
   - record retry count and which trigger fired.
4. Adaptive behavior must be deterministic and exposed as an explicit config/
   CLI mode until representative quality is established.
5. Retain a new default only at 7 restarts and after an explicit spec
   update.

## Phase 6: Unit-Vector Math Trial

1. Remove redundant normalizations only where the immediate caller has already
   normalized every vector.
2. Treat this as numeric-relaxed: direct dot products may differ by float
   rounding from renormalizing the same value again.
3. Require the current 7-restart quality baseline and compare candidate/
   generation counts to identify any changed search path.
4. Retain only with a material speedup; otherwise restore the defensive
   normalization helpers.

## Testing

- Build `vc_fiber_trace_metric`, `test_fiber_trace3d`,
  `test_chunked_plane_sampler_fallback`, and `test_lasagna_normal_sampler`.
- Run all three focused test binaries after each retained code change.
- Add exact output-parity tests for top-K selection, compact frontier mapping,
  spatial permutation/scatter, cube reuse, and persistent sessions.
- Run `git diff --check` and review portability for Ubuntu/macOS and amd64/arm64.
- Do not add architecture-specific SIMD in this task unless it is guarded with
  a portable fallback and separately measured after the listed phases.

## Performance Protocol

- Apply the host-load gate immediately before every representative benchmark.
  Run directly only when the host is quiet; otherwise defer until resources are
  available.
- Use the exact existing command, fiber manifest, fiber JSON, normal manifest,
  and remote cache path. Do not substitute paths or add experimental flags;
  compile isolated defaults for trials when required.
- Record wall/CPU time, stage timing, candidates, generations, dependencies,
  unique chunks/cubes, evaluated parents, and restarts.
- When resources are stable, obtain three retained-final repetitions and report
  min/median/max. Mark runs overlapping unrelated compilation or heavy work as
  contaminated and exclude them.
- Log failed and neutral experiments as well as improvements.

## Spec Update

- Document spatial reordering and compact storage as internal result-neutral
  implementations that must preserve original global indices and tie behavior.
- If retained, document shared-cube interpolation semantics and the bounded
  two-depth pinned-session lifetime/budget rules.
- Update cap/adaptive-search semantics only for retained behavior. Preserve
  `--lookahead-parent-cap 0` exact lazy mode and full exhaustive mode.

## Docs Updates

- Replace the active task files with this continuation; retain prior results
  only in `planning/changelog.md` and git history.
- Update `docs/code_structure.md` for any retained sampler-session or grouping
  architecture.
- Add one durable changelog entry summarizing retained improvements, not every
  experiment.

## Review

- Review this plan directly against `planning/specs.md`, `planning/plan.md`,
  `planning/task.md`, and the volume-cartographer portability requirements.
- Independent-agent review is unavailable in the current tool context; record
  that workflow deviation and perform a direct consistency review before code.
