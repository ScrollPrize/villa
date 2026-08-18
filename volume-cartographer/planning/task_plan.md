# VC3D render attribution and lookup repair plan

## Phase 1: deterministic paired attribution

1. Keep the benchmark binary unchanged. Run Callgrind and DRD with the same
   `--fair-sched=yes` and `--scheduling-quantum` values and the same benchmark
   arguments. Enable periodic Callgrind dumps at the fixed basic-block interval
   without adding application markers.
2. Parse the benchmark's existing paired `steady_clock`/`clock_gettime` calls
   from the DRD syscall trace and trim the dependency graph to the measured
   render only. Require one unambiguous boundary pair; drop outside dependencies
   as already-satisfied and renumber the retained graph.
3. Parse periodic Callgrind dumps as chronological per-thread event-cost slices.
   Preserve every event counter and require their sum to equal the complete
   measured Callgrind profile.
4. Keep main thread 1 as the stable control role. Capture the passive scheduler
   stream in the Callgrind execution as well as DRD, enumerate all four-worker
   assignments, and score them from normalized activity share and cumulative
   measured-window activity. Replay every mapping within scheduler-quantum
   resolution of the best score and use the maximum replay makespan. Fail the
   case when compatible mappings exceed the predeclared makespan-spread limit.
5. Resample each matched chronological Callgrind cost trace over that DRD
   worker's measured eligible windows while preserving order and exact total
   cost. Keep original DRD thread IDs, program order, blocking, and dependency
   edges. Implement direct per-window attribution in the native replay engine.
6. Keep the regular collection and evaluation path native: the C++ tool parses
   raw periodic Callgrind profiles and the raw DRD trace, validates and matches
   them, performs attribution/replay, and writes the evaluation artifact.
   CMake/Ninja invokes Valgrind and the native tool directly. Python remains
   available only for offline calibration and is not part of the CI estimate.
7. Collect repeated paired runs for every parallel scenario. Require stable
   canonical signatures, exact cost conservation, and stable modeled makespan.
   A pair that cannot prove reconstruction is rejected rather than averaged.
8. Retain generic per-thread attribution for other replay users and keep serial
   evaluation unchanged. Keep role pooling only as a diagnostic comparison,
   not the renderer gate's final attribution.
9. Freeze a new reference only after deterministic pairing and repeatability are
   accepted, then re-enable the workflow gate in a separate reviewable change.

## Phase 2: lookup performance

1. Build the pre-render-order commit, the render-order commit, and the repaired
   head with the same GCC Release configuration. Run serial and parallel native
   trials repeatedly and collect Callgrind counters/checksums for all scenarios.
2. Separate synthetic-fixture overhead from production behavior. The synthetic
   array currently inherits the contextual `tryGetChunk()` forwarding overload,
   while production `ChunkCache` overrides it directly; benchmark both the
   production-equivalent direct overload and the generic compatibility path.
3. Profile before choosing among these compatible optimizations:
   - a per-tile/per-level successful-chunk cursor containing chunk bounds,
     resolved payload, strides, and full key, checked before coordinate division
     and key construction;
   - a source-bound prepared lookup handle containing render-job request,
     source ID/state, and level invariants;
   - a fixed-capacity linear/small-vector local chunk cache in place of hashing
     for the current maximum of eight pinned chunks.
4. Implement one optimization at a time. Keep the old path as the correctness
   oracle and compare exact image checksums plus requested/missing/error counts.
5. Accept only changes with repeatable native and modeled improvement. Report
   command, build type, input scenario, repetitions, and mean plus distribution
   statistics.

## Testing and validation

- Native replay unit tests for direct window-cost conservation, chronological
  placement, invalid window vectors, and unchanged generic per-thread
  attribution.
- Native parser tests for periodic Callgrind dumps and passive measured-window
  trimming, plus native matching tests for scheduler-selected assignments,
  scheduler ties, material makespan ambiguity, chronological attribution, and
  exact event-counter conservation.
- Repeated complete render matrices before freezing any reference.
- Exact synthetic-render checksums and existing `ChunkedPlaneSampler` and
  `ChunkCache` tests for speed changes.
- Build relevant C++ targets with all 32 cores and run `git diff --check`.

## Specification updates

Update `specs/benchmarks.md` so the renderer-gate contract requires passive
deterministic paired reconstruction, measured-window trimming, canonical
logical-worker matching, chronological per-window cost placement, exact cost
conservation, and rejection of ambiguous pairs. Raw worker IDs never cross
independent-run boundaries. Update `planning/spec.md` only if the accepted
lookup optimization adds a renderer requirement not already covered there,
including the measured no-numeric-change requirement.

## Documentation updates

Update `docs/thread_sync_replay.md` and
`docs/benchmarks/render_valgrind_ci.md` with paired-trace attribution,
repeatability/rebaseline procedure, matching diagnostics, and the reason raw
worker IDs cannot cross independent Valgrind runs. Document any new prepared
lookup API beside the renderer implementation.

## Changelog update

Add one dated line when attribution is corrected and the stable gate is
re-enabled. Add a separate measured lookup-performance line only if the second
phase produces an accepted code change.

## Independent review

Pending review of this plan against the current replay implementation, passive
measurement constraint, benchmark specification, complete-cost invariant, and
strict ambiguity rejection.
