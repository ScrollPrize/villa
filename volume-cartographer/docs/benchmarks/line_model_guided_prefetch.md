# Guided line-model prefetch experiment

## Plan and constraints

Continue locally from `cdbffc05b` on `experiment/line-model-guided-prefetch`.
The independent `line-annotation-strip-vertical-pin` branch is unchanged. No
push, installation, original-fiber edit, or application-cache clearing is allowed.
The frozen PHerc1299 eight-control, 3735-point fiber and model manifests under
`/tmp/villa-pherc1299-prefetch.IJmtzD/baseline-on/input` remain the canonical input.

1. Obtain fresh Codex/Fable plan reviews and incorporate relevant feedback.
   Reuse the existing shared scheduler, isolated real-remote benchmark runner and
   exact-output validator. Recheck the frozen previous candidate build as baseline.
2. Extract a bounded, no-I/O path predictor in `ModelPrefetch.*`. Keep a selectable
   straight mode reproducing the old scheduling policy for same-binary A/B.
   Add direction-change refresh before the distance-only threshold, with a small
   minimum displacement to avoid repeatedly restarting plans on beam jitter.
3. Pass an immutable view of the existing CP-span reference curve into native
   one-way prefetch only. Orient it to the trace direction, retain bounded samples,
   and follow nearby, direction-compatible reference geometry. Bound search work
   and progress to avoid jumping to another branch at a crossing. Fall back to the
   current tangent when the new trace leaves the reference. Never modify trace
   requests' numerical fields, candidates, endpoint acceptance, or solver math.
4. Independently test a curvature-aware mode for paths without usable reference
   guidance, especially open tails. Use several recent observed directions with
   bounded history/turn and damped, short-horizon extrapolation. Do not sample or
   download model data just to make a prediction. Keep all existing byte, request,
   planning and cancellation bounds and required-read failure behavior.
5. Add counters for refresh causes and reference/curvature use. Record the chosen
   mode in GUI/CLI logs and benchmark configuration. For coverage, compare the
   disabled run's observed remote payload set with prefetched cache inventories;
   report additional fetched bytes for this input separately from total bytes and
   wall time. This is an excess-traffic proxy, not proven unused bytes: legacy
   normal prefetch remains active even with the newer prefetch disabled.
   Arrival-before-use is not the same as eventual coverage;
   do not infer per-request usefulness from process-wide remote counters.
6. Unit-test geometry orientation, bends, invalid/duplicate inputs, deviation,
   reference endpoint/length caps, refresh throttling, curvature limits, optional
   failure and cancellation. Run exact cold/warm comparisons on the frozen fiber,
   first short screening pairs, then at least three paired trials per retained
   mode, alternating groups and reporting min/median/max and mean. Include a
   full-retrace equivalence check if practical; do not call saved-state solves a
   replay of the user's unavailable historical click sequence.
7. Keep a new default only if measured latency/traffic supports it. Experimental
   curvature can remain opt-in or be removed if it regresses. Repeat completed-code
   Codex/Fable reviews, address findings, run regression tests and build VC3D.
   Preserve the overall `VC3D_LINE_MODEL_PREFETCH=0` startup opt-out.

First scope is reference guidance, turn refresh, and measured tail curvature.
Multi-candidate corridors, extra model/coarse-level reads, and adaptive network
horizons are follow-on candidates, not bundled with this comparison: otherwise
we cannot attribute a benefit or regression to the projection change.

## Reviews, implementation and measurements

Codex plan review: no blocker. Incorporated explicit subspan orientation by control
indices, bounded local arc-progress matching (not global nearest), independent
limits on input inspection/reference storage/search/output/history, no jumping
across invalid data or appending an endpoint after truncation, and preserving the
old exact scheduling in straight mode. Direction thresholds compare against the
current plan, and curvature uses spatial history rather than call count.

Fable plan review: no starting blocker. Incorporated: the frozen saved-span
reference is optimistic compared with a fresh edit, so savings on it cannot be
called historical click-replay savings. Native tails can use the already-built
Lasagna tails as guidance too; pass them without altering or recomputing them.
Use the leading beam's existing smoothed history direction for guided forecasts
to reduce refresh churn on beam swaps. Keep normal-growth windows straight in
all modes to isolate this experiment to native projection. Straight mode keeps
the exact previous instantaneous direction and distance-only policy; guided adds
turn refresh and span/tail reference guidance; curved adds bounded spatial-history
extrapolation when reference guidance is unavailable. No adaptive network horizon
or beam-union change is included. The earlier excess-traffic wording follows
Codex's caution: disabled fetches are not an exact oracle of true demand.

Completed-code reviews found no correctness blocker. Codex requested a frozen,
hashed inventory and resolved path for the disabled traffic reference, recording
the effective `straight` mode when omitted, and a stronger crossing assertion
whose endpoint differs on the wrong branch; all incorporated. Fable requested
avoiding turn-only refresh when the current corridor already follows a matched
reference bend. Those plans now retain the 64-trace-voxel distance refresh;
unmatched forecasts can refresh after a >15-degree turn and at least 16 voxels of
movement. A stale reference is reconsidered at the next distance refresh, not
at every direction change. Added a regression test for that tradeoff. Also
excluded zero-horizon reference fallbacks, clarified the 15-degree integrated
curvature cap, and added optional prediction/traffic aggregates to the runner's
summary. Both Codex and Fable follow-up reviews found no remaining blocker.

Reference inspection/storage is capped at 2048 oriented input points; each
reference match and emitted path is capped at 128 segments. Invalid samples end
the usable prefix; duplicate points do not extend the inspection budget. Dense
or truncated references can therefore produce a shorter horizon than 256 trace
voxels. This intentionally trades speculative coverage for bounded work; it
does not truncate the actual fiber or relax its numerical criteria. Curvature
keeps at most 8 spatial observations and emits 8 segments, with coherence checks,
reversal/jump resets, and damped integrated turn capped at 15 degrees.

The implementation lives in the shared `ModelPrefetchPredictor`/window. Native
span/tail callers pass borrowed references that are copied only inside the
existing optional-allocation guard; no solver input or trace arithmetic changes.
GUI and CLI logs expose the projection, replans, turn refreshes, matched and
fallback reference plans, and curved plans. Original scheduling remains the
default; the two new modes are explicit startup experiments.

## Reproduction and interpretation

Build with the existing configured dependencies (no installation):

```bash
AGENTS_AGENT_MODE=1 cmake --build volume-cartographer/build \
  --target VC3D vc_fiber_trace_metric test_model_prefetch -j6
```

Launch the GUI with one projection per process:

```bash
VC3D_LINE_MODEL_PREFETCH=1 VC3D_LINE_MODEL_PROJECTION=guided VC3D_LINE_PERF_LOG=1 \
  volume-cartographer/build/bin/VC3D
```

Replace `guided` with `straight` (default) or `curved` (reference guidance plus
curvature fallback). Unrecognized projection values fall back to `straight`.
`VC3D_LINE_MODEL_PREFETCH=0` still bypasses all new speculative planning in every
projection mode. Existing legacy scalar normal prefetch is unaffected by it.

Example isolated real-remote dirty-span benchmark, refusing an existing output
directory and never clearing the application cache:

```bash
AGENTS_AGENT_MODE=1 python3 volume-cartographer/scripts/benchmark_line_model_remote.py \
  --output-dir /tmp/villa-guided-prefetch.ff9dfn/user-guided-trial \
  --binary /tmp/villa-guided-prefetch.ff9dfn/final-build/bin/vc_fiber_trace_metric \
  --fiber-json /tmp/villa-pherc1299-prefetch.IJmtzD/baseline-on/input/fiber.json \
  --fiber-manifest /tmp/villa-pherc1299-prefetch.IJmtzD/baseline-on/input/fiber/PHerc1299-20260309130042-las-sd1-aac02eb8.lasagna.json \
  --normal-manifest /tmp/villa-pherc1299-prefetch.IJmtzD/baseline-on/input/normal/PHerc1299-20260309130042-lasagna-20260724.lasagna.json \
  --dirty-span 6 --prefetch 1 --projection guided --threads 1 --trials 3 \
  --compare-run /tmp/villa-guided-prefetch.ff9dfn/baseline \
  --payload-reference-run /tmp/villa-pherc1299-prefetch.IJmtzD/baseline-off
```

Each cold run starts with only frozen manifests and remote markers. Its paired
warm run starts a fresh process using the cold run's disk cache, with empty
decoded caches. A warm run may still fetch extra optional objects; it is not a
claim of zero-network or fully decoded-memory warmth. `remote_bytes` measures
that process's traffic after draining optional requests. An inventory describes
all files retained at its snapshot: the warm inventory is cumulative across the
pair. `extra_observed_bytes` compares that inventory against the union of a
compatible disabled run's inventories; it is not incremental warm traffic, a
true demand hit rate, or proof that bytes arrived before use. The runner stores
the merged reference inventory and hash, exact commands and input/build hashes,
per-stage metrics, output signatures, cache hashes, and summary statistics.

For the four-thread batch-sampling check, use `--threads 4` for every variant and
compare only compatible four-thread runs. For a full retrace, omit `--dirty-span`
and compare only full-retrace runs. The comparison validator intentionally
rejects mixing different solver/thread/span settings.

## Measurements and decision

Ubuntu amd64, GCC 13.3, existing **RelWithDebInfo** (`-O2 -g`, unchanged LTO and
architecture flags), public S3 models with no artificial delay. Native scoring
threads were fixed separately at 1 and 4; OpenBLAS used 1, corner readers 64, legacy
scalar readers their default 8/channel, configured cache 512 MiB. These are not
Release-build claims. The existing RelWithDebInfo build was retained for
consistent baseline/profile comparisons. Built-in stage profiling was used;
system `perf` is unavailable on this host (`perf_event_paranoid=4`). No builds
overlapped timings. Binaries and all six repository shared libraries were frozen
and hashed for each version; the final snapshot is `final-build` in the new
scratch root. System libraries/toolchain are the same local environment.

The exact original fiber SHA256 remains
`2e9a9fc9f59a6c9a7cc0187b7ddf8365e4bc67b4fa113d7b67824dd34ffae4a1`.
All dirty-span runs start from its 3735 points and 8 controls, retrace span 6 and the
open tails, and produce 2835 points with 1 native span, 2 native tails and 0 fallbacks.
This is an optimistic saved-curve re-solve, not a reconstruction of the old click
sequence. The scan volume is not part of the workload or cache manipulation.

Final same-binary measurements have **three cold/warm pairs per mode per thread
setting**. For each thread setting, run order was straight/guided/curved,
curved/guided/straight, straight/guided/curved. Each entry below is solve mean
with min / median / max; MB are decimal, per-process bytes including optional
requests drained after the solve. All samples, including slow ones, are kept.

| Threads | Projection | Cold solve, s | Disk-warm solve, s | Cold MB mean | Warm MB mean |
|---:|---|---:|---:|---:|---:|
|1|Straight|4.041 (3.658 /3.686 /4.778)|1.337 (1.319 /1.345 /1.348)|17.294|0.614|
|1|Guided|5.605 (3.909 /4.002 /8.903)|1.398 (1.327 /1.334 /1.532)|15.931|0|
|1|Curved|5.165 (3.739 /3.978 /7.778)|1.357 (1.317 /1.338 /1.415)|15.931|0|
|4|Straight|7.319 (5.254 /8.322 /8.380)|0.449 (0.382 /0.482 /0.484)|16.804|1.104|
|4|Guided|7.878 (6.649 /7.513 /9.471)|0.484 (0.422 /0.488 /0.543)|15.931|0|
|4|Curved|7.279 (6.081 /6.788 /8.969)|0.577 (0.514 /0.597 /0.620)|15.923|0.008|

Paired solve deltas relative to straight (seconds; negative is faster), in trial
order, rather than only differences between group means:

| Threads | Mode | Cold deltas | Warm deltas |
|---:|---|---|---|
|1|Guided|+0.223, +0.344, +4.125|-0.011, -0.021, +0.213|
|1|Curved|+0.292, +0.081, +3.000|-0.007, -0.031, +0.096|
|4|Guided|-0.867, -1.673, +4.217|+0.061, +0.106, -0.062|
|4|Curved|-2.299, +0.647, +1.534|+0.115, +0.238, +0.030|

Cold paired byte savings for guided were 1.121/1.346/1.622 MB at 1 thread and
1.900/0.360/0.360 MB at 4 threads. Curved saved the same at 1 thread and
1.925/0.360/0.360 MB at 4 threads. Total cold-plus-warm pair bytes converge to
17.908 MB for straight versus 15.931 MB for both new modes, an 11.0% reduction on
this workload. This is a traffic benefit, **not a demonstrated solve-speed win**.
Mean cold process times (metadata/startup/solve/drain/output) were
4.833/6.431/6.141s at 1 thread and 8.566/8.936/8.578s at 4, in mode order.
Warm process means were 1.518/1.483/1.442s and 1.278/0.573/0.733s respectively;
the straight mode's optional post-solve downloads explain why process and solve
comparisons can point in different directions.

At 1 thread, extra retained payload relative to the disabled observed corpus
averaged 5.427 MB cold for straight versus 4.064 MB for guided/curved; after the warm
run, 6.041 MB versus 4.064 MB. No reference-corpus payload was absent. This supports
reduced excess traffic relative to that corpus, not a proven reduction in truly
unused bytes. The 4-thread runs intentionally do not reuse a 1-thread corpus as a
compatible traffic reference.

Stage profile: at 4 threads, straight/guided/curved prediction materialization
averaged 2.067/2.591/2.483s, normal materialization 0.627/0.780/0.833s, and native
start sampling 1.115/1.086/0.532s. Candidate scoring itself was ~0.028s and Ceres
compute ~0.004s. Full reinitialization averaged 2.032/1.980/2.261s, including
normal preparation 0.947/0.751/1.192s. At 1 thread, scalar reads are included in
candidate scoring (1.650/2.425/2.206s), so zero batch-materialization counters
do not mean no model I/O. These nested stage totals are not additive to wall
time. Forecast planning/admission averaged at most 1 ms per dirty-span cold solve.
Model access remains the dominant streamed cost; smarter geometry alone did
not remove it.

Final guided/curved dirty-span runs made 34 native corridor plans, 22 reference-following and
12 reference-fallback plans, with 9 turn refreshes. Curvature actually activated
once per solve in curved mode (zero in guided); that is weak coverage for a
general curvature performance claim.
Straight made 29 plans. Before the review-driven refinement, guided made 38 plans
with 20 turn refreshes; final output and completed pair payload sets are unchanged.
Initial screening (also 3 pairs/mode at 1 thread) had cold means 6.279/6.903/10.901s
and warm 1.369/1.467/1.431s. A separate initial 4-thread screening pair had cold
3.440/3.522/3.526s. These are retained under `straight-*`, `guided-*`, `curved-*`
in the scratch root, not pooled with the revised code's results. The frozen
pre-change `cdbffc05b` recheck had cold 5.585s (3.741 /5.095 /7.918), warm 1.389s
(1.339 /1.361 /1.466), 3 pairs; it is provenance/equivalence context, not the
same-binary comparison table. Network variation makes cross-group speed claims
particularly unsafe.

**Decision:** keep `straight` as the default. Keep `guided` and `curved` as
explicit local experiment switches for GUI testing, with no claim that either
makes streamed reoptimization nearly as fast as cached. The single-thread cold
results regressed in every paired comparison, four-thread results were mixed,
and curvature did not add a convincing benefit over reference guidance. Do not
promote either mode based on downloaded-byte counts alone.

## Full-retrace and correctness checks

One additional full-retrace cold/warm pair per mode, 1 native scoring thread,
same frozen final build and exact input, compared against the frozen previous
version's full-retrace result:

| Projection | Cold solve, s | Warm solve, s | Cold MB | Warm MB |
|---|---:|---:|---:|---:|
|Straight|60.089|17.944|106.711|4.878|
|Guided|35.576|18.203|96.984|2.699|
|Curved|32.021|17.677|99.030|1.320|

These single-pair checks establish equivalence on all seven control spans, not
a speedup claim. The previous version's full-retrace cold sample was 27.436s,
illustrating why the apparent new-mode improvement over this particularly slow
60.089s straight sample is not reliable evidence. All modes produced the same
2835 points, seven native spans and two native tails, no fallback. Guided/curved
used 284 reference plans and 178 fallbacks; curved extrapolation activated 16
times. Curved fetched more pair-total bytes than guided on this full-line case,
another reason not to promote the curvature mode.

The new code with `--prefetch 0 --projection curved` also matched the disabled
reference exactly and recorded zero new plans/submissions. Its single cold/warm
solve sample was 14.745/1.427s with 11.867/0 MB; this is an opt-out correctness
check, not a performance baseline to pool with the alternating matrix.

All 44 final benchmark processes passed the exact-output checks within their
compatible span/thread configuration. All recorded shared cache payload hashes
matched, every remote failure/pending-request count ended at zero, and frozen
input and build hashes stayed unchanged. The comparison includes geometry,
metadata, normalized decisions/messages and candidate/generation counts, not
just a geometric tolerance. No candidate scoring, solver arithmetic, precision,
acceptance threshold, or numerical build flag was changed.

The original application fiber and application cache were not modified. Test
caches and result artifacts are retained under `/tmp/villa-guided-prefetch.ff9dfn`
so this experiment can be inspected or abandoned without deleting user data.
Code was reviewed independently by fresh Codex and Fable reviewers and refined
until both reported no remaining blockers; Codex additionally recomputed the
final dirty-span measurement tables from the artifacts.

The rebuilt 13-target regression suite passed, including the 17-case corridor
test binary, HTTP failure/coalescing fixtures, cache scheduling, line optimizer,
native tracer and GUI warmup queue. All 31 Python harness tests passed. The
offscreen VC3D `--help` startup check passed. Two older dataset-specific optimizer
checks skip because las008/5.json are absent; the exact PHerc1299 benchmark above
is the real-data validation for this change. macOS/arm64 were not executed here;
the new predictor uses portable C++ and retains existing scheduler/threading
behavior. There is no interactive GUI-test claim until the user tries the build.

Exact regression build/run commands used:

```bash
AGENTS_AGENT_MODE=1 cmake --build volume-cartographer/build --target \
  VC3D vc_fiber_trace_metric test_model_prefetch test_lasagna_remote_prefetch \
  test_chunk_cache test_remote_file_cache test_lasagna_line_optimizer \
  test_lasagna_batch_window test_lasagna_manifest test_chunk_cache_persist \
  test_chunked_plane_sampler test_fiber_trace_review test_fiber_trace3d \
  test_lasagna_normal_sampler test_line_model_warmup_queue -j6
AGENTS_AGENT_MODE=1 ctest --test-dir volume-cartographer/build --output-on-failure \
  -R '^(test_model_prefetch|test_lasagna_remote_prefetch|test_chunk_cache|test_remote_file_cache|test_lasagna_line_optimizer|test_lasagna_batch_window|test_lasagna_manifest|test_chunk_cache_persist|test_chunked_plane_sampler|test_fiber_trace_review|test_fiber_trace3d|test_lasagna_normal_sampler|line_model_warmup_queue)$'
AGENTS_AGENT_MODE=1 python3 -m unittest discover \
  -s volume-cartographer/scripts/tests -p 'test_line_model*.py'
AGENTS_AGENT_MODE=1 QT_QPA_PLATFORM=offscreen volume-cartographer/build/bin/VC3D --help
```
