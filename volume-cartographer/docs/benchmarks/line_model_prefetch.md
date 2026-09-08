# Line model prefetch experiment

## Scope and isolation

The user observed that VC3D line reoptimization becomes much faster when the
Lasagna and fiber models are downloaded, even with the raw scan still streamed.
This experiment starts at `02bd2098d` on local branch
`experiment/line-model-prefetch`. Keep all changes, builds, and benchmark outputs
local until the user tests them. Do not push, publish a PR, install dependencies,
or replace the installed application.

## Implementation plan (initial review version)

1. Establish a real-data baseline using a saved fiber and the matching Lasagna
   and fiber manifests. Prefer the user's PHerc1451 reproduction; otherwise use
   the available PHercParis4 data. Measure identical solves with model data local,
   through a fresh HTTP cache, and through a warm HTTP cache. A local HTTP server
   serving the actual model chunks with controlled request latency makes the
   streaming comparison reproducible without changing source datasets. Record
   wall time, download counts/bytes, and the existing tracing profile. Use an
   isolated cache for each cold trial; never clear the user's caches.
2. Add a reusable, bounded model-chunk prefetch helper in the Lasagna sampling
   layer. Expose the existing corner sampler's actual shared chunk cache rather
   than building a second cache or duplicating interpolation logic. Plan chunks
   along a polyline with a small spatial margin in each channel's own grid;
   support differing model spacings/chunk shapes. Request only relevant model
   channels, with lower priority than required reads. Bound planned work, bytes,
   and outstanding requests; resident and already-requested chunks are skipped.
   Chunk planning must handle boundaries, negative/outside/nonfinite points,
   degenerate lines, and unusually long segments without enormous loops.
3. Start corridor prefetch when a stored fiber is opened and when edits schedule
   a solve. Use existing line geometry plus affected control spans; capture
   immutable inputs and sampler ownership. Keep I/O and substantial planning off
   the GUI thread. Coalesce obsolete requests and stop generating speculative
   work on session close, dataset change, or superseding edits. Do not cancel
   other consumers' chunk requests or make a solve wait for speculative work.
4. Keep fetches ahead of tracing using a moving window along the current trace
   direction/reference line, including endpoint extensions. Reuse the same
   helper as the GUI corridor warmup. Start a bounded next stretch while the
   current generation computes; replenish as tracing progresses. Prediction and
   normal channels must both be covered even if they use different grids. A
   cache miss outside the corridor must still use the unchanged exact blocking
   sampling path. Speculative errors must not change solver/fallback results.
5. Add performance diagnostics for combined corner preparation, chunk wait,
   gather/scoring, and speculative work, alongside existing total solve time and
   remote-store counters. Label aggregate/process-wide timings accurately. Add
   an opt-out switch for controlled A/B testing and easy local abandonment.
6. Build the local VC3D executable and relevant test/benchmark targets using the
   existing toolchain. Add focused regression tests for planning and asynchronous
   prefetch behavior (bounded work, cache reuse, priority, failure handling), and
   verify identical line coordinates and acceptance/metadata between enabled
   and disabled runs on real data. No precision, candidate-order, interpolation,
   solver-tolerance, or approximation changes are permitted.
7. Repeat real-data trials (at least three per condition), report mean and
   min/median/max, build type, exact commands, dataset, line and cache condition.
   Profile representative runs to distinguish networking from CPU work. Report
   cold first-use time separately from reoptimization after background warmup;
   include warmup time/bytes so shifted work is visible. Prefer Release for final
   numbers and RelWithDebInfo for profiling. Document any environmental limits.
8. Obtain independent fresh Codex and Fable reviews of this plan before changing
   production code. Incorporate applicable findings. Obtain fresh reviews of the
   completed diff and validation evidence, fix significant findings, and repeat
   until reviews have no unresolved actionable correctness or scope findings.

## Review and validation record

### Fresh Codex plan review

The independent review identified these requirements, accepted for implementation:

- Ordinary `prefetchChunks` is insufficient: it leaves terminal errors in the
  cache. Track speculative fetch origin and retry its failure once as ordinary
  demand, including when demand joins the speculative operation before it fails.
- Add a queue-only cache request API with shared process-wide count/byte
  admission. Reservations must follow the actual task lifetime, including a
  source read still running after logical invalidation. Do not use one waiting
  thread per chunk or futures whose destruction waits for I/O.
- Initial background batches must be conservative. Priority cannot preempt
  transfers already running, so an admission ceiling alone does not ensure
  responsive viewer requests. Preserve the model cache's existing eviction policy.
- Resolve GUI selections on the GUI thread; initialize missing model samplers
  from immutable locations off-thread. Extract reusable dataset-opening helpers
  from the existing initialization code. Reuse returned resources only after
  validating the session and dataset identities. Warmup cancellation is separate
  from solve cancellation, and close must cancel warmup even without a solve.
- Use each model channel's actual grid and the existing coordinate adapter.
  Keep an independent planning-iteration budget even when all chunks are cached.
- Log fused corner preparation/layout/pin/gather separately: pin includes queue
  and allocation time, gather includes the scoring visitor, and dependency counts
  are not download counts. Report actual remote-store deltas separately.

### Fresh Fable plan review

The local authenticated Claude CLI accepted the `fable` model. Both an initial
bounded plan-only review and a fresh repository-reading review completed before
implementation. Accepted findings reinforce speculative-origin retries (also
for scalar readers joining the embedded HTTP store), global task-lifetime
reservations, independent cancellation, coordinate conversion, and precise timing
labels. The review also requires counting shifted warmup time and overfetch bytes,
and checking a bandwidth-limited workload rather than latency alone.

The proposed full scheduler cancellation-token redesign is intentionally narrowed:
replacement/close stops new issuance immediately, but up to 16 already admitted
requests may finish. Running network reads cannot be preempted safely. Admission
is additionally capped at half the configured shared read maximum (minimum one),
32 MiB decoded estimates, and half the shared decoded-cache capacity. This is not
a reservation of network bandwidth. Warmup owns no GUI session or QObject.

Speculation deliberately skips sharded arrays because one logical chunk can fetch
a much larger physical shard. It also skips incompatible normal-channel grids
that use the existing scalar fallback. Required reads remain unchanged in both
cases. The legacy scalar cache is not unified with the decoded corner cache;
those paths benefit from the shared embedded persistent HTTP cache.

The first stored-line open retains its existing synchronous normal pass. Warmup
targets subsequent reoptimization, not that open operation. Missing prediction
metadata is opened on a coalescing background worker from immutable locations.
A locally selected manifest with remote groups cannot be classified without I/O;
fresh warmup skips ambiguous local identities until the sampler already exists.

### Baseline

The available PHercParis4 saved fiber (29 controls, 4,834 points) was used because
no PHerc1451 reproduction path was supplied. Three trials per condition, existing
RelWithDebInfo build, real byte-identical models served with 50 ms/request latency:
local trace mean 1.729 s; cold streamed mean 27.393 s; warm disk streamed mean
1.795 s. Cold requests total 558 / 75,163,667 bytes, including 537 chunk GETs;
warm runs make no requests. Candidate/generation counts match in all trials.
The preserved executable/libraries, exact commands, raw profiles, HTTP logs and
source-byte validation are in `/tmp/villa-line-prefetch-baseline.nyyoFy/README.md`.
This baseline is a pre-existing build snapshot, not a source rebuild. Controlled
enabled/disabled comparisons of the newly built same executable will remove that
qualification for final results. Hardware `perf` is blocked by host settings;
built-in stage profiling is used instead.

### Implementation and completed-code reviews

The three changes share `vc::lasagna::ModelPrefetchPlan`: GUI corridor warmup,
moving native-trace lookahead, and accurately labelled diagnostics. The helper
does no I/O while planning and uses each sampler's actual grid. It bounds keys,
planning iterations and estimated decoded bytes, deduplicates dense curve boxes,
clips outside coordinates, and interleaves channels. Numerical trace generation,
scoring, sampling, tolerances, and iteration order are unchanged.

The implementation is concentrated in these areas:

- `core/{include/vc,src}/lasagna/ModelPrefetch.*`: shared bounded corridor planner;
  `ChannelSampler` and `LasagnaNormalSampler`: expose existing cache descriptors.
- `core/src/render/ChunkCache.cpp`: low-priority queue-only requests, bounded
  task-lifetime reservations, demand promotion and retry. `Dataset.cpp` and
  `RemoteFileCache.cpp` preserve required-reader recovery when sharing optional I/O.
- `apps/VC3D/LineAnnotationController.*` and `LineModelWarmupQueue.hpp`: immutable,
  coalesced background jobs and accurate preparation/wait/scoring diagnostics.
- `core/src/fiber_tracer/FiberTrace.cpp`: moving speculative window;
  `ZarrChunkFetcher.cpp`: reuse actual HTTP-byte observation for adaptive reads.
- `apps/src/vc_fiber_trace_metric.cpp`: headless GUI optimizer mode, explicit
  warmup/reader controls and exact output artifacts. Shared existing defaults and
  the existing GUI optimizer implementation are reused rather than duplicated.
- Focused core/GUI tests, the two benchmark scripts, analyzer unit tests and
  corresponding CMake wiring provide repeatable validation without new dependencies.

The GUI snapshots the existing/edited curve and selected model identities,
coalesces edits for 50 ms, and pumps two directions from the focus for at most
five seconds (metadata opening may extend worker lifetime). Cancellation never
waits for I/O. There is at most one worker and one latest pending job per session;
workers use the global Qt pool at low priority, whereas solves use their separate
pool. On machines with very small pools, unrelated global-pool work may wait.

Fresh Codex and Fable completed-code reviews both identified permanent rejection
stalling a plan. The cache API now distinguishes permanent `Skipped` from temporary
`Rejected`, and a regression checks that an errored/oversized first channel cannot
block subsequent valid channels. Codex additionally found background metadata
errors leaking into foreground initialization. Optional metadata now carries the
same scoped source-origin marker through the manifest and embedded Zarr stores;
ordinary followers get one bounded retry outside registry locks. Real HTTP tests
cover successful sharing, speculative failure, ordinary failure, and failure of
the extra attempt for chunks, array metadata and manifest files. Scope restoration
on nested calls and exceptions is tested too.

Successful data must give identical numerical results. Under transient errors,
speculation can perform one extra attempt before the ordinary attempt; this is an
explicit recovery policy, not a claim of identical request counts or behavior
under an arbitrary changing/failing server. A retry may share another newly raced
operation and remains bounded: repeated failure is terminal, not an infinite loop.

A second fresh Codex completed-code review and a fresh focused Fable review found
no remaining actionable correctness findings after these fixes. Fable requested
one additional chunk-store second-failure test, which was added. Its question
about malformed decoded sizes was checked: the existing store function marks
them `Error`, so ordinary retry remains applicable.

### Performance-driven second iteration

The first headless GUI-optimizer matrix (three trials per condition) revealed a
real scheduling problem: cold time only improved from 31.368 s to 30.574 s while
bytes rose 27.4%. Five-second warmup was serialized and shifted work without a
useful overall improvement. Normal-channel omission was initially suspected but
disproved by the request log: **all six normal/prediction channels were present**.
The different grids disable the combined fused visitor, not the normal descriptors.

Two causes were identified and fixed without modifying the scheduler algorithm:

- Shared-array Lasagna cache factories did not mark their embedded HTTP store as
  remote. They now reuse the existing HTTP download observer. Only observed HTTP
  body bytes provide adaptive bandwidth evidence; disk hits do not.
- Capping all queued speculation at half the *current* adaptive admission kept
  optional-only work below the saturation threshold forever. The queue cap now
  uses half the *configured maximum*, still globally limited to 16 / 32 MiB.
  Actual reads remain subject to the existing adaptive gate, and required queued
  work retains priority. At a low initial gate, running speculative reads can
  temporarily occupy it; a required request may wait for one running read but
  overtakes the queued speculative backlog. A deterministic regression verifies
  initial admission two, a bounded pending backlog, and both forms of demand
  promotion. No global scheduler/persistent-download policy was changed.

Moving rays are also clamped to the remaining segment distance or tail length.
The final window uses a 256-trace-voxel lookahead, replans after 64 voxels, and
has a 16-voxel radius. An intermediate 512/128/32 window reached near-local
optimizer time but downloaded excessive extra data, motivating the smaller
window. These distances affect speculative requests only, not trace candidates.

The user's saved GUI configuration was inspected read-only and uses manual
**64** download workers, whereas the standalone CLI starts with fresh adaptive
history at **two**. Final representative comparisons must use the same 64-worker
configuration on both sides. Fresh-adaptive startup remains a separate, explicitly
labelled stress case. No GUI preferences were modified.

### Local build and validation

No dependencies were installed, no source datasets or user caches were cleared,
and no branch or build was published. The existing toolchain was reused with
FetchContent disconnected. All changes remain in `volume-cartographer/`.

From the repository root:

```bash
AGENTS_AGENT_MODE=1 cmake -S volume-cartographer -B volume-cartographer/build \
  -DFETCHCONTENT_FULLY_DISCONNECTED=ON -DFETCHCONTENT_UPDATES_DISCONNECTED=ON \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
AGENTS_AGENT_MODE=1 cmake --build volume-cartographer/build --target \
  VC3D vc_fiber_trace_metric test_model_prefetch test_lasagna_remote_prefetch \
  test_chunk_cache test_remote_file_cache test_lasagna_line_optimizer \
  test_lasagna_batch_window test_lasagna_manifest test_chunk_cache_persist \
  test_chunked_plane_sampler test_fiber_trace_review test_fiber_trace3d \
  test_lasagna_normal_sampler test_line_model_warmup_queue -j 6
AGENTS_AGENT_MODE=1 ctest --test-dir volume-cartographer/build --output-on-failure \
  -R '^(test_model_prefetch|test_lasagna_remote_prefetch|test_chunk_cache|test_remote_file_cache|test_lasagna_line_optimizer|test_lasagna_batch_window|test_lasagna_manifest|test_chunk_cache_persist|test_chunked_plane_sampler|test_fiber_trace_review|test_fiber_trace3d|test_lasagna_normal_sampler|line_model_warmup_queue)$'
```

All 13 target-level tests passed. The HTTP fault suite also passed ten consecutive
runs (`ctest --test-dir volume-cartographer/build -R '^test_lasagna_remote_prefetch$'
--repeat until-fail:10 --output-on-failure`). That suite binds only loopback and
requires permission outside the tool's socket-restricted sandbox. The first
sandboxed attempt failed to create its socket; it passed after permission was
granted. The fixture is POSIX-guarded for Ubuntu/macOS. Neither macOS nor arm64 was
available for execution. VC3D's offscreen `--help` startup smoke check passed.

For a local GUI A/B test, start separate processes with the same package and line:

```bash
VC3D_LINE_PERF_LOG=1 volume-cartographer/build/bin/VC3D
VC3D_LINE_MODEL_PREFETCH=0 VC3D_LINE_PERF_LOG=1 volume-cartographer/build/bin/VC3D
```

The experiment is enabled by default; the opt-out is read once at process startup.
After opening a line, allow roughly five seconds before testing reoptimization,
then also test an immediate reoptimization and rapid edits/close. Compare separate
fresh model caches if testing cold behavior; do not delete the user's normal cache.
Raw scan fetching and numerical solver settings are not changed.

### Reproducible benchmark harness

The new standard-library-only scripts require Python 3.9 or newer and an existing
`vc_fiber_trace_metric` build; they install nothing. The runner serves original
model bytes on loopback, writes only a **new** output directory, and records source
and manifest hashes, binary hash, commands, stage profiles, HTTP logs and exact
trace artifacts. `--bandwidth-mib` limits aggregate response-body bandwidth, not
individual connections. Local model files and the user's caches are never changed.

`--gui-reoptimize` calls the existing GUI optimizer helper, including its default
endpoint tails. It is a headless optimizer benchmark, not an interactive GUI test:
it excludes controller post-solve display-normal refresh. The controlled warmup
uses the same planner over the saved whole corridor and explicitly waits for its
submitted requests within the specified time budget. The GUI instead starts at
the focus in both directions, has a separate half-budget per direction, coalesces
edits, and never makes the solver wait for warmup. Thus warmup numbers characterize
available prefetch benefit, not an exact measurement of controller scheduling.
The GUI cannot transfer unused planning budget between its halves; an off-center
focus or uneven corridor can truncate one half even when the CLI whole-line plan
fits. Cold metadata opening also consumes the GUI's five-second submission budget,
but precedes the CLI warmup timer. Check the GUI log's `truncated=0` and
`submission_done=1`, and allow outstanding reads to settle, before assuming
equivalent coverage. Required reads always handle any remaining misses.

The analyzer rehashes the recorded input and manifests, checks every cache payload
against its source (including files absent from HTTP logs), verifies requested
reader settings, rejects incomplete/failed trials and failed request drains, and
compares deterministic outputs across variants. Floating-point output values are
serialized as their exact bit patterns. Timing-bearing GUI prose is checked
separately after masking only recognized timing fields; acceptance, fallback,
solver status and other diagnostic content must still match. Binary hashes may
differ for an intentional before/after comparison but are reported explicitly.
Historical missing request-drain flags are accepted only with a compatibility
warning; final runs use the hardened runner and have explicit drain records.

The five original harness tests plus eighteen hardening regressions passed:

```bash
AGENTS_AGENT_MODE=1 python3 -m unittest discover \
  -s volume-cartographer/scripts/tests \
  -p test_line_model_prefetch_analysis.py -v
```

Fresh final Codex and Fable reviews checked the throughput fixes. Codex found that
cross-variant analysis without saved traces needed explicit metric/count equality;
that check and its regression were added. Fable's separate harness review found
missing provenance revalidation, incomplete cache/drain validation, and exceptional
HTTP-handler accounting leaks. All three were fixed with regressions. A subsequent
Codex review of those fixes and the final speculative window found no actionable
issues. Fresh Fable verification independently found no actionable correctness
issues; its direct-header-include nit was addressed. An additional independent
Fable pass found that copied cache manifests also needed hash verification, and
that deliberately slow bandwidth tests needed a configurable request-drain limit.
Both were fixed with regressions. The runner now accepts positive, finite
`--drain-timeout-s` (default five seconds); exceeding it is still a hard failure.
Analyzer provenance also reports threads, latency, bandwidth and drain settings,
and intentionally comparing different settings emits an explicit warning.
Final Codex and fresh Fable verification confirmed these fixes with no remaining
blocking correctness findings. Cross-setting warnings go to stderr; each variant's
settings remain in its persisted validation. Fable noted a low-severity diagnostic
attribution race for a late accepted HTTP connection around a trial boundary; this
does not change optimizer output, and complete cache-byte validation is independent
of request attribution. No such misattribution was observed in the final matrix.

## Final performance measurements

### Representative manual-64-reader configuration

All comparisons below use the **same frozen v3 executable and libraries**,
RelWithDebInfo (`-O2 -g`, GCC 13.3, existing LTO/OpenMP/x86-64-v3 settings), on an
AMD Ryzen 9 7950X, Ubuntu x86_64. Four scoring threads and one OpenBLAS thread
were fixed across trials. RelWithDebInfo was retained for consistent before/after
profiling rather than switching one side to Release. No new architecture-specific
or floating-point optimization flags were introduced.

Input: the complete PHercParis4 saved fiber
`/media/djosey/nvme2/fibers/PHercParis4.volpkg.json/sj_20260815T134259532_000072.json`,
with fiber manifest `/media/djosey/nvme2/fiber_vols/fiber_s1_002.lasagna.json` and
normal manifest
`/media/djosey/nvme2/PHercParis4.volpkg/las008_s1_full/las_008.lasagna.json`.
The input has 29 controls and 4,834 saved points. Each run optimizes all 28 spans
and both default 1,200-base-voxel tails, producing 4,634 points without fallback.

Three trials per row; 50 ms per HTTP request, unlimited response-body bandwidth.
Cold uses a new isolated persistent cache; warm is a new process using only its
paired cold run's cache. Local reads the original downloaded model files. OS page
caches were not dropped. Timings are seconds; MB means decimal megabytes.

| Prefetch | Model cache | Optimizer mean (min / median / max) | Process mean | Download MB mean |
| --- | --- | ---: | ---: | ---: |
| Disabled | Local | 3.634 (3.497 / 3.657 / 3.747) | 3.733 | 0 |
| Disabled | Cold HTTP | 20.897 (20.828 / 20.927 / 20.935) | 21.963 | 80.632 |
| Disabled | Warm HTTP | 3.663 (3.640 / 3.651 / 3.699) | 3.768 | 0 |
| Moving only | Local | 3.637 (3.622 / 3.637 / 3.651) | 3.737 | 0 |
| Moving only | Cold HTTP | 13.860 (13.580 / 13.632 / 14.367) | 14.930 | 150.109 |
| Moving only | Warm HTTP | 3.667 (3.534 / 3.662 / 3.804) | 3.771 | 1.250 |
| Corridor + moving | Local | 3.628 (3.482 / 3.608 / 3.795) | 3.725 | 0 |
| Corridor + moving | Cold HTTP | 3.738 (3.632 / 3.705 / 3.876) | 8.184 | 153.656 |
| Corridor + moving | Warm HTTP | 3.752 (3.665 / 3.680 / 3.912) | 4.100 | 2.064 |

Cold corridor warmup took 3.352 s mean (3.322 / 3.358 / 3.377 min/median/max),
completed all 756 planned chunks, and downloaded 104.770 MB before optimization.
The remaining downloads are included in the table's total bytes. The optimizer
then ran within about 3% of local time: 5.59x faster than the same-binary cold
baseline. Including metadata initialization, warmup, optimization and output
serialization, process time improved 2.68x, from 21.963 to 8.184 s. Immediate cold
optimization with moving prefetch alone improved 33.7%, but was not near-local.

This is not free caching: corridor-plus-moving used **1.91x the baseline's cold
model bytes**. Warm repeated processes also fetched about 2.1 MB of additional
speculative chunks. These are reasons to keep the experiment opt-out available
and validate it on the user's actual network and PHerc1451 line. The raw scan was
not part of this headless model-I/O benchmark; concurrent scan network traffic
could compete for bandwidth in the real application.

### Remaining network validation

The planned final 8 MiB/s and fresh-adaptive-startup comparisons were **not run**.
Two requests for permission to start the loopback bandwidth matrix remained
unapproved and were cancelled before execution; no bandwidth result is inferred
from the latency-only matrix. The harness supports these follow-up comparisons
using the commands below. Near-local performance is established only for the
controlled manual-64-reader, 50-ms/request case after corridor warmup. With nearly
twice the model bytes transferred, a sufficiently bandwidth-constrained connection
may show a smaller benefit or a regression. This remains a limitation for the
user's local testing, alongside PHerc1451 and the interactive GUI workflow.

### Instrumented hotspots and numerical validation

Hardware profiling could not run: `perf stat -e task-clock true` is denied with
host `perf_event_paranoid=4`; no host settings were changed. Existing built-in
stage timers identify these largest instrumented regions (three-trial means):

| Region | Local, disabled | Cold, disabled | Cold, after corridor warmup |
| --- | ---: | ---: | ---: |
| Normal batch (`sampleCandidateNormals`) | 1.110 s | 7.510 s | 1.102 s |
| Prediction batch (`FiberPredictionField::sampleBatch`) | 1.341 s | 6.025 s | 1.284 s |
| Candidate scoring | 0.466 s | 0.516 s | 0.459 s |
| Frontier handling | 0.199 s | 0.226 s | 0.207 s |
| Pruning | 0.167 s | 0.187 s | 0.169 s |

Normal materialization within that batch dropped from 7.488 to 1.084 s, and
prediction materialization from 5.838 to 1.133 s. These inclusive stage timers
are not a complete mutually exclusive decomposition of GUI optimizer wall time.
The available models use different prediction and normal grids, so this workload
exercises the non-fused path; fused diagnostics are covered by existing tests.

All 27 final outputs have identical exact geometry, normals, generated metadata,
acceptance/fallback decisions and result metrics. They visit 14,767,434 candidates
and 9,157 generations. Canonical result SHA-256:
`b0c01b9666de29ab7228ed2e3a84909f77ea14573e4aaa8a3dbea683e8334083`.
Separately timing-normalized GUI diagnostic SHA-256:
`f43ef20f25670da25c3a8bdb2c217d24904c7eafb2fbb789216800c2291633cc`.
The final hardened analyzer verified recorded input hashes, 18 seeded manifest
copies and 8,451 cache payloads totaling 1,163,133,123 bytes. Every trial recorded
`requests_drained=true`. These runs preceded the configurable drain-timeout flag,
so its configuration value is explicitly reported as unrecorded (the runner then
used five seconds). The analyzer also warns about the intentional warmup-argument
difference between variants; neither warning weakens output or byte validation.

### Commands and retained artifacts

Frozen binary SHA-256:
`46a638a08036bf68d640da1207caa6697a0220b9b4a2b91ee8c7594421ea51c2`.
It and its repository libraries are retained in
`/tmp/villa-line-prefetch-baseline.nyyoFy/candidate-gui-v3-build/`, with a
`sha256.json` manifest. The final source adds only an explicit, already-transitive
header include after this snapshot; subsequent version-stamp rebuilds also do not
change the optimizer or prefetch implementation measured here.

The actual sequential matrix invocation was:

```bash
AGENTS_AGENT_MODE=1 python3 /tmp/villa-line-prefetch-baseline.nyyoFy/run_matrix.py \
  --binary /tmp/villa-line-prefetch-baseline.nyyoFy/candidate-gui-v3-build/bin/vc_fiber_trace_metric \
  --prefix gui-v3-final --trials 3 --readers 64
```

Each `gui-v3-final-{disabled,enabled,warmup}/` directory records its exact CLI
commands in `results.json`, settings in `configuration.json`, source hashes in
`workload.json`, summaries, validation, HTTP logs and exact traces. The wrapper
only invokes the repository runner sequentially. A portable equivalent for the
warmup variant is shown below; choose a **new** output directory for each run.

```bash
AGENTS_AGENT_MODE=1 VC3D_LINE_MODEL_PREFETCH=1 python3 \
  volume-cartographer/scripts/benchmark_line_model_prefetch.py \
  --output-dir /tmp/CHOOSE-NEW-LINE-BENCHMARK-DIRECTORY \
  --binary volume-cartographer/build/bin/vc_fiber_trace_metric \
  --fiber-json /media/djosey/nvme2/fibers/PHercParis4.volpkg.json/sj_20260815T134259532_000072.json \
  --fiber-manifest /media/djosey/nvme2/fiber_vols/fiber_s1_002.lasagna.json \
  --normal-manifest /media/djosey/nvme2/PHercParis4.volpkg/las008_s1_full/las_008.lasagna.json \
  --threads 4 --trials 3 --delay-ms 50 --save-traces \
  --extra-arg=--gui-reoptimize --extra-arg=--model-readers --extra-arg=64 \
  --extra-arg=--prefetch-warmup-ms --extra-arg=5000
```

For moving-only, omit both `--prefetch-warmup-ms` arguments. For disabled, also
set `VC3D_LINE_MODEL_PREFETCH=0`. Add `--bandwidth-mib 8 --modes cold` for the
bandwidth-limited cold comparison, or omit both `--model-readers` arguments for a
separately labelled fresh adaptive-startup test. Analyze variants together:

```bash
AGENTS_AGENT_MODE=1 python3 volume-cartographer/scripts/analyze_line_model_prefetch.py \
  /tmp/villa-line-prefetch-baseline.nyyoFy/gui-v3-final-disabled \
  /tmp/villa-line-prefetch-baseline.nyyoFy/gui-v3-final-enabled \
  /tmp/villa-line-prefetch-baseline.nyyoFy/gui-v3-final-warmup
```
