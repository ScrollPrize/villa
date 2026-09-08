# PHerc1299 interactive-edit prefetch follow-up

## Scope and implementation plan

Continue locally on `experiment/line-model-prefetch`, starting from `fa167075d`.
Do not publish, install, change solver mathematics, edit the user's fiber, or
clear the user's application cache. The user explicitly requested testing the
fiber they just created with controlled cold versus warm caches.

Canonical input:
`/home/djosey/.VC3D/remote_cache/open_data/projects/fibers/PHerc1299.volpkg.json/dj_20260908T152535598_000001.json`
(eight controls, 3,735 saved points at discovery). Freeze its exact bytes and
matching normal/prediction manifests and remote markers in a new scratch folder.
The previous PHercParis4 benchmark remains historical, not the acceptance input
for this follow-up.

1. Extend the existing headless GUI-optimizer validation path with a selectable
   dirty span and per-stage/model-store diagnostics. Reuse existing optimization
   request logic and defaults; do not copy GUI algorithms. A dirty-span re-solve
   of the saved fiber approximates an edit transaction, but does not reconstruct
   the user's unavailable historical pre-edit curves or click sequence.
2. Add a reproducible direct-remote cold/warm benchmark workflow using shared
   existing harness helpers. A cold trial copies only manifests/remote markers to
   a new isolated cache; its warm partner is a fresh process reusing that cache.
   Run the exact same frozen fiber, settings, and span on both sides. No loopback
   server is necessary for the primary actual-network comparison. Record exact
   commands, input/manifest/marker/binary hashes, per-stage wall time, downloaded
   bytes and complete cache payload hashes. Verify shared payload keys have
   identical bytes and deterministic result payloads match across all variants.
   Network variability is a limitation; alternate/repeat at least three cold/warm
   pairs where practical. Do not assume a partially populated user cache is a
   complete local dataset and silently turn missing chunks into fill values.
3. Establish current-code baselines before changing production prefetch logic.
   Use the user's fixed64 download-reader setting and matching thread/default
   solver settings. Freeze the baseline executable and repository libraries.
   Measure both prefetch-enabled and disabled conditions, plus dirty-span versus
   full-line behavior if needed to localize the bottleneck.
4. Extend bounded model lookahead to Lasagna's normal-guided extension/reinit
   path. It currently uses short step-based legacy prefetch while the new helper
   is confined to native tracing. Extract shared scheduling behavior where needed;
   preserve every computed point, normal sample, iteration, candidate and fallback.
   Do not remove temporary Lasagna tails merely because native tails replace them:
   they participate in the existing global solve. Preserve required-read error
   behavior; optional work stays lower priority, bounded and cancellable.
5. Investigate prefetch coverage for the newly constructed reinit geometry and
   upcoming endpoint tails. Add only improvements supported by stage measurements;
   distinguish read scheduling from changes to solver behavior. Preserve the
   startup opt-out and avoid an unbounded background worker or per-chunk threads.
6. Add focused regression tests for the new prefetch integration, including
   exact old/new output equivalence, local-only behavior, cancellation and optional
   failure handling. Build with the existing RelWithDebInfo toolchain, disconnected
   dependencies, no precision or optimization-flag changes.
7. Repeat paired cold/warm measurements on the frozen exact PHerc1299 fiber with
   the same binary configuration and report mean/min/median/max, stage hotspots,
   bytes and warm-cache downloads. Report regressions or incomplete coverage
   candidly. Obtain fresh Codex/Fable plan and completed-code reviews, fix relevant
   findings, and finish with a local VC3D build ready for interactive testing.

## Evidence motivating the follow-up

In the user's four-control edit, corridor warmup found all159 planned keys already
resolved, but the solve still recorded165 remote objects (~20MiB) and took9.53s.
At five controls, reinitialization took7.25s and native tails5.64s out of13.58s.
The normal preparation reported inside reinit took5.00s, while reported Ceres
compute took9ms. The previous enabled/disabled GUI logs are not matched trials:
different controls were added and the disabled run reused the enabled disk cache.

## Review, measurements and validation

Fresh Codex and Fable plan reviews completed. Incorporated: source-grid spacing
is not interchangeable with tracer spacing; the existing 12-step normal ray is
already ~379 base voxels here. The added normal window uses 512/128/32 base-voxel
lookahead/refresh/radius. The shared scheduler retains native 256/64/16 trace-voxel
settings. Legacy scalar prefetch stays: scalar and corner decoded caches differ,
sharing only the physical remote store. Required scalar reads/errors are not
intercepted. Custom sampler discovery remains the existing concrete-Lasagna
pattern, avoiding a new dependency in the abstract NormalSampler interface.

Also incorporated: a post-timing bounded request drain; fresh-process disk-warm
label; canonical manifest identities; GUI-matching 512MiB cache, fixed64 readers
and one native scoring thread; recorded legacy worker override (default8/channel,
distinct from the corner-cache setting); a separately timed corridor-warm condition.
The saved dirty-span re-solve is explicitly not a replay of the historical insert
(which can dirty two spans). Only span6 is retraced; the other saved spans persist.

Artifacts are in `/tmp/villa-pherc1299-prefetch.IJmtzD`. Baseline binary and six
repository shared libraries were copied before production changes, preserving
relative loader paths. The baseline has CLI diagnostics only beyond fa167075d.
All test caches contain only seeded manifest/marker files before cold runs; source
arrays remain remotely backed. Full post-exit cache inventories are hashed and
common keys verified, including across variants. These checks detect changed
shared payloads, not an independent authoritative hash of the remote dataset.

Initial paired baseline (three cold/warm pairs, last saved span6, 3735-point input):
mean optimizer cold4.697s (min4.263/median4.446/max5.382), disk-warm1.311s
(1.304/1.305/1.325). Initial normal moving-window rung: cold4.247s
(4.139/4.194/4.408), warm1.335s (1.327/1.335/1.342), exact outputs unchanged.
This modest change alone does not achieve near-cached cold speed.

The baseline's 10s-budget saved-corridor warmup already reaches ~cached solve
speed on this frozen final fiber (mean1.283s), but takes time before the solve:
mean process6.743s and90.55MB downloaded versus17.40MB without corridor warmup.
This upper-bound coverage condition must not be presented as a successful
reproduction of the user's new-control coverage gap.

Second production rung: parallel independent reinit tails for
concurrent-capable samplers when enabled, preserving scalar math, vector ordering,
the same temporary tails, and the later global solve. The existing startup switch
disables this rung as well. New phase byte diagnostics include overlapping work;
they are not exclusive per-phase request attribution.

Fresh completed-code Codex review found two issues, now fixed and re-reviewed:
optional thread-launch failure falls back to serial tail construction, and the
harness records/verifies repository shared-library hashes in addition to the
executable. Added concurrent-tail cancellation and required-error coverage.
Early exploratory runs predate library hashes; the final comparisons use frozen
builds and the stronger provenance checks. Fresh completed-code Fable review found
no blocking correctness bug. Its test findings were fixed: required seed-growth
failure now occurs at the sixth sample (inside growth), and the moving-window test
requires a positive admission after a refresh. Added an explicit Dataset include.
Usage/build-layout requirements below were added during the review.
Fable's focused follow-up verified the fixes and reported no remaining review
blocker; Codex's follow-up likewise found no blocker. Final rebuilt tests and the
offscreen VC3D `--help` startup smoke passed after these fixes.

Accepted failure-path limitation from Fable: on a required error in one parallel
tail, exception propagation can wait for the other tail's future to finish. This
keeps sampler/vector lifetime safe, is consistent with existing parallel seed
construction, and does not suppress the error. User cancellation stops both tails
at the next per-step checkpoint. No wall-time bound is promised for in-flight HTTP.

## Final measurements

Ubuntu amd64, GCC13.3, existing **RelWithDebInfo** (`-O2 -g`, unchanged LTO and
architecture flags); no installs, precision changes or altered solver settings.
Native scoring uses1 thread; this is not a claim to pin Ceres concurrency.
Fixed64 corner-model readers, legacy default8/channel,512MiB configured cache.
Actual public S3 sources, no artificial delay or bandwidth limit. Baseline and
candidate binary/library layouts were frozen separately. Builds did not overlap
timed measurements. Built-in stage profiling was used: system `perf` counters are
unavailable on this host (`perf_event_paranoid=4`, established earlier).

Six paired cold/warm trials per enabled version, in groups before and after the
other version; the disabled reference has three pairs. Same exact saved3735-point
fiber, dirty span6. Every output has2835 points, one native span and two native
tails, zero fallback spans. These are saved-state re-solves, not recorded clicks.

| Version | Cold solve mean (min / median / max), s | Disk-warm solve mean (min / median / max), s | Cold reinit mean, s | Cold process mean, s | Cold downloaded MB, mean |
|---|---:|---:|---:|---:|---:|
| Previous branch, prefetch off (3 pairs) |5.991 (5.856 /6.029 /6.089)|1.312 (1.302 /1.313 /1.322)|2.052|6.771|11.867|
| Previous branch, prefetch on (6 pairs) |4.452 (4.107 /4.298 /5.382)|1.324 (1.304 /1.323 /1.355)|2.051|5.306|17.310|
| New normal window + parallel tails (6 pairs) |3.796 (3.597 /3.760 /4.123)|1.324 (1.307 /1.322 /1.344)|1.443|4.606|17.373|

The follow-up reduces mean cold solve time by **14.7%** relative to the previous
enabled branch, reinit by29.7%, with essentially unchanged warm timing and cold
download volume (+0.36%). Relative to turning all prefetch off, cold solve time
is36.6% lower, but download bytes are46.4% higher. These are observations from a
small real-network sample, not latency guarantees. It still does **not** make a
fully cold solve nearly as fast as cached (3.80s versus1.32s).

Final-three-pair phase profile: span1.281s, reinit1.398s, native tails1.043s;
normal preparation inside reinit0.672s, Ceres compute~0.003s. Completed payloads
observed during phases: span2.366MB, reinit5.202MB, tails~9–10MB, including overlap.
Thus there remains both a normal-read bottleneck and a native model-read cost.

Final repeat with full binary/library hashes: baseline-repeat cold4.207s
(4.107 /4.182 /4.333), candidate-final3.722s (3.597 /3.671 /3.899), an11.5%
mean reduction. Earlier groups are baseline-on and candidate-parallel; the initial
window-only and oversized-cache smoke are excluded from the combined table.
All exact geometry/metadata/decision signatures and candidate/generation counts
matched across versions/cache states. Every recorded shared cache payload hash
matched. All final remote failures and pending-request counts were zero.
Disk-warm enabled runs still fetched optional extras (baseline mean0.517MB,
candidate mean0.535MB); they are not labeled zero-network or decoded-memory warm.

The rebuilt13-target regression suite and29 Python harness tests passed. The
optimizer fixture includes concurrent-tail cancellation/required-error coverage;
the HTTP fixture covers scalar reads joining failed speculative I/O. Historical
las008/5.json dataset-specific tests skip when those fixtures are absent; the new
PHerc1299 exact-output benchmark supplies the requested real-data validation.

## Reproduction

The direct-remote runner requires the existing shared-library CMake build layout
(Ubuntu `.so` or macOS `.dylib`, including the codec libraries). No dependencies
are installed. It snapshots input bytes itself, refuses existing output
directories and never deletes caches. Fresh-process warm runs can still download
optional keys admitted differently from the cold run; report those bytes rather
than calling these fully memory-cached GUI runs. Known missing remote objects
can also produce zero-byte requests.

Example using the frozen exact input (change only output name, binary, switch,
and optional warmup between labeled variants):

```bash
AGENTS_AGENT_MODE=1 python3 volume-cartographer/scripts/benchmark_line_model_remote.py \
  --output-dir /tmp/villa-pherc1299-prefetch.IJmtzD/new-trial \
  --binary /tmp/villa-pherc1299-prefetch.IJmtzD/baseline-build/bin/vc_fiber_trace_metric \
  --fiber-json /tmp/villa-pherc1299-prefetch.IJmtzD/baseline-on/input/fiber.json \
  --fiber-manifest /tmp/villa-pherc1299-prefetch.IJmtzD/baseline-on/input/fiber/PHerc1299-20260309130042-las-sd1-aac02eb8.lasagna.json \
  --normal-manifest /tmp/villa-pherc1299-prefetch.IJmtzD/baseline-on/input/normal/PHerc1299-20260309130042-lasagna-20260724.lasagna.json \
  --dirty-span 6 --prefetch 1 --trials 3 \
  --compare-run /tmp/villa-pherc1299-prefetch.IJmtzD/baseline-repeat
```

Omit `--dirty-span` for full retracing (a separate workload); `--warmup-ms 10000`
adds the separately timed saved-corridor warmup. Each trial's exact executable
command, model identities, stdout/stderr, trace output, stage profile and complete
post-exit cache inventory are saved automatically. `summary.json` reports
mean/min/median/max. Timeout or read/drain failure aborts rather than continuing
with an invalid pair. Compare against a run with matching fiber/manifests/span,
thread count, cache budget and legacy reader setting; prefetch/warmup may differ.
The runner passes CLI `--gui-dirty-span 6` to select that span and
`--settle-prefetch-ms 10000` for the bounded drain outside optimizer timing.

Build/test commands (existing configured dependencies only):

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
VC3D_LINE_MODEL_PREFETCH=1 VC3D_LINE_PERF_LOG=1 volume-cartographer/build/bin/VC3D
```

Set `VC3D_LINE_MODEL_PREFETCH=0` to bypass the moving/corridor prefetch and the
new parallel reinit tails at startup. No raw-scan volume is read by this benchmark.
Nothing has been published or installed; the original fiber/cache are unchanged.
macOS/arm64 runtime behavior has not been measured on this Ubuntu machine.

The source fiber SHA-256 is
`2e9a9fc9f59a6c9a7cc0187b7ddf8365e4bc67b4fa113d7b67824dd34ffae4a1`.
The original and frozen copies were verified identical after measurements.
