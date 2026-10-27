# Reusable fiber test and experiment runner

`native-volume-direct` and `native-volume-direct-report` reuse the local direct
audit worker for completed9ef89ac3 Paris4 native-resolution predictions. Both
FWoff/FW512 run CPU-only on56reference fibers/112directions with unchanged tuned
parameters. No inference, GPU or source cache writes. Metadata hashes bind the
new output `paris4_native_volume_9ef89ac3_direct_20261027_v1` under evaluations.

`extraction-direct-audit` runs the local Paris4 same-volume direct control,
with historical prediction-peak seeding and optional full-prefix reverse checks.
It uses the existing native training replay implementation, not simplified beam
replay. CPU-only, no inference/cache writes; outputs and recovered historical
reports go to `paris4_extraction_audit_20261026_v1` under evaluations.

`fusion-route-pilot`, `fusion-route-full`, `fusion-route-report`, and
`fusion-route-resume` run the local fusion-only cached export harness. They retain
exactly the prior duplicate-fusion decisions but restore unresolved groups, verify
serial/parallel parity and unchanged resume outputs, and audit all route handoffs.

`reduced-route-pilot`, `reduced-route-full`, `reduced-route-resume`, `reduced-route-report` use the local
`test_reduced_routes.py` harness for cached endpoint-compatible in-chunk reduction
and full routing. The pilot checks exact 1/64-worker parity and resume without
rewrites; the report audits every emitted representative and boundary connection.
No inference or GPU construction; raw chunks and prior exports stay unchanged.

`unambiguous-route-pilot`, `unambiguous-route-full`, `unambiguous-route-report`
use the local `test_unambiguous_routes.py` harness and the completed Paris4 chunk
cache. They verify two-sided connection filtering, serial/parallel parity and
resume, then audit every exported transition against the original-degree graph.
These are CPU-only, reference-independent exports; no inference/GPU construction.
Use the existing `full-route-build`/`full-route-tests` suites for native validation.

`paris4-repair-build`, `paris4-repair-tests`, `paris4-repair-pilot`,
`paris4-repair-launch`, `paris4-repair-status` and `paris4-repair-report` use the optional local
`test_paris4_repair.py` harness to verify physical-volume boundary handling,
resume the interrupted production chunk cache, and then export full routes.
The pilot/construction use GPU-manager admission; full-route export is CPU-only.
The launch supervisor records each stage and stops on failure. Generic boundary
regressions live in the native channel test and CUDA benchmark test modes.

`full-route-build`, `full-route-tests`, `full-route-pilot` and `full-route-report`
validate the reference-independent full-length route exporter. They use the
existing ephemeral native build, native/Python regression tests and a connected
Paris4 cached region at 1/16/64 workers, with fresh output and exact geometry/stop/
join comparison. No inference or GPU resources are requested. These optional
suites require `local/test_full_route_export.py` and its local cache fixtures;
generic regressions live in the committed native/Python test directories.

`trace-fast-decoder` rebuilds the current native tracing targets on ephemeral
storage and compares full GPU chunk construction/filtering/publication against
the frozen prior reader. It reuses the local 512-chunk Paris4 fixture, three
repeats and a single manager grant, checking exact geometry/seeds/stops/flags.
`trace-fast-decoder-report` regenerates the report without rerunning tracing.
Like other local benchmark suites, these require the uncommitted local harness
and fixture; they are not standalone tests on a fresh clone.

`reader-multichannel` runs paired1/7-channel public ROI workloads,16/64callers,
10GiB/512MiB cache budgets, independent exact oracle, and matched memcpy/decode
controls. `reader-multichannel-report` generates output-byte throughput and
physical-decode amplification tables. No production tracing/layout changes.

`reader-layout` builds/tests the generic blocked ROI copier and runs the locked
1/16/64-caller standalone reader benchmark plus bounded-cache streaming. It uses
the same no-argument command below; local inputs/results/builds remain on the
evaluation filesystem. No tracer or CUDA kernel runs in these measurements.
`reader-layout-pool` validates partial idle-buffer trimming and reruns bounded
streaming against the frozen pre-layout binary. `reader-layout-report` and
`reader-layout-pool-report` generate the respective Markdown/JSON reports.

Approve this command once and reuse the identical command:

```sh
/home/hendrik/villa3/scripts/run_fiber_tests
```

It takes no CLI arguments. Select named suites in the untracked
`local/fiber_tests.json`, for example:

```json
{"suites": ["runner-tests", "manager-tests"]}
```

Without a config it runs only the launcher unit tests. The runner prints the
configuration and selected tests, runs sequentially, and stops on the first
failure. Unknown keys/suites, duplicate suites and missing harnesses fail before
execution. There are no arbitrary shell commands or pytest argument passthrough.
No installs, commits, production restarts or direct GPU allocation are performed.

| Suite | Work |
|---|---|
| `runner-tests` | Launcher validation/unit tests; CPU only |
| `manager-tests` | Manager command/partial-grant/cleanup regressions; synthetic GPU inventory, temporary local sockets |
| `loader-build` | Existing isolated queued-wake CMake configure/build harness |
| `loader-tests` | Native cache/queue and related regression harnesses; may build their test targets |
| `loader-scaling` | Existing matched 1/16/64-worker experiment; requests its GPU/CPU grant through the manager |
| `loader-report` | Recreate the scaling report from saved measurements |
| `reader-isolated` | Build and measure the frozen independent native-reader baseline |
| `reader-isolated-after` | Intermediate reader candidate and native regressions |
| `reader-final` | Final native reader build, regressions, 1/4/16/64-worker scaling |
| `reader-fixed-pool` | Compare frozen/final bulk readers with a fixed 64-worker pool |
| `reader-capacity` | 64/128/224-worker native read/decode/copy/cache controls |
| `reader-roi` | Actual `Volume::readZYX` ROI reads, cold decoded cache and resident cache |
| `reader-regressions` | Rebuild/run native queue, cache, hints, Zarr and ROI regression targets |
| `reader-report` | Recreate independent reader tables from completed measurement JSONs |
| `reader-roi-multi-before` | Frozen public multi-chunk ROI benchmark (do not rebuild after source changes) |
| `reader-roi-after` | Build/regressions and matched public multi-chunk ROI benchmark with keyed wakeups |
| `reader-roi-single-after` | Matched single-chunk public ROI benchmark |
| `reader-wake-capacity` | Internal diagnostic confirming the wakeup fix at64/128/224workers |
| `reader-roi-report` | Public ROI before/after tables and full timing distributions |
| `reader-scaling-diagnosis` | Matched ROI/control inputs at1/16/64callers, physical IO/page faults and reused-buffer controls |
| `reader-scaling-report` | Summarize completed diagnostic counters and timing distributions |
| `reader-decode-reuse` | Isolated decode allocation candidate, native regressions and matched ROI/reused-decode controls |
| `reader-decode-reuse-report` | Before/after distributions including512MiB bounded-cache streaming and allocation counters |
| `reader-decode-reuse-tests` | Rebuild/rerun native regression targets without repeating performance measurements |

Loader suites require the existing `local/test_queued_*` harnesses and build/data
fixtures in this workspace; they fail explicitly on fresh clones without those
fixtures. Scaling retains the harness's resume behavior: existing completed
measurements are reused, not silently retimed. Change the experiment's output
directory when fresh timings are intended. New experiments should add a named
suite, not an arbitrary command option. Announce changed suite selection before
running it; approval of this launcher does not authorize unrelated work.

The `reader-*` suites likewise require the local `test_reader*` /
`test_isolated_chunk_reader*` harnesses and locked Paris4 inputs. They run a
standalone C++ benchmark, **without tracing, inference or CUDA kernels**. The
manager currently requires a GPU promise even for CPU work; these harnesses
request a nonexclusive 1 MiB token on one device plus the measured CPU resources.
Builds and scratch for new reader candidates live under the evaluations root.
The generic benchmark target and portable configuration are documented in
[`chunk_channel_hints.md`](../volume-cartographer/docs/chunk_channel_hints.md).

The launcher uses the existing `/home/hendrik/.venv_las/bin/python` environment.
The `stitched-reference` suite compares legacy online replay, complete export-rule
replay and frozen exported routes with the same Paris4 fields/chunks and reference
protocol. `stitched-reference-audit` measures independent raw-path proximity;
`stitched-presence-audit` inspects stored presence around reference locations.
These require `local/test_stitched_reference.py` and its locked local data. They
run CPU-only, without inference or new chunk construction. Result reuse is bound
to reference/config hashes, export generations and the actual native library;
choose a fresh output directory after implementation or input changes.
The `owned-route-pilot`, `owned-route-full` and `owned-route-report` suites use
`local/test_owned_route_export.py` with the completed Paris4 cache. They test
serial/parallel ownership parity and resume, run a CPU-only export into a separate
directory, and compare output counts/lengths/stops. They do not rebuild chunk paths
or request GPUs. Select `full-route-build` and `full-route-tests` first when changing
native code. Local inputs and measurements remain outside committed source.
On other hosts set `VILLA_TEST_PYTHON` to an existing interpreter with pytest.
It sets repository imports and single-thread BLAS defaults, and never installs
an environment. Long GPU work stays under the existing manager-aware harness.
# Direct candidate batching validation

`direct-batch-before` captures the scalar control on four frozen 1024-base
reference prefixes from native Paris4 volume 9ef89ac3. Its lifecycle check stops
only the identified old audit worker. `direct-batch-build`, `direct-batch-tests`
and `direct-batch-after` build, run native regressions and compare the batched
implementation with that control. Each timing uses 16 traces, one or 16 outer
workers, three warm repetitions, with no GPU/inference. Results and exact
geometry comparisons are under `evaluations/direct_candidate_batch_20261027_v1`.

## Native-volume chunk benchmark

`native-direct-width-sweep` runs eight direct native-volume Paris4 variants:
cosine width tolerances30/20/15/10% crossed with post-modulation cutoffs0.4/0.25.
It reuses the existing32-case parallel audit worker and completed predictions,
with all reverse verification disabled; no inference or GPUs. Raw/min512 metrics,
case geometry, effective settings and baseline comparisons are saved under
`paris4_native_direct_width_sweep_20261027_v1`. The existing seed-presence floor
also follows the cutoff. Select the suite in `local/fiber_tests.json`, then run
`scripts/run_fiber_tests`; completed cases resume under a matching binding.
The suite also runs a fresh40%/0.4 control and verifies the native-library and
reference identities match, with zero reverse checks across all cases.
`native-direct-width-lower` runs only the30% extension at cutoff0.1,0.01 and
off(0), then extends the same report using the completed controls. Its separate
`lower_cutoffs30` binding preserves the original experiment's resume identity.

`direct-separate-gates-lower` extends the same experiment with presence cutoffs
0.15/0.10/0.05 at both actual width limits, reusing the parent control after
native/input/config identity validation. Output is its `lower_presence` directory.

`direct-joint-hard-gates` tests cosine width targets 20/30/40% crossed with
actual hard width limits 5/10%, with the original joint cutoff .25 (including
width modulation). Same native worker/control; output is `joint_hard_width`.

`direct-joint-hard-fine` runs targets25/35% at hard10%/joint.25 and reuses
compatible20/30/40% results in the comparison. Output: `joint_hard_width/targets25_35`.

`direct-joint-hard-relaxed` runs hard actual width15/20% at cosine30%/joint.25,
comparing to cached no-hard and hard10% results. Output: `joint_hard_width/hard15_20`.

`direct-reverse-delayed` reruns the recorded hard20/FWoff baseline with the current
native binary and compares delayed future-history reverse verification with
convergence (512interval,4future edges,1536minimum reverse travel,4vx/5degrees,
4consecutive steps). Baseline case geometry must match historical output exactly.
Use `direct-separate-gates-build` first. Output: `paris4_direct_separate_gates_20261027_v1/reverse_delayed_v1`.

`direct-reverse-tail` tests presence/width stop-tail retention (four accepted
history segments retained on pass), with minimum1024base reverse travel and
8base/10degree convergence, four consecutive matches. It reruns FWoff for exact
historical parity and compares the prior1536/4/5 experiment. Output:
`paris4_direct_separate_gates_20261027_v1/reverse_delayed_tail_v2`.
Build first with `direct-separate-gates-build`; select the suite in
`local/fiber_tests.json`, then run `/home/hendrik/villa3/scripts/run_fiber_tests`.
`direct-reverse-tail-report` regenerates validation/statistics without tracing.

`direct-reverse-single` changes only convergence count4→1 at1024base/8base/10degrees,
reusing the corrected stop-tail policy. Runs112directions with32native case threads;
hash-checks cached four-match/FWoff controls. Output:
`paris4_direct_separate_gates_20261027_v1/reverse_single_convergence_v3`.
`direct-reverse-single-report` validates and regenerates the report without tracing.

`direct-sampling-profile` measures native timers on32distinct1024base Paris4
prefixes, one compute thread per trace, plus a4096base long-prefix diagnostic.
Build first with `direct-separate-gates-build`. Data/config source is the completed
single-match reverse trial; no inference/GPU. Output: `sampling_profile_v1` under
`paris4_direct_separate_gates_20261027_v1`. `direct-sampling-profile-report`
regenerates statistics only. `direct-sampling-profile-finish` preserves the first
completed32case serial control, stops only that exact diagnostic worker, and
finishes three32worker repeats and the long diagnostic; avoids repeating the
expensive serial pass. Reports explicitly list repeat counts. Optional
`TraceProfile` counters reset per call and reject active reads/reuse.

`sampling-overhead` builds/runs `bench_fiber_sampling_overhead` and native
prediction/normal sampling regressions. It compares scalar-equivalent cached
batches with the existing owned `FiberChunkTensor`, plus raw scalar interpolation
from retained chunk bytes and a pregathered arithmetic lower bound. Six real
Paris4 sites, deterministic synthetic clouds of 648/52,488 points, five repeats,
one native worker, no inference/GPU. Values/validity are checked exactly and
storage-seam crossing is asserted. Output: `sampling_overhead_v1` beside the
profile above. Setup, memory footprint and approximate break-even points are
reported separately; these are not end-to-end tracer timings.

`direct-sampling-window` builds and tests the opt-in direct CPU sampling window,
then compares32real Paris4 prefixes (1024 base) and a4096base reverse-check case
with sampling_window_voxels=0/64. All non-timing replay outputs must match exactly.
`direct-sampling-window-full` follows a successful pilot with all112complete
Paris4 directions, comparing against the saved single-match reverse baseline.
`direct-sampling-window-checks` builds/runs regressions without rerunning tracing.
One native thread per case,32cases concurrently,no inference/GPU; output under
`paris4_direct_separate_gates_20261027_v1/sampling_retained_v1`. The earlier
`sampling_window_v1` measured copied ROI arrays; retained mode instead holds
decoded cache payloads plus immutable pointer tables, with zero ROI voxel copies.
The prepared-address implementation now writes `sampling_retained_prepared_v1`
instead, retaining both earlier versions for comparison. The full suite can be
repeated after a successful pilot; it replaces only its own full-run row.

`retained-sampling-diagnosis` builds the existing sampling benchmark and compares
both production storage backends in one executable:64stored windows, six Paris4
sites including seams, five rotated repeats,648/52,488points. Diagnostic variants
isolate the retained dense-index roundtrip versus direct/prepared XYZ, using the
same grid/buffers and shared interpolation. Every repeat asserts exact parity;
native prediction/normal regressions follow. Perf availability is probed, not
changed. Output `sampling_retained_diagnosis_v1` includes setup, memory, paired
timing deltas and saved full112phase attribution. No production settings change.
The current prepared-address comparison writes `sampling_retained_prepared_diagnosis_v1`
and compares against the preserved initial diagnostic and full replay profiles.

`unified-sampling-diagnosis` builds/runs the same native sampler benchmark with
isolated coordinate/address/gather, scalar interpolation, compact-vector tensor
accumulation, axis solving/normalization and prediction materialization stages.
It also checks complete combined-channel versus separate-channel scalar parity.
Results: `sampling_unified_diagnosis_v1`. Stage timings include staged-buffer
traffic and are not additive production cycle attribution. Five rotated repeats
at six real/seam sites,648 and52,488points. No inference/GPU.
Current `direct-sampling-window`/`direct-sampling-window-full` runs write
`sampling_unified_v1`, preserving previous retained/copy baselines.

`sampling-inpass-profile` measures exclusive intervals inside actual combined
sampler calls, including normal and width work, output/loop/control and gather
operations. Integer stage sums must equal their instrumented root elapsed time.
It separately measures uninstrumented controls: per-point clocks significantly
perturb runtime, so these stage costs must not be passed off as uninstrumented
costs. Results:`sampling_unified_inpass_v1/REPORT.md`; validation replay:
`sampling_unified_inpass_replay_v1`. Production stage instrumentation is off.

`direct-separate-gates-build` builds the native binding and tests independent
width/presence eligibility, committed-width history and replay stop handling.
`direct-separate-gates` runs the matched30%/0.25 control and actual-width hard
gates5/10% crossed with width-free direction-modulated presence0.25/0.5/0.75,
on cached Paris4 predictions. It reuses the native parallel replay worker and
verifies control outputs exactly against the preceding experiment. No inference
or GPU use. Outputs: `paris4_direct_separate_gates_20261027_v1`.

`native-chunk-width-audit` samples cached case171 paths with the native prediction
field and fiber-normal samplers. It builds only a small diagnostic helper against
the existing benchmark build, performs no inference/retracing, and writes width
statistics under the benchmark's `failure_diagnosis/width_audit.json`. Select it
in `local/fiber_tests.json` and run `scripts/run_fiber_tests`.

`native-chunk-failures` audits the three saved native-volume chunk failures using
cache-only native route replay, saved direct coverage and conflict-pair checks.
It does not run inference or construct new GPU paths; evidence is written only
to the benchmark's `failure_diagnosis` directory.

`native-chunk-build` builds the current CUDA plugin and tests the generic
evaluator/workflow. `native-chunk-pilot` runs eight reference directions through
the existing native on-demand chunk queue; `native-chunk-full` expands to the
112 Paris4 directions after the pilot gate. The local harness is
`local/test_native_chunk_benchmark.py`; its configurations/results live in
`evaluations/paris4_native_chunk_9ef89ac3_20261027_v1`.

GPU work uses existing manager admission, nonexclusive partial grants and the
actual available device/CPU budgets. It reads completed native-resolution Zarr
predictions and constructs only requested chunks; it does not run inference or
whole-volume production. Existing chunks are reused, and results report raw and
512-base filtered metrics against the same-case direct and historical controls.
