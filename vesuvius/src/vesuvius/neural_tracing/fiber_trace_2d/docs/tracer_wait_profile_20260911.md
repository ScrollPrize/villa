# Native wait profiling and readiness fast path —2026-09-11

## Measured on the existing run

Native PID246841; classic50-fiber/100-direction cohort, exclusive manager task
232a816c-345a-4e17-b7ae-ac9260cdcc6c. Existing RelWithDebInfo GCC13 executable.
No shutdown, GPU launch, inference-cache deletion or settings change occurred.

| Observation | Window | CPU cores | CPU-active threads | Physical reads | Observed TCP receive |
|---|---:|---:|---:|---:|---:|
| Initial aggregate |10.085s|18.94|not recorded|8.4MiB|0.28MiB|
| Per-thread sample13:54:45UTC |10.079s|22.12|38|92KiB|318bytes|
| Later tail13:59:57UTC |5.082s|9.50|11|24KiB|see raw report|

In the38-thread sample, several hot threads alternated CPU and futex waits.
In the later sample, leading threads consumed~4.9CPU seconds over5.08s,
consistent with a late tail dominated by individual single-thread traces.
At the subsequent check94/100 directed cases were complete.

These samples do not identify a particular C++ lock. Idle service workers are
also included in aggregate futex counts. GDB attachment was denied; perf is
absent/restricted, and per-thread syscall addresses were denied independently
of wchan access. No permission workaround was attempted.
The observed negligible I/O applies only to these windows, not the whole run.

Raw reports live under:
`/ephemeral/hendrik/fiber3d_train2/evaluations/classic_s1_91500_20260911_unified_cache_v1/process_probe_20260911T135445768446.json`
and `process_probe_20260911T135957022148.json`.

The approved observation command remains:

```bash
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py observe --config /home/hendrik/villa2/vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/configs/eval_classic_collection_unified_cache.json --pid 246841 --seconds 10
```

## Confirmed unnecessary work and fix

The scalar candidate path invokes readiness for each candidate. Previously the
case callback always released its CPU permit, acquired the shared bridge mutex
to examine acknowledged chunks, then reacquired its permit—even for ready data.

FiberCaseReadiness now stores a bounded set of exact array-path/chunk-index
publication proofs owned by one live case thread. Known hits return before
bridge or permit operations. Only successfully acknowledged dependencies are
remembered; missing/new/evicted entries are checked through the existing bridge.
The default is4096 entries/case (up to409600 entries across100cases); paths,
hash nodes and buckets determine actual metadata bytes. This is not another
decoded cache. Set evaluation.readiness_cache_entries=0 for the unmemoized
comparison. The underlying shared bridge still coalesces knowledge across cases.

Atomic cancellation is checked on hits and after waits. Error text is accessed
under the bridge mutex only when cancellation is detected. The enforced single
thread per live case remains necessary for this case-owned helper.
No sampling math, tracer settings or first-read readiness policy changed.

This does NOT yet eliminate dependency-list construction in FiberPredictionField,
change scalar scoring to batch scoring, or enable nested per-trace parallelism.

## Opt-in native profiling

Set evaluation.profile_tracing=true in the workload config; use the same
approved full/pilot runner. No new executable command is required.

Per-case counters include readiness calls, local hits, bridge calls, submitted
dependencies and proof-cache clears. Timers include local lookup, explicit
bridge-state mutex acquisitions, ready waits and inclusive bridge/permit time.
Existing native stage timings and candidate/generation counts are also recorded.
Completed cases write profile.json. Progress callbacks print JSON at most once
per10s; they cannot produce a heartbeat during a single stuck wait/generation.

Do not sum nested timings. Bridge/permit time contains mutex and ready wait;
ready wait includes condition-variable reacquisition/scheduling. Explicit
bridge-lock time excludes sendMutex and condition-variable reacquisition.
Scalar candidate scoring contains readiness and sampling. Disabled profiling
does not add clocks/counters in hot paths.

The original process had none of these new counters or optimizations and
completed naturally before the requested stop (100/100,2316.144s native wall).
The shortened cache-only comparison below supersedes the proposed full rerun.

## Short cache-only comparison

Completed corrected results and complete timing table: [cached_trace_profile_20260911.md](cached_trace_profile_20260911.md). Native mean71.968→65.703s (8.71% less time); exact original replay equality, zero inference.

`cached_trace_profile.py` uses the shared authenticated `coordinate_native`
loop, but supplies read-only published stores. It validates the completed
inference inventory, checkpoint identity and locked annotation membership.
A missing channel chunk is an error: no inference worker, model, accumulator
or inference write may be created. Actual decoding remains in the native cache.

The first attempted128-base-voxel prefixes failed closed: their reverse traces
started from new endpoints and requested uncached chunks. No inference ran.
Failed evidence is retained in `classic_cached_readiness_20260911_v1`.
The final v3 cohort instead uses the four shortest complete references,
eight directions, spanning PHerc0139, PHerc0332 and Paris4. It is a small
cached-tracing pilot, not a hundred-case concurrency/scaling benchmark.

Run order is readiness-cache disabled/enabled/enabled/disabled, with unchanged
beam8,lookahead2,threads1,step4 and shared201.54GiB decoded-cache budget.
Each native process gets a fresh decoded cache but reuses the same published
inference and OS source cache. One exclusive manager allocation covers all
four passes; no GPU inference executes. Every case artifact is required and
the entire serialized replay geometry/failures must match between modes and
the original completed full evaluation. Whole-fiber mode reads original files
directly, preserving precision and metadata. Preliminary v2 off/on results
matched each other, but Python reserialization slightly rounded reference
coordinates; those timings are superseded rather than mixed into final results.

```bash
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py profile --config /home/hendrik/villa2/vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/configs/eval_classic_collection_cached_trace_profile.json
```

Output is configured under
`/ephemeral/hendrik/fiber3d_train2/evaluations/classic_cached_readiness_20260911_v3`.
Use a fresh profile.output_root for another invocation: evidence is never
overwritten. Cohort hashes, per-case profiles, native summaries and incremental
results.json are retained. Both modes enable identical timing instrumentation.
Individual runs are bounded to300s after coordinator start.

## Validation

The dry-run build showed app/test executable work only; no core shared library
mapped by the running process was rebuilt.

```bash
cmake --build /tmp/vc-live-native-cmake --target vc_fiber_trace_metric test_fiber_trace3d -j 8
bash /home/hendrik/villa2/vesuvius/scripts/run_fiber3d_native_replay_tests /tmp/vc-live-native-cmake /home/hendrik/.venv_las/bin/python
/home/hendrik/.venv_las/bin/python -m pytest --noconftest -o pythonpath=/home/hendrik/villa2/vesuvius/src -q vesuvius/tests/neural_tracing/test_live_process_probe.py
```

318 native cases,71 integration Python tests (31.06s),3 observer tests (1.60s).
Independent plan/code review passed.

Synthetic2-direction replay, both beam widths1 and4:
18 readiness calls; without memo18 bridge entries; with memo6 bridge entries
and12 local hits. Full replay geometry matched offline exactly in all cases.
Unit tests additionally cover1000 repeated hits, failed/cancelled acknowledgements,
new paths/indices, separate cases, eviction and disabled mode.
This verifies behavior and call counts, NOT a full-dataset throughput gain.

Linux execution only; macOS/arm64 runtime untested. Observer syscall decoding
recognizes x86_64/aarch64 futex WAIT forms only; unsupported/denied data is absent,
never represented as an identified lock.
