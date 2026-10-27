# Synthetic Rendering Benchmark

`bench_render_synthetic` measures the production
`ChunkedPlaneSampler::sampleCoordsFineToCoarse` path without storage, network,
decompression, or asynchronous chunk scheduling. It uses the production
`ChunkCache` with a four-level deterministic `IChunkFetcher`. The fetcher
materializes pseudo-random chunk bytes during preload, and the benchmark fails
if rendering reaches it after measurement setup. Production cache lookup,
locking, and request-context handling remain inside the measured path.

## Scenarios

Each case uses trilinear sampling and validates every output pixel against its
expected source pyramid level.

| Scenario | Residency pattern |
| --- | --- |
| `full_res` | Every coordinate is available at level 0. |
| `fallback_3` | Levels 0-2 are missing; every coordinate uses level 3. |
| `mixed_correlated` | Equal spatially coherent quarters use levels 0, 1, 2, and 3. |
| `mixed_shuffled` | The same coordinates and level mix, deterministically shuffled across output pixels. |

The `serial` fixture is `96x96`, below the sampler's `128x128`
parallelization threshold. The `parallel` fixture is `256x256` and uses
`VC_RENDER_SAMPLER_THREADS=4`. Native validation confirms that multiple worker
threads execute the latter path. The environment override accepts `1..8`, is
read once per process before the first sampler call, and leaves automatic worker
selection unchanged when unset.

## Native Timing

Configure a generic Release build and build with the desired local parallelism:

```bash
cmake -S volume-cartographer -B volume-cartographer/build-release \
  -DCMAKE_BUILD_TYPE=Release \
  -DVC_MARCH_NATIVE=OFF \
  -DVC_TESTING=ON \
  -DVC_RUN_RENDER_BENCHMARKS=ON \
  -DVC_BUILD_APPS=OFF \
  -DVC_BUILD_UI_TRACER=OFF \
  -DVC_BUILD_FLATBOI=OFF
cmake --build volume-cartographer/build-release --parallel 32 \
  --target bench_render_synthetic
```

Run one native case:

```bash
VC_RENDER_SAMPLER_THREADS=4 \
  volume-cartographer/build-release/bin/bench_render_synthetic \
  --fixture parallel --scenario mixed_correlated --native-trials 9
```

The JSON output reports mean, median, and p95 wall time, median pixels/second,
checksum, observed worker count, compiler, build type, and architecture target.
Native timing depends on the host and is not the portable CI gate.

## Deterministic Cost

Configure the non-default Release benchmark graph and run the complete gate:

```bash
cmake -S volume-cartographer -B volume-cartographer/build/ci-render-benchmark \
  -G Ninja -DCMAKE_BUILD_TYPE=Release -DVC_MARCH_NATIVE=OFF \
  -DVC_TESTING=ON -DVC_RUN_RENDER_BENCHMARKS=ON \
  -DVC_BUILD_APPS=OFF -DVC_BUILD_UI_TRACER=OFF -DVC_BUILD_FLATBOI=OFF
cmake --build volume-cartographer/build/ci-render-benchmark \
  --target bench_render_synthetic --parallel 32
ctest --test-dir volume-cartographer/build/ci-render-benchmark \
  --output-on-failure -R '^test_render_synthetic_fixture$'
jobs=$(nproc)
cmake --build volume-cartographer/build/ci-render-benchmark \
  --target render_valgrind_ci --parallel "$jobs"
```

Ninja owns the complete artifact graph. It schedules at most four independent
Valgrind collectors at once. This is valid even for four-worker guest fixtures
because Valgrind serializes each guest; native verification remains a separate
non-oversubscribed test.

Callgrind starts with instrumentation disabled. The executable materializes and
warms the fixture, allocates one output and coverage matrix per measured
repetition, then uses Callgrind client requests to measure only the production
sampling calls. Validation happens after instrumentation stops.

The fixed simulated cache is:

- I1: 32 KiB, 8-way, 64-byte lines
- D1: 32 KiB, 8-way, 64-byte lines
- LL: 8 MiB, 16-way, 64-byte lines
- Branch simulation: enabled

### Fresh Valgrind replay gate

The CI gate uses `scripts/run_render_valgrind_ci.py` and writes under
`build/ci-render-benchmark/render-valgrind-ci/<fixture>/<scenario>/`. Every case
has one separate-thread Callgrind execution. Parallel cases use the scheduler
and futex stream from that same execution. Collection manifests are written
atomically after validation, and each raw file is hashed before evaluation.

The frozen model is `core/test/data/render_valgrind_ci_model.json`. It uses the
seven synthetic-only data-read event coefficients. Serial score is total
modeled thread work per render call. Parallel score is the native FIFO replay
makespan per call with these fixed controls:

```text
workers = 4
cores = workers + 1 = 5
split = equal
residual_fraction = 0.5
wake_latency = 0
cross_thread_release = 3157.1798928563853 ns
replay_idle_scale = 1
dependency_excess_scale = 1
```

Process startup and native timing are not part of this score. The calibration
did not pass its absolute-timing promotion gates, so the metric is a relative
modeled-runtime score only. It must not be described as estimated native wall
time. Each candidate score must fall in `[0.90, 1.10]` times the checked
reference.

Historical model hashes, checksums, compiler, cache geometry, architecture,
fixture shape, repetition count, worker count, and Valgrind version are
diagnostic and do not fail comparison with the reference. Current-run metadata,
the measured window, and successful futex dependencies must still be internally
valid. Material profiler effects remain subject to the modeled-score tolerance.
CI retains the complete `render-valgrind-ci/` tree even on failure.

The C++ replay engine (`bench_thread_sync_replay evaluate-render`) parses the
periodic profiles and scheduler log, computes event features and per-thread
modeled costs, then performs serial aggregation or parallel replay. Python only
drives collection retries and freezes references.

The 2026-08-07 GCC 15.3.0/Valgrind 3.25.1 reference collection took 15.73 s on
the four-core calibration host. A second fresh collection and gate took 16.38
s. Its parallel score ratios ranged from 0.993 to 1.016; all serial scores were
identical. CI derives Ninja concurrency from `nproc`; four was a property of
that reference runner, not a workflow constant.

## Operating And Updating The Gate

See [VC3D Rendering Performance Gate](render_valgrind_ci.md) for local use,
artifact diagnosis, CI activation, model changes, eight-case reference
refresh, and intentional tolerance changes.
