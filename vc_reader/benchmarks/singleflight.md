# Chunk-load coordination benchmarks

Build with the existing native-test configuration in Release mode:

```sh
cmake --build <reader-build> --target bench_chunk_singleflight bench_singleflight_contention test_native_singleflight --parallel 8
ctest --test-dir <reader-build> -R test_native_singleflight --output-on-failure
```

`bench_chunk_singleflight [users=32] [waves=8]` tests 13 full/selected/raw-range cases against disposable generated local chunks and computed sources. Each decoded chunk is 1 MiB, with four uint8 channels, compressed with Blosc/Zstd or stored raw. Three repetitions contain eight cold waves (or resident hits for the hot cases). JSON lines report mean, median, p95, min/max, throughput and actual source loads. Input preparation and thread creation are excluded from the timed waves. Filesystem-cache results are not device-bandwidth measurements. Fixtures are automatically removed, including on exceptions. Set `TMPDIR` to select the fixture device. The process rejects starting another case after its 25-second workload budget.

`bench_singleflight_contention CASE USERS [seconds=1]` tests sustained coordination without a large backing volume. Cases are `independent`, `collision`, `shared`, `failures`, `probe`, `whole`, and `selected`. Collision keys deliberately share a former 256-way stripe to catch accidental serialization of source work. Shared/failing loads use eight keys with 50 µs simulated source latency. Probe requests inspect already busy keys without waiting. Whole/selected cases exercise actual reader sampling/loading and decoded-cache eviction using 256-byte generated chunks and a 64 KiB cache; scalar point requests have no reader helper threads. The small payloads and bounded sample vectors isolate coordination without allocating a large volume per user.

Use users `1 8 32 64 192 384 1920` on a 192-CPU host, or adapt the last count to ten times the available CPUs. Each invocation creates one team, warms up for 250 ms, then measures three windows; completion of operations already in flight is included in elapsed time. No maintenance drain is included. JSON reports median/mean/min/max throughput, sampled p50/p95/max latency, producer calls, operations, errors and actual measured durations. Latency sampling is bounded to 4,096 entries per thread per window. Thread-creation failure releases and joins already created threads. Use an external 28-second process timeout to detect stalls.

Report each case across all user counts, including the slowdown at ten-times-CPU oversubscription relative to that case's maximum throughput. Shared-key workloads deliberately contain genuine dependence on a single producer: throughput need not scale linearly with more consumers. Compare a bounded real extraction workload separately before making full-volume runtime claims.

## Measured reference

Release build, Linux, 192 available logical CPUs, three one-second measured windows after warmup. Median million requests/s; 1,920 users is ten times CPU concurrency. Final suite: zero errors in all 49 invocations. These tiny-payload workloads measure coordination, not disk bandwidth or full extraction speed.

| Case | 1 | 8 | 32 | 64 | 192 | 384 | 1920 | Peak/1920 slowdown |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| independent | 4.98 | 5.03 | 9.03 | 10.93 | 17.70 | 18.83 | 19.77 | 1.00× |
| collision | 5.00 | 5.33 | 9.41 | 13.78 | 18.07 | 19.13 | 20.19 | 1.00× |
| shared | 0.01 | 0.08 | 0.29 | 0.54 | 1.53 | 2.43 | 2.19 | 1.11× |
| failures | 0.01 | 0.07 | 0.28 | 0.53 | 1.52 | 2.41 | 2.25 | 1.07× |
| probe | 35.01 | 24.49 | 30.40 | 35.49 | 23.63 | 19.53 | 22.34 | 1.59× |
| whole | 0.80 | 1.00 | 4.05 | 6.22 | 11.06 | 12.64 | 5.25 | 2.41× |
| selected | 0.58 | 0.67 | 2.35 | 4.96 | 8.50 | 11.30 | 10.78 | 1.05× |

Independent-key and former-stripe-collision throughput does not collapse under oversubscription. Shared-key cases have genuine dependencies. The whole-reader stress case drops 2.41× from its peak and also performs more source loads per request at 1,920 users; this does not isolate the cause of its entire slowdown. Whole/selected cases intentionally churn a tiny cache and should not be read as representative capacity/hit-rate predictions. General 1 MiB read benchmarks at 128 users improved same-chunk whole reads 8.15×, colliding selected reads 5.55×, and uncached same-chunk selected reads 14.76×, but distinct selected reads regressed ~17% in that sample. No claim that every workload improves.

On a bounded real 50,000-fiber compound-normal extraction using 50% auto cache, the normal sweep changed from 2.93 to 2.70 seconds at 64 workers (two repeats), and 2.86 to 2.37 seconds at 128 workers (one run). Whole-process time changed only from 13.56 to 13.06 seconds and 12.79 to 12.59 seconds respectively, including output persistence and cleanup. Exact prior arrays matched. These were warm, non-interleaved controls on a shared host; modest percentage improvements are indicative rather than a rigorous small-difference claim. Synthetic gains do not establish full-volume runtime.
