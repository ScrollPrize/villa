# Cache capacity and steady-pressure benchmark

Build `bench_cache_scale` in a Release native-test build. It exercises the actual cache maps, LRU nodes, keys, locks and accounting at full entry count, sharing immutable 4-KiB payloads rather than allocating the modeled payload capacity. A separate reference-count block per producer avoids making one shared payload a global atomic hotspot. This isolates bookkeeping; it does not measure I/O, decompression or physical chunk allocation/destruction.

```
bench_cache_scale ENTRIES WORKERS SHARDS MAINTENANCE SECONDS [SCAN [PROFILE [WINDOWS [full]]]]
```

For one uint8 channel in 16³ chunks:

|Modeled capacity|Entries|Example|
|---|---:|---|
|0.25 GiB|65,536|`bench_cache_scale 65536 192 192 4 2 0`|
|64 GiB|16,777,216|`bench_cache_scale 16777216 192 192 4 2 0`|
|512 GiB|134,217,728|`bench_cache_scale 134217728 192 192 4 2 0`|

`MAINTENANCE=0` selects synchronous eviction. Positive values select background workers with foreground assistance. `SCAN=0` uses full-directory oldest-16 cohorts; positive values select the oldest available tail from a bounded randomized-start scan, with age-dependent batch size. `PROFILE=1` enables lock wait/hold instrumentation; use separate diagnostic runs because profiling changes throughput.

`WINDOWS` defaults to six and accepts 1–15. Use fifteen 2-second windows for larger capacities that have not stabilized after twelve seconds; this gives thirty seconds per pressure workload.

Parallel prefill is outside measurement, reports progress and peak RSS, and aborts at 60 seconds if the requested entry count cannot be constructed. An incomplete prefill is not a result for that capacity. Keys retain the synthetic volume/chunk/channel string format. There is no giant additional key vector or payload array.

After prefill, run one read-only window, then six mixed windows (80% reads, 20% new-key insertions), then six insertion/eviction-only windows. `SECONDS=2` gives 12 seconds of each pressure workload. Explicit insertions have fresh keys. Mixed reads select the initial first quarter of keys and refill misses, as a reader would; otherwise permanent hot-set loss would make later windows artificially easy. The puts count includes these refills. Throughput counts logical requests, so actual cache calls are reads plus puts. Within each workload, the cache and maintenance workers persist across windows without any drain or capacity reset. Foreground threads restart between windows, so this has small scheduling gaps, not an uninterrupted service trace. Report progression and the last three windows rather than only the initial rate; do not assert steady state if those windows still trend.

CSV includes throughput, read hits, put latency p50/p95/p99 (one in 64 puts), eviction entries/batches/scans/claims, assisted admissions, headroom, resident/pending-reclamation bytes at window end and sampled hard-cap excess. Optional profiling adds total worker lock wait/hold times. The `drain_seconds` field is always zero in this workload: headroom replenishment is not a deferred capacity violation and no drain is timed. Async fixed-budget sampled excess is a failure. Synchronous mode can temporarily exceed its budget during concurrent writes; report that separately. Sampling is not a proof against every transient.

All setup and cache cleanup are excluded from throughput. Large fixtures use a flushed process exit after measurement so the OS releases their memory without serial traversal of 134 million nodes; there are no benchmark files or external resources to clean up. The small CTest smoke case takes the normal destructor/thread-join path. Use an outer process timeout that includes setup. Run one process at a time when comparing policies and record machine contention.

Aggregate last-window throughput with mean, median and range; inspect p99, hit retention and caller assistance alongside it. Equivalent decoded bandwidth is operations/s × 4096 bytes, not measured data transfer. A 10×-margin supported bandwidth divides that figure by ten; it is not a reason to dismiss contention or poor retention.

## Full workload suite at virtual capacities

```bash
python benchmarks/run_cache_suite.py --virtual --binary build/bench_cache_scale --output /path/to/new/results
```

This runs all thirteen workload types at 0.25, 64 and 512 GiB equivalent capacity, with foreground worker counts 1, 8, 32, 64, available CPU count, twice CPU count and ten times CPU count. `--workers` can override that matrix. Shards remain fixed at the CPU count, with four maintenance workers and an eight-tail candidate scan: only foreground concurrency varies. Each ordinary case runs six consecutive two-second windows; the compact and per-capacity scaling tables use the median of the last three and retain every window, latency, hit count and sampled excess. These windows show whether the workload has stabilized; they are not independent repetitions. The suite allows up to 300 seconds for setup (`--prefill-seconds`), because constructing 134 million entries on one worker can exceed the standalone native default of sixty seconds. The total timeout adds 180 seconds for measurements. Failures remain explicit while the other combinations continue. Optional headroom draining is never awaited.

Use `--reuse-from /path/to/previous/results` to reuse completed capacity/worker combinations. The runner checks the executable hash, host, shard count, timing, maintenance settings, payload representation and memory limit before accepting them, and records the source in each reused result. Raw logs are retained alongside the new results. This avoids repeating valid baselines without silently mixing different benchmark configurations.

The full mode extends the same native fixture, not a second cache implementation. Cases are uniform reads, sixteen hot keys, absent reads, replacement, mixed reads/writes, insertion/eviction, burst writes, paced mixed, paced bursts, mixed budget shrinking, allocating insertion/eviction, and isolated shrinking. Finite insertion throughput is measured during parallel full-capacity prefill, before maintenance starts. It is a synchronous setup measurement, not steady-state asynchronous insertion. Hot keys are explicitly populated before their case; their hit count is checked. Replacement uses initial key IDs, so a small fraction already evicted for headroom can be reinserted.

Mixed workloads refill missed hot keys, as in the existing capacity test. Bursts switch write share 20% → 90% → 20% within each window. Paced targets are one million mixed requests/s and 250,000 burst requests/s, respectively. Budget shrink uses a controlled auto-cache probe without smoothing: mixed shrinking reduces the full budget by 5% halfway through each window and restores it before the next. Isolated shrinking reduces current residency by 5% once and measures time to meet that new budget (not extra headroom); it fails after 28 seconds. Large budget changes necessarily leave previously admitted entries temporarily above the new budget. The hard-cap excess column checks against the original full capacity, not this deliberate shrink transient.

Most payloads remain shared, preserving real LRU/maps/key entry counts without allocating the equivalent payload volume. The allocation/eviction case puts a fresh real 4 KiB payload, then immediately replaces it at the same key with the shared dummy. Thus real payloads are bounded by concurrent operations and reclamation batches, rather than accumulating with virtual capacity. This case performs **two puts per logical request** and includes allocation, replacement and free costs; it is not directly comparable to the former retained-allocation case. The ordinary eviction case is unchanged. Peak process RSS is logged after prefill and every measurement window, outside the timed region.

The suite runner applies a hard **256 GiB address-space limit per native process** using `RLIMIT_AS` (configurable with `--memory-limit-gib`) and, on Linux, a **64 GiB RSS watchdog** sampled every 250 ms (`--rss-limit-gib`). Core dumps are disabled. Address space includes allocator reservations and thread stacks, not just physical memory: a 96 GiB address-space ceiling can prevent creation of 1,920 threads even with much lower RSS. The RSS watchdog kills a process over its threshold; it is sampled protection, not an instantaneous kernel-enforced RSS cap. Non-Linux runs default to address-space protection only; requesting the Linux RSS watchdog elsewhere fails explicitly. Per-process `.limits.json` files record sampled peak RSS/address space and limits. If the fixture cannot fit, failures and partial logs are retained. Run capacities sequentially, not simultaneously, and put reports on persistent storage.

`--only-failed-from /path/to/results` retries failed capacity/worker combinations without rerunning successes. `--combine-from /path/to/original /path/to/retries --output /path/to/new/combined` produces a combined report, preserving each result's source directory, executable hash and resource limits. It verifies measured workload settings and refuses to overwrite an already successful result. Changed setup/resource guards are recorded separately; they must not be described as changes to measured window duration or cache policy.

The compact three-capacity table and full diagnostics are written to `report.md`; exact commands, every window and executable hash are preserved in JSON/CSV. The small-cache contention matrix remains available without `--virtual` but is not a substitute for this suite.
