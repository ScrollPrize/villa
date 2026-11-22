# Complete cache contention benchmark

Build `bench_cache_contention` in an existing Release build. The reusable runner launches the C++ workloads, retains every repetition and exact command, and generates full throughput tables and a detailed latency/accounting table. Python only orchestrates processes and reports results; the measured cache and workload are C++.

```bash
cmake --build build --target bench_cache_contention -j 8
python benchmarks/run_cache_suite.py --binary build/bench_cache_contention --output /path/to/new/results
```

The output directory must be new. Defaults are three repetitions, 0.5 seconds per ordinary workload, one second for bursts and 1 Mops/s paced traffic, and two seconds for 250 kops/s adaptive bursts. Worker counts are 1, 8, 32, 64, CPU count, twice CPU count and ten times CPU count. Override with `--workers 1 8 32` or `--seconds 1 --repeats 3`. Each process has a 28-second timeout; failures remain explicit in the table and cause a nonzero suite exit. Run on an otherwise idle machine for controlled comparisons.

The suite compares synchronous sharded eviction with four maintenance workers sampling eight shard tails. Both use one shard per available CPU. Policies are interleaved to reduce host-load bias. This is a benchmark selection, not a change to production defaults.

| Case | Workload |
|---|---|
| hits | Uniform cached reads |
| hot | Reads contending on sixteen keys |
| miss | Absent-key reads |
| insert | Exactly 65,536 distinct insertions; finite, includes worker wakeup overhead |
| replace | Replace existing cached values |
| evict | Pressure writes with retained payloads, isolating bookkeeping |
| allocate_evict | Pressure writes with fresh payload allocations |
| mixed | 80% hot reads, 20% writes; missed reads are not refilled |
| mixed_shrink | Mixed traffic with available-memory changes every 10 ms |
| burst | Write share changes from 20% to 90% and back to 20% |
| paced | Mixed traffic paced at 1 M operations/s across all workers |
| adaptive | Burst traffic paced at 250 k operations/s |
| shrink_only | Single-thread synchronous budget shrink, 32 refill/shrink cycles |

Fixtures use 65,536 keys and 256-byte payloads. Pressure cases hold 4,096 entries (1 MiB); non-pressure cases have a 32 MiB budget. Fixture creation and prefill are outside timing. The isolated shrink row measures entries evicted/s, not foreground read/write operations. It is synchronous by definition and is not falsely duplicated as an asynchronous result. Read hits and puts are both counted as operations in the other rows. An operation rate alone does not establish useful-cache performance: compare read-hit rates, accepted puts, bypasses and eviction statistics in the raw results too.

Put latencies sample every 64th put per worker. Reports include mean/median/min/max rates, throughput per worker, p50/p95/p99 put latency, read hits, foreground-assist fraction, batch size and sampled budget excess. No optional post-workload headroom drain is timed or awaited. End-of-work budget and hit/insertion invariants are checked. A budget reduction can temporarily place existing contents above the new budget; sampled excess also includes those transitions. Synchronous insertion may transiently exceed its budget under concurrent writes. A zero sampled excess is not proof that every instant was sampled.

`metadata.json` records the executable hash and configuration; `results.json` retains raw repetitions and commands; individual CSV/stderr files retain native output. `report.md` contains every case and worker count, including failures. These small-fixture tests isolate contention and allocation; use [the capacity benchmark](cache-scale.md) separately for realistic entry counts and longer steady-state checks.

Native positional arguments are `WORKERS REPEATS SECONDS CASE SHARDS OPERATIONS MAINTENANCE_WORKERS OPS_PER_SECOND CANDIDATE_SCAN`. Zero `OPERATIONS` selects duration-based operation and zero `OPS_PER_SECOND` disables pacing. `CANDIDATE_SCAN=8` selects the sampled policy; zero selects full-directory candidate selection. The retained CSV `drain_seconds` column is zero because optional headroom draining is not measured.
