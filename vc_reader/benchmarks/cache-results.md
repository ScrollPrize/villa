# Decoded cache maintenance benchmark

Run the [virtual-capacity suite](cache-scale.md) in a Release build:

```sh
cmake -S vc_reader -B build/cache-bench -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DVC_READER_BUILD_NATIVE_TESTS=ON -DVC_READER_BUILD_PYTHON=OFF -DVC_READER_ENABLE_CUDA=OFF
cmake --build build/cache-bench --target bench_cache_scale
python vc_reader/benchmarks/run_cache_suite.py --virtual --binary build/cache-bench/bench_cache_scale \
  --workers 1 8 32 64 192 384 1920 --output build/cache-results
```

The fixture uses actual cache keys and records for 65,536, 16,777,216 and 134,217,728 entries, with shared 4 KiB payloads so the nominal capacities are 0.25, 64 and 512 GiB. It measures CPU cache bookkeeping, not I/O or decode. Allocation/eviction allocates a real payload and then replaces it with a shared dummy, counting two puts as one request.

Measured on shared x86-64 Linux, GCC 13.3, 192 logical CPUs and 192 shards. Four maintenance workers use eight-tail sampling and a bounded inline attempt. Prefill deliberately remains synchronous for a comparable setup control; the selected policy runs during the steady workloads. Rates are medians of the last three of six consecutive two-second windows, not independent trial medians. The runner saves per-window throughput, p50/p95/p99 put latency, hits, assisted puts, eviction batches, memory, commands and binary hashes.

## Real-data validation

Three extractions of the same 50,000 Paris4 fibers, with compound sheet normals, Lasagna winding-distance input, one shared 50% auto budget, and default maintenance, produced the same 1,180,316 constraints (201,604 perpendicular and 978,712 parallel) and rejection counts. End-to-end times were 19.21, 10.99 and 10.45 s. The previous single-lock auto-cache run on this cohort took 13.69 s. These are shared-host runs, not an interleaved A/B; the first new run was slower than the two subsequent runs. The real-data check used the existing local extractor and sampler integration; this PR changes only the reader. The sample fits in cache, so this example verifies the read path, not an eviction speedup.

The clean branch passed 12 focused native tests plus 17 Python tests and 15 subtests, including fork and loopback HTTP coverage.

## Full suite results

All 273 measurements completed (13 workloads × 3 capacities × 7 worker counts), with no sampled excess above the original fixed capacity. Peak physical RSS was 45.53 GiB under a 64 GiB RSS watchdog and 256 GiB address-space limit. Deliberate budget reductions are separate from this fixed-capacity check.

Throughput is million requests/s, except isolated shrink (million entries/s). Prefill and isolated shrink are finite measurements.

### 0.25 GiB equivalent

| Workload | 1 | 8 | 32 | 64 | 192 | 384 | 1920 |
|---|---:|---:|---:|---:|---:|---:|---:|
| hits | 6.94 | 8.61 | 23.96 | 37.54 | 49.84 | 46.61 | 47.90 |
| hot | 16.15 | 6.61 | 7.97 | 6.31 | 5.09 | 4.55 | 5.32 |
| miss | 10.07 | 27.99 | 81.13 | 115.42 | 143.55 | 128.91 | 135.60 |
| insert | 3.16 | 3.22 | 5.19 | 5.90 | 4.45 | 3.39 | 0.99 |
| replace | 7.09 | 7.78 | 14.96 | 22.23 | 30.24 | 34.08 | 35.73 |
| evict | 2.36 | 1.57 | 2.91 | 4.04 | 4.44 | 4.50 | 4.46 |
| allocate_evict | 1.67 | 1.58 | 3.47 | 3.65 | 4.01 | 4.19 | 4.16 |
| mixed | 4.43 | 3.75 | 8.50 | 11.69 | 13.94 | 14.18 | 11.23 |
| mixed_shrink | 4.18 | 3.89 | 10.60 | 12.07 | 13.60 | 14.29 | 13.15 |
| burst | 3.75 | 3.09 | 6.53 | 9.00 | 10.72 | 11.00 | 10.52 |
| paced | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.99 |
| adaptive | 0.25 | 0.25 | 0.25 | 0.25 | 0.25 | 0.25 | 0.25 |
| shrink_only | 1.03 | 2.41 | 2.70 | 2.78 | 2.95 | 2.84 | 2.63 |

### 64 GiB equivalent

| Workload | 1 | 8 | 32 | 64 | 192 | 384 | 1920 |
|---|---:|---:|---:|---:|---:|---:|---:|
| hits | 0.74 | 5.62 | 16.83 | 25.93 | 41.41 | 37.93 | 38.65 |
| hot | 16.80 | 8.02 | 6.11 | 6.06 | 5.60 | 4.69 | 5.44 |
| miss | 1.68 | 12.27 | 31.30 | 47.57 | 68.11 | 66.51 | 67.71 |
| insert | 0.92 | 2.13 | 3.16 | 3.57 | 3.73 | 3.32 | 3.22 |
| replace | 0.71 | 6.26 | 17.66 | 25.96 | 39.05 | 38.77 | 39.55 |
| evict | 0.62 | 1.44 | 2.69 | 3.96 | 4.78 | 4.89 | 4.83 |
| allocate_evict | 0.57 | 1.36 | 2.56 | 3.46 | 4.37 | 4.41 | 4.40 |
| mixed | 0.83 | 3.20 | 7.89 | 10.41 | 13.51 | 13.97 | 13.18 |
| mixed_shrink | 0.93 | 3.13 | 7.24 | 10.27 | 13.57 | 13.96 | 13.09 |
| burst | 0.82 | 2.64 | 5.70 | 8.30 | 10.64 | 1.57 | 10.41 |
| paced | 0.65 | 1.00 | 1.00 | 1.00 | 1.00 | 0.97 | 0.99 |
| adaptive | 0.25 | 0.25 | 0.25 | 0.25 | 0.25 | 0.25 | 0.25 |
| shrink_only | 2.03 | 2.80 | 2.55 | 2.48 | 2.53 | 2.52 | 2.55 |

### 512 GiB equivalent

| Workload | 1 | 8 | 32 | 64 | 192 | 384 | 1920 |
|---|---:|---:|---:|---:|---:|---:|---:|
| hits | 0.91 | 4.33 | 13.19 | 19.94 | 30.18 | 28.52 | 30.92 |
| hot | 16.56 | 7.75 | 6.13 | 6.95 | 5.59 | 4.74 | 5.41 |
| miss | 1.09 | 6.00 | 18.26 | 28.06 | 42.23 | 40.20 | 39.27 |
| insert | 0.94 | 1.92 | 3.33 | 3.71 | 3.68 | 3.33 | 3.05 |
| replace | 0.91 | 4.36 | 13.90 | 21.26 | 31.14 | 32.24 | 29.58 |
| evict | 0.52 | 1.49 | 2.76 | 3.27 | 4.32 | 5.00 | 4.79 |
| allocate_evict | 0.44 | 1.27 | 2.39 | 3.37 | 4.20 | 4.40 | 4.35 |
| mixed | 0.80 | 2.93 | 6.92 | 9.90 | 13.10 | 14.05 | 13.05 |
| mixed_shrink | 0.77 | 2.78 | 6.68 | 9.17 | 13.06 | 13.78 | 12.92 |
| burst | 0.48 | 2.51 | 5.30 | 7.33 | 10.17 | 11.04 | 10.16 |
| paced | 0.74 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.99 |
| adaptive | 0.25 | 0.25 | 0.25 | 0.25 | 0.25 | 0.25 | 0.25 |
| shrink_only | 1.28 | 2.31 | 1.91 | 1.86 | 1.83 | 1.89 | 1.85 |
