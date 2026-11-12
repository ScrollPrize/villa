# Pre-reader decoded-cache control

This optional standalone target compiles the actual VC `ChunkCache`, `DecodedChunkCacheBudget`, and supporting sources from a supplied historical Villa checkout. It runs the **same** `cache_scale.cpp` workload generator as the current cache; it does not substitute a reconstructed single-lock cache.

```bash
git worktree add --detach /path/to/pre-reader 3339e9cf2
cmake -S vc_reader/benchmarks/pre_reader -B build/pre-reader -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DPRE_READER_SOURCE=/path/to/pre-reader \
  -DDELTA3D_INCLUDE_DIR=/path/to/vc-delta3d/include \
  -DDELTA3D_LIBRARY=/path/to/libvc_delta3d.a
cmake --build build/pre-reader --parallel 8
ctest --test-dir build/pre-reader --output-on-failure
python vc_reader/benchmarks/run_cache_suite.py --virtual --backend pre-reader \
  --historical-source /path/to/pre-reader --binary build/pre-reader/bench_cache_pre_reader \
  --output /path/to/new/report
```

The generated header adds a benchmark-only friend declaration. No historical cache routine is changed. The adapter creates one source state, then calls the original decoded-result storage, result lookup/touch, and shared-budget eviction routines. Opaque decoded payloads let the fixture share immutable 4-KiB dummy payloads without allocating the modeled capacity. The source's separate metadata-entry ceiling is configured as unlimited so that the byte budget determines residency. Fixture string IDs are parsed into the original numeric source/chunk keys; historical cache maps do not use strings. The original `Entry` structure and its memory overhead remain intact.

This compares decoded-cache bookkeeping, **not** complete historical sampling: fetch scheduling, I/O and decompression are bypassed. Historical eviction remains synchronous, with no simulated background workers or auto-sizing mechanism; the fixture explicitly supplies its controlled budget reductions. Maintenance-specific counters are unsupported and omitted from the report. A synchronously enforced byte budget can have transient excess during concurrent admissions; the fixture records it without treating it as a failure of the new cache's hard-admission guarantee. Read and put lock scopes are preserved. The asynchronous current cache and synchronous historical cache therefore implement different eviction policies; compare read-hit retention as well as throughput and latency.

The full suite uses the same foreground-worker matrix, entry counts, six consecutive two-second windows, last-three-window median, allocation semantics and bounded process guards as [the current-cache benchmark](../cache-scale.md). Incomplete setup or exceeding the RSS watchdog is an explicit failed capacity result, not a reduced-entry-count substitute. The report records the historical commit and executable checksum. This target is for benchmarks only and does not reintroduce the removed cache into production.
