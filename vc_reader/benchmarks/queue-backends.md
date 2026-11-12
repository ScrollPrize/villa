# Queue backend benchmark

`queue_backends.cpp` compares Boost.Lockfree, Rigtorp MPMCQueue and moodycamel ConcurrentQueue with pointer-sized messages, concurrent producers and consumers. It is separate from `thread_pool.cpp`, which measures the actual reader scheduler, including callback execution, completion, wakeups and descriptor allocation. Neither benchmark reads volumes or writes databases in its timed region.

Build `bench_queue_backends` with native tests enabled. Boost is required by the benchmark; the reader's native configuration also requires the concurrentqueue CMake package. Benchmark adapters are detected from installed headers: `VC_READER_BENCH_RIGTORP_INCLUDE_DIR` must contain `rigtorp/MPMCQueue.h`, and `VC_READER_BENCH_MOODYCAMEL_INCLUDE_DIR` must contain `concurrentqueue.h`. Ubuntu supplies moodycamel in `libconcurrentqueue-dev`; Rigtorp can be supplied from upstream commit `b9808ede08f26fa9df4df4e081d19cace8f6c6ea`. The benchmark does not download dependencies.

```bash
cmake --build build --target bench_queue_backends -j 8
ctest --test-dir build -R '^test_queue_' --output-on-failure
build/bench_queue_backends moodycamel tickets 128 16 4096 0.5
python benchmarks/run_queue_suite.py --binary build/bench_queue_backends --output /path/to/new/results --build-type Release
```

Native arguments are backend (`boost`, `rigtorp`, `moodycamel`), mode (`check`, `transfer`, `tickets`), producer count, consumer count, initial capacity and seconds per measured window. The suite runs a separate exact-once bitmap check for every backend before the timed cases. Each timed case has a warmup and three measured windows, validates counts/checksums and includes final queue draining. Each native process has a configurable timeout (28 seconds by default). Setup, token initialization and report writing are outside timing.

`transfer` measures successful enqueue/dequeue transfers. `tickets` adds preallocated descriptor references, completion flags, stale-ticket observation and safe descriptor reuse. It does not execute callbacks or simulate four callback writes: use `bench_thread_pool` for actual batch-size effects. Successful transfers—not failed polls—are the throughput unit; two queue operations constitute one transfer. Failures, empty polls, busy descriptors, CPU time and stale fractions are reported separately. Sampled operation latencies include failed attempts; histogram p50/p95 values are bin upper bounds.

Boost and Rigtorp enforce the configured global capacity. Moodycamel does not: its producer-local blocks require additional reserve, so the benchmark preallocates `round_up(capacity + 64 × producers, 32)` message slots and warms two reusable blocks per explicit producer before timing. This prevents an early producer monopolizing the shared block pool. Timed publication uses non-growing `try_enqueue`. Reports mark this capacity difference explicitly. Moodycamel also does not guarantee globally linearizable cross-producer FIFO order. Its benchmark uses explicit producer tokens; the reader scheduler uses queue-owned implicit producer state so caller tokens cannot outlive a pool. Queue-only speedups therefore must not be treated as scheduler or extraction speedups.
