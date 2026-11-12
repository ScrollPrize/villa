# Worker-local decoder scratch

Channel decoding reuses temporary compressed-range, decompressed-block and block-directory buffers through `scratch_pool.hpp`. Raw multi-channel reads and full-codec paths use the same scratch mechanism. Scratch never aliases a returned/cached channel buffer. Source ranges, interpolation arithmetic and worker scheduling are unchanged.

Blosc decoding uses reusable Blosc2 decompression contexts; Zstd uses reusable `ZSTD_DCtx` instances. Contexts are lazy resources of the same leased frames as byte scratch, so nesting gets independent codec state without a global codec mutex. Resources expire with the same idle policy and are destroyed outside the frame claim. The reader requires the `blosc2 >= 2.13` pkg-config package (Ubuntu: `libblosc2-dev`). Blosc1 remains the encoder, preserving the existing written format and encoding settings; Blosc2 reads those buffers. Delta3D decoding is unchanged.

For one-byte Blosc1 data with byte shuffle enabled, use a Blosc2 build containing the upstream identity-filter optimization. Older releases, including 2.13.1 and 3.4.1, can spend substantial time unshuffling bytes that do not need rearranging. The validated source revision is [`9168d0c0cc835c0a94b60b17bae53c4854e1b73f`](https://github.com/Blosc/c-blosc2/tree/9168d0c0cc835c0a94b60b17bae53c4854e1b73f). Supply its installed `blosc2.pc` through `PKG_CONFIG_PATH`; no source-code workaround or global library replacement is needed. Decoder contexts request one internal thread. Blosc2 honors `BLOSC_NTHREADS`, so leave it unset or set to 1 when parallelism should come only from the reader workers.

## Retained outputs and direct decoding

Decoded chunks use `Chunk`, an owned contiguous-byte buffer that can adopt a source vector's allocation without copying. Ordinary sized construction retains zero/fill initialization, including compute callbacks and partially assembled output spans. Validated full decodes instead request `Chunk::Overwrite`, which allocates uninitialized bytes. Every visible byte must be filled before cache publication; failed decodes never publish their output. Missing chunks still use the array's fill value. There are no payload pools or slabs.

Selected Blosc blocks are read in the same coalesced physical ranges. A reusable prefix before the current block constructs its single-block frame; subsequent prefixes overwrite only bytes of already-decoded physical blocks, eliminating the extra compressed-payload copy. Short final blocks explicitly preserve their unsplit encoding when expressed as individual frames. Blocks entirely inside a selected channel decode directly into that retained channel's destination. Blocks spanning channel boundaries still need bounded scratch and scatter; no scatter/gather destination API is assumed. Thus aligned blocks need neither zeroing nor a decoded-scratch-to-cache copy. Decoder-internal filter workspace may still be necessary, but reusable contexts retain it across calls.

Each executing thread owns a pool. A stack-scoped lease selects a frame by nesting depth, so a read callback may perform another decode without overwriting its parent's scratch. Frames remain stable when the frame list grows. Spans are uninitialized scratch: the decoder writes every byte before reading it. Spans may be invalidated when their slot grows and must not escape the lease. Leases must be destroyed in nesting order. This is a CPU allocation-lifetime optimization, not an alternate decoding algorithm or GPU kernel.

## Retention policy

- Capacity grows on demand, with up to 50% growth slack; no buffers are preallocated for unused threads or slots.
- Smaller requests do not immediately shrink capacity. If requests stay below half of capacity for ten seconds, the next request shrinks to the largest smaller request observed in that interval. There is no immediate "one-off request" guess.
- Buffers unused for two seconds are released, including an unused slot on a worker that continues using other slots. Task completion opportunistically expires unused slots.
- One process-wide scavenger checks inactive frames every 250 ms, including those belonging to live but idle caller threads. It skips active frames, detaches expired buffers under a brief claim and frees them after releasing that claim. Expiry is approximate, not a hard real-time deadline. Buffers in an active lease cannot be reclaimed.
- Thread exit releases its pool once any concurrent scavenger snapshot releases its reference. A child process creates a fresh scratch service/pool rather than acquiring inherited maintenance locks.

No worker consumes another worker's free-list. Reuse takes only a worker-local atomic frame claim; there is no global mutex or global accounting update per reuse. The registry mutex is used only for new worker/frame registration and maintenance snapshots. Global byte/allocation counters update only on growth, shrinking and release. A worker could briefly encounter the scavenger's claim on its own inactive frame; expensive freeing is outside that claim.

This first implementation has no hard process-wide byte ceiling for scratch: retained capacity follows recent per-worker demand and the expiry policy. Active work must still have enough memory for its requested buffers. Automatic cache budgets see scratch as other process memory; they do not count it as decoded-cache occupancy. Returning buffers to the C++ allocator does not promise an immediate RSS reduction, and the implementation does not force allocator-wide trimming.

The existing output-buffer pool is not used here: it serves externally owned output lifetimes with a shared exact-size free-list, whereas these buffers are private to executing workers and cannot escape a lease.

Scratch byte counters cover the reader-owned byte buffers, not private allocations inside codec contexts. Codec libraries can retain their own bounded per-thread workspace after a context is released; process RSS includes that memory. In particular, freeing a Blosc2 context need not immediately free workspace retained by its internal per-thread cache.

## Verification and measurement

```sh
cmake --build <reader-build> --target test_native_scratch_pool test_native_channel_decode test_native_channel_cache test_native_singleflight bench_scratch_reuse --parallel 8
ctest --test-dir <reader-build> -R '^test_native_(scratch_pool|channel_decode|channel_cache|singleflight)$' --output-on-failure
```

Tests cover repeated reuse, nested leases/decodes, sustained-small-demand shrinking with an injected clock, no trimming of active frames, automatic idle expiry on a live caller thread, unused-slot expiry while other slots stay busy, exceptions, concurrency and process recreation after fork. Test fixtures require no persistent data.

`bench_scratch_reuse USERS [seconds=0.5]` runs one warmup and three measured windows. Every lease requests four private buffers and touches their endpoints; it isolates reuse/coordination rather than decompression or memory bandwidth. JSON reports mean/median/min/max leases/s, peak retained bytes and growth allocations during measured windows. Run `1 8 32 64 192 384 1920` users on a 192-CPU host with an external 28-second timeout per invocation. High-throughput steady reuse should have no measured growth allocations. Thread creation/destruction and idle cleanup are outside the measured windows. Compare actual coordinate sampling separately.
