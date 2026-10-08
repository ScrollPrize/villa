# Demand inference chunk accumulator arena — 2026-09-17

## Why it changed

The prior demand backend created sparse mmap arrays with the full logical scan
shape under each output Zarr's `.live_tmp/session-*`. A killed full-resolution
run left multi-terabyte sparse files. Resume synchronously removed them and
blocked in ext4 `wait_transaction_locked` while one `ext4-rsv-conversion` worker
consumed a CPU core. Device throughput was near zero: the work was serialized
extent-tree cleanup, not useful inference I/O.

## Current data path

`GPU result slot -> stable chunk-owner process -> chunk-local hot/spill record -> immutable publication snapshot -> atomic Zarr chunk`

GPU result transport is a separate reusable ring. Large result slots use fixed
offsets in one ordinary sparse file-backed mmap under the hidden accumulator
directory, mapped directly by both producer and consumer processes. This avoids
the host-global POSIX shared-memory ceiling while retaining direct process
sharing. It does not create per-result files, serialize arrays, copy through the
coordinator or issue fsync. Small input slots remain POSIX shared memory. The
pool unlinks the result-ring file only after child processes detach.

- One pool-level logical arena serves every compatible source. Logical keys are
  `(store key, output chunk origin)`; physical slots never appear in queue work.
- A record is one padded output chunk: float32 blend weight plus every raw
  product channel in the configured float16/float32 accumulator dtype.
- The fixed process-local anonymous-RAM hot tier is divided deterministically
  across accumulator owners. A hot record is used only by its owner process,
  so putting this storage in POSIX shared memory provided no sharing benefit
  and unnecessarily consumed the finite `/dev/shm` mount. Each owner
  exclusively operates its range in FIFO order.
- A new/reused record is fully zeroed. Deterministic LRU eviction copies the
  complete record into the pool-wide spill arena. Production process-local hot
  storage is a page-aligned private anonymous mmap: untouched pages are logical
  zero pages and consume no RAM. Released/evicted slots are reset on Linux with
  `MADV_DONTNEED`, restoring sparse zero semantics without a record-sized
  memset; other platforms retain the exact memset fallback.
- Spill consists of fixed-size mmap views into one anonymous regular file under
  the cache filesystem. It is not tmpfs/memfd and is not one file per chunk or
  owner. A central broker hands out the lowest reusable record slot and performs
  one cheap logical `ftruncate` when a new segment is required; the kernel lazily
  allocates physical blocks as mmap pages are dirtied. There is no synchronous
  `posix_fallocate`. The file grows only beyond the prior concurrent cold
  high-water mark, and owners never create or grow files during eviction.
- A completed hot or cold record is copied once to a generation-leased shared
  publication slot. The owner then returns its accumulator slot immediately.
  The publisher finalizes the immutable snapshot and atomically writes the Zarr
  siblings. A failed publication aborts the run and never marks readiness.
- New reusable result-slot pools use Linux `O_TMPFILE` backing and keep the
  parent-owned descriptor alive while spawned workers attach through its
  descriptor path. These files therefore have no directory entry during the
  run and cannot leak into an exported Zarr view. Filesystems without
  `O_TMPFILE` retain the named-file fallback until pool close.
- Legacy `.live_tmp/session-*` directories remain subject to an explicit,
  writer-locked cleanup scan at cache startup. The scan examines only known
  channel `.live_tmp` directories and direct `session-*` children; it does not
  follow symlinks or touch published Zarr data.

The arrival-order adds, float16 per-add rounding, float32 weights, finalizer,
output cache format, explicit zero chunks, and atomic sibling replacement are
unchanged. Legacy `accumulator_layout` and `accumulator_prefault` remain in cache
identity for prepared-cache compatibility but no longer select demand scratch.
Sequential full-volume inference retains its existing circular-Z mmap backend.

## Tuning and telemetry

- `inference.accumulator_hot_slots`: total hot records; must be at least the
  accumulator-worker count. It is divided quotient/remainder across owners.
  Capacity is reserved as anonymous virtual memory inside the owner processes;
  physical RAM is committed only as records become live. It is reported
  separately from transport `shared_slot_bytes` as
  `accumulator_hot_capacity_bytes`.
- `inference.accumulator_spill_segment_slots`: records per mapped view of the
  single spill file. Total growth is unbounded by the application; freed slots
  are reused globally before the broker extends the file.
- Publication snapshot count equals `flush_workers`.
- Result-slot count need not exceed `GPU count * max_batch_size`; additional
  slots increase mapped backing without increasing possible GPU concurrency.
- `result_slot_backing` records whether the result ring is POSIX shared memory
  or a regular-file mmap. `shared_slot_bytes` includes both input and result
  interprocess buffers even when their backing differs.
- Queue telemetry includes hot/spill occupancy and high-water, spill capacity,
  eviction/promotion counts, copied bytes, and pending publication snapshots.

These knobs affect memory, storage high-water, and throughput only. They do not
enter persistent inference identity or alter numerical results.

## Validation

Focused tests cover float16/float32 arrival-order output equivalence, late
requests, multi-source key isolation, hot eviction/promotion and lowest-slot
reuse, absence of named spill files, publication backpressure/failure, exact
cache identity, worker failure, and the synchronous adapter. The full approved
native/Python replay command remains the release gate before resuming the
50-fiber benchmark.

The completed full-resolution v22 run used 32 file-mapped result slots, 96
POSIX-shared input slots and 163,840 lazily committed hot records. It inferred
1,066 missing tiles, published 412 chunks, did not spill, and completed all 180
directed cases. See
`benchmarks/fullres_refinement_213753_best_ema999_20260917.md`.
