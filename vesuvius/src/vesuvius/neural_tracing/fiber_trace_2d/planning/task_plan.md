# Plan

1. Lock `best_ema_0p999.pt` by SHA-256 and verify the model output scale from
   its training configuration.
2. Derive a new immutable evaluation config from the last comparable full
   50-fiber no-embedding refinement evaluation. Preserve the six source
   collections, 50 fibers/90 eligible pieces/180 directions, 40/80 ellipse,
   160-base excursion grace, 80-base reference-distance restart grouping, and
   tracer parameters. Require `metric_definition=restart-episodes-80-base-voxels-v1`
   and exactly 6867.627253 directed reference mm.
3. Set inference scale to the model's native full-resolution output
   (`scaledown_power=0`), use a fresh cache namespace/output root, all eight
   managed GPUs, and rendering/embeddings disabled. Use batch cap 1 as an
   explicitly recorded VRAM/throughput-only deviation for the much larger
   full-resolution 256-cubed decoder output. Runtime tuning established batch
   cap 4 within the measured 62-GiB-per-GPU grant.
4. Freeze the exact checkpoint, patch the config to that immutable copy, then
   prepare/audit and run only through the existing pre-approved `freeze-snapshot`,
   `prepare`, and `full` runner modes. Validate checkpoint hash, completion,
   case count, directed reference length, combined metrics, and per-scroll metrics.
5. Persist the exact command, checkpoint/config identity, results, and any
   deviation in the task log and a benchmark report.

Outcome: completed in v22 after replacing the large POSIX shared-memory GPU
result ring with one fixed-offset regular-file mmap ring. All 180 directions
completed; see
`../docs/benchmarks/fullres_refinement_213753_best_ema999_20260917.md`.

## Full-resolution accumulator correction

The first managed launch exposed a real scale-1 blocker before any tile was
inferred: PHerc0139's float32 weight accumulator is about 27 TB, beyond ext4's
16 TB single-file limit. Preserve the existing demand accumulator rather than
introducing another implementation:

1. Extend the shared mmap descriptor with backward-compatible per-channel
   Z-shard metadata. Split only at storage-axis-0 boundaries, which are whole
   Z-brick planes for chunk-major layout.
2. Have `DemandStore` create sparse shard files below a conservative,
   test-injectable per-file ceiling and map shards lazily. Bound each worker's
   mmap cache with LRU eviction, and use a per-publication local cache so the
   full logical volume is never simultaneously mapped. Accumulation/finalization
   must route regions across shards while preserving the same per-voxel
   operations, rounding, and ordering.
3. Keep unsharded descriptors and the full-volume inference identity unchanged;
   shard files are temporary implementation details under `.live_tmp`.
4. Add exact flat/chunk-major cross-shard read/add tests, padded-edge coverage,
   overlapping f16/f32 accumulation, lazy/bounded mapping and cleanup coverage,
   descriptor validation, and end-to-end DemandStore publication equivalence.
   Run the existing approved replay-test runner before relaunching the exact
   prepared benchmark.

## Full-resolution host-output correction

1. Store only product tensors in per-result shared slots. Build one immutable
   shared blend-weight template per worker pool and have accumulator tasks read
   it without repeating a GPU-to-host transfer for every tile.
2. Copy weighted CUDA tensors directly into their shared CPU destinations,
   preserving float32 weighting and eliminating the temporary CPU tensor plus
   second host memcpy.
3. For aligned single-chunk writes, encode the padded C-order Zarr-v2 payload
   directly with the compressor recorded in `.zarray`, write a same-directory
   temporary file, and atomically replace the final chunk. Retain the general
   Zarr fallback for unsupported/multi-chunk writes. Do not fsync regenerable
   inference-cache chunks.
4. Use 64-cubed output chunks, 128 accumulator workers under a 192-logical-CPU
   exclusive grant, and 512 product-only result slots. These are throughput-only
   settings and must not enter numerical cache identity.

## Reusable hot/spill accumulator arena

The interrupted full-resolution run demonstrated that deleting globally
addressed, multi-terabyte sparse mmap shards can serialize ext4 reserved-extent
conversion for minutes with one kernel worker saturated and no useful device
I/O. Replace that temporary representation without changing inference,
accumulation arithmetic, finalization, or persistent Zarr cache identity.

1. Move physical demand accumulators from `DemandStore` to one pool-level arena
   shared by all compatible stores. `DemandStore` retains only scheduling and
   publication metadata. Key every live record by `(store/source, chunk_origin)`
   so equal coordinates from different sources cannot alias. Give the
   synchronous `DemandStore.complete()` adapter an explicit tiny arena using the
   same record/accessor implementation.
2. Represent every incomplete output chunk by one padded `chunk^3` local record:
   float32 weight plus every product's raw channels in the configured float16 or
   float32 accumulator dtype. Page/cache-line align records, report exact
   `record_bytes`, validate positive hot/segment slot counts, and address data by
   `(slot, local ZYX)` rather than global volume coordinates. A newly assigned
   record is completely zeroed before its first add; eviction and promotion copy
   the complete record including padded edge bytes.
3. Allocate a configurable fixed process-local anonymous-RAM hot-slot pool
   partitioned into owner-exclusive ranges among stable accumulator processes
   and shared logically across all stores. Only the owning process operates on
   its range; no other process ever needs to map hot data. Anonymous virtual
   capacity is reserved up front and physical pages are committed lazily. There
   is no application-level file-backed hot fallback. Each owner processes
   its FIFO queue serially, so record allocation, accumulation, eviction,
   promotion and completion cannot race. Spill contains only records first
   admitted to and later evicted from hot storage; at least one hot slot per
   owner guarantees progress because an idle incomplete victim can always be
   evicted before the next queued contribution. Require total hot slots at least
   equal owner count and partition quotient/remainder deterministically.
4. Use a separate fixed shared-memory publication-snapshot pool. Once all
   contributions acknowledge, enqueue completion on the same stable owner FIFO;
   the owner copies the complete hot or cold record into a generation-leased
   publication slot and immediately releases its accumulator slot. The publisher
   reads only that immutable snapshot and returns the lease after its atomic Zarr
   write. This removes movable-location publication pinning and cross-process
   spill-fd sharing; validate publication-slot generation on every ready/done
   event to reject ABA reuse. Never copy complete records on the coordinator pump
   thread. Accumulation/completion messages carry only `(store key, chunk origin)`
   plus local source/destination bounds, never a physical arena slot descriptor;
   the owner resolves locations at FIFO execution time.
5. Add an on-demand spill arena made of fixed-size views into one pool-wide
   regular-file mmap under the existing `/ephemeral` cache. It is never one
   file per output chunk and never globally volume-addressed. Eviction copies a
   full hot record into a spill slot; promotion copies it back and releases the
   cold slot. Publish a completed cold record directly without needless
   promotion. A central broker reuses the lowest free spill slot and extends the
   arena only when none is free. Thus capacity tracks the high-water number of concurrently
   cold incomplete records; because incidental partial chunks may remain live,
   that high-water can legitimately grow with explored work.
   Select victims by deterministic least-recently-used order among incomplete
   hot records, tie-broken by store key and chunk origin.
6. The pool creates one backing file directly on `/ephemeral` via `O_TMPFILE`
   where supported, or `mkstemp` followed by immediate unlink. Spawn-safe fd
   duplication gives owners access without reopening a pathname. A broker thread
   serializes cheap logical `ftruncate` growth once per segment; physical block
   allocation and writeback remain kernel-managed through the mmap and are never
   synchronously preallocated per record. Owners mmap fixed-size segment views.
   This prevents an eviction wave from making every owner create/grow an inode
   concurrently without serializing its data writes. Never
   use memfd/tmpfs for spill. Last-close reclamation is bounded by compact spill
   high-water rather than the volume lattice. Stop scanning/removing legacy
   `.live_tmp/session-*` during startup; legacy cleanup is an explicit offline
   maintenance action, not part of evaluation resume.
7. Extract shared local-record add/read/finalize accessors used by both demand
   arena accumulation and existing whole-volume helpers; do not fork numerical
   arithmetic or output finalization. Preserve stable owner FIFO arrival order,
   float16/float32 per-add rounding, atomic Zarr publication, late-request reuse,
   and one application of each contribution per session. Keep the persistent
   cache identity JSON byte-for-byte compatible: legacy layout/prefault identity
   fields remain accepted for existing prepared caches while no longer defining
   the internal demand scratch representation. Hot capacity, spill segment size
   and eviction policy are throughput-only settings outside cache identity.
8. Add telemetry for hot/spill occupancy and high-water, pinned/queued counts,
   promotions, evictions, segment/capacity growth, bytes copied, direct cold
   publication snapshots, publication-slot occupancy/waits, allocation/free
   balance and generation failures.
9. Validate exact f16/f32 scripted-arrival equivalence over multiple eviction/
   promotion cycles and padded edges. Cover global multi-store sharing/key
   isolation, zero-before-reuse, ABA rejection, queued-task movement exclusion,
   one-hot-slot eviction forward progress, direct cold publication snapshots,
   publication-slot ABA rejection, publisher failure/close ordering, spill
   free-list/high-water reuse across repeated publications, multiprocessing
   `spawn`, worker failure, no named scratch/startup traversal,
   exact cache-identity compatibility, telemetry invariants and synchronous
   completion. Then run the approved native/Python replay suites and resume the
   same prepared 50-fiber evaluation through the existing approved runner.

## Spec update

Replace the `specs.md` demand clauses that require full-logical-shape
`.live_tmp/session-*` mmap, chunk-major/flat scratch translation, prefaulting,
synchronous restart cleanup, and uncapped sparse-volume storage. Specify the
pool-wide chunk-local hot/spill arena, owner-FIFO state machine, publication
snapshot leases, eviction forward
progress, anonymous regular-file `/ephemeral` segments, explicit legacy cleanup,
slot reuse/high-water semantics and telemetry. Retain FIFO owner arrival order,
late-request partial reuse, exact precision/rounding, bounded GPU-result RAM,
RAM-only coordinator metadata, atomic persistent cache publication, cache
identity compatibility and all tracer/evaluation invariants.

## Docs updates

Document arena ownership, eviction/promotion/publication pinning, spill growth
and reuse, crash behavior, telemetry, and relevant tuning knobs. Add a benchmark
report for the full-resolution best EMA999 result and update the task log/status
and changelog after successful completion.

## Testing

The benchmark itself is the requested integration test. Use the existing
pre-approved command:

`run_classic_fiber_evaluation.py {freeze-snapshot,prepare,full} --config <config>`

Use the approved native replay-test runner for the accumulator correction.
Validate the produced report and all 180 case artifacts before reporting
success.
