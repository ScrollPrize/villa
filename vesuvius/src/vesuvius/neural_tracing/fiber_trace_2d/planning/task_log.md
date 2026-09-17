# Full-resolution EMA999 evaluation log

Task started 2026-09-17. Requested training run:
`5scroll_json_eligible_fullres_refinement_128_sd1_16x8_ema_20260916_213753`.

Resolved checkpoint:
`/ephemeral/hendrik/fiber3d_train2/runs/5scroll_json_eligible_fullres_refinement_128_sd1_16x8_ema_20260916_213753/snapshots/best_ema_0p999.pt`.
SHA-256: `5ec278f71b15b27187b55805310176fe2fa3b00c5264fc528b95bc6d0690206e`.

The training config on `fiber-tweaks` commit `39b7277cd` has
`model_3d.output_stride=1`; inference must therefore use
`scaledown_power=0`, not the prior refinement models' value 2. The evaluation
otherwise retains the established full-50 no-embedding protocol. The standard
50-fiber path is the runner's `freeze-snapshot`/`prepare`/`full` sequence (not
the separate extracted-manifest `benchmark-full` mode). Only that existing
pre-approved runner will be used.

Independent review approved the plan with explicit provenance gates. The frozen
snapshot confirms EMA999 best at update 17000 / 2,176,000 samples, selection
loss `0.19905350849894451`, `output_stride=1`, and the expected refinement
weights. Immutable snapshot path:
`/ephemeral/hendrik/fiber3d_train2/evaluations/fullres_refinement_213753_best_ema999_off_20260917_v1/frozen/live_inference/_models/5ec278f71b15b27187b55805310176fe2fa3b00c5264fc528b95bc6d0690206e/snapshot.pt`.

The config preserves workers32, native threads1, beam8/lookahead2/step4,
smoothness normal0.1/tangent10, embedding0, match refinement1, normal/tangential
ellipse40/80, grace160, and no rendering/embedding emission. The restart metric
must be `restart-episodes-80-base-voxels-v1` over exactly 50 annotations,
90 eligible pieces, 180 directed cases, and 6867.627253mm. Inference batch cap
is conservatively reduced from8 to1 because a 256-cubed stride1 decoder has
substantially larger activation/output residency; this is a throughput/VRAM
setting only and is recorded as a deviation from prior stride4 runs.

Preparation and validation succeeded. The prepared runtime config matches the
prior u22000 full evaluation after excluding checkpoint/cache/output and the
model-required scale/batch settings. Native reference planning confirms
50 fibers, 90 CP-bounded/no-Lasagna pieces, and 180 directions.

First managed full launch acquired all eight GPUs but failed before inference
while constructing PHerc0139's temporary weight accumulator: level-1 output
shape `[38477,13256,13256]` requires about 27 TB at float32, exceeding ext4's
16 TB single-file maximum (`OSError: [Errno 27] File too large`). Process file
size is unlimited; this is not quota or eager disk allocation. No benchmark
tiles were consumed. Continue by transparently Z-sharding the existing sparse
temporary mmap without changing numerical operations or persistent inference
identity, test it, then rerun the identical prepared workload.

Independent correction review approved Z-sharding as the smallest correct fix
and required strict descriptor validation, whole-plane partitions, bounded lazy
mapping, numerical-equivalence tests, and cleanup/security coverage. The design
keeps each 32-cubed brick contiguous, uses channel-major ordered shard paths
with a common complete axis-0 partition per descriptor, bounded lazy maps, and
per-publication local maps. Product and weight descriptors may have different
partitions because their dtypes differ. Sharding remains temporary and absent
from cache identity.

Exact sharding validation passes for flat and chunk-major layouts, float16 and
float32 accumulators, overlapping additions, circular-Z wrap, terminal padding,
descriptor rejection, bounded-LRU eviction/remapping, lazy sparse creation,
publication, and failure cleanup. The complete approved regression runner
passes native suites 23/85/17 and Python 280 passed / 8 skipped.

The first sharded launch crossed the former `EFBIG` point but exposed aggregate
virtual-address pressure: 128 publication threads could each map multi-terabyte
8-TiB shards. It stopped after 142 inferred tiles / 18 published chunks with
`ENOMEM`. The corrected default is 64 GiB per temporary shard (still split only
between complete storage-axis-0 planes), and publication retains only one map
at a time. Targeted regression: 12 passed.

The next clean-root run proved that mapping correction stable, but the native
stride-1 output makes each 256-cubed tile fan out over as many as 512 32-cubed
accumulation tasks. The default of one accumulator process per GPU meant only
eight CPU consumers; after four minutes it remained result-slot bound at about
0.2 completed tiles/s, a roughly 20-hour projection. This throughput-only run
was stopped with its published cache chunks retained. The audited retry adds
`accumulator_workers=64`, matching the granted CPU budget and preserving
per-chunk stable ownership and contribution order. Early sustained throughput
is about 3--4 completed tiles/s (roughly 60--80 minutes projected) and the full
run is active in the clean `...20260917_v3` evaluation root while reusing the
same persistent inference-cache identity.

Host measurements disproved storage as the remaining cause: CPU was 91--99%
idle, I/O wait and I/O PSI were zero, while 124--125 of 128 result slots stayed
in accumulation. The stride-1/32-cubed pairing generated up to 512 separately
queued multiprocessing tasks per tile, making the serial Python coordinator
and per-task IPC the bottleneck. Accumulator handoff now groups all same-tile,
same-owner contributions into one process message/acknowledgement while keeping
the identical per-chunk owner and contribution order. All 71 DemandStore tests
and the full approved 23/85/17 + 280-pass replay suite pass.

The 128-cubed trial reduced accumulation fan-out but expanded the requested
inference footprint from roughly 14.9k to 27.1k tiles. It also exposed the true
post-forward path: each stride-1 tile synchronously copied seven float32 output
channels plus a repeated blend-weight volume through a temporary CPU tensor,
about 512 MiB per tile, before mmap accumulation. GPU samples were 0--3% while
the host spent 50--60% in system time. The trial stopped at 7,527 tiles / 1,884
published chunks; its distinct 128-cubed cache remains intact.

The corrected shared-result contract writes each weighted product tensor
directly from CUDA into its shared NumPy allocation, removing the temporary CPU
tensor and second host copy. Blend weights are deterministic, so DemandWorkerPool
now creates one immutable 64 MiB shared template consumed by every accumulator
instead of transferring/storing it in every result slot. Product arithmetic and
per-chunk ownership/order are unchanged. The persistent output path now directly
encodes an aligned Zarr-v2 chunk with the recorded numcodecs compressor, writes a
same-directory temporary file, and atomically replaces the final chunk. It no
longer creates a temporary Zarr array/copies `.zarray`, and the regenerable cache
does not issue file/directory fsyncs. Boundary chunks remain full padded codec
chunks. Validation: 73 DemandStore tests; approved native 23/85/17 and Python
282 passed / 8 skipped.

At user direction the benchmark now uses 64-cubed output chunks as the balance
between demand over-inference and chunk I/O. Blosc-Zstd level 3 with byte shuffle
remains the persisted compressor; publication queues repeatedly drain to zero,
so compression is not the sustained limiter. The 64-cubed trial reduced the
initial footprint to about 18.7k tiles. With 64 accumulator workers, throughput
was about 17 tiles/s but result slots saturated and GPUs starved. A 192-logical-
CPU exclusive grant plus 128 accumulator workers raised it to roughly 22--25
tiles/s. The final active configuration uses 512 product-only result slots
(about 224 GiB, 232 GiB total shared buffers) so up to 256 GPU-queued tiles can
overlap roughly 256 accumulating tiles. Early telemetry keeps about 288 jobs
submitted, accumulation mostly below 20, and sampled aggregate GPU utilization
near 65%. Published 64-cubed chunks are reused across these runtime-only trials.

The interrupted run left globally addressed sparse accumulator shards under
`.live_tmp`. On resume, `DemandStore.__init__` synchronously validated and
removed every stale session before creating the new one. The evaluator blocked
in kernel `wait_transaction_locked` while a single md0 ext4 worker consumed a
core performing reserved-extent conversion/cleanup, despite negligible useful
device I/O. `/dev/md0` is clean, mounted read-write, and has free space; the
failure is pathological extent lifecycle complexity rather than evidence of a
damaged filesystem. Moving scratch to another filesystem was explicitly
rejected.

The replacement design approved by the user is a fixed, large shared-memory hot
pool plus an unbounded-on-demand file-backed mmap spill arena on `/ephemeral`.
Both tiers contain fixed-size chunk-local accumulator records, not globally
addressed volume arrays. Cold incomplete chunks map to reusable spill slot IDs;
promotion/publication returns slots to a lowest-first free list, and new arena
segments are allocated only beyond the peak simultaneous cold working set. A
pure bounded pool was rejected because arbitrary demand order can leave an
unbounded number of partially accumulated chunks; the spill tier therefore has
no hard capacity. Per-chunk backing files are explicitly disallowed.

Independent review required and the plan now specifies one pool-level arena
across sources, generation-checked leases, owner-queue movement serialization,
zero-before-reuse, direct cold publication, deterministic LRU eviction, explicit
spawn-safe backing-fd transfer, cache-identity compatibility, and comprehensive
lifecycle/equivalence tests. The reviewed correction keeps the user's stricter
contract that spill contains only previously hot records: if all hot records are
transiently busy/pinned, completed GPU output remains leased in a
`waiting_for_hot` state while releases continue to be pumped. No new logical
chunk is initialized directly in spill.

The first full 50-fiber arena run exposed a flaw in the initial physical spill
implementation. At the first broad eviction wave, 128 owners independently
created and grew 128 anonymous 1.2-GiB logical segment files. All owners entered
ext4 `wait_transaction_locked`, GPU utilization fell to zero, and the run was
stopped. The corrected implementation uses one pool-wide anonymous regular file
and a central broker. The broker allocates physical extents serially per record,
globally reuses the lowest released slot, and grows only at the simultaneous
cold-record high-water. Workers receive a spawn-safe duplicate fd and mmap only
already prepared large segment views; eviction performs no inode creation or
file growth. This more directly implements the requested single large,
re-indexed mmap arena and eliminates the observed transaction storm.

Filesystem-cleanup monitoring baseline (UTC): at `2026-09-17T13:21:29Z`, md0
reported `delayed_allocation_blocks=272626`, `reserved_clusters=4096`,
`session_write_kbytes=193478374660`, `nr_dirty=283670`, and `nr_writeback=0`.
At `2026-09-17T13:22:14Z` the ext4 counters were unchanged and
`nr_dirty=283635`; `kworker/u449:2+flush-9:0` was still running at about 72% of
one CPU. The earlier untimestamped sample had 15,420,146 delayed-allocation
blocks, so most work drained before the timestamped baseline, followed by a
plateau at 272,626 blocks.

Further progress audit from `2026-09-17T13:25:42Z` through
`2026-09-17T13:26:13Z` found zero changes in delayed-allocation blocks,
session-write KiB, jbd2 transaction count, md0 read/write sectors or md0 I/O
milliseconds; dirty memory remained about 1.08 GiB and I/O PSI remained zero.
A publication-shaped canary on the actual `/ephemeral` cache filesystem
(create, 1-MiB write, close, atomic rename, stat, unlink; no fsync) was still
blocked at `2026-09-17T13:28:53Z`. It had not obtained a file descriptor and its
wait channel was `wait_transaction_locked`, proving that new file creation was
still blocked despite the lower delayed-allocation count. The canary was sent
SIGKILL and will exit when the uninterruptible kernel call returns.

The md0 journal recovered without a reboot by `2026-09-17T13:51:00Z`:
`nr_dirty=166`, `nr_writeback=0`, `delayed_allocation_blocks=1`, the journal
thread was sleeping normally, and ext4 still reported zero errors. A real
`/ephemeral` create/write/file-fsync/readback/unlink canary then completed
inside its 15-second bound. The first post-recovery replay run exposed one
arena startup defect: the one-second child-attachment status poll leaked
`queue.Empty` instead of continuing to the actual 60-second deadline. The poll
now retries while checking child exit status. Targeted validation passes 75
tests with 28 skipped; the complete approved replay suite passes native
23/85/17 and Python 282 tests with 8 skipped.

The first resumed v14 full run reused the intended persistent cache and reached
1,743 inferred tiles / 1,036 published chunks, but repeatedly starved all GPUs
with all 128 result slots waiting on accumulation. The central broker was still
calling synchronous `posix_fallocate` once per newly occupied cold record; a
128-record eviction wave therefore serialized tens of seconds of filesystem
allocation. The run was stopped with published cache chunks retained. Per user
direction, spill now uses only once-per-segment logical `ftruncate`; physical
allocation and writeback are left to ordinary file-backed mmap page faults.
The single arena file, global lowest-slot reuse, high-water growth, and failure
propagation contracts are unchanged.

The resulting v15 diagnostic made the workload size visible: the run held about
4,096 hot plus 17,000 cold records concurrently, while md0 accumulated about
67 GiB of dirty/delayed-allocation pages. This was genuine incidental partial-
chunk state, not merely owner imbalance; record-slot high-water reached about
20,000. The run was stopped at 654 inferred tiles / 304 newly published chunks,
and close reclaimed the anonymous spill file, returning dirty/delayed state to
roughly 36/33 MiB within seconds. With 2.0 TiB total RAM and about 1.1 TiB
available, v16 raises the fixed hot tier from 4,096 to 32,768 records (about
144 GiB total hot storage for 4.5-MiB records). The unbounded mmap tier remains
the correctness fallback.

The v16 run disproved that initial extrapolation. At 1,545 inferred tiles / 906
published chunks it held all 32,768 hot records plus about 19,600 cold records,
and the spill slot high-water was still rising past 23,500. Once hot filled,
128-result-slot waves again stalled while thousands of independent 4.5-MiB
records were copied through sparse-file faults and broker round trips. The run
was stopped before this became another large dirty-file workload. This is a
real broad partial-chunk working set, not a raw memcpy benchmark: roughly 100
GiB had already crossed the eviction path in about one minute.

Hot records are private to exactly one stable accumulator-owner process. They
therefore no longer use POSIX shared memory: each owner reserves process-local
anonymous RAM and only touched live records consume physical pages. This
removes the `/dev/shm` ceiling and an unnecessary cross-process backing layer.
Transport result/input/publication slots remain shared. The v17 full run uses
131,072 hot records (576 GiB virtual capacity, lazily committed) so the measured
working set can remain resident; the single unbounded file-backed arena remains
only as the correctness fallback.

The first v17 attempt exposed that merely reserving anonymous hot storage was
not enough: `mutable_record(create=True)` still called `fill(0)` over every new
4.5-MiB record. At 56k live records this had first-touched about 250 GiB even
though many incidental chunks contained only small contributed subregions.
Host telemetry showed only 2--3% user CPU but 56--58% system CPU, memory PSI
`full` near 87%, about 18 GiB immediately free, and roughly 1% I/O wait. The
stall was tens of millions of 4-KiB allocation/zero/reclaim faults, not DRAM
memcpy, GPU compute, or filesystem I/O. v17 was stopped at 1,563 tiles / 970
published chunks, all reusable from the persistent cache.

Hot storage is now a page-aligned private anonymous mmap. Its initial pages are
kernel-provided logical zeros, so first use skips full-record clearing; only
actually accumulated subregions fault physical RAM. Linux resets released or
evicted slots with `MADV_DONTNEED`, returning pages and restoring zero semantics;
the portable fallback still explicitly clears. An isolated 4.5-GiB logical
arena with 100 records touching 64 KiB each grew RSS by only 7 MiB rather than
450 MiB. Focused eviction/spill/zero-reuse tests pass (3 tests).

The next full-run attempts exposed two independent capacity effects. First, a
131,072-record (576-GiB virtual) hot arena filled at about 150,000 live records;
roughly 19,000 records entered spill, and the resulting full-record copies made
GPU execution bursty. The final configuration therefore uses 163,840 lazily
committed hot records (720-GiB virtual capacity), based on the measured live set.
Second, all eight GPU workers could SIGBUS together inside
`_copy_tensor_to_shared_array`: the 128 large output slots were POSIX
shared-memory objects while the host already retained about 987 GiB of shared
memory from a paused training process. Reducing the ring did not make that
host-global limit reliable.

GPU results now use 32 reusable fixed offsets in one ordinary sparse file-backed
mmap under the hidden accumulator directory. Eight GPUs times max batch four
means this retains all useful output concurrency. GPU workers and accumulator
workers map the same bytes directly; there are no staged files, serialization,
per-tile files, explicit fsyncs or coordinator copies. Input transport remains
the much smaller 96-slot POSIX shared-memory ring. A spawn regression verifies
independent and child-process mappings plus unlink-on-close.

The v22 resumed run completed successfully. It inferred 1,066 missing tiles in
382 batches, published 412 new chunks, replayed all 180 directions, and produced
the exact prepared 6,867.627253 directed reference mm. Incremental inference
took 143.056 s and native replay took 106.502 s. The combined result is 127
restart episodes, 0.018493 restarts/mm, 53.653-mm mean, 8.922-mm median,
0.241-mm P05 and 88.464-mm P95. This is a resumed-cache integration result, not
a clean cold-cache throughput number. Full per-scroll metrics and reproduction
are in `../docs/benchmarks/fullres_refinement_213753_best_ema999_20260917.md`.

Release validation after the result-ring change: native 23/85/17 passed; Python
283 passed with 8 skipped; focused file-mapped/arena selection 4 passed with 70
deselected; `py_compile` and `git diff --check` passed. A separate broad
multiprocessing-backpressure selection blocked during setup on the loaded host
and was interrupted; it is not reported as passed.
