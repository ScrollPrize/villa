# Short failure benchmark plan

## Scroll-local parallelism correction

GPU lifetime clarification: benchmark-full passes existing DemandGpuLease settings
to the shared replay session instead of acquiring an outer whole-run grant.
Request on ready inference input, release after30seconds with no GPUactivity,
reacquire on further demand. Retain CPU queues/mmap buffers and continue tracing.
Reuse already-tested idle policy; no second manager implementation.

Stop128storage run, retain all caches. Sequential physical-scroll groups (both
Paris collections together) retain one inference session/GPUgrant,224logical CPU
budget per active scroll. First measured correction: let anchor and path fetch
services each admit up to the whole budget but share CPU permits with tracing;
path dependencies are resolved BEFORE acquiring compute permits, and prediction
readiness waits release them. This avoids source/stage CPU reservations without
putting blocking dependency tasks onto one saturated pool. Existing cache request
queues dispatch work; do not claim reuse of full-volume's finite-Z-slab dispatcher.
Replace64stripe generation locks with per-owner single-flight paired results;
remember empty owners so prefix/routes cannot regenerate them. Preserve exact
sampling/DP math and all publication/cache identities. Existing path sampler
deduplicates corners and batches spatially within chunks; repeated halo reads
between chunks are cache reuse, not proof of duplicate DP. Profile/measure before
further cross-chunk batching changes. Test same-owner prefix/route/empty/error,
independent-owner concurrency, budget/dependency safety and scroll grouping.
Build via approved workspace runner; rerun locked200benchmark and report actual
throughput, CPU use, construction work and quality. Warm cache comparisons must
be identified, not attributed entirely to scheduler changes.

## Classic then fiberlet comparison (next requested execution)

Run Classic on the immutable200-piece set, then u22000 fiberlet replay on the
same set. Reuse benchmark-full and historical model caches/bindings. Fiberlet
uses existing production pipeline, storage128basevoxels (earlier requested
smaller construction chunks), workers24/source, beam16/lookahead384/step48.
Only backend/fiberlet configuration is overridable; preserve source/model and
evaluation protocol. Existing historical source entries carry canonical fiberlet
identity/cache root; require these for graph execution. Graph storage changes
construction fingerprints, not inference cache identity. Run sequentially under
exclusive managed all8GPU/128CPU grants, without rendering. Verify200cases and
662.089848mm each; report restarts, shared metrics and credited traced coverage
for fiberlets (missing seeds must not look like successful tracing). Add binding
override tests, log full results and comparison. No new scientific specs needed;
document runner option and configs. Preserve existing extraction/history logs as
this is completion of the same targeted-benchmark task.

## Accepted benchmark evaluation

Extraction completed:200 pieces,662.089848mm,41best/50other/109Classic.
Evaluate all200 forward-only annotations using the same frozen best EMA999
u22000 snapshot, without embedding. Add a small approved-runner mode reusing
CachedReplaySession/extend_cached_replay and the historical source bindings;
verify manifest membership/hashes and checkpoint/cache identities. Preserve
the existing acceptance, grace and restart counting protocol. One exclusive
manager grant covers any missing inference, all8GPUs,128trace workers.
Do not re-extract or change the accepted dataset. Persist exact config/provenance,
walltime and shared complete_report overall/per-scroll metrics. Validate directed
coverage equals200 and source grouping, then run to completion. Document that
this selected failure set is not comparable directly with the original full set.

## Persistent queue (current user request)

Keep a spawned coordinator process per GPU lane. Assign snapshots to lanes with
cache-overlap components co-located, so retained writer locks cannot conflict.
Each lane caches up to two model sessions (LRU); a session owns shared source
readers, stores/accumulators and DemandWorkerPool across successive native replay
batches. Only one CUDAmodel perlane stays active (measured~50GiBreserved/batch8);
CPU/mmap sessions stay retained when switching models. Group available work by
resident model, without fixed batch-count rotation; switch when its queue drains.
Reuse the shared coordinate_native loop with an explicit borrowed
session; default single-run callers still clean up as before. A session opens
the model's source catalog once after the first actual miss, then reuses its
loaded model/workers without close/reopen at each batch. Cached-only models do
not initialize inference. Drain at batch boundaries; errors invalidate a session;
eviction/final shutdown closes pool before stores. One outer GPU grant continues.
Evict all overlapping retained cache owners before opening a session; co-location
alone does not release their writer locks. Bound retained sessions, preserve exact producer/cache identity, and report
session reuse/setup counts and per-batch inference deltas. Native replay process
remains per batch (shared native implementation), not a second tracing engine.
Test borrowed-pool reuse/cleanup, cache-affinity, model eviction, cancellation,
and repeated warm native replay. Profile bounded real workload, then continue
the extraction when validation passes; ten minutes is not a hard timeout.

## Parallel extraction correction

Stop current extraction; preserve all trial/state/cache evidence. Replace blocking
32-candidate windows and sequential model batches with a rolling asynchronous
model queue. Keep deterministic candidate-order selection and exact100voxel
search/confirmation. Refill available slots immediately; no slow-tail group barrier.
One outer GPU grant remains; divide its devices among concurrent model jobs,
never overlap cache writers (including different historical runs sharing caches).
Bound native worker count and decoded-cache budget across jobs, with additional
per-job caps for auxiliary loading/accumulation/publication pools. Warm geometry on
the coordinator only; worker threads only write distinct trial/batch artifacts.
Propagate cancellation through shared native/inference coordinator, drain jobs
before releasing the outer grant. Preserve implementation-only resume compatibility.
Test independent-job overlap, slot/cache exclusion, exception/cancellation cleanup,
deterministic selection and resume. Measure using existing trials/input, without
claiming a ten-minute runtime before observing it. Update docs/log/status.

Updated inference policy (user clarification): caches are an optimization, not an
eligibility constraint. Fill missing chunks with the original verified snapshot
through shared demand inference under one exclusive, priority-100 manager grant
for the entire extraction, using all configured GPUs. No per-batch grants. Retry old
uncached trials and rebuild winner decisions, retaining valid prior trial evidence.
Current output/config v3; v2 remains immutable history of cache-only pilot.

Self-proximity prefilter: native uniform-grid segment index on eligible GT spans,
query samples at<=4basevoxels, exact anisotropic point-to-segment minimum using
shared acceptance measurement/normal sampler. Ignore original-arc separations
under500basevoxels; compare across eligible pieces but never include excluded
Lasagna/extrapolated geometry. Exclude entire original fiber on any hit; record
original arcs/positions/ratio, spacing, minimum separation and timing. Sampling
is explicit (not a certified continuous curve/curve intersection test). Apply
before inventory/failure selection; new scientific protocol/output root.

1. Stop fiberlet evaluation, archive planning and preserve partial artifacts/caches.
2. Inventory current-protocol Classic/improved evaluations. Reuse reported restart episodes and source-oriented GT arcs. Deduplicate overlapping ±100 failure windows within original eligible pieces. Best priority, then other-model cases without Classic failure nearby, then Classic backfill.
3. Minimal opt-in single-direction annotation support in replay-set; default both directions unchanged. Extract shared cached replay coordination, never duplicate tracer or readiness logic. Add mode to approved runner.
4. Search both directions at100-base-voxel start spacing, increasing distance until target failure reproduces and also reproduces with start100 earlier. No monotonicity assumption/binary search. Preserve original model/normal/trace parameters, acceptance/grace. Match failure±100 original arc. End100 beyond target clipped to eligible bounds. Reject directions where extra100 start cannot fit. Checkpoint trials and batch using existing native parallel replay and PublishedStore; cache misses fail explicitly, never zeros.
5. Export200 unique validated directed fibers with manifest: source fiber/span, original arcs/coordinates, direction, model/settings, historical failure and both confirming trials. Backfill rejected cases; report genuine candidate shortage rather than duplicate. Validate exports and report per-source/category lengths. No training/original-data changes.

## Spec update

Add directed targeted benchmark contract to specs.md; keep regular evaluation defaults/metrics unchanged. Minimum is on100-voxel grid, not continuous. Targeted regression set is not unbiased full evaluation.

## Docs updates

Document runner/config, provenance, cache-only policy, search resolution, final counts/lengths. Update log/status/changelog. Record unfinished fiberlet run explicitly.

## Validation

Review safeguards: accept only first failure from fresh seed; independently audit saved matched points to require actual160-voxel outside travel (not terminal-only unresolved excursion). Search each direction independently before comparing. Other-only selection excludes raw Classic failure windows and requires completed Classic reference coverage. Reserve dedup locations after successful validation. Bind resumable trials to native binary, protocol and source/model hashes.

Independent review before implementation. Unit tests: arc slicing/orientation, dedup/quotas, target matching, nonmonotone search/resume. Native default-bidirectional and directed tests. Approved build/tests, cached pilot then complete search and export validation.
