# Parallel native reference evaluation with live inference

First standalone implementation, 2026-09-10. **Offline evaluation and on-demand inference are connected; synthetic cold/warm tests and a real classic-checkpoint PHerc0139 pilot pass.** Source bindings are configured; short eight-GPU inference profiles are recorded in the [chunk-major report](live_inference_chunk_major_20260910.md) and linked prior reports. The full evaluation has not been restarted during these performance experiments. See [classic evaluation report](classic_fiber_evaluation_20260910.md). Training and VC3D live-view integration remain later work. Current task records/status are in ../planning/.

## Current live implementation and command

Dynamic model batches are available through `inference.max_batch_size` (positive integer, default1). This is the number of inference tiles **per GPU model call**, not a queue depth. A worker takes whatever descriptors are currently ready, up to that cap, without waiting for a full batch. Per-tile CT normalization and result/accumulator ownership remain independent. Keep enough input/result slots for the desired concurrency; a cap does not guarantee that many ready tiles or enough GPU memory. The cap is included in cache provenance. Summaries report actual batch histograms, amortized tile timings and peak allocated/reserved GPU memory. See [the dynamic batching benchmark](dynamic_inference_batching_20260911.md) for256³ measurements and practical limits.

The classic50-fiber workload explicitly uses256³ tiles and dynamic cap8 after a
complete100-direction validation on8H100 GPUs. Other models retain generic cap1.
Pending work uses exact-order priority heaps (foreground, remaining length, then
source/tile ties); no scheduling-policy change. The classic request timeout is
3600s because longest-first scheduling can legitimately leave a short trace behind
more than600s of healthy queued work. See [full-set measurements](full_dynamic_inference_batching_20260911.md).

Use the existing environment directly; activation is unnecessary:

```sh
PYTHONPATH=/home/hendrik/villa2/vesuvius/src:/home/hendrik/villa2 \
  /home/hendrik/.venv_las/bin/python \
  -m vesuvius.neural_tracing.fiber_trace_3d.live_evaluate \
  --config /absolute/path/evaluation.json \
  --checkpoint /absolute/path/completed-snapshot.pt \
  --devices 0,1,2,3,4,5,6,7 \
  --native-binary /absolute/path/vc_fiber_trace_metric
```

Add `--render` to this command to generate reference/traced top/side strips as each case completes. It uses the completed in-memory replay directly, not a reload of saved traces. Images land in `cases/<index>/rendered/strips_*.jpg`; `render.json` describes the panels. The default is off; top-level config `"render": true` is equivalent. Each source needs a valid local concrete uint8 OME CT `ct_volume` with parent multiscale metadata when rendering is enabled. All CT bindings are checked before tracing starts. Strip extraction is serialized per source to bound texture work; it does not alter prediction/search decisions. Render failure makes the batch incomplete.

Paths/checkpoint/devices above are placeholders, not a selected real workload. `vc_cache_root` must be beside the native executable. Production requires explicit CUDA indices; CPU identity models are injected only in smoke tests, never for training. `--model-config` supports legacy checkpoints lacking embedded config.

Use the offline example below, but remove `fiber_manifest` and per-source `inference_scaledown_power`; add each source's `input_volume` (OME **group root**, not a bare `/0` array), `input_level` (default0), and `source_revision` if the locator can change content. Add this top-level section:

```json
"inference": {
  "tile_size": 192,
  "overlap": 64,
  "border": 16,
  "scaledown_power": 2,
  "chunk_size": 32,
  "precision": "auto",
  "accumulator_dtype": "float16",
  "compressor": "blosc-zstd",
  "slots_per_gpu": 2,
  "input_workers": 8
}
```

Set top-level `workers` for active CPU cases (e.g.16) and include `--threads 1` in `trace_args`. Tile/stride/border must align to output scale, which must be divisible by model native stride. Chunk size is in **output voxels**. Initial input support is explicit OME ZYX, isotropic power-of-two scales, zero translation. Unsupported affine/anisotropic metadata is rejected. Bind correct sheet normals explicitly; matching shapes alone cannot prove source identity. Real checkpoint/devices and six source/normal bindings remain to be chosen before running the downloaded cohort.

### Implemented architecture

File-backed accumulator scratch has no application size cap. Sparse files grow
their physical footprint as pages are written; there is no per-tile byte-budget
accounting or free-space polling. Input/result shared RAM pools remain bounded.
Actual filesystem errors still fail the run; mmap write faults on exhausted
storage may terminate a worker with SIGBUS, which is treated as failure, not
successful chunk publication. Scratch is removed on normal error cleanup; a
process crash may leave owned temporary scratch for existing restart cleanup.

- `FiberTraceEvaluationCli.cpp` extends the existing native wrapper; `FiberTrace.cpp` remains the same VC3D greedy/beam search. Whole fibers run in both directions. Existing ellipsoid failure/reset policy and forward matching are preserved. Beam checks only irrevocable prefixes and selected terminal paths, never reference-guided speculative pruning.
- `FiberLiveBridge.{hpp,cpp}` and `FiberPredictionField` gate scalar, batch, optimized corner and prefetch reads before cache access. The exact float visitor and double sampler corner sets are covered; unavailable data never becomes fill-zero or a geometric failure. Live stored-grid scans are rejected.
- Version1 localhost TCP exchanges authenticated `hello`, batched `demand` descriptors (source/case/chunk/remaining length), `ready`, and cancellation `error`. No image tensors cross it. Tokens are redacted in saved config; native ready knowledge is bounded to100000 entries.
- **Reviewed deviation:** one stack-preserving OS thread per directed case, default maximum256, with bounded active CPU permits. Waiting releases its permit without holding sampler locks. This preserves beam state but is not a stackless session engine, persistent trace recovery, or a solution for thousands of case threads.
- `live_evaluate.py` freezes a checkpoint by SHA256/copy, validates bindings, creates the source/model cache, starts native replay and persistent spawned GPU workers lazily, prioritizes demand by longest remaining length, then drains reference-intersection prefetch. Reference prefetch starts after authenticated source validation even if tracing produces no demand. No whole-array scan.
- `lasagna/demand_predict3d.py` reuses shared tile geometry, GPU worker, raw input conversion, blending/filtering, per-add accumulation and output flush/finalizer from `tiled_predict3d.py`. Native input sampling is the same selected-level primitive as fast training. All model options/products publish coherent ordinary Zarr channels.
- Full logical output-shape sparse mmap holds raw products and weights inside hidden output-Zarr directories. Fresh holes already read as zero: no zero reservation, scratch fsync or NPZ staging. Shared-RAM results go directly to the existing accumulator processes. Demand uses arrival-order additions (same f16/f32 precision, not guaranteed bitwise equality across schedules). There is no application scratch cap or per-tile storage polling; actual disk exhaustion can still kill an accumulator and abort the run.
- Optional `inference.accumulator_layout: "chunk-major"` stores a flat grid of contiguous output-chunk bricks within each existing per-channel mmap. Boundary bricks are padded, without scratch byte accounting; logical output shape and Zarr encoding do not change. Both additions and finalizer reads use shared addressing helpers. Default remains `flat`; sequential circular-Z storage is unaffected. See the [layout benchmark](live_inference_chunk_major_20260910.md).
- Optional `inference.accumulator_prefault: true` requires chunk-major and Linux writable-prefault support. The owner advises a full brick once per channel mapping immediately before its first addition. This can allocate sparse backing, but adds no Python zero writes or scratch sync. Cost is inside task timings; failures are explicit, never silently replaced by read-prefetch. Default is false. Both options participate in identity and have isolated `profile` overrides in the classic runner config.
- A tile feeds all touched chunks, including unrequested partial neighbors, **without scheduling their missing tiles**. Late demand reuses that partial accumulation. Shared result slots release only after all touched consumers apply them. Coordinator-owned SQLite WAL tables track identity, chunks, tiles and scratch under an exclusive writer lock; pending/applied contribution edges stay in RAM.
- Ready means all contributors applied, every sibling physically published (including all-zero chunks), integrity checked/fsynced, then ledger committed. Restart validates ready file size/SHA256, discards unfinished scratch/edges and rebuilds incomplete chunks. An active ready chunk must never be edited/evicted externally.
- Cache identity covers unsigned source locator/revision/metadata/lattice, frozen checkpoint, effective model/inference parameters, runtime versions and relevant adapter/model/network/normal-encoding implementation hashes. Mutable source data needs a new revision; source voxels are not globally hashed.

### Outputs and first-version limits

`inference_summary.json` records coordinator completion, wall time, worker/load/tile counters, slot bytes, cache roots and prefetch counts; `inference_incomplete.json` records failure. `native.log` preserves subprocess output. **Native `summary.json` alone does not prove live completion:** require inference summary and absence of incomplete markers too, since background inference can fail after tracing finishes. All per-case geometry/strip outputs below remain available.

Only the inference level is generated; no coarser pyramid production. Metadata explicitly labels partial coverage. Ordinary Zarr viewers still see uncomputed chunks as fill and need future live-readiness integration.

Other limitations/deferred hardening: active shared RAM slots are bounded, but file-backed scratch and queued descriptors have no application size cap; priorities are monotone promotions rather than exact aging/decrementing waiter priorities. Accumulation uses chunk-owner processes and publication runs on bounded background threads; each GPU call still handles one tile. Short pipeline profiles measure throughput/backlogs, not full-cohort sustained saturation. Input reads are threaded and shutdown can wait for a stuck native/remote read; **the user explicitly deferred this edge case**, and it is not being worked on now. No stuck read has been observed in passing tests. Broader crash/worker-death/ENOSPC stress remains future work. No persistent trace resume, training hook or GUI live integration yet. Ubuntu/macOS source paths are implemented, but only Ubuntu amd64 tests have run; the optional writable-prefault experiment is Linux-only. The requested batch visualization is the simple `--render` flag, not a separate selection framework.

Validation covers exact offline/live beam/greedy geometry, all cases waiting with one CPU permit, cold/warm spawned identity-model runs, scale1/2 odd-shape arrival-order oracle checks in f16/f32, reverse completion, late demand, physical zeros, identity/writer/corruption checks, uncapped sparse scratch planning and storage-error propagation. Flat/chunk-major addressing and optional prefault have regression coverage. Timing p50/p95 and OS counters are in the short profile reports. No full real-model output parity, peak-memory survey or full-cohort over-inference measurement is claimed.

## Available offline commands

After building `vc_fiber_trace_metric` in your configured native build:

```sh
vc_fiber_trace_metric replay-set --config evaluation.json
vc_fiber_trace_metric replay-set --config evaluation.json --render
vc_fiber_trace_metric render-replay evaluation/cases/0/trace.json --config render.json
```

Minimal `evaluation.json` (paths are relative to this file):

```json
{
  "version": 1,
  "output": "evaluation",
  "workers": 8,
  "error_threshold_base_voxels": 20,
  "match_refine_steps": 1,
  "trace_args": ["--beam-width", "8", "--beam-lookahead-steps", "2"],
  "sources": [{
    "id": "collection-id",
    "coordinate_space": "scroll/volume@L0",
    "fiber_manifest": "predictions/inference.lasagna.json",
    "normal_manifest": "normals/dataset.lasagna.json",
    "ct_volume": "ct.zarr/0",
    "base_voxel_size_um": 2.4,
    "inference_scaledown_power": 2,
    "fibers": ["test/fiber.json"]
  }]
}
```

Use the actual validated source/normal binding and the same inference scale used to produce the predictions; these strings are placeholders, not a mapping for the downloaded dataset. `workers: 0` uses hardware concurrency capped by case count. Each selected whole fiber creates two directed cases; no cross-fiber branch following. Outputs must not exist. `summary.json` contains per-source metrics and physical aggregate counts; `cases/<index>/` contains `trace.json`, `trace.obj`, and `reference.obj`. Reset spans stay disconnected. `incomplete.json` reports worker failure without claiming completed evaluation.

`render.json`:

```json
{
  "normal_manifest": "normals/dataset.lasagna.json",
  "normal_working_to_base": 1,
  "ct_volume": "ct.zarr/0",
  "threads": 1,
  "output": "rendered-case-0"
}
```

For the optional post-hoc command, normal scale must equal saved `provenance.trace_to_base`, not an assumed one. Normals and CT must match the evaluated source binding. CT needs a **local concrete uint8 OME-Zarr array with group multiscale metadata**. It produces the same paginated reference-top/reference-side/traced-top/traced-side JPG strips as inline rendering; predictions/model are unnecessary. Batch `--render` automatically uses the correct source scale and needs no extra render config or artifact reload.

Native builds should use the project's existing configured CMake tree:

```sh
cmake --build /path/to/configured-build --target vc_fiber_trace_metric vc_cache_root test_fiber_trace3d test_global_cache_root -j 8
bash vesuvius/scripts/run_fiber3d_native_replay_tests /path/to/configured-build /home/hendrik/.venv_las/bin/python
```

The smoke wrapper locates the four executables in the build tree. HiGHS is now optional, ported from `/home/hendrik/villa`: absent/disabled HiGHS disables only LP/MILP labeling with an explicit error if requested, not beam tracing. `VC_ENABLE_HIGHS` defaults ON (auto-detect); OFF explicitly disables it. The native evaluator now builds under the headless `VC_BUILD_FIBERLETS=ON` path too. The current-source validation build is `/tmp/vc-live-native-cmake` (GCC13, RelWithDebInfo), replacing the earlier hand-linked smoke build. Exact configure/build commands and test results are recorded in task_log.md; no dependency installation performed.

Shared cache root is now configured with `vc_cache_root --set /absolute/root`, and `vc_cache_root` prints it. This writes `cache.json` under `VC3D_CONFIG_DIR` or `~/.VC3D`, backing up a previous setting. Explicit root wins over automatic `/volpkgs`/`/ephemeral` fallback. This host was set to `/ephemeral/hendrik/vc3d_cache`; old caches remain untouched. GUI and Python source use the same resolver but installed binaries require rebuilding.

## Original architecture plan (historical design, not a completion checklist)

The remaining sections preserve the reviewed design and pre-implementation findings. **Current behavior, deliberate deviations and unsupported items are described above.** In particular the planned stackless bindings, coarser pyramids, priority aging and detailed profiling are not implemented in this first version.

### Existing implementation to reuse

| Area | Existing implementation | Required extension |
|---|---|---|
| Native search/sampling | volume-cartographer/core/{include/vc,src}/fiber_tracer/FiberTrace.{hpp,cpp} | Resumable session; preserve greedy/beam search, stable candidate ties and optimized corner sampling |
| Reference metric | FiberReplayMetric, PolylineGeometry, FiberReferenceReplayBenchmark | Reuse both directions, forward matching, failures, units and aggregation |
| CLI | apps/src/vc_fiber_trace_metric.cpp; vc_fiber_trace_chunk reference-replay-benchmark | Multi-source manifest and common live/offline evaluator; no copied tracer |
| Python binding | volume-cartographer/python/vc/volume.cpp and shared extension structure | Batched native session/events, GIL-free search |
| Model/output semantics | fiber_trace_3d/infer.py, inference_adapter.py, prediction.py; lasagna/normal_encoding.py | Same checkpoint model/preprocessing, all options, raw accumulation and compact final outputs |
| Tiled inference | lasagna/tiled_predict3d.py | Demand scheduler and random-access scratch backend; retain shared GPU/accumulator/flush implementations |
| Fast input loading | fiber_trace_3d/sample_pipeline.py, loader.py; ordered_sample_pipeline.py; native VC sampling | Dynamic inference descriptors using shared worker/slot/cache primitives, no training augmentations/targets |
| Cache root | apps/VC3D/VCSettings.hpp, RemoteVolumeCachePaths.cpp | Shared non-GUI resolver with explicit configured override |

The optimized scorer currently recognizes concrete FiberPredictionField with a dynamic_cast. Merely wrapping it in an arbitrary FiberPredictionSource could lose the fast corner path. Put readiness/dependency support below or within the shared field/corner preparation interface; preserve the optimized visitor for offline and live sources.

## Evaluation semantics

- Read the locked test partition of split_95_5_length_seed20260910: 50 fibers, normally 100 directed cases. Treat Paris4 source collections separately in reports even where they share volume/model cache. Do not follow branches across the split.
- Validate each collection's source volume locator, coordinate transform, input scale and physical spacing. Labels alone are not reliable volume identities: PHerc1229 files were observed with PHerc1299 metadata during split discovery. Surface conflicts before running; never guess a substitution.
- Bind real Lasagna sheet-normal manifests separately. Fiber nx/ny is the fiber direction and must not substitute for sheet normals used in the ellipsoid or normal-aware smoothness. Validate transforms and coverage. Existing invalid-normal fallback remains explicit, not a silent workaround for a missing whole normal source.
- Start at each exact endpoint toward the interior, using reference tangent initialization and shared reset policy. Evaluate the whole line, not independent CP pairs.
- Shared metric default normal radius is 20 base voxels; tangential radius is 4 times that. Existing comparison is strict greater-than; invalid local normals use the existing Euclidean fallback. Inherit/serialize effective config instead of hardcoding a new metric.
- On a failing committed step, record predicted and matched reference points, threshold measurement and reason; resume on reference at max(matched arc, prior segment start + nominal step), bounded at endpoint. Data-not-ready is waiting, NEVER a geometric failure or fill-zero prediction. Infrastructure failures are separately reported as incomplete evaluation.
- Pre-implementation finding: traceFiberReplay required beamWidth=1 and beamLookaheadSteps=1, while its observer saw the current best frontier endpoint. This has now been replaced by common irrevocable-prefix and selected terminal/fallback-path observation, tested against the full-selected-path oracle without changing native candidate scoring/search.
- Beam parity oracle: run the unchanged native search for a directional segment, examine its selected returned path in order with the existing forward matcher, truncate at the first failing step and reset normally. An online optimization may examine only a proven irrevocable prefix shared by every state still eligible for final return (including fallback state), then validate the remaining selected path on all terminal return paths. It must match this oracle. Do NOT freeze the current best beam each iteration or prune beams using ground truth: either would change the tracer. The oracle may infer beyond the eventual first failure; measure this speculative cost. If safe incremental commitment cannot be established, report that limitation rather than silently substitute a different search.
- Report per-source and aggregate failures, complete directed length, failures/mm, seeded/failed span statistics and existing mean-segment reliability. Convert differing voxel scales before physical aggregation; do not sum incomparable base-voxel lengths. Preserve detailed failure locations and partial-run status.

## Execution architecture

### Inspection artifacts

Retain every case's complete evaluated geometry by default, both successful and failed spans, indexed by source/fiber/direction and stable segment IDs. Store reference geometry, predicted points, cumulative losses, forward matches, failure measurements and reset markers in base XYZ with source/normal/model/config identities. Persist each completed segment/case atomically to bound RAM and preserve inspection data on interrupted evaluation. A reset is a discontinuity, never an extra predicted line between two segments.

Reuse/extract existing FiberReplay.cpp replay serialization, segmented OBJ and reference/trace strip-rendering helpers; the current heavyweight writeFiberReplayBundle also requires graph/fiberlet artifacts, so factor out the generic pieces rather than fake empty graph outputs. The present multi-case benchmark retains outcomes but discards replay geometry after measuring each case: extend this point explicitly. Support all-case exports and filters for source, fiber, direction and failure. CT-backed top/side strips and overview images are optional postprocessing; exported line geometry remains available even without textures/normals. Rendering must consume saved traces and must not invoke inference/search again. This is offline inspection, not live GUI integration.

Confirmed GUI call sites: LineAnnotationFiberSegments.cpp invokes traceFiberSegment and traceFiberExtrapolation with its traceConfig; these reach the same traceOneWayCore beam implementation. The evaluator extends the test wrapper around this implementation. It does not replace it with an independent tracer or force greedy search.

Python coordinator owns the scheduler/DB and starts persistent spawn-context PyTorch GPU workers. C++ owns search, sampling and reference metrics. Prefer in-process native bindings with batched request/events, not callbacks per voxel; no LibTorch dependency required. The CLI can use a local versioned control transport to the same session API. No bulk image/prediction arrays travel through a JSON control stream.

All directional cases are logically active. A bounded native pool advances ready sessions. At a sampling batch, enumerate ALL required interpolation-corner chunks, register missing dependencies, and suspend with search state intact. Resume on relevant readiness events; do not restart a trace from its beginning. Do not block candidate workers waiting on Python or inference. Preserve frontier/history and deterministic generation/tie order. Limit nested candidate parallelism under multi-trace execution.

Requests carry session/case identity, output product and chunk key, approximate remaining directional reference length and reason. Merge by chunk key. Demand requests outrank reference prefetch; among demand, longest remaining length first, stable key ties. For shared dependencies use maximum remaining length of their demand waiters, not sum duplicated requests. Propagate priority to all unfinished contributor tiles. Update priorities when traces progress, cancel, or finish. Support aging within priority classes and dependency boosting so blocked finalization cannot starve. Bound queued descriptors and resident work independently of logical case count.

Reference-prefetch producer traverses polyline segments through the output lattice, not merely stored vertices. Include interpolation-corner/support requirements. It is lowest priority and fills spare pipeline capacity. Publish the union once per source/cache identity and promote existing work on demand. Define completion to drain this requested reference-intersection set as well as trace demand; report any cancelled/unfinished background set explicitly. No whole-volume chunk scan.

Input workers use the same native requested-level cache/sampling primitives as fast training; inference normalization/boundaries remain the authoritative shared inference behavior. Factor common worker/slot operations out rather than duplicate private training classes. Batch compatible tiles by source/model/config/shape; one persistent model per selected GPU, no DDP. Reserve enough CPU capacity for input and output so native traces cannot monopolize it. Model/devices are explicit run parameters; none selected or launched during planning.

## Two lattices and the chunk database

Output chunk coordinates are on the persisted prediction lattice. Model tiles are on the canonical source lattice. Native stride, configured inference scaledown, clipping, blend border, repeated resampling-filter support and odd edges all affect the dependency relation. Reuse the shared runner's exact geometry, not just bbox intersection of nominal tiles.

For each requested output chunk, calculate its complete set of contributing canonical tiles. An illustrative initial output chunk size is 32 cubed **output voxels**, with 16/32/64 candidates for profiling; do not confuse these with input voxels. Actual default must respect model output footprint and codec/read overhead. Smaller output chunks may reduce excess inference but increase I/O and metadata.

Suggested SQLite tables (sparse rows for discovered work only):

- cache_identity: schema/version, source identity, checkpoint SHA256, canonical effective parameters, manifest/output geometry.
- session: writer lease/generation, lifecycle and progress.
- model_tiles: tile key, queued/inflight/accumulating/consumed state. Shared result slot and remaining-consumer count are coordinator memory, not temporary files.
- output_chunks: chunk key, partial/ready state and sibling integrity metadata. Requested/publishing flags and pending/applied contributor sets are coordinator memory, rebuilt on restart.
- request_waiters: trace/case key, requested chunk, demand/background class and remaining length.

Use a single coordinator writer and batched transactions; DB is not a hot-path per-voxel service. Native readiness lookup is an in-memory synchronized index updated by event. Coarse process ownership lock guards all channels/products in one cache identity. WAL/sidecars stay with the DB. Lock behavior must be implemented/tested for Ubuntu/macOS; don't assume a shared network filesystem provides the same guarantees.

Model inference complete is NOT output ready. Ready requires every contributing tile applied, all coherent sibling channels/options finalized and atomically published, and publication acknowledged. Reader never probes absent live files as ordinary Zarr fill. Gate initial samples, speculative candidate corners, prefetch and fast visitors consistently. Preserve explicit valid-empty representation; missing does not mean computed zero. On notification invalidate any stale negative lookup or avoid creating it in the first place.

## Scratch, ordering and late requests

The existing circular Z mmap requires monotonic Z order and cannot serve arbitrary demand. Live mode uses full-logical-shape, sparse-on-touch file-backed accumulation. Preserve accumulator precision; demand update order is explicitly allowed to vary. No full-volume zero initialization or dense in-memory allocation. Sparse-file support is required; actual filesystem failures propagate without an application byte cap or per-tile free-space checks.

GPU completion and addition order may be arbitrary. Feed shared result descriptors directly to the SAME `_accumulate_process_main` workers as whole-volume inference; the shared task executor is also used by the synchronous test adapter. Each output chunk has one stable owner, hashing its chunk-grid coordinates (voxel coordinates divisible by32 would collapse modulo8 ownership). FIFO owner queues prevent overlapping read-modify-write races without file synchronization. Retain each RAM result until ALL touched unready chunks acknowledge it, including incidental neighbors. Apply weight and every raw channel before acknowledging. No missing-predecessor waiting and no intermediate tensor files.

Discover the finite set of output chunks touched by each completed tile and register/apply its contributions there. Do NOT recursively request every neighbor's missing contributors merely because partial data exists: that would turn on-demand inference into whole-volume inference. Unrequested partial chunks remain scratch, not ready. A later demand schedules missing contributors; already-run tiles are never inferred again in the session. If all incidental contributors are already applied, a late request queues publication without more GPU work. Never silently evict unfinished contributions.

Accumulator workers serialize updates per chunk. Input slots release immediately after H2D independently of result slots. Default accumulator count equals GPU count; each descriptor queue is bounded32 tasks, with pending descriptors bounded by active result slots and tile geometry. Four background publication threads finalize fully acknowledged chunks; the scheduler never performs accumulation or encoding. `inference.accumulator_workers` and `inference.flush_workers` override these counts. Pending accumulation/publication counts as busy. On shutdown stop/join workers and publishers before closing shared memory or deleting scratch. Final Zarr publication still syncs files/directories and commits the ready ledger; scratch does not. OS mmap page faults/writeback can still perform filesystem I/O.

## Cache layout, identity and recovery

Requested host root: /ephemeral/hendrik/vc3d_cache.

    <global-root>/live_inference/<source-key>/<inference-key>/
      inference.lasagna.json
      live.sqlite
      writer.lock
      <existing per-channel OME-Zarr groups>/
        .live_tmp/<session-id>/   # mmap/staging only; never published data

Place each accumulator with its corresponding output Zarr; designate one group's hidden scratch directory for shared weights and cross-product staging, recording ownership. Use existing output adapter's per-channel names and manifest conventions, not a new flattened schema. Ordinary finalized channels stay presence/nx/ny for every option; no raw seven-channel public volume. Coarser pyramids are produced only where all required children are ready using existing scalar/paired-normal pyramid helpers; don't advertise an entirely complete array while sparse.

Source key: canonical locator plus selected level, immutable source revision where available, shape, transform and metadata fingerprint. Model key: checkpoint bytes SHA256, never mutable current.pt path or mtime alone. Effective inference key covers architecture/config, preprocessing, precision, tile/stride/overlap/border, output scale, recurrent/options, accumulation precision, codec/chunk geometry and implementation schema. Keep trace/search parameters in evaluation identity, not prediction identity. Exclude credentials/signed URL tokens; do not conflate unrelated sources through basename alone.

Pre-implementation configuration conflict: VCSettings.hpp forced /volpkgs/remote_cache, then /ephemeral/remote_cache before considering any suggestion. Resolved with the explicit shared `cache.json` override described above; host configured to /ephemeral/hendrik/vc3d_cache. No existing mirror/cache data moved or deleted. Installed GUI/Python binaries still require rebuild.

The DB persists coordination history but scratch is temporary and never resumable truth. On restart take writer ownership, validate cache identity and affected output sibling presence/integrity, retain verified final outputs, reset stale inflight/partial records, delete only stale owned temporary session files, and rebuild incomplete chunks from scratch. Scope reconciliation to ledger/request keys, not recursive scans. A crash mid-publication must not make mixed generations ready: unfinished transaction entries force whole coherent product rewrite; a ready record still requires intact siblings. Never overwrite a valid ready group in a running reader session.

Sparse unfinished caches can look like zeros to ordinary Zarr-only viewers, which cannot understand the ledger. Preserve existing Zarr/manifest tooling compatibility, but clearly label partial coverage in metadata and require this evaluator's readiness gate. Actual GUI live-readiness support is deferred, not claimed.

## Staged validation and decisions still required

1. Offline synthetic reference parity (greedy), then explicit beam committed-step fixtures; no reference-guided speculative pruning.
2. Exhaustive tiny tile/output dependency oracle, randomized completion order and single/multi-worker byte parity using deterministic mock model.
3. Real shared-runner vs demand output parity on identical model/settings; same persisted bytes and trace failures where backend determinism permits. Document backend nondeterminism rather than relax tests invisibly.
4. Duplicate/late-demand, zero-valued chunks, odd boundaries, branches, missing normal/source data, cache hit, partial publication, process death, ENOSPC, cancellation, restart and queue saturation tests.
5. One tiny real bidirectional case, cold/warm timings, then larger held-out set only after correctness and resource bounds. Checkpoint/GPU selection and all six source/normal bindings remain unresolved before a real run.

No throughput estimate is substituted for measurement. Report pipeline utilization and over-inference (source voxels evaluated across model tiles, including repeated overlap, divided by source-space footprint of the explicitly requested prediction chunks), alongside unique tile and output counts. Report native trace, queue/load, GPU, accumulation, Zarr encode/decode and total time separately, plus RAM/VRAM/scratch peaks. Use this to decide output chunk size, GPU batch sizes and trace worker count.
