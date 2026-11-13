# Fiber-layout workflows

## End-to-end spatial extraction jobs

Joint normal extraction (`orient_normals:true`) uses the spatial chunk pipeline by default. `chunk_pipeline:false` explicitly selects the older globally staged diagnostic path. One shared parallel geometry stage discovers candidates, refines and deduplicates them on original whole curves, then hands finalized encounter positions to the spatial jobs through `FiberTraceConstraintStream::geometryReady`. No prediction sampling happens before this handoff. Each subsequent chunk independently measures winding evidence, samples/orients local normal pieces, immediately emits completed records into the shared sharded importer, and frees its records. It never searches or refines geometry again. Chunk IDs and event identities are independent of dispatch order; estimated-heavy sampling jobs start first.

`chunk_size_base_voxels` defaults to 2048. Sampling jobs have no geometry halo: half-open final attachment-A ownership assigns each encounter once. `chunk_geometry_halo` is no longer used by this path. Prediction queries retain their exact coordinates and can read outside the owning core. The existing scorer, refinement, winding integration and evidence normalization are reused, not reimplemented. Parallel represented-support weights are computed over the complete finalized geometry neighborhood before partitioning; missing predictions discard observations without reallocating their support to neighbors. This intentionally differs from the diagnostic path's post-measurement normalization when predictions are invalid or winding checks reject encounters. With valid retained measurements, weights are unchanged. The original fiber endpoints remain the only scientific endpoints.

Only the sign connecting consecutive spatial pieces is reconciled across jobs, as a final pass. Both sides must use the identical shared boundary arc/coordinate, and their normals must match exactly up to sign. Invalid shared references are explicitly unresolved (zero correction), never matched to a nearby point. Regular original-arc samples plus exact cut references define each local frame. Attachment normals use the preceding regular sample's orientation without altering that frame, making it independent of which chunk owns an attachment. This partition-independent rule differs from the old sweep in which attachment samples themselves influenced subsequent orientation. No persistent regular normal array or second volume sampling pass is needed. The output consumer borrows each chunk's endpoint normals in its local-piece gauge; a missing batch is an error, never a resampling fallback.

During construction, each shard writes measurement rows into a private SQLite TEMP table using the ordinary measurement columns. After all spatial jobs finish, the final pass reconciles piece signs and each shard transfers its staged rows into the final measurement table, multiplying endpoint normals by their piece sign. An unresolved piece instead stores a false validity flag, NULL XYZ and reason `unresolved_piece_frame`. Source indices and attachment arcs locate transient pieces; no additional per-edge piece reference is stored. The TEMP table is an importer implementation detail, not a registered extension or published artifact. Final output uses measurement format 2 on core vcobj schema 21, with no normal-frame table or read-time piece correction. Existing writer connections perform the bounded transfers before sealing; there is no completed-graph reread or shard reopening.

### Native commands, artifacts and reports

The two production commands each take one JSON configuration file:

```sh
vc_object_import_afv import.json
vc_fiber_constraints extraction.json
```

The first configuration names the readonly AFV `source`, new geometry `output`, catalog `volume_id` and optional JSON `report`. For parallel packed bulk import, use `shards:32`, `readers:32`, `packed:true`, `memory_first:true`, `background_flush:false` and a sufficient `resident_bytes` budget. `fiber_limit:0` imports the full source; `verify_readback:false` is the production setting. See the [AFV import options](../../vc_object_store/docs/afv-import.md) for the complete schema and bounded-memory choices. Import uses32 readers/writers by default; these are independent of extraction's compute-worker count.

The second configuration names `fibers` (the sealed geometry store), `coordinate_space`, `normal_manifest`/`normal_provenance`, and for a compound normal source an independent `winding_manifest`/`winding_provenance`. Set `normal_source:"fiber"`, `orient_normals:true`, `repetitions:1`, `umbilicus` and its `umbilicus_sha256`, plus `remote_cache`, `database`, `output` and `normal_prior_output`. `full_volume:true` selects all source fibers; otherwise provide the intended spatial bounds. The shared Python `fiber_wrapped_full.extraction_settings` constructs these settings and preserves explicit overrides. Relevant controls are `chunk_size_base_voxels:2048`, compute `threads`, `import_workers:32`, `packed_store:true`, `normal_cache_mode:"auto"`, `normal_cache_fraction:0.5` and `normal_channel_cache:true`; `pair_owned_geometry:true` selects the measured fast geometry scheduler. Compute workers, reader helpers, geometry-loading workers and SQLite writers are independent pools, not a process-wide thread limit. Do not enable `saturated_chunk_benchmark` in a production import: that mode intentionally discards output.

| Artifact | Contents |
|---|---|
| Geometry `output`, e.g. `fibers.vcobj` | Original whole-fiber objects, native line-point chunks and source-extension data. A packed file contains the independent SQLite shards and routing metadata. |
| Extraction `database`, e.g. `constraints.vcobj` | One native edge and standard format-2 typed measurement row per retained encounter, with piece-sign corrections already baked into endpoint normals. It references the sealed geometry store rather than copying/splitting its fibers. Keep the referenced geometry available when moving the constraint store. |
| Extraction `output`, e.g. `extraction-report.json` | Effective settings, selection IDs/counts, source identities/provenance, load/geometry/pool/fusion timings, per-chunk metrics, import summary and reader/RSS statistics. This is a verbose diagnostic report, not only a short timing summary. |
| `normal_prior_output`, e.g. `outward-priors.json` | Per-fiber `source_ids`, `agreement`, `valid_arc_weight`, source-store pin, umbilicus identity and provenance. It contains no persistent regularly sampled normal array. |

The import report ends with `stage:"complete"` and includes `store_uuid`, `mode`, `fibers`, `points`, `constraints`, `output_bytes`, `ingestion_seconds`, `manifest_seconds`, `persistence_seconds`, `validation_seconds`, `total_seconds`, peak RSS and per-reader/per-shard metrics. Ingestion and final persistence are reported separately.

The extraction report's `chunk_pipeline` includes `chunk_size_base_voxels`, `chunk_count`, `chunk_workers`, `normal_pieces`, `geometry_accepted_encounters`, `measured_encounters`, `geometry_stage_seconds`, `pool_seconds`, `sign_fusion_seconds`, `unresolved_pieces` and per-chunk data. Its top-level `export` includes the constraint count, source-store identity, writer/preparation worker counts, `ingestion_seconds`, `persistence_seconds` and the final path. `measurement_format:2` and `measurement_staging:true` identify this construction path. `measurement_transfer_rows` counts transferred observations; `measurement_transfer_worker_seconds` sums shard transfer callback durations and `measurement_transfer_max_shard_seconds` is their maximum. These callbacks include waiting for sign fusion, not only SQL transfer CPU time. Transfer occurs within ingestion, before sealing and final persistence; do not add its overlapping timings to ingestion. Other overlapping worker timings likewise must not be summed as sequential wall time. The console prints chunk progress, a `normal sign fusion pieces=... seconds=...` line, and a verbose `chunk extraction complete` JSON summary; use the report file for structured consumption.

Piece merging changes neither fiber topology nor the public measurement representation. Raw SQL, generic extension scans and native graph readers all see the same final endpoint normals. This final pass establishes continuity along each fiber only; a later inter-fiber normal-sign solver still chooses the fiber's global sign. It is not an additional winding assignment or a core schema upgrade.

### Stored normals and sign layers

| Data | Storage | Meaning |
|---|---|---|
| Constraint endpoint normals | `ext_fiber_graph_solve_constraints.normal_a_x/y/z`, `normal_b_x/y/z`, plus validity/reason columns | Expanded XYZ unit vectors in the common along-fiber gauge after piece fusion, not encoded prediction bytes or a duplicate unsigned/signed pair. Unresolved piece frames have false validity and NULL XYZ. |
| Later whole-fiber solve sign | `ext_fiber_graph_solve_fibers.normal_sign`, `sign_probability`, keyed by `(fiber_uuid,run_uuid)` | A separate per-run−1/+1 choice for the fiber's overall normal orientation. Extraction does not yet populate a solved whole-fiber sign. |

For an extraction endpoint, the fully oriented result is `stored_normal × solved_fiber_sign`. Native rich/compact graph readers, Python `CollectionReader.fiber_constraints()` and raw extension scans agree on the stored along-fiber normal; no consumer must reconstruct spatial pieces or apply their signs. A consumer selecting a later sign solution applies that run's whole-fiber sign separately. There is no second permanently stored signed-normal vector per endpoint. Dense regular along-fiber normal samples and piece frames are transient in this extraction path; the optional samples result table belongs to explicit sample-export workflows and is not populated here.

There is no independently optimized normal sign per constraint. `local_orientation` is the geometric tangent-alignment diagnostic (sign of the two tangents' dot product), not a normal flip or winding-order label. Optional `signed_winding_delta`/`signed_parallel_winding_delta` diagnostics are distinct from both sign layers; piece fusion does not populate them with merged normal signs. Winding ordering in downstream solvers must use the relevant geometry and selected normal-sign solution, not interpret those diagnostics as the piece correction.

Generated measurement `source_id` values are also chunk-based and sparse (`chunk_id × 2^32 + local_index`), rather than a single dense sequence. These are opaque diagnostic identifiers; original fiber UUIDs and one-edge-per-observation graph semantics are unchanged. Readers must not allocate arrays indexed by the maximum measurement source ID. Packed/sharded storage, store references to geometry, run tables and outward-prior/report sidecars are existing graph workflow features, not additional topology caused by piece fusion.

The shared reader uses 50% auto cache (`normal_cache_mode:"auto",normal_cache_fraction:0.5`) with selected compound normal channels and an independent Lasagna winding channel in the same cache. Each chunk invokes sampling with one caller worker; vc_reader owns chunk coordination/decoding. Import shard writers are separate from these job workers. `CHUNK_COMPUTED` and `CHUNK` progress records identify the actual chunk/worker. The result distinguishes global geometry-stage time and initial/refined candidate counts, catalog time, per-chunk winding/normal/output times, peak concurrent chunks, thread CPU time when supported, process CPU time (including reader helpers and writers), ingestion and final persistence. Do not interpret process CPU/wall as the number of chunk jobs or add overlapping worker durations to elapsed time.

For bounded diagnostics, `profile_stages:true` adds `GEOMETRY_PROFILE` phase-boundary wall/process-CPU snapshots, per-chunk winding/normal/output caller CPU times where supported, and `IMPORT_PROFILE` after producers, shard sealing and queue draining finish but before manifest/final persistence. Its queue-wait counter includes queue push overhead, not exclusively blocked time. Geometry process CPU includes OpenMP spin and other process threads: a serial index build must not be described as parallel merely because CPU/wall exceeds one. Combine these records with the existing vc_reader coordination/scheduler trace to distinguish file opens/reads, decode scopes, exact-key dependency waits and queue/lock overhead. Trace flags are described in `vc_reader/docs/coordination-profiling.md`; compare tracing-enabled cases with disabled controls at unchanged cache and scientific settings. Diagnostics do not change encounter selection, interpolation, winding measurements or import records.

Build `vc_fiber_constraints`, `test_fiber_trace_winding_bp`, and `test_fiber_normal_publication` in the normal volume-cartographer CMake build. The extraction regression covers exact boundary ownership with original arc grids; the normal-publication regression runs end-to-end chunk extraction on a synthetic signed field with one/four workers and checks stable geometry/normals, exact sign-boundary matching, no unresolved valid boundaries, and zero endpoint rereads in output. Benchmark real spatial subsets (not arbitrary fiber prefixes), record source preparation separately, and bound diagnostic runs before attempting a full volume.

### Geometry-only scheduling comparison

`geometry_only:true` on the existing `vc_fiber_constraints` command uses the ordinary production object-store catalogue, UUID ordering, volume transforms and parallel geometry loader, then returns before constructing prediction samplers or exporting graph records. Source manifests/provenance are still validated. Its `geometry_schedules` array accepts `"global"` (the existing schedule) and `"pair"` (experimental pair-owned hash buckets); for example `["global","pair","pair","global"]`. Import input through the normal fast `vc_object_import_afv` workflow first; do not substitute a standalone AFV fixture reader. Both schedules use the same option parser and the same scientific scoring, refinement, deduplication and evidence-normalization routines. The benchmark's prediction callback throws if invoked, and it writes zero output graph rows.

Pair-owned scheduling uses complete canonical fiber pairs as the ownership key. Discovery workers count/prefix/scatter candidates into independent hash buckets without a global candidate sort. Bucket jobs sort locally, score, deduplicate, perform all configured refinement rounds and normalize pair evidence; buckets progress independently. Accepted positions/pointers are assembled with parallel prefix/scatter. The original immutable spatial index and the barrier after discovery remain in this first version; it does not yet overlap discovery with pair processing or replace the R-tree. Normal production scheduling stays unchanged unless explicitly selecting `pair_owned_geometry:true` with the chunk pipeline.

Each benchmark run reports geometry wall time, process CPU/wall effective parallelism, candidate/refinement/event counts, and separately measured verification/teardown time. `verify_geometry:true` (default) computes a canonical SHA256 of source-piece identities, attachment arcs/XYZ, distance/orientation scores, support intervals, evidence weights and local orientation. Hashes must agree across schedules in one invocation; cross-invocation comparisons should also check them. Hashing is diagnostic work after the geometry-ready timestamp, not part of production stage2 throughput. Preserve within-pair candidate order and refinement-round semantics when changing scheduling. Unit regressions compare multi-pair geometry/weights/audits at one/four workers and reject any unexpected prediction sampling.

### Saturated chunk-window diagnostics

Joint extraction defaults to `chunk_size_base_voxels:2048`; this is spatial sampling-job size, not the independent winding-integration region or importer batch size. When `threads` is omitted, the native extraction command defaults to the full available physical-core count, not the SMT/logical CPU count. Linux uses affinity-filtered sysfs sibling groups; macOS uses `hw.physicalcpu`. If physical topology cannot be determined, set `threads` explicitly rather than guessing an SMT factor. Explicit chunk/worker overrides are preserved. The final chunk-pipeline report records the effective size, and `settings.threads` records the resolved compute-worker count. Compute `threads`, shared-reader helpers and SQLite `import_workers` are independent pools; a compute-worker count is not a process-wide thread cap.

`saturated_chunk_benchmark:true` exercises the real chunk sampler and sharded importer but intentionally never publishes a complete graph. A window starts after all requested workers have been observed active and `benchmark_warmup_seconds` has elapsed (default1s). It stops when unfinished jobs become strictly fewer than the worker count, or after `benchmark_window_seconds` (default28s). With `benchmark_require_worker_completion:true`, the duration cutoff waits until every chunk worker has completed at least one job; the low-job cutoff still stops the measurement and reports whether this requirement was met. Use a separate parent emergency timeout, treating a timeout without a cutoff checkpoint as incomplete, not a valid result. Active jobs include winding, normal work and encoding/queue emission; activity is not CPU utilization. Effective parallelism uses whole-process CPU/wall, including reader helpers and SQLite writers. Report preparation/geometry separately; do not use whole-process wall time as the window duration. Chunk sizes with fewer jobs than workers cannot provide a valid saturated comparison. Cutoff diagnostics include unique decoded-cache residency (shared caches counted once), VFS resident extent bytes/peak and configured VFS/SQLite cache budgets. These are not an exact heap/RSS partition: untracked live work, allocator-retained pages and other resident mappings remain outside these counters.

Synchronous reader requests can take longer than the window to unwind. A supervised test should provide `benchmark_cutoff_file`: the process atomically checkpoints its window there immediately at cutoff, before waiting for in-flight calls. The test harness then terminates that process and removes its owned temporary staging directory, rather than allowing the rest of the workload to finish. A cutoff checkpoint is a partial diagnostic result, not a completed extraction/store. The harness must preserve unexpected errors, accept only its expected termination signals with a valid checkpoint, and verify that no graph was published. Do not use this mode with a destination containing user data. Without external supervision, the API drains in-flight work before unwinding its temporary importer.

For bottleneck profiling, `benchmark_pool_start_file` optionally writes a small monotonic-clock marker immediately before chunk workers start. Cutoff summaries include `pool_start_monotonic_seconds`, allowing existing reader coordination/scheduler traces to be restricted to the measured window rather than preparation or cleanup. `normal_cache_profile:true` also includes reader/cache statistics in the cutoff memory snapshot. Profiling changes timing: report unprofiled controls separately, and do not attribute differences solely to chunk size when instrumentation or read/cache ordering differs.

Progress rates distinguish completed winding batches, completed normal-sampling jobs and completed encoding/emission calls. Large jobs can have long pipeline latency, so a28s capped window with few emitted jobs does not establish steady-state output throughput. Report the stop reason, total/remaining job counts and occupancy with rates; never treat a time-capped case as having reached the low-job-count cutoff. When worker completion is required, freeze counters at the low-job boundary but wait outside the measurement until every worker has completed a job. Emergency timeouts remain failures. A partial-work estimate may calibrate winding seconds/constraint, normal seconds/sample and emission seconds/constraint on completed chunks, then sum completed stage units at these costs and divide by the total calibrated seconds/constraint. Label this as estimated constraint-equivalent progress, not actual emitted rows or a durable-write rate; heterogeneous chunks and unfinished sub-batches limit its accuracy. Counter snapshots are sampled about every10ms. Pure-clock regression tests check warmup exclusion, the strict128→127 cutoff, worker completion and the duration target; the integration fixture checks that deliberate cancellation publishes neither a graph nor normal priors.

The preparation, tuning, diagnostics and surface-export workflows are shared
Python modules, not local runner scripts. Configuration and datasets remain
external. The solver implementations and scientific defaults are retained;
this restoration does not select a new model or change its losses.

## Run

From an environment configured for this checkout:

```sh
python -m vesuvius.neural_tracing.fiber_trace_3d.fiber_workflows --config /configs/fiber-layout/job.json
```

The JSON contains **one** workflow key and its settings, as below. Relative paths
are relative to the working directory, not the config file. Use absolute paths
for jobs that launch child processes. A local approval wrapper should only choose
the interpreter, environment and configuration; all workflow logic lives here.

Use the existing Vesuvius ML environment, `lasagna` and `villa_gpu_manager` from
this repository. Individual tools additionally use SciPy, threadpoolctl, psutil,
matplotlib, tifffile/imagecodecs, scikit-image (connectivity) or boto3 (S3 download).
No command installs dependencies or downloads inputs implicitly. CPU diagnostics
do not acquire GPUs. GPU solve stages use the existing manager launcher; configure
`devices` and `gpu_manager` in their solver configs. Non-exclusive grants are
supported. The optional matrix performance harness requires an exclusive parent
grant, unlike ordinary search trials.

Native extraction uses `vc_fiber_constraints` on `PATH`, or `native_binary`.
Command-list jobs also accept `native_binaries` (name-to-path mapping) and
`binary_dir`. Supply `native_regression_binary` to the
normal/axial-prior wrappers to run the native regression executable first.
Other native tools, such as reference replay, must be available separately.

## Store format and cache boundary

Workflow `fibers` inputs and `database` graph outputs use schema21 vcobj. The shared native readers support ordinary SQLite files, unpacked sharded collections and packed multi-database containers, including readonly store references. Supply the actual root locator returned by import; an unpacked collection's locator is its manifest, not the surrounding directory. Do not open a composed store directly with SQLite or look for the removed AFV/AFC tables.

Fibers/relations retain global UUIDs. Python IDs are canonical UUID strings; dense integers are local array indices, not persistent fiber identities. Measured constraints are native relations plus typed `ext_fiber_graph_solve_constraints` rows. Run/input/participant tables capture source snapshots and cohort membership; supported stage results use the same extension family. Native import can still read an AFV source, but internal graph/normal/layout workflows neither write AFV nor retain the old AFC backend.

The full-source workflow discovers constraints once, then samples each fiber once at regular arc positions and its exact constraint attachments. That same sweep establishes the fiber-local minimum-turn gauge and accumulates outward priors. It bulk-imports the final oriented constraint normals directly into `oriented.vcobj` and writes priors pinned to that final snapshot. Regular along-fiber normal samples are transient; no curve dataset or raw-normal intermediate graph is written. This local orientation is not the later global cross-fiber sign solve. Standalone normal alignment of an existing graph remains available through the native orientation API and writes an immutable typed measurement overlay. The retired JSON-normal SQLite UPDATE/publication pipeline is not an alternative backend. NPZ checkpoints and solver tensors remain supported alongside typed graph stores.

Native measured constraints use extension format version 2 within schema21: explicit typed endpoint XYZ values are in the declared base-voxel coordinate space, while native relation anchors use the source object's geometry frame. Numerical evaluation reads those exact measured points, not transformed anchors. Format version 1 is retained only for the external legacy constraint import adapter. Normal source identity is a checksum; oriented-normal reuse also requires the recorded orientation run and sampling spacing. The [typed measurement format contract](../../vc_object_store/docs/extensions.md#native-extraction-measurement-format-2) defines the fields, endpoint ordering and frame metadata.

Geometry-based solver preparation uses that measurement volume for source curves and the umbilicus, including registered source-to-target transforms. Prepared-only inputs without a graph can name `working_volume`; if both are supplied they must agree. Fixed-assignment UV staging preserves this setting. The neighborhood-selection workflow still requires source fibers already imported into the measurement volume and rejects a different registered source frame before selection; normalize that source with the import transform first. This avoids interpreting source-space bounds or frozen GT arcs as target-space measurements.

Rebuild caches produced before this port: fiber IDs are UUIDs and packed geometry keys now contain UUID-catalog ordinals, not numeric AFV IDs. Numeric extraction event IDs remain separate. Store snapshot identity and ordered fiber-version digests detect incompatible reuse; memory-mapped graph caches also check their format and required columns. Preserve referenced source stores with derived outputs; a lone overlay need not contain its geometry. See [storage and extraction](fiber_layout.md#storage-and-identities) for native import options, reference resolution and validation scope.

## Preparation and evaluation

### Experimental wrapped-IRLS classification and normal signs

Full-source comparisons can use `fiber_wrapped_full --config FILE`. It calls the same native importer/extractor and records command settings, code checksums, wall times and native phase timings. The `actions` list selects preparation stages; completed native steps can be reused only with identical settings. `import_pilot` uses an automatically cleaned directory and a 28-second execution budget plus two seconds for termination. Paths, scan identity, published artifact locator, local compound prediction and worker/cache budgets are configuration, not bundled experiment data.

Native extraction optionally accepts `winding_manifest` and `winding_provenance` separately from `normal_manifest`/`normal_provenance`. Both must name the same scan and base dimensions. With `normal_source: "fiber"`, the compound `sheet_nx`, `sheet_ny`, `sheet_grad_mag` channels provide masked sheet normals; a separate Lasagna winding source is required for extraction because `sheet_grad_mag` is a validity mask, not winding density. Winding-distance integration continues to use the Lasagna sampler and unchanged thresholds. No missing compound normal is filled from Lasagna. Both source identities are recorded. Omitting the separate source preserves the existing Lasagna-only behavior.

Constraint export uses the shared `ParallelImportStream` and `importInitialCollection` APIs. Normal sampling uses spatial ROI jobs before arc-ordered sign propagation and typed record preparation. Prepared batches emit directly to bounded UUID-routed shard queues; independent SQLite writers are controlled by `import_workers`. Materialized-report export uses `import_readers` and `import_batch_rows` for preparation. Native records go directly to the typed extension encoder without a JSON round-trip. Exact measurements, invalid normals and endpoint ordering are unchanged. Live source event IDs encode `(job << 32) + eventWithinJob`, leaving gaps while preserving deterministic source ordering independently of scheduling; they are not dense array indices. Rebuild graph caches when replacing an earlier extraction.

Scoring volume reads partition actual constraint positions into Z/Y/X ROIs (`scoring_region_base_voxels`, default 4096), with at most `candidate_batch_size` candidates per spatial batch (default 65536). Fiber identity never selects a read job. Pure geometry reads no volume data and keeps cache-friendly source order. CPU-only deduplication and refinement use coarse blocks that retain complete pair evidence. Winding integration independently groups its actual sample points into spatial batches (`winding_region_base_voxels`, default the scoring ROI; `winding_batch_points`, default 262144), then integrates in the original connector order. ROI planning counts points per coarse source block and uses stable region prefixes, sorting only distinct region keys. This is scheduling, not clipped tile extraction: complete geometry remains available, and probes crossing ROI boundaries are neither clipped nor dropped. Compound sheet-normal aliases retain one runtime group per source packed array. Diagnostics expose encoded channel bytes read and decoded bytes separately from remote network traffic.

Refinement visits every configured scale, including scales after an empty round: later rounds lower the eligible arc-gap threshold. The former early stop depended on unrelated candidates sharing a small CPU batch and could omit finer events. Removing that stop intentionally corrects batch-dependent event selection; it does not change measured winding distances or interpolation. Historical graphs can therefore differ by a small number of additional refined events, rather than matching byte-for-byte.

Constraint extraction now completes geometric scoring, deduplication and all refinement rounds before the first prediction-volume read. Refinement considers geometric validity and parallel/perpendicular classification only. The sampling stage measures the retained encounters in large spatial batches and applies invalid-data/winding rejection afterwards; it never triggers another geometry round. Progress marks `geometry_complete` before `winding_roi` and `measurements_complete` afterwards. This intentionally removes the former coupling to measurement validity (and certified winding categories): missing predictions can no longer suppress geometry work, and the resulting candidate/constraint counts can change. Source curves, projection windows, probe locations, interpolation and integration mathematics are unchanged.

`VC_FIBER_PROJECTION_PROFILE=1` enables worker-local projection histograms, printed as `PROJECTION_PROFILE` JSON after each geometry pass. Counts distinguish parallel/perpendicular, geometry acceptance/rejection and initial/refined passes. They measure final arc movement, maximum intermediate excursion, movement from the original source sample (including refinement), Cartesian maximum-axis displacement, and queried projection-window reach. Histogram bounds are inclusive; the final count is above the last bound. These are evaluations before winding checks and deduplication, not final graph constraints; instrumented timings are not baseline throughput measurements.

The optional `bench_fiber_projection` CMake target runs this geometry diagnostic on an AFV source using the shared source reader, array decoder and production extractor. Its JSON config requires `source` and `output`, with optional `threads` (default32) and `bounds` (`xmin,ymin,zmin,xmax,ymax,zmax`). It selects whole original curves intersecting the box, never clipped curves. Refinement uses explicitly marked constant valid winding evidence, so no normal or prediction-volume reads are required; its output must not be used as a physical winding solution. For example: `VC_FIBER_PROJECTION_PROFILE=1 bench_fiber_projection projection.json`.

The configured CPU budget is shared between concurrent ROI callers and each caller's sampler loops; a large ROI is not restricted to one CPU. `reader_threads` separately configures the shared reader helper pool; zero retains the reader's default. `VC_FIBER_WINDING_PROFILE=1` emits per-call `WINDING_PROFILE` JSON lines separating connector/sample preparation, spatial planning, reading/interpolation, materialization and integration. Use these instrumented runs for attribution, separately from ordinary throughput measurements.

On bounded inputs, winding-read ROIs may be subdivided to provide about four coarse jobs per configured CPU; full-volume inputs can reach the configured candidate maximum. Normal read jobs retain their independent point maximum. Geometry payloads and probe buffers use worker-local arenas, released after constraint assembly. Rejected candidates carry no full constraint payload; deduplication compacts coarse spans in place. These choices affect scheduling and temporary storage only, not sample positions, certification, integration order, or normal sign propagation.

Packed CPU normal batches use a shared Lasagna interpolation policy executed inside `vc_reader`'s existing sampling traversal. Only final normals and validity return to the caller; interior points read cached channel planes directly, and only chunk-boundary points gather temporary corners. Required channel registrations still select just sheet-normal components and their validity mask. CPU winding integration also consumes interpolated density/normal values from shared reader policies, including sources in independent arrays. Independent encoded normal components use bounded reader-internal scratch to preserve projective tensor interpolation; no corner batches return to the integrator. Fractional sample placement, projection, decode scaling and trapezoidal accumulation remain unchanged. `normal_point_interpolation: false` and `winding_point_interpolation: false` independently select their respective corner-buffer reference paths for comparisons. Scalar double-precision sampling and CUDA retain their existing implementations.

For a bounded sampler-only benchmark, add `normal_sampling_benchmark` to an ordinary `vc_fiber_constraints` request. It loads the selected geometry and sources but does not extract constraints or write a result store. For example, `{"points":65536,"unique_points":4096,"batch_points":64,"workers":64,"repetitions":3}` repeats real source positions in pipeline-sized batches. The corner-buffer and policy methods share a warmed decoded cache, alternate timing order, and must produce bit-identical normals and validity. Each measurement repeats passes for at least `minimum_measurement_seconds` (default 0.25); the report records pass count and actual elapsed/CPU time, plus min/median/max/mean time per pass, valid count and setup time. `workers` is outer sampling concurrency; the reader may also use its shared helper pool. End-to-end extraction timings should be measured separately, retaining setup and final persistence.

Packed constraint imports use disposable memory-first buffers, no background spill, and a final durable export. `import_resident_bytes` is a hard RAM budget (default 64 GiB); budget exhaustion fails rather than silently switching to synchronous disk writes. Size it for the final database and SQLite working set. `import_export_workers` defaults to 8; `sqlite_cache_bytes` defaults to 128 MiB per writer. Standalone extraction may set `sqlite_memory_statistics:false` before opening any database to avoid SQLite's process-global allocator-accounting lock. Export reports ingestion and persistence separately, plus summed worker time for sampling, encoding, source waits and shard-queue waits; summed worker time is not wall time.

Normal orientation and outward-prior sweeps first construct the unchanged arc-ordered samples, then sample them in spatial ROIs (`normal_region_base_voxels`, default 4096; `normal_batch_points`, default 262144). After scattering samples back, CPU-only scans preserve each fiber's sign continuity and prior accumulation. Reports separate coordinate preparation, ROI planning, sampling, orientation and persistence, with actual ROI/batch counts. Binary curve export remains ordered. Derived normal stores use the same fast import settings. Collection relationship readback bulk-loads endpoint lifecycle and paged edge records instead of querying both endpoint objects for each relation; overlay tombstones and retired endpoints are still respected.

For solver-only comparisons on archived tensors, the benchmark accepts `cache_only: true`, `normal_cache`, `hv_cache`, and `length_catalog` (an NPZ with matching `ids` and whole-fiber `lengths`). This bypasses geometry import and preserves the original tensor IDs, evidence, and H/V exclusions. It sets `write_result_store: false`: complete solution NPZs, histories and metrics are saved, but no new vcobj result overlay is created. Production stage defaults still write typed result stores. Cached H/V length retention is relative to the stage's active input, not fibers excluded by the earlier normal stage.

The existing `fiber_normal_sign` and `fiber_hv_stage` configurations accept `optimizer: "wrapped_irls"`. Adam remains unchanged and remains the default. Preparation, node identity, measured evidence, normal convention, typed vcobj outputs and managed GPU launch are shared with the existing stages. This backend changes the soft objective; it is not a claim of equivalence to the independent-probability SGD model.

The scalar phase has period two. The loss evaluates `wrap2(phase_b - phase_a - target)` on every outer iteration: target zero for equal labels, one for opposite labels. Normal alignment uses the sign of the endpoint normal dot product for the target and its magnitude for the weight; H/V uses the existing parallel/perpendicular evidence. Neither saved labels nor continuous phases are rounded during optimization. The periodic L1 cost is continuous at the wrapping seam, but has a derivative corner there. IRLS uses the same inverse-residual weights as the winding solver, with an epsilon-Huber treatment near zero; it does not use a trigonometric or quartic replacement. Both that optimized objective and exact wrapped-L1 cost are logged.

Each outer iteration constructs the local quadratic from the current wrapped residuals, then solves a linear graph system with Jacobi-preconditioned CG. Wrapping is never applied inside the CG matrix-vector product. Per-fiber keep variables multiply all their incident pair terms and pay the existing linear disablement cost. They are updated with a bounded, vectorized diagonal quadratic majorizer of the coupled keep objective, not independent edge rejection. Backtracking checks the actual periodic objective. Five successive small objective decreases stop the solve; this is a local stopping criterion, not a global-optimum certificate.

Settings (in addition to the existing prior/cost settings):

```json
{
  "optimizer": "wrapped_irls",
  "irls_iterations": 100,
  "irls_cg_iterations": 100,
  "irls_epsilon": 0.05,
  "irls_cg_tolerance": 0.00001,
  "irls_tolerance": 0.0000001,
  "irls_log_every": 10,
  "entropy_weights": [0]
}
```

The experimental H/V backend requires explicit `entropy_weights: [0]`; nonzero entropy hardening is rejected, not silently ignored. It inherits the same prior-stage exclusions as Adam. Phase and keep are initialized from the same axial/radial evidence and initial logits. The saved ternary/class probabilities are circular interpolation scores, **not calibrated uncertainty**. `original_expected_objective` evaluates the unchanged entropy-free SGD objective on those scores; its value is not interchangeable with the optimized wrapped objective.

For a real-data stage-isolated comparison, `fiber_wrapped_benchmark` imports a supplied whole-fiber AFV cohort through the native packed/sharded importer and induces matching subgraphs from archived normal/HV tensors. Configure `source`, `selected_cache`, `normal_cache`, `hv_cache`, `native_import_binary`, `volume_id`, `workers`, a fresh `output_root`, `parameters`, `devices`, and `gpu_manager`. The caches must refer to the same original source-ID catalog; existing H/V exclusions stay fixed across all trials. This intentionally measures solver replacement on frozen evidence, not a fresh extraction or an end-to-end change of the selection cohort. Inputs/configs and experimental outputs are external, not checked into the repository.

```sh
python -m vesuvius.neural_tracing.fiber_trace_3d.fiber_wrapped_benchmark --config /configs/wrapped-stage-comparison.json
```

The comparison runs Adam and wrapped IRLS for signs and entropy-free H/V, plus a separately labelled Adam H/V entropy-.01 control. `parameters.steps` controls Adam's existing iteration budget; `irls_iterations` is an upper bound on IRLS outer iterations. It reports import/preparation time, solve/wall time, memory, losses, violations, retention and typed-result readback. GPU reservation is configured explicitly; nonexclusive timings are operational, not isolated hardware measurements. Each method is run once. Synthetic regressions in `test_fiber_wrapped_irls.py` additionally cover seam continuity, gradients away from corners, exact dense/CG agreement, self-edges, isolation, rejection/reactivation, descent and CPU/CUDA parity.

The full-source `fiber_wrapped_full` workflow prepares shared evidence outside GPU grants and launches the two managed solver supervisors in separate **spawned processes**. A thread pool is not suitable for managed launches: the supervisors would register the same process tree. Spawn also avoids forking a populated graph or initialized runtime. Caller scripts must protect their entrypoint with `if __name__ == '__main__':`. Completed stage summaries are reused only when their saved settings match the requested settings.

`fiber_selected_solver_comparison` can reuse an existing `selection_root` with
frozen sign/H/V/broken classifications and extracted constraints. Optional
`minimum_length_mm` and `minimum_width_base` strictly filter whole fibers before
the existing largest-component/perpendicular-support selection. Length uses the
explicit `voxel_size_mm` configuration; width uses the stored piece-local
length-weighted `fiber_attributes.width_mean` in base voxels (not the parent
coverage-policy width). Missing widths are excluded and counted. The selection
report records threshold counts and final mean/median/total physical length.
No sign estimation, H/V classification or constraint extraction is repeated.
Use `include_sgd:false`, `exact_lp:false`, `capacity_scales:[]` and
`uv_methods:["irls"]` for IRLS only followed by fixed-label U/V fitting.
These filters must match when reusing a `prepared_root`.

### Normal-source reader cache

Native extraction/orientation defaults to `normal_cache_mode: "auto"` with `normal_cache_fraction: 0.5`; compound workflows also request `normal_channel_cache: true`. Auto mode uses one shared vc_reader decoded-cache budget for the normal and winding datasets, including all channels. Its target is `fraction * (available_memory + resident_cache_bytes)`; subsequent budget changes use the reader's default 10-second EMA, capped by current headroom. It adjusts to memory pressure rather than allocating a separate fraction per volume. An explicit `normal_cache_mode: "fixed"` selects the `normal_cache_bytes` budget when required by another caller; the workflow and extraction performance comparisons use 50% auto, not cache-size sweeps. The general vc_reader library cache default is unchanged. This is a cache budget, not a process memory limit: active sampling buffers and cache entries still held by in-flight readers can temporarily occupy additional RAM.

For controlled profiling, `normal_cache_shared: true` also shares one fixed-size cache in fixed mode; by default fixed mode retains separate caches. `normal_cache_profile: true` enables reader LRU acquisition-wait/hold counters and memory-probe timing in `reader_stats`. These switches do not change sampling values. Shared-cache counters appear under both readers and must only be counted once. Compare profiling-on/off controls because the timing instrumentation adds work.

`normal_cache_maintenance_workers: 4` opts extraction into background eviction with foreground assistance; `normal_cache_candidate_scan: 8` bounds sampled shard-tail selection. The default remains synchronous eviction (zero maintenance workers). Shared normal/winding caches start maintenance only once. `reader_stats` includes eviction batches, evicted entries, assisted puts and headroom for checking whether a run actually exercised eviction.

Channel caching registers only the channels bound by the sampler (the sheet-normal components and validity for compound input). CPU corner sampling submits channels belonging to the same packed array together so vc_reader can coalesce codec range reads. Sampling coordinates, validity masks, tensor interpolation and winding measurements are unchanged. These are reader options, not extra filesystem or compressed-byte caches.

### Joint normal extraction

The full-source `extract` action includes normal orientation and outward priors; `complete` proceeds directly to global sign/classification solves. Separate `orient`/`priors` actions are rejected by this workflow to prevent duplicate sampling. Native extraction enables the combined path with `orient_normals: true`, `normal_sample_spacing`, `umbilicus`, `umbilicus_sha256`, and `normal_prior_output`, plus the final graph `database`. All exact attachment points are known after global geometry discovery. Spatial jobs then sample and emit into the sharded import concurrently; final piece fusion precedes the staged measurement transfer and sealing. Reports distinguish geometry discovery, spatial pool, sign fusion, measurement transfer, overall ingestion and final persistence; `export.sampling_seconds` is zero when the encoder reuses supplied normals. The explicitly selected globally staged diagnostic path completes its normal sweep before ingestion. The import returns its sealed store UUID, lineage generation and commit sequence directly, without reopening the finished graph for identity/readback validation.

### Same-input IRLS winding comparison

`fiber_irls_benchmark` compares an external winding-field `sheetgraph.solve_l1`
implementation with the existing LEMON backend on the identical aggregated
fractional-target graph. Supply `solver_source` and its `solver_sha256`, a
`dataset` (`fibers`, `tensor_cache`, `umbilicus`, measurement calibration/weights),
`sgd_solution`, `events`, `gt_config`, `gt_solution`, and a fresh `output_root`.
Optional `sgd_report` validates the historical SGD score and records its timing.
GPU settings use the standard nonexclusive `devices`/`gpu_manager` launcher.

The external module is imported directly and must already have its dependencies
installed. No code/data is downloaded. Defaults are 15 IRLS reweightings,
500 CG steps, epsilon .05, and the upstream 1e-6 zero-prior weight. Both raw
continuous potentials and upstream-style component-phase-rounded labels are
reported. The tiny prior and finite epsilon mean IRLS is an approximate solver,
not a certified exact L1 optimum. The exact fractional-capacity LP and LEMON
capacity scales 1000/1 provide controls on the same graph.

No track splitting, surface inference, winding-inference seeds, or UV refinement
is used. The saved independent SGD winding checkpoint is scored, not retrained;
its source IDs, intervals and families must exactly match. Continuous endpoint
GT L1/correctness and geometric sign errors use existing shared helpers and a
single global integer gauge. Report full graph-solver coverage and the identical
SGD-retained event subset separately. SGD has rejection and geometry losses;
its training objective is not numerically comparable to graph L1. Timings from
nonexclusive grants are operational, not isolated hardware benchmarks.

`fiber_length_histogram` takes `comparison` (the per-fiber NPZ from
`fiber_hv_label_comparison`) and `report` (output JSON). It reports source arc
lengths in base voxels: count histograms and quantiles for all/H/V/broken fibers.

For a CPU-only 3D preview of changed source-family labels, use:

```json
{"fiber_hv_crop": {
  "fibers": "/data/fibers.vcobj",
  "comparison": "/results/hv_comparison.npz",
  "output_root": "/results/change_crop",
  "size_base_voxels": 2048,
  "minimum_each_change": 3
}}
```

The comparison is the per-fiber output of `fiber_hv_label_comparison`. The crop
is selected deterministically from cells containing both change directions;
alternatively supply `bounds_xyz: [[xmin,ymin,zmin],[xmax,ymax,zmax]]`.
The new output directory contains five line OBJs (`h`, `v`, `broken`,
`h_that_were_v`, `v_that_were_h`), an MTL and `index.json` with bounds and source
IDs. Changed subsets also occur in their final H/V group. Original raw curves
are segment-clipped, without repairs, rescaling or translation. Pairwise OBJ
line elements avoid accidental connections across gaps and work with MeshLab.
For a whole-volume preview instead, replace the size/minimum-change options
with `"sample_per_group": 100, "seed": 42`. This independently samples each
of the five groups uniformly without replacement, reads the original full
fibers without clipping, and records the seed and selected IDs in the index.
Changed-group samples need not also be selected in the independent H/V samples.

| Workflow key | Inputs / result |
|---|---|
| `fiber_json_solve` | Reviewed JSON directory + `volume` id and `volume_shape_zyx` + explicit ordered stages; creates a fiber store (`fibers.vcobj`, each fiber keeping its document `uuid`) and dispatches extraction, normal/HV/direction and layout requests. |
| `fiber_anchored_dataset` | Reference-curated source graph; constructs an anchored dataset with fixed reference events. |
| `fiber_selected_dataset_eval` | `dataset`, `base_config`, `gt_root`, `output_root`; filter unsupported sources, split if configured, initialize, solve and score. |
| `fiber_selected_score` / `fiber_selected_checkpoint_report` | Score completed or published intermediate states without rerunning optimization. |
| `fiber_joint_gt_eval` | `baseline_root`, `base_config`, `gt_root`, `output_root`; attach GT, with `force_gt_enabled: true` or `false`. `report_only` reads completed output. |
| `fiber_gt_inpainting` | Fixed-reference solve and optional post-solve deformation. |
| `fiber_neighbor_expansion` | `seed_solve`, whole `parent_tensors`, `fibers`, `gt_root`, `z_range`, extraction/orientation templates; MAP-select neighbors, re-solve, re-filter per round. |
| `fiber_neighbor_pair` | Run and compare no-GT/forced-GT variants on the expanded dataset. |
| `fiber_full_slab` / `fiber_full_slab_soft` / `fiber_full_slab_progress` | Whole-slab dataset construction, soft-only control and checkpoint reporting. |

Minimal selected-set job:

```json
{
  "fiber_selected_dataset_eval": {
    "dataset": "/data/fiber-layout/selected",
    "base_config": "/configs/fiber-layout/base.json",
    "gt_root": "/data/fiber-layout/gt",
    "output_root": "/data/fiber-layout/results/free",
    "devices": "0"
  }
}
```

`base.json` wraps the solver config under `fiber_implicit_winding`. The selected
dataset contains `selected.vcobj`, `tensors.npz` and `cross/cross_events.json`.
`gt_root/uv_whole` contains the trusted `config.json`, `solution.npz` and referenced
tensors. A new joint solve also needs that layout's `summary.json` with the
saved tensor source signature and the prepared baseline's `inputs.json` with its
source digest, alongside `tensors.npz` and `gt_events.json`. Paired-neighbor
preparation emits the manifest after validating the dataset's saved source.
Neighbor expansion verifies the parent `tensors.json` signature before reading
its source curves, so the exported source digest describes the original tensor
evidence rather than whichever geometry is current at expansion time.
Regenerate old preparation artifacts that never recorded this evidence; do not
stamp them with current geometry. Existing report-only runs do not need new input
manifests. These schemas are also checked by the
[three-arm replay benchmark](fiber_layout_benchmark.md).
Editable samples: [selected-set evaluation](examples/fiber_workflows/selected_dataset.json),
[GT inclusion](examples/fiber_workflows/joint_gt.json), and
[neighbor expansion](examples/fiber_workflows/neighbors.json).

For reviewed JSON, provide `expected_manifest_tokens` (source volume/scan tokens)
and supply `umbilicus` in stages that require it. Do not infer scan identity from array shape alone.
Use `stages`, an ordered list of `{name, config_file}` or `{name, request}`
entries. Each request is a complete workflow object (or native command-list
job). Stage names are caller-selected output labels, not preset selectors.
`{fibers}` and `{output_root}` are expanded in request string values; referenced
native configuration files must already contain their own correct paths. No
solver settings, enablement policies, pass counts or GPU reservations are injected.
An empty list converts JSON to a fiber store only. `layout_output` optionally requests the
categorical-layout report; omit it for other checkpoint formats. See the
[JSON pipeline example](examples/fiber_workflows/json_pipeline.json).

For surface projection, `reference_xyz_scale` is required in anchored-dataset and
full-slab jobs (including slab scoring jobs). Supply the reference-to-base-voxel
scale for the actual input, rather than assuming a particular scan resolution.
Projection stride, candidate count and sampling distance are configurable.
`fiber_gt_inpainting` runs deformation only when `deformation_parameters` is
provided. `fiber_current_gt_solve` renders only when `visualization` is supplied.
Both settings objects pass caller choices through to the existing implementations.

## Iterative search and regression problems

`fiber_split_weight_search` restores asynchronous three-arm scheduling, screening,
candidate reuse, score recomputation, coupled proposals, discrete split choices
and coarse-to-ultrafine coordinate search. The objective remains summed GT
perpendicular sign violations **divided by** selectable retained-GT arc fraction;
summed GT perpendicular L1 breaks ties. The threshold is probability >= 0.5.

The [minimal search example](examples/fiber_workflows/split_search.json) shows
bounds, discrete choices and a coupled proposal. It is an interface illustration,
not a tuned parameter set. Unspecified solver settings come from `base_config`.
Set `background: true` for a persistent coordinator; it writes its PID,
config and log. Otherwise the process remains attached. `accepted_improvements:
null` keeps searching; an integer is an explicit stop limit. Resume with identical
settings and code in the same output root. Changed scientific inputs/code require
a new root; reuse checks must not be bypassed. Linux process-control actions verify
the exact module/config before signalling; use normal process control elsewhere.

The earlier `fiber_paired_weight_search`, `fiber_density_weight_search`,
`fiber_selected_weight_search` and `fiber_uv_layout.loss_weight_search` remain
available. They retain their own ranking protocols; do not mix their scores.

The small-GT matrix is data-independent code:

1. `fiber_current_gt_solve`: construct all-event or enclosure-only graphs.
2. `fiber_gt_width_initialization`: generate explicit width-factor variants
   (`kind: "scale"`, factors 0.5 and 2.0), preserving all other initialization arrays.
3. `fiber_gt_convergence_probe`: run a `trials` list from supplied checkpoints
   using the configured optimizer and full objective.
4. `fiber_reference_check`: construct/use consensus regression references;
   reports geometry, signs, length, angle and meeting checks.
5. `fiber_uv_weight_search.regression_guard`: apply the recorded acceptance
   limits. Supply limits/reference data and any excluded source IDs explicitly.

Choose regression modes, initialization variants, sampling and acceptance limits
in external configurations. The [width-regression sample](examples/fiber_workflows/gt_width_regression.json)
illustrates two caller-selected trials; it is not a required test matrix.
The [no-initialization-width-prior rule](fiber_layout_no_width_prior.md) applies
to every workflow. In paired searches, `gt_parameter_overrides` explicitly maps
regression names to mode-specific overrides; names never select hidden behavior.
Non-screened paired searches use two large-run GPU lanes and the remaining lanes
for the supplied regression suite. `gt_gpu_manager` optionally overrides regression
resource requests; otherwise those come from each supplied regression request.

## Diagnostics, visualization and export

| Workflow/API | Functionality |
|---|---|
| `fiber_sign_rejection_report` | Compare supplied `roots` against `baseline`; `internal_audits` maps each root to its audit summary path. No implicit grids or audit-directory discovery. |
| `fiber_density_sign_report` | Internal or frozen-GT sign counts, distributions and detailed error breakdown. |
| `fiber_implicit_match_report` | Same-sheet meeting metrics over probability cutoffs. |
| `fiber_reference_rounding_audit` | Floating vs integer scoring and alternative references. |
| `fiber_forced_selectable_audit` | Exact parent-geometry/hypothesis transfer and full objective comparison on selectable graphs. |
| `paired_winding_review` | Matched-cohort changed-winding plots, dotted fibers and sign-neighbor highlighting. |
| `fiber_direction_review` | Direction disagreement audits and OBJ exports; `normal_visualization` configures the shared normal-arrow exporter. |
| `fiber_hv_label_comparison` | Compare an H/V-stage `solution` NPZ with `fibers` source family labels. Writes JSON to `report`; optional `per_fiber_output` NPZ and `probability_cutoffs`. Reports count/arc disagreement and confusion by family, keeping normal-stage and H/V-stage exclusions separate. Source labels are comparison-only. |
| `fiber_uv_refinement_inspection` | Matched before/after deformations, movement in voxels and windings, plots. |
| `fiber_fixed_metric_probe`, `fiber_optimizer_probe`, `fiber_lbfgs_audit` | Fixed-meeting/length/angle probes and optimizer/conditioning comparisons. Fixed-metric rendering is opt-in via `visualization`. |
| `fiber_uv_surface` / `fiber_uv_surface_audit` | H-anchored tifxyz surface construction and audit from historical categorical U/V solutions. |
| `fiber_reference_surface` | Explicit S3 reference acquisition with integrity/provenance checks. |
| `fiber_pair_connectivity` | Existing two-fiber threshold/component/skeleton diagnostic with reference strips and OBJ export. |

Reference downloads may resume or refresh the same S3 prefix. Reusing an output
directory for a different source, or leaving local files absent from the current
source listing, is rejected before downloading. Use a new/clean output directory;
the downloader does not delete old reference files automatically.

Surface usage:

```json
{
  "fiber_uv_surface": {
    "solutions": [{"label": "candidate", "run_root": "/data/fiber-layout/uv-solve"}],
    "normal_curves": "/data/fiber-layout/normal-curves",
    "coordinate_space": "YOUR_VOLUME/YOUR_SCAN",
    "output_root": "/data/fiber-layout/surfaces",
    "grid_step": 100, "max_search": 1000, "joint_threshold": 0.5
  }
}
```

This exporter retains its original checkpoint contract; it does not silently
reinterpret a fixed-length/probability-model checkpoint as categorical geometry.
The separate `fiber_exact_deformation` and `fiber_uv_surface_refinement` workflows
retain their own checkpoint schemas.

Extra read-only tools have module CLIs (use `--help`):
`fiber_constraint_database_report`, `fiber_constraint_database_validation`
(requires `--fibers`, the source fiber store, and the matching C++ header), `fiber_oriented_normals_validation`,
and `fiber_violation_neighborhoods` (consumes `paired_winding_review` output).

Performance modes restored in `fiber_uv_layout`: `benchmark_pair_fusion`,
`diagnostic_benchmark`, `distributed_suite` and `experiment_matrix`.
`indexing_audit` is restored in soft/density optimization. They reuse the normal
solver algebra; enable them explicitly. No new speed or convergence claim is made.

## Local archive and verification

### Whole-fiber winding-range selection with minimum-cost flow

`fiber_flow_uv` fits conditional U/V geometry after flow selection. It takes
`flow_root`, `umbilicus`, `output_root`, `devices`, and `gpu_manager`. Fibers with
no perpendicular events are explicitly removed and recorded in `selection.json`;
neither remaining integer labels nor enabled status are optimized. A common
integer gauge shift puts width lookup indices in range; all relative assignments
are unchanged. The conditional targets at event bases include the flow's exact
seam transport, not rounding of evolving U/V coordinates.

This mode reuses `SharedCurve` straight fixed-length geometry, monotone widths,
width regularizers, the scaled PyTorch L-BFGS optimizer and TIFF/PDF drawing sinks.
It does **not** replay the probabilistic assignment objective: fixed integer
event separations supervise approximate cross-winding coordinates, while fixed
same-sheet perpendicular events additionally supervise physical U/V meetings.
Parallel constraints supervise transverse coordinates only. Angular/Z priors
are sampled along curves; radii initialize widths but never provide a width prior.
`meeting_weight`, `meeting_scale`, `distance_weight`, `metric_weight`,
`prior_weight`, `v_prior_weight` and `width_smooth` control this conditional fit.
There is no free bending or length change in this first geometry mode.

`initial_solution` continues a compatible fixed-label fit after checking every
fiber/target identity. `visual_initial` optionally renders the original initial
checkpoint alongside a continued solution. Outputs include `fibers.vcobj`, tensors,
`solution.npz` (fixed labels, widths, shared U/V anchors), complete loss/mismatch
history, and equal-scale `visualization/{initial,solution}.{tif,pdf}`. TIFFs use
LZW compression. Re-running an identical completed request only exports/renders;
it does not allocate a GPU again.

The `fiber_min_flow_selection` workflow combines an oriented constraint database
with final normal/HV decisions and a small intact GT fiber set. Required settings:
`fibers`, `database`, `normal_solution`, `comparison`, `gt_root`, `umbilicus`,
`extract_template`, `orient_template`, `native_binary`, `output_root`, `z_range`.
Optional settings include `threads`, `seed_spacing` (64), `maximum_distance` (128),
`max_rounds` (10), `perpendicular_relative_weight` (1),
`perpendicular_offset` (0.5), and `lasagna_winding_scale` (1).

It finds spatial GT neighbors, extracts their GT cross-events using the existing
native workflow, and reads source/source measurements through the shared UUID-indexed typed graph reader.
Each round jointly solves candidates, filters to GT-connected fibers intersecting
the GT H winding range within the Z slab, then jointly re-solves the filtered set.
Subsequent rounds include one-hop eligible neighbors. All output curves remain
whole; only the selection test is restricted to the slab. The flow solver has no
probability/disabling model: upstream normal/HV rejections apply, and all seven
(or however many supplied) GT fibers remain enabled. No SGD confidence threshold
is substituted. Stable membership or the explicit round limit ends expansion;
a monotone prune/re-solve cleanup then ensures the exported set satisfies its
own final range/connectivity test. `finalize_only: true` can finish this cleanup
from an existing `selected_tensors.npz` and `summary.json` without re-expansion.

`history.json` records counts, timings, geometric-order violations and certified
flow objectives. Its `measured_target_l1` is fit to calibrated measured targets,
**not** error against independent GT winding labels. `selected.vcobj` contains
original source IDs/geometry and the resolved H/V family labels;
`selected_with_gt.vcobj` includes GT using its original UUIDs; `gt_id_mapping.json` records the source association, not a synthetic integer identity. Matching tensors and integer solution arrays are saved
separately. Original source files are read-only.

An optional `frozen_gt_cross` directory can reuse historical signed GT normal
samples only after checking the GT fiber store, normal solution, tensors and solution
hashes against its `inputs.json`; this avoids silently changing the GT normal
orientation when extraction implementations have evolved.

The `fiber_min_flow_full` workflow solves all upstream-eligible source fibers,
without a spatial crop or added GT. Required settings are `fibers`, `database`,
`normal_solution`, `comparison`, `umbilicus`, and `output_root`.
`experiment_link` optionally creates a convenient output-directory symlink.
Preparation uses `workers` (32), `geometry_batch` (4096 fibers),
`constraint_batch` (500000 database IDs), and `read_batch` (32768 rows).
The same `perpendicular_relative_weight`, `perpendicular_offset`, and
`lasagna_winding_scale` settings as selection control the shared flow objective.
Set `prepare_only: true` to publish the packed geometry and `graph.npz` without
starting HiGHS. The summary then records `prepared_only: true` and no solution is
written. This graph can be passed as `source_root` to `fiber_irls_benchmark` with
`full_graph: true` for an IRLS-only solve of its largest connected component.

Exact original polylines and seam transport are packed into memory-mapped arrays;
workers stream disjoint constraint-ID ranges. Only identical L1 arcs are combined,
preserving fractional-target terms. Preparation blocks are resumable with matching
input identities and configuration; the optimizer itself restarts if interrupted.
`progress.json` tracks preparation/solve state, `solver.log` retains native solver
progress, and `summary.json` is written after completion. `solution.npz` contains
source IDs, resolved family, integer winding, flow and connected-component IDs.
Component winding origins are arbitrary and normalized independently. This is a
CPU solve with no GPU request; full-volume optimizer time need not scale linearly
from cropped runs. Source fiber/graph stores remain read-only.

`fiber_flow_threads` benchmarks the saved full-volume graph without extracting
anything again. Set `source_root`, a fresh `output_root`, optional
`experiment_link`, `thread_counts` (default `[1,2,4,8]`) and `iterations`
(60000 per trial). It presolves once, benchmarks serial dual simplex and parallel
PAMI from the same reduced LP, then continues the selected trial's basis to
completion. Both pivot throughput and logged objective improvement per second
are recorded in `trials.json`; selection uses objective improvement per second
so unproductive cheap pivots do not automatically win. Native progress intervals
provide min/median/p95/max pivot rates, not independent repeated-run confidence
intervals. The installed HiGHS simplex concurrency is capped at eight even if
more scheduler threads are requested; both are recorded. The best basis is
checkpointed to `best.basis`. The final solution must pass the same original-graph integer
primal/dual certificate as the ordinary solver.

This workflow uses SciPy's private bundled HiGHS interface for presolve/postsolve
and basis retention; compatibility is covered by backend-equivalence tests and
must be checked when updating SciPy. It runs synchronously and must not share a
process with another active HiGHS solve (thread scheduler is global). Optional
`stop_previous_pid` plus exact `previous_command` can replace an earlier owned
workflow after preparation succeeds; process identity checking uses Linux
`/proc`. Omit these options elsewhere. Benchmark logs, selection and final solve
log are retained separately; it never overwrites the source graph or old logs.

### Optional specialized integer-capacity flow benchmark

`fiber_lemon_benchmark` compares LEMON network simplex and cost-scaling on an
already solved dataset. It requires the optional system LEMON development package,
`pkg-config`, and a C++17 compiler; it never installs these automatically. The
explicit benchmark builds its small native adapter in the output directory with
Release optimization and records source hash, compiler and LEMON versions.
No GPU is required. Configure, for example:

```json
{
  "fiber_lemon_benchmark": {
    "output_root": "/results/flow-backend-comparison",
    "dataset": {
      "fibers": "/data/fibers.vcobj",
      "tensor_cache": "/data/tensors.npz",
      "umbilicus": "/data/umbilicus.json",
      "perpendicular_relative_weight": 1,
      "perpendicular_offset": 0.5,
      "lasagna_winding_scale": 1
    },
    "baseline_solution": "/data/solution.npz",
    "baseline_certificate": "/data/certificate.json",
    "scales": [1000, 1],
    "repeats": 3
  }
}
```

Fractional target expansion is unchanged. Identical arcs are aggregated first,
then capacities become `floor(scale * weight + 0.5)`; zero-capacity arcs are
dropped, not clamped to one. Scale 1 is direct integer rounding, while larger
scales preserve finer weight distinctions. This deliberately changes the objective
and is not the earlier variant that rounded measured winding targets.

`results.json` records dropped arcs/weight, repeated native graph/setup/solve/export
timings, complete certification wall time, and original-weight L1 and sign metrics.
Each result must pass the existing primal/dual certificate for its quantized graph
and match HiGHS on that exact graph. Original-weight scores are evaluated separately;
certification of the quantized graph is not certification of the original objective.
Overall mean L1 divides by the total contributing weight, including valid unsigned
constraints; sign diagnostics necessarily include only known-sign constraints.
If quantization disconnects components, comparison gauges are explicitly aligned
to the baseline and cross-component original events are reported.

Run the optional native regression tests with:

```sh
python -m pytest -q vesuvius/tests/test_fiber_lemon.py
```

They skip if the optional compiler/package is unavailable; no setup is performed.

### Full cached-graph LEMON solve

`fiber_flow_perpendicular` creates a perpendicular-only experiment from a full
graph and calls the same LEMON runner. Configure `source_root`, `output_root`,
`minimum_perpendicular_constraints` (default2), `scale`, and `reference_mean`.
It relies on the full builder's H/V compatibility filter: H–V arcs are
perpendicular, same-family arcs parallel. Before capacity rounding, summed
fractional-target arc weights divided by the original perpendicular event weight
recover original constraint counts. Integrality is checked, not assumed.
Fibers below the minimum are recursively removed until the remaining induced
graph satisfies it. Multiple measured events to the same neighbor count
separately; target-expansion arcs do not. Selection counts and original IDs are
saved. The derived `geometry/catalog.npz` contains identities only, not a new
packed-geometry copy; use original packed geometry for subsequent evaluation.
This changes the objective/cohort, so a normalized factor against a mixed
parallel/perpendicular reference is only a diagnostic, not a like-for-like score.
Optional `minimum_length_base` filters by strict full-polyline arc length before
recursive degree pruning, using the original packed-geometry arc coordinates.
`fiber_flow_length_sweep` accepts `output_root`, optional `experiment_link`,
`lengths_base`, `workers` and a `common` perpendicular-workflow configuration.
It executes independent trials in a bounded process pool. Optional
`common.reference_evaluation` selects cached reference scoring after each solve;
its `comparison_solution` can name the prior no-length-filter solution for
matched-cohort comparisons. Errors are recorded per trial in `results.json`.

`fiber_lemon_full` reuses `graph.npz` and `geometry/catalog.npz` from a completed
full-volume preparation. It does not repeat extraction or run HiGHS first.

```json
{"fiber_lemon_full": {
  "source_root": "/results/full-preparation",
  "output_root": "/results/cost-scaling",
  "scale": 1,
  "reference_mean": 0.27,
  "progress_interval": 20
}}
```

Choose `reference_mean` from the actual small reference report, not the example
above. Capacity quantization is explicit and reported, including discarded arc
weight. Progress and final L1 use **all original arc weights**, including arcs
discarded by quantization. `factor_vs_small` divides that normalized L1 by the
supplied reference mean. Intermediate snapshots evaluate rounded current node
potentials: they are not the intermediate flow objective printed by HiGHS and
are not a convergence percentage. Excess imbalance is separately recorded in
quantized capacity units. Final output requires the ordinary primal/dual,
conservation and capacity certificate for the quantized problem.

The optional progress build generates a local instrumented copy of the installed
LEMON header, preserving its license and algorithm. It inserts read-only hooks
at cost-scaling phase starts and periodically within partial augmentation; it
does not edit system headers or change solver arithmetic. Header source hashes
are recorded, and incompatible layouts fail closed. Reporting is same-thread
and time-throttled, so no unsynchronized access to solver state occurs. A long
internal heuristic may delay a snapshot. Scanning the original graph has some
overhead; no per-arc Python callbacks are used. The backend remains single-core.

### Reference evaluation of cached whole-fiber flow solutions

`fiber_flow_reference_eval` accepts `geometry` (the packed geometry directory),
`solution`, and `output_root`. Optional `gt` supplies `config` (GT tensor/fiber-store configuration), `reference` (explicit frozen offsets/exclusions), and `events`
(cached cross-constraint JSON). It reuses the existing sign evaluator and reports
both single-global-offset and explicitly separate component-aligned diagnostics.
`spiral` supplies `reference` (tifxyz), `xyz_scale`, `surface_stride`, `workers`,
`candidates`, `sample_spacing`, `max_distance`, and `fiber_batch`.
Set `allow_solution_subset: true` for a component-only solution drawn from the
full geometry catalog; source IDs and family labels are validated before mapping
to the original geometry. Set `continuous_gt_protocol: true` to additionally
report the same continuous-endpoint, single-global-integer-offset GT metrics as
the IRLS comparison workflow, alongside the explicit-integer reference metrics.

The spiral check samples all fibers uniformly in arc length, including endpoints,
and reuses the nearest-surface projection implementation. Continuous fitted
winding combines the saved integer label, exact polyline seam transport, and
local umbilicus phase. The primary error uses the arc-weighted median pointwise
reference-minus-fit per fiber, then one global median translation. It does not
align disconnected components independently. Reports include coverage, L1,
within-two-windings percentage, largest-component and half-length-covered subsets,
and the historical median-coordinate metric separately. This is approximate
reference association, not optimization or a certificate of reference correctness.
Projection batches are cached for resuming identical requests.

`fiber_flow_reference_analysis` reads `evaluation_root`, writes `output_root`,
and accepts `winding_band_width` (default10). It reports absolute-error
histograms, quantiles, reference-winding bands and low-error subset counts.
Per-band median removal and a held-out linear drift fit are diagnostic only:
they never change the saved solution or replace the single-global-offset score.
It also exports `per_winding.md` and `per_winding_{all,main_component}.csv` using
nearest-integer reference winding IDs. Each row includes the L1-optimal additional
shift, total shift from raw labels, support count, pre/post-shift L1 and residual
p90. Positive additional shifts increase fitted winding. These reference-derived
regional shifts are diagnostic only; they are not applied to the saved solution.

`fiber_flow_reference_rescore` accepts `evaluation_root` from a prior full
projection, a new whole-fiber integer `solution` on the same IDs or a subset,
and `output_root`. It reuses reference geometry exactly: changing a fiber's
constant integer label translates all its projected residuals by that constant.
It validates source identities and H/V classifications, recomputes GT signs,
and reports old/new scores on the identical retained cohort, each with one
global offset. It is not suitable for solutions that change fiber geometry.

`fiber_flow_stop` accepts an explicit `pid` and exact `expected_command`; it
reuses the guarded process-stop helper and never terminates a pattern of jobs.

### Optional parallel network-simplex benchmark

The `fiber_pns_benchmark` workflow compares the external MIT-licensed
[PNS implementation](https://github.com/gokcehan/pns) against a completed
`fiber_lemon_benchmark` report. Supply the upstream `pns.cpp` explicitly; the
workflow does not download or install dependencies.

```json
{"fiber_pns_benchmark": {
  "source": "/dependencies/pns/pns.cpp",
  "comparison_root": "/results/lemon-comparison",
  "output_root": "/results/pns-comparison",
  "thread_counts": [1, 2, 4, 8, 16],
  "scales": [1000, 1], "repeats": 3, "time_limit": 10,
  "avx2": false
}}
```

The adapter reuses upstream initialization and pivot routines. Signed flow
bounds `[-q,q]` become `[0,2q]`, with corresponding node supplies; recovered
flows and potentials receive the same primal/dual, conservation and bound
checks as the other backends. This bypasses the upstream printer, which omits
nonpositive-cost arcs. Every completed result must match the prior certified
quantized optimum. Time-limited trials are marked incomplete, never optimal.
The limit is checked every 64 pivots and may overshoot. Reports include repeated
timings, pricing/tree/potential-update profiles, original-weight metrics and
compiler/source hashes. OpenMP pivot tie-breaking may select different optima.

This optional backend requires an OpenMP C++17 compiler and Boost headers.
Upstream unconditionally includes x86 intrinsics, so other architectures are
explicitly rejected without affecting the portable default backends. AVX2 is
opt-in and CPU-checked. Conservative int32 flow/potential range guards reject
unsupported inputs rather than truncating them; this is not yet an unrestricted
full-volume backend. Parallel pricing does not parallelize every pivot step.

```sh
FIBER_PNS_SOURCE=/dependencies/pns/pns.cpp python -m pytest -q vesuvius/tests/test_fiber_pns.py
```

### Parallel measured-prior preparation

The predicted-only managed L-BFGS solver accumulates independent edge, phase-prior,
Z-prior and global-regularizer objective parts. `lbfgs_prior_batch_size` defaults
to `lbfgs_edge_batch_size`; original sample weights and full-fiber denominators
are preserved. Each part's gradients are accumulated before releasing its graph,
and priors/globals are evaluated once per full objective, not once per edge batch.
This changes floating-point reduction order, not the mathematical objective.
`lbfgs_objective_progress_every` optionally reports first-objective batch progress.

Geometry initialization uses sorted, segmented exact-location topology for zero
merge distance and batched lexicographic interpolation queries rather than
whole-array scans per fiber. Nonzero-distance greedy merging retains its original
semantics. Exact-location duplicates use their shared coordinate directly, avoiding
roundoff from averaging repeated identical coordinates. Straight fixed-length
queries do not construct unused knot-interpolation indices or packed query copies;
the same physical winding/arc evaluation and losses are retained. Startup logs
separately report data loading, geometry construction and query-plan readiness.

`fiber_soft_winding.prepare_samples_geometry` accepts `prior_sample_workers`
(default 1) and `prior_sample_batch` (default 512). It batches all interpolation
queries for each fiber in native NumPy, and uses a bounded process queue when
workers exceed one. Results retain input fiber/sample order and identical weights.
Linux workers share read-only preparation arrays through fork; other platforms
can use spawn, which copies worker initialization state and requires more memory.
Run this CPU preparation before initializing CUDA. The full-volume SGD workflow
defaults to 64 prior workers; `prepare_only: true` stops before optimization and
records compute/save times in `prior_sample_timing.json`. Sample output is published
by atomic rename so interrupted writes cannot be reused as completed caches.

Only reusable implementations, specifications, synthetic tests and minimal usage
examples belong in the shared tree. Dataset/scan names, snapshot IDs, historical
trial names, tuned winner configurations and experiment-result tables do not.
Keep those artifacts externally, for example under ignored `experiments/docs/`
and `experiments/data/`. Workflows must take experimental choices as explicit
inputs; report metadata must describe those inputs, not a historical experiment.
Changing a request requires a new output root when provenance checks reject reuse.

```sh
python -m pytest -q vesuvius/tests/test_fiber_*.py vesuvius/tests/test_paired_*.py
```

Tests require the configured Vesuvius import path. GPU tests skip without CUDA;
the two-rank CPU Gloo test needs local socket access. Preparation/extraction and
full GPU solves require user-provided data and native binaries. Old tests that
required the deleted width prior/fallback geometry were updated to assert the
current rejection contracts, not used to reintroduce those models.

Read-only real-checkpoint regression (recomputes every full-objective term as well
as all historical scoring metrics):

```sh
FIBER_LAYOUT_REFERENCE_SPEC=/configs/fiber-layout/benchmark.json python -m pytest -q vesuvius/tests/test_fiber_workflows_real_data.py
```

## Route-page import

`fiber_stream_import` imports completed `fiber-stream-attributes-v2` tile exports.
Set `tiles`, `output_root`, `frame_template`, and optionally `workers` (16).
Optional `z_range: [lower, upper]` selects whole fibers intersecting that slab;
omit it or use null for the full volume. The importer verifies completion,
coordinate-space identity, unique source IDs, and source counts, preserves the
native base-voxel geometry and stable source-derived UUIDs, and sets family to unknown for independent
H/V classification. Tile decoding is parallel with bounded outstanding work;
the shared initial-fiber API constructs line-only records for the native UUID-routed bulk importer. The Python adapter currently materializes these records before native import; its memory/time is separate from native shard throughput.

`fiber_route_import` converts native `vc-fiber-routes-v4-reduced` page exports
into packed `fibers.vcobj` using the shared initial-fiber API and native bulk importer. Configure `routes`, `output_root`,
`frame_template` (a vcobj fiber store with the independently verified same coordinate frame),
`z_range`, and optional `workers`. It keeps whole intersecting routes, preserves
`points_base` without rescaling, and does not read component indexes as fibers.
Family labels are unknown until the normal/H/V preparation stages are run.

## Registered AFV comparison

`fiber_afv_transform` applies a supplied XYZ affine registration to AFV curves.
Here `afv` is deliberately an external source: it is read through native `AfvSource`, not the removed Python AFV backend. Output is a packed line-only `fibers.vcobj` store with stable UUIDs derived from source identity and source fiber IDs. Set `afv`, `transform` (published `transformation_matrix` JSON), `invert`,
`fixed_volume`, `source_coordinate_space`, `target_coordinate_space`,
`frame_template`, `z_range`, and `output_root`. It validates frames, preserves
source-derived UUIDs and whole polylines intersecting the target slab, and recomputes
native-coordinate bounds and lengths. Source annotations remain attached to each imported fiber; the import report records source identity and transform checksum; original H/V labels remain available for
comparison with independently inferred labels.

After the normal/H/V and `fiber_min_flow_selection` workflows,
`fiber_selected_solver_comparison` consumes `selection_root`, `base_config`
(the independent winding model), `gt_root`, `umbilicus`, the hash-bound external
IRLS `solver_source`/`solver_sha256`, `devices`, `gpu_manager`, and `output_root`.
GT fibers are removed before solving. All methods use the same largest
perpendicular-supported connected component, with exclusions reported explicitly.
The existing independent winding model is initialized from radial geometry, not
selection-time winding assignments. LEMON and IRLS use the same fractional-edge
graph; reports include both all-event and common SGD-enabled GT cohorts.
Set `include_sgd: false` to run only LEMON/IRLS winding assignment: no SGD
checkpoint is required, and scores use all eligible GT events (no SGD-based
filter). `exact_lp: false` also skips the optional LP control solve. Both flags
default to true to preserve existing three-way recipes. `prepared_root` can
reuse completed preparation in a new output directory after checking selection,
reference, initialization-config and coordinate-axis bindings; it does not reuse
the old winding solution. Keep the old run directory intact for provenance.
`uv_parameters` controls the separate fixed-label geometry fit, and `uv_methods`
selects `lemon_1000` and/or `irls`. These produce equal-axis-scale TIFF/PDF UV maps.
Keep dataset-specific recipes and outputs in the external experiment archive.

To fit/render an existing graph solution without rerunning winding assignment,
use `fiber_flow_uv_saved` with `tensor_cache`, `solution` (NPZ containing matching
`source_id`, integer `winding` and `component` arrays), source `fibers`, `umbilicus`,
`output_root`, `devices`, `gpu_manager`, and UV `parameters`. It selects the saved
largest component, converts aligned endpoint arcs back to unsigned input arcs,
and calls the same fixed-label UV workflow. Input hashes are recorded; labels
are not re-estimated. Only whole unsplit fibers are supported.

`fiber_dataset_union` combines prepared independently classified cohorts. Supply
`sources` (each with `name`, `fibers`, `tensor_cache`, saved graph `solution`, frozen
GT `events`, and `normal_solution`), native extraction/orientation templates,
`native_binary`, `output_root` and nested `comparison` settings. Original UUIDs are preserved; duplicate UUIDs across input cohorts are rejected, and coordinate frames must agree.
Within-source constraints and reference events are retained exactly. Native
extraction plus the existing local normal sweep supplies only new cross-source
constraints, using the inherited inward normal signs and family compatibility.
This requires unchanged source geometry and the same normal sampler/sweep settings
as the original classification. No deduplication, GT-in-solve, winding warm start,
or reclassification occurs. The saved solution selects membership only; its
winding labels are not passed to the combined solve.
# Enclosure-only constraints and frozen-reference normal signs

The `fiber_eval_enclosures` workflow imports a directory listing of annotated
fiber JSONs, preserves full `line_points`, and uses the existing native local-event
extractor and fiber-local normal sweep. It does not fit winding widths or UV.

Set `sign_alignment: "reference_constraints"` and provide
`reference_constraint_alignment` (the `fiber_reference_constraint_signs` workflow
configuration). The reference specifies its `fibers`, original oriented `database`,
normal `solution`, and prepared `geometry` cache. Only new signs are fit.
The existing native line/normal construction exports frozen normal curves,
validating their gauge against the original constraint endpoints. The native
extractor's `cross_query_fiber_ids` mode explicitly selects target UUIDs and queries new lines against reference lines only, in parallel, with no reference-reference search. Spatial reference selection
uses conservative block bounds, without nearest-fiber or per-fiber sample caps.
The shared `CurveBank` interpolates both frozen normal curves at actual extracted
constraint base points. A single batched dot-product reduction independently fits
each new sign. `minimum_abs_dot` filters weak evidence and `minimum_agreement`
leaves contradictory signs unresolved. Perpendicular evidence is the default,
matching the original sign stage; `include_parallel` explicitly adds parallel
evidence. Reference signs are never reoptimized. The slow Python nearest-endpoint
lookup implementation was removed; there is no per-sample SQLite normal lookup.

`constraint_mode` selects `ordinary`, `additional`, or `only`. Ordinary native
events are always saved unchanged in the raw/oriented databases. Additional
relations have `constraint_type="perpendicular_enclosure"`, one exact integer
anchor step per H/V pair and two witness event indices into `ordinary_tensors.npz`.
The local relation is same winding from the inner H encounter to its backing V.
The anchor step includes full polyline seam transport AND connector transport.
Neither measured Lasagna distance nor an estimated winding assignment is used
to establish these exact bounds. Opposite signs alone are insufficient: maximum
lower bound must equal minimum upper bound. Contradictory and nonunique pairs are
reported separately. Connected-component outputs include isolates and audit
integer cycle consistency without silently deleting conflicting relations.
Saved `relative_winding` values are a spanning-tree audit gauge, not a solved
assignment for components with conflicts. Disconnected components have independent
integer offsets; the enclosure-only graph does not determine their mutual placement.

Required configuration also includes `url`, `output_root`, `frame_template`,
`umbilicus`, `extract_template`, `orient_template`, and `native_binary`.
An explicit `allowed_source_shape_zyx` permits a known annotation metadata shape
discrepancy only; coordinates are never rescaled or clipped. H/V uses manual tags
where present, otherwise saved automatic tags. Fibers without tags are reported.

Run through the normal `fiber_workflows --config FILE` entry point (or a locally
approved fixed runner). Output `graph.npz` is the selected tagged graph;
`enclosures.json` retains witnesses/rejections; `enclosure_components.npz` and
`results.json` describe the enclosure graph independently of ordinary edges.
# Enclosure-component reference checks

`fiber_enclosure_reference_eval` compares a saved enclosure graph with a frozen
full-volume IRLS solution and a reference tifxyz surface. It does not run a new
solver or modify either reference. Supply `enclosures`, `cross_root`,
`irls_solution`, `irls_geometry`, `reference_normal_solution`, `umbilicus`,
`spiral` (the usual projection configuration), and `output_root`.

The evaluator reuses native new-to-reference cross events and frozen signed
normal caches. It reports perpendicular/parallel distance L1, rounded-step
agreement and geometric-order violations against IRLS, plus arc-weighted
surface winding errors, projection coverage and internal constraint checks.
One L1-optimal translation is fitted per component/reference (integer for the
integer IRLS realization, real for continuous spiral coordinates);
there is no rescaling or per-fiber correction. Internal pair differences are
offset-invariant. Contradictory enclosure components are explicitly flagged:
their spanning-tree potentials are diagnostic, not a consistent solution.
IRLS cross-event metrics and spiral projection metrics have different cohorts
and are not an apples-to-apples ranking of the references. Full counts and
definitions accompany the JSON/Markdown output.
