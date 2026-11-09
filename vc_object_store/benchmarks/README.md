# Prepared-input initial-import benchmark

`vc_bench_prepared_import` diagnoses serial and sharded import overhead independently of source decoding. It uses the existing C++ `AfvSource`/`AfcSource` readers, loads native records and little-endian float64 geometry buffers into RAM, and finishes source UUID generation before any import timer. It does not resample, round, bundle constraints, invent control points or change application import semantics. A separate on-disk binary cache is unnecessary for this test: all repetitions reuse fully materialized C++ records.

Build with `-DVC_OBJECT_STORE_BUILD_BENCHMARKS=ON`, then build target `vc_bench_prepared_import`. It requires only the core object-store dependencies, not adapters, Python or GPU support.

Example local configuration (paths are supplied by the caller):

```json
{
  "source": "/path/to/fibers.afv",
  "constraints": "/path/to/constraints.sqlite",
  "output_root": "/path/to/disposable-fixtures",
  "workload": "relations",
  "count": 100000,
  "variant": "native_packed",
  "repetitions": 3
}
```

Run `vc_bench_prepared_import config.json`. The output directory must already exist and must be disposable. The benchmark refuses existing destination files. A supervising script should impose a 30-second hard limit and remove its own private fixture directory on both failure and success. Keep persistent configuration and JSON-line reports outside that directory. Use a RAM-backed fixture directory to isolate CPU/import overhead from physical disk throughput; final publication to RAM-backed storage is not a durable-disk throughput measurement.

Use a matrix to prepare the input once for all variants and repetitions:

```json
{
  "source": "/path/to/fibers.afv",
  "constraints": "/path/to/constraints.sqlite",
  "output_root": "/path/to/disposable-fixtures",
  "workload": "combined",
  "count": 10000,
  "max_prepared_xyz_bytes": 1073741824,
  "repetitions": 1,
  "cases": [
    {"variant": "prototype"},
    {"variant": "native_packed"},
    {"variant": "native_packed_sharded", "workers": 8},
    {"variant": "native_packed_sharded", "workers": 16}
  ]
}
```

Each case may override only `variant`, `workers`, `sql_batch_rows`, `repetitions`, `label`, `export_workers`, `pre_size_export`, `resident_bytes`, and `sqlite_cache_bytes`; source data cannot silently change between cases. Prototype JSON/binary-ID preparation runs once only if the matrix actually contains a prototype case. Native-only matrices do not serialize the entire source into prototype JSON. Completed private output files are removed after their size and any explicitly requested diagnostics are recorded, so RAM usage does not accumulate across the matrix; `remove_completed_outputs=false` retains them until the supervising runner cleans its fixture directory. Preparation, all cases, inventory and readback must fit the same runner invocation budget (28 seconds before termination, with a two-second kill grace in the local fixed runner). Start with a small input and one repetition; do not infer that the example count is universally fast for long fibers.

Workloads:

- `relations`: the first `count` individual AFC constraints and their distinct endpoints as typed opaque fiber handles. All constraints, original metadata and anchors remain present; there is no pair bundling. This intentionally isolates graph import, not geometry or hydration of rich parent metadata.
- `combined`: the first `count` AFC constraints and every distinct endpoint's complete original AFV geometry, name, annotation and detail fields. Preserved source IDs identify the exact fibers; a native bounded-ID cursor reads only those fibers, merging consecutive IDs into one interval. `AfvSource::selectRange` reuses the same validated snapshot and prepared statement across intervals, without repeatedly opening or validating the source. Every required fiber is decoded once and reused by every case. Missing geometry or inconsistent source identities fail explicitly. The import contains both full geometry and relations, not opaque stand-ins.
- `geometry`: `count` AFV fibers, with original line coordinates, source names, annotations and detail fields. `geometry_sampling="prefix"` (default) selects the first fibers. `geometry_sampling="uniform"` calls the native catalog partitioner once and reads the first fiber in each balanced catalog interval, providing deterministic coverage across the volume without scanning intervening XYZ payloads. It reuses one `AfvSource` snapshot/cursor via `selectRange`; reports distinguish reader count from interval count. Sampling selects whole fibers; it never resamples their coordinates. Empty control-array descriptors carry no points/chunks. All variants use the native chunk iterator and content-hash implementation.

For a matched comparison against `vc_object_import_afv`, use `workload:"geometry"`, `afv_records:true`, and the same `volume_id`/`scale_factor`. This uses the shared AFV record converter, including coordinate/source metadata and the collection record, instead of the older benchmark-only annotation layout. `count` remains the fiber count; reported objects include one extra collection object. Optionally add `include_constraints:true`, `constraints`, and a positive `constraint_limit`: this imports the first eligible constraints whose endpoints both lie in that fiber prefix, exactly like the streaming command with the matching `fiber_limit` and `constraint_limit`. This option requires prefix geometry sampling and reports actual constraint counts; it does not change the existing `combined` workload's constraint-first selection.

Both paths use the same `ParallelImportStream` queues, work stealing, UUID routing and native shard writes. Prepared objects/XYZ are borrowed without copying, while edges are batched in parallel. Fixture materialization remains a separate serial benchmark setup step, not a production importer. `uuid_partition_preparation_seconds_excluded` now measures only temporary benchmark-ID bookkeeping; UUID routing and queue work are included in ingestion. Older reports excluded serial payload partition/copy. Compare prepared `source_seconds + uuid_partition_preparation_seconds_excluded + total_seconds` with streaming `end_to_end_seconds - validation_seconds` for complete processing, or compare native ingestion separately while explicitly noting that streaming ingestion includes producer waits. Both report `peak_rss_bytes` (process high-water mark; null on unsupported platforms). In a multi-case prepared process, later cases inherit the process's earlier high-water mark. Prefix geometry preparation additionally reports source open, selection, decode/validation and record-construction wall times. Those detailed fields do not include optional constraint decoding; that remains included in `source_seconds`. Warm-source and first-read results should not be mixed into one speedup claim.

`max_prepared_xyz_bytes` defaults to 1 GiB and rejects preparation above that total; it does not truncate fibers or drop constraints. A single fiber must still be decoded before this guard can inspect its bytes. Reports include actual points, XYZ bytes, native geometry reader count, and the number of position/whole-geometry/indexed anchors. AFC currently supplies position plus whole-geometry anchors, not stable point-index anchors; the benchmark preserves that distinction and does not manufacture stronger measurements. Prefix fibers can be much longer than typical volume fibers, so do not project geometry cost from fiber count alone.

`AfvSource::selectRange(first, pastLast)` explicitly abandons any unfinished previous scan and resets that scan's counters while retaining its SQLite read snapshot. Each returned fiber still receives all native validation. Declared whole-collection counts are checked only when a full `[0, INT64_MAX)` selection reaches EOF; a sampled interval does not pretend to validate unread fibers. Tests cover repeated seeks/EOF, invalid intervals, reset accounting and snapshot consistency across concurrent source writes.

Variants:

- `prototype`: diagnostic reconstruction of the earlier five-table immutable layout: objects, fibers, edges, geometry chunks and object-to-chunk mappings. Original record attributes are pre-serialized into opaque JSON payloads; this serialization is reported separately. It uses 64-KiB pages, journal/sync off, append-time foreign keys, one transaction and final endpoint indexes. **This is not a supported vcobj format or an application writer.** It has no provenance columns, approval logic, lifecycle bindings, geometry manifests, mutation-dependency index, collection routing or v2 generation machinery.
- `native`: current `importInitialCollection`, one ordinary SQLite file, history/spatial off, all native correctness semantics retained. Private immutable builders disable journaling/sync during construction and durably publish at finalization. They use 64-KiB SQLite pages. SQL statements use the existing default batch size of one within 256-record API batches, not one transaction per record. Optional `sql_batch_rows` tests larger SQL batches without changing API batch size or data.
- `native_packed`: the same API with disposable RAM-first packed VFS and background flushing disabled. `resident_bytes` defaults to 4 GiB; exhaustion fails rather than silently spilling. `export_workers` defaults to 1 and controls final packed payload export independently of importer workers. `pre_size_export` defaults to false; true sizes the compact target to its computed payload length before export. These settings may appear at the configuration root or in individual cases, and map directly to `PackedStoreOptions::residentBytes`, `exportWorkers`, and `preSizeExport`. Final export/publication is reported separately. Results echo the settings and `packed_export_options_applied`; plain/prototype cases do not apply them.
- `linked`: publish the prepared handles as a sealed dependency first, then import constraints referencing it. Report dependency construction separately; do not compare the sum against an import-only number without saying so.
- `native_sharded` / `native_packed_sharded`: phased construction with independent workers (`workers`, default 4). Prepared objects are routed by the parallel queue without payload copies; relations reference sibling shards without object duplication. Reports include actual cross-shard relation/reference counts, wall ingestion and persistence, and summed worker times. Worker sums overlap and must not be read as wall time. Optional inventory checks verify that relation-only output owns no copied objects.

Reports include compiler/SQLite/build information and the configuration. Supply `source_revision` for revision provenance; otherwise it is explicitly unrecorded. Historical `direct` trigger-removal diagnostics are removed because their schema assumptions no longer apply.

Native results include `per_worker` object/edge counts, creation, producer, insertion, seal/commit durations and exclusive phase timers. Endpoint directories, their merge/barrier, and per-endpoint reference resolution are removed; older reports containing those phase names describe the retired implementation. Native sharded cases now route through the same bounded parallel stream as AFV; routing is included in ingestion. The compatibility field `uuid_partition_preparation_seconds_excluded` contains only benchmark-ID bookkeeping, not data copies. This changes placement from older contiguous-source partitions, so comparisons are end-to-end implementation comparisons, not isolated removal measurements. Aggregate worker durations include overlapping work and waits, not CPU time or critical-path wall time. Use individual workers to distinguish contention and workload imbalance; never sum those aggregates into an import latency.

All variants precompute source identities. Native APIs still validate/canonicalize UUID strings and bind their 16-byte representations; that work intentionally remains timed. The prototype has precomputed binary UUIDs. The benchmark therefore measures the batch APIs as they exist, not equivalent SQL instructions or identical schemas.

## Parallel-stream scaling measurement

A warm-source regression trial used 50,000 complete fibers (15,486,987 XYZ points, 371.7 MB), 300,000 individual constraints and identical native metadata on both paths. Build: GCC 13.3, RelWithDebInfo, SQLite 3.45.1; host: 192 logical CPUs, no CPU quota. Packed memory-first ingestion used a 4 GiB resident budget, no background flushing, API batches of 256, SQL row batches of one, SQLite memory accounting off, and separate final export to a self-cleaned RAM-backed directory. Each subprocess had a 30-second hard cap. There was one repetition per case, so min/median/max equal the individual measurement; these are not confidence intervals or p95 measurements. Source configuration and raw reports belong to the caller's local experiment archive.

| Readers/writers | Prepared ingestion s | AFV+AFC streaming ingestion s |
|---:|---:|---:|
| 1 | 4.12 | 8.34 |
| 8 | 0.72 | 1.47 |
| 16 | 0.37 | 0.98 |
| 32 | 0.22 | 0.93 |
| 64 | 0.33 | Not tested |
| 128 | 0.16 | Not tested |

Prepared routing/queues are now included in ingestion, with no serial deep copy; serial fixture materialization and final export are excluded. AFV includes source decoding and queue waits but excludes initial catalog/range setup and final export. Prepared caches were 512 MiB per writer; streaming caches were 1 GiB per writer and 128 MiB per source cursor. These are limits, not reservations. The 64-writer prepared result was nonmonotonic; do not infer an exact saturation point from one short run. Thirty-two writers gave approximately 19× prepared speedup over one. AFV's useful scaling knee was around 16; 32 gained another 5%, while 64 readers feeding 32 writers were slower. Final RAM export was approximately 0.26–0.38 seconds and is not included in this table or evidence of durable disk performance.

Before the changes, prepared import at 32 writers took 0.297 seconds plus 0.151 seconds of excluded serial routing/copy, versus 0.219 seconds including routing afterward. AFV streaming fell from 1.434 to 0.934 seconds at 32 writers. The main proven object-write bottleneck was `CHECK(state IN (0,1,2))`: SQLite constructed a temporary B-tree for each inserted object and contended on the process-wide PMEM mutex even across independent databases. Equivalent scalar comparisons preserve validation without that tree. Aggregate object-row time previously grew to 4.02 worker-seconds at 32 writers; after the fix it stayed near 0.35 worker-seconds from 1 through 32. A regression checks both the accepted/rejected states and the INSERT bytecode.

The remaining AFV source profile attributed approximately 76% of measured constraint-decoding time to JSON parsing/serialization, 19% to SQLite stepping, 2% to UUID construction, and 3% to other decoding. These exclusive elapsed worker sums include waits, not hardware CPU samples. UUID contexts and fixed metadata nodes are reused; no source fields are dropped. Larger caches (512 MiB→1 GiB writer, 32→128 MiB source) did not improve this sample (0.924→0.926 seconds); optional readonly source mapping was worse (1.24 seconds) and remains off. These observations do not establish a memory-bandwidth ceiling. No claim of ideal whole-volume scaling is implied by the small warm-source trial.

Reports separate preparation, native object/edge work, producer copying, index completion, commit/publication, and readback. Phase timings are exclusive nested timers; do not add a parent elapsed total to its phases. Each fresh builder worker installs its profiler during the object callback and retains it until thread exit, covering later seal/commit/checkpoint work as well as API callbacks. The builder joins those threads before reading their counters; linked base preparation remains outside that profile, and final container publication has its separate wall timer. Private batch construction does not run ordinary editing triggers. Optional table inventory counts include all persisted rows. Prototype geometry hashing is inside object time. JSON-line result records include database bytes; table row counts are included only with `inventory:true`. Constraints are always counted individually, never as equivalent events hidden inside bundled records.

Output reopening is disabled by default. Set `verify_readback:true` for explicit diagnostic readback outside import timers: prototype relation payloads are compared in full; native constraints are sampled for original metadata and endpoint identity, and geometry is sampled byte-for-byte. Independently, `inventory:true` scans table counts and checks total object/edge counts. Both flags default to false; normal performance trials do not reopen completed output. This is a performance diagnostic, not a replacement for the core regression suite (`bash vc_object_store/scripts/run_cpp_tests.sh`).

For comparisons report the dataset prefix, endpoint/point counts, build type, writer count, repetition count, mean/median/min/max, page/cache/journaling differences, and whether final export/readback/preparation is included. `source_totals` reports the actual AFV fiber/point and AFC constraint totals, which may describe only a prepared cohort, not a full traced volume (constraint total is zero/not queried when a geometry-only configuration omits `constraints`). Label projections with their target scope. Do not extrapolate a long-fiber prefix to whole-volume geometry without accounting for its different point-count distribution. Combined geometry-plus-relation time cannot be scaled by constraint count alone. Projection from a constraint prefix is only a rough estimate, not measured whole-volume throughput.
