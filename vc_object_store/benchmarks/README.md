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

Each case may override only `variant`, `workers`, `sql_batch_rows`, `repetitions`, `label`, `export_workers`, `pre_size_export`, and `resident_bytes`; source data cannot silently change between cases. Prototype JSON/binary-ID preparation runs once only if the matrix actually contains a prototype case. Native-only matrices do not serialize the entire source into prototype JSON. Completed private output files are removed after their size and any explicitly requested diagnostics are recorded, so RAM usage does not accumulate across the matrix; `remove_completed_outputs=false` retains them until the supervising runner cleans its fixture directory. Preparation, all cases, inventory and readback must fit the same runner invocation budget (28 seconds before termination, with a two-second kill grace in the local fixed runner). Start with a small input and one repetition; do not infer that the example count is universally fast for long fibers.

Workloads:

- `relations`: the first `count` individual AFC constraints and their distinct endpoints as typed opaque fiber handles. All constraints, original metadata and anchors remain present; there is no pair bundling. This intentionally isolates graph import, not geometry or hydration of rich parent metadata.
- `combined`: the first `count` AFC constraints and every distinct endpoint's complete original AFV geometry, name, annotation and detail fields. Preserved source IDs identify the exact fibers; a native bounded-ID cursor reads only those fibers, merging consecutive IDs into one interval. `AfvSource::selectRange` reuses the same validated snapshot and prepared statement across intervals, without repeatedly opening or validating the source. Every required fiber is decoded once and reused by every case. Missing geometry or inconsistent source identities fail explicitly. The import contains both full geometry and relations, not opaque stand-ins.
- `geometry`: `count` AFV fibers, with original line coordinates, source names, annotations and detail fields. `geometry_sampling="prefix"` (default) selects the first fibers. `geometry_sampling="uniform"` calls the native catalog partitioner once and reads the first fiber in each balanced catalog interval, providing deterministic coverage across the volume without scanning intervening XYZ payloads. It reuses one `AfvSource` snapshot/cursor via `selectRange`; reports distinguish reader count from interval count. Sampling selects whole fibers; it never resamples their coordinates. Empty control-array descriptors carry no points/chunks. All variants use the native chunk iterator and content-hash implementation.

`max_prepared_xyz_bytes` defaults to 1 GiB and rejects preparation above that total; it does not truncate fibers or drop constraints. A single fiber must still be decoded before this guard can inspect its bytes. Reports include actual points, XYZ bytes, native geometry reader count, and the number of position/whole-geometry/indexed anchors. AFC currently supplies position plus whole-geometry anchors, not stable point-index anchors; the benchmark preserves that distinction and does not manufacture stronger measurements. The phased directory receives the complete accepted geometry's dependency information. Prefix fibers can be much longer than typical volume fibers, so do not project geometry cost from fiber count alone.

`AfvSource::selectRange(first, pastLast)` explicitly abandons any unfinished previous scan and resets that scan's counters while retaining its SQLite read snapshot. Each returned fiber still receives all native validation. Declared whole-collection counts are checked only when a full `[0, INT64_MAX)` selection reaches EOF; a sampled interval does not pretend to validate unread fibers. Tests cover repeated seeks/EOF, invalid intervals, reset accounting and snapshot consistency across concurrent source writes.

Variants:

- `prototype`: diagnostic reconstruction of the earlier five-table immutable layout: objects, fibers, edges, geometry chunks and object-to-chunk mappings. Original record attributes are pre-serialized into opaque JSON payloads; this serialization is reported separately. It uses 64-KiB pages, journal/sync off, append-time foreign keys, one transaction and final endpoint indexes. **This is not a supported vcobj format or an application writer.** It has no provenance columns, approval logic, lifecycle bindings, geometry manifests, mutation-dependency index, collection routing or v2 generation machinery.
- `native`: current `importInitialCollection`, one ordinary SQLite file, history/spatial off, all native correctness semantics retained. Private immutable builders disable journaling/sync during construction and durably publish at finalization. They use 64-KiB SQLite pages. SQL statements use the existing default batch size of one within 256-record API batches, not one transaction per record. Optional `sql_batch_rows` tests larger SQL batches without changing API batch size or data.
- `native_packed`: the same API with disposable RAM-first packed VFS and background flushing disabled. `resident_bytes` defaults to 4 GiB; exhaustion fails rather than silently spilling. `export_workers` defaults to 1 and controls final packed payload export independently of importer workers. `pre_size_export` defaults to false; true sizes the compact target to its computed payload length before export. These settings may appear at the configuration root or in individual cases, and map directly to `PackedStoreOptions::residentBytes`, `exportWorkers`, and `preSizeExport`. Final export/publication is reported separately. Results echo the settings and `packed_export_options_applied`; plain/prototype cases do not apply them.
- `linked`: publish the prepared handles as a sealed dependency first, then import constraints referencing it. Report dependency construction separately; do not compare the sum against an import-only number without saying so.
- `native_sharded` / `native_packed_sharded`: phased construction with independent workers (`workers`, default 4). Objects are partitioned once; relations reference sibling shards without object duplication. Reports include actual cross-shard relation/reference counts, wall ingestion and persistence, and summed worker times. Worker sums overlap and must not be read as wall time. Optional inventory checks verify that relation-only output owns no copied objects.

Reports include compiler/SQLite/build information and the configuration. Supply `source_revision` for revision provenance; otherwise it is explicitly unrecorded. Historical `direct` trigger-removal diagnostics are removed because their schema assumptions no longer apply.

All variants precompute source identities. Native APIs still validate/canonicalize UUID strings and bind their 16-byte representations; that work intentionally remains timed. The prototype has precomputed binary UUIDs. The benchmark therefore measures the batch APIs as they exist, not equivalent SQL instructions or identical schemas.

Reports separate preparation, native object/edge work, producer copying, index completion, commit/publication, and readback. Phase timings are exclusive nested timers; do not add a parent elapsed total to its phases. Each fresh builder worker installs its profiler during the object callback and retains it until thread exit, covering later seal/commit/checkpoint work as well as API callbacks. The builder joins those threads before reading their counters; linked base preparation remains outside that profile, and final container publication has its separate wall timer. Private batch construction does not run ordinary editing triggers. Optional table inventory counts include all persisted rows. Prototype geometry hashing is inside object time. JSON-line result records include database bytes; table row counts are included only with `inventory:true`. Constraints are always counted individually, never as equivalent events hidden inside bundled records.

Output reopening is disabled by default. Set `verify_readback:true` for explicit diagnostic readback outside import timers: prototype relation payloads are compared in full; native constraints are sampled for original metadata and endpoint identity, and geometry is sampled byte-for-byte. Independently, `inventory:true` scans table counts and checks total object/edge counts. Both flags default to false; normal performance trials do not reopen completed output. This is a performance diagnostic, not a replacement for the core regression suite (`bash vc_object_store/scripts/run_cpp_tests.sh`).

For comparisons report the dataset prefix, endpoint/point counts, build type, writer count, repetition count, mean/median/min/max, page/cache/journaling differences, and whether final export/readback/preparation is included. `source_totals` reports the actual AFV fiber/point and AFC constraint totals, which may describe only a prepared cohort, not a full traced volume (constraint total is zero/not queried when a geometry-only configuration omits `constraints`). Label projections with their target scope. Do not extrapolate a long-fiber prefix to whole-volume geometry without accounting for its different point-count distribution. Combined geometry-plus-relation time cannot be scaled by constraint count alone. Projection from a constraint prefix is only a rough estimate, not measured whole-volume throughput.
