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

Workloads:

- `relations`: the first `count` individual AFC constraints and their distinct endpoints as typed opaque fiber handles. All constraints, original metadata and anchors remain present; there is no pair bundling. This intentionally isolates graph import, not geometry or hydration of rich parent metadata.
- `geometry`: the first `count` AFV fibers, with their original line coordinates, source names, annotations and detail fields. Empty control-array descriptors carry no points/chunks. All variants use the native chunk iterator and content-hash implementation.

Variants:

- `prototype`: diagnostic reconstruction of the earlier five-table immutable layout: objects, fibers, edges, geometry chunks and object-to-chunk mappings. Original record attributes are pre-serialized into opaque JSON payloads; this serialization is reported separately. It uses 64-KiB pages, journal/sync off, append-time foreign keys, one transaction and final endpoint indexes. **This is not a supported vcobj format or an application writer.** It has no provenance columns, approval logic, lifecycle bindings, geometry manifests, mutation-dependency index, collection routing or v2 generation machinery.
- `native`: current `importInitialCollection`, one ordinary SQLite file, history/spatial off, all native correctness semantics retained. Native journaling remains enabled. SQL statements use the existing default batch size of one within 256-record API batches, not one transaction per record.
- `native_packed`: the same API with disposable RAM-first packed VFS, 4-GiB resident budget and background flushing disabled. Final export/publication is reported separately. Failure on exhaustion, not silent spill.
- `linked`: publish the prepared handles as a sealed dependency first, then import constraints referencing it. Report dependency construction separately; do not compare the sum against an import-only number without saying so.
- `native_sharded` / `native_packed_sharded`: phased construction with independent workers (`workers`, default 4). Objects are partitioned once; relations reference sibling shards without object duplication. Reports include actual cross-shard relation/reference counts, wall ingestion and persistence, and summed worker times. Worker sums overlap and must not be read as wall time. The linked variant additionally verifies that relation-only output owns no copied objects.

Reports include compiler/SQLite/build information and the configuration. Supply `source_revision` for revision provenance; otherwise it is explicitly unrecorded. Historical `direct` trigger-removal diagnostics are removed because their schema assumptions no longer apply.

All variants precompute source identities. Native APIs still validate/canonicalize UUID strings and bind their 16-byte representations; that work intentionally remains timed. The prototype has precomputed binary UUIDs. The benchmark therefore measures the batch APIs as they exist, not equivalent SQL instructions or identical schemas.

Reports separate preparation, native object/edge work, producer copying, index completion, commit/publication, and readback. Phase timings are exclusive nested timers; do not add a parent elapsed total to its phases. Native edge-row time includes SQLite-trigger work. Table counts include such implicitly inserted rows. Prototype geometry hashing is inside object time. JSON-line result records include database bytes and nonempty table row counts. Constraints are always counted individually, never as equivalent events hidden inside bundled records.

Readback runs outside import timers: prototype relation payloads are compared in full; native constraints are sampled for original metadata and endpoint identity, geometry is sampled byte-for-byte, and every result checks the total edge count. This is a performance diagnostic, not a replacement for the core regression suite (`bash vc_object_store/scripts/run_cpp_tests.sh`).

For comparisons report the dataset prefix, endpoint/point counts, build type, writer count, repetition count, mean/median/min/max, page/cache/journaling differences, and whether final export/readback/preparation is included. Do not extrapolate a long-fiber prefix to whole-volume geometry without accounting for its different point-count distribution. Projection from a constraint prefix is only a rough estimate, not measured whole-volume throughput.
