# Reusable unsigned fiber constraint database, version 1

Current artifact: `data/raw_winding_v1/full.afc.sqlite` (moving16, continuous
median parallel estimate, no winding-value processing filters). See
[current results and field semantics](raw_winding_results.md). The original
`data/constraints_v1/full.afc.sqlite` is retained as a historical filtered result.

The native `vc_fiber_trace_chunk afv-constraints CONFIG.json` adapter supports
`full_volume: true` and `database: PATH`. It uses the complete AFV catalog and
unchanged shared unsigned local-event extractor. It does not cut fibers, trace,
infer, align normals or run a winding BP solve. Export requires `repetitions: 1`.

## Artifact and schema

SQLite `.afc.sqlite`, schema `user_version=1`, metadata `format="fiber-constraints"`.
This is **not an AFV geometry file and cannot be opened as one in VC3D**. Retain
the source `Paris4_v1.afv`; constraints refer to its original IDs and UUID.
Coordinates and arc lengths are native Paris4 L0/base voxels. The application ID
is the same VCFC container-family value; consumers must check the format metadata.

- `metadata(key,value)`: JSON provenance, complete flag, schema and statistics.
  Provenance contains source UUID, all selected IDs, source config and supplied
  SHA256 identities (AFV, normals, binary, tracer library), normal manifest and
  inference provenance, resolved local settings and extraction timings/counts.
  Hashes are also independently recorded by the external approved runner.
- `fibers(id,source_index,length_base)`: every selected original source fiber,
  including isolated fibers. `source_index` is the native extractor's input index.
  Degenerate/dropped inputs cause the adapter to fail, not silently omit fibers.
- `constraints`: stable event ID; original `fiber_a`/`fiber_b` foreign keys;
  `arc_a`/`arc_b`; `ax,ay,az,bx,by,bz`; parallel/perpendicular scores;
  unsigned winding and parallel-winding distances; evidence weight;
  endpoint `normal_a`/`normal_b` JSON; full `measurement` JSON.
- Endpoint indexes support querying all events incident on a fiber.

The `measurement` object preserves **every** `FiberTraceConstraint` field using
its C++ name: support spans, local orientation, flags, all optional/null signed
fields, diagnostics, scores, positions and distances. Piece indices refer to
native whole-piece order; use explicit fiber IDs for external association.
JSON doubles are round-trip serialized; primary typed columns allow fast SQL.

Each endpoint normal is `{valid,xyz,reason}`. Coordinates are passed to the shared
double-position batch sampler without rounding to voxel centers. `xyz` is an
unoriented unit sheet-normal axis, not a fiber tangent or globally oriented normal.
Invalid samples retain their reason and `xyz:null`; zeros are not substituted.
Unsigned distances are integrals of winding density times absolute normal
projection along the connector. They do not determine winding-order sign.

## Publication and reuse

Source inputs are read-only. The writer uses a transaction and bounded8192-event
normal-sampling batches. The entire extraction result is still resident in RAM;
batching export does not make extraction streaming. A sibling `.partial` file is
closed and renamed only after integrity/FK validation; existing final or partial
outputs are refused. Only consume the final filename. One writer per output is
required; concurrent publication to the same path is unsupported.

Example query (Python sqlite3 or any SQLite client):

```sql
SELECT id,fiber_a,fiber_b,parallel_score,perpendicular_score,winding_distance,
       normal_a,normal_b,measurement
FROM constraints WHERE fiber_a=123 OR fiber_b=123;
```

## Reproduction

```bash
bash /home/hendrik/villa2/vesuvius/scripts/run_fiber3d_native_build
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py crop-benchmark --config /home/hendrik/villa2/global-fiber-sgd/run_extract_pilot.json
/home/hendrik/.venv_las/bin/python global-fiber-sgd/validate_constraint_database.py global-fiber-sgd/data/constraints_v1/pilot.afc.sqlite --all-measurements
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py crop-benchmark --config /home/hendrik/villa2/global-fiber-sgd/run_extract_full.json
/home/hendrik/.venv_las/bin/python global-fiber-sgd/validate_constraint_database.py global-fiber-sgd/data/constraints_v1/full.afc.sqlite
```

Choose new output paths before rerunning; do not overwrite an existing artifact.
The recorded build hashes identify this run; regenerate them after code changes.
Pilot validation checked all19,889measurement field sets/values plus200original
AFV endpoint/arc reconstructions. Full validation samples ~1000measurements and
200endpoints, checks all source IDs, database integrity/FKs and degree sum=2E.
Native writer stats use lower-order-statistic median/p95 for event degree.
Event degree counts repeated local encounters; it is not unique-neighbor degree.
