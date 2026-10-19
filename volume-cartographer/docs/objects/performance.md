The main migration risks are unbounded edge/metadata materialization, geometry
sampling across the chunk-cache boundary, and small writes issued as separate
durable transactions. Small indexed graph queries and narrow spatial queries
already scale well in these checks. Keep Spiral's resident graph and training
arrays, and add bounded queries for vc3d before replacing its current readers.

These baseline measurements were collected on 2026-10-19 at `fee3f2a7b`, using the
actual native Python bindings from `build/python`, before implementation changes.
Descriptions of missing APIs and current behavior below refer to that baseline.
See [implemented improvements and follow-up measurements](performance-improvements.md)
for the subsequent changes; consumer migration remains separate.
The [raw results and validation logs](performance/2026-10-19/) include every
timing sample, query plans, profiles, module paths and build information.

**Actual consumer entry points**

| Workflow | Existing consumer | Store data/access needed |
| --- | --- | --- |
| vc3d fiber catalog and slice display | [`FiberCollectionController.cpp`](../../apps/VC3D/FiberCollectionController.cpp), `listPage`/`catalogByLength` call and region worker | Family, physical length, stable IDs; 200-row length-sorted cursor pages; slice bounds and bounded geometry blocks. The current generic selector cannot express family/length predicates or length ordering. |
| vc3d editing and fit input reload | [`SpiralWorkspace.cpp`](../../apps/VC3D/SpiralWorkspace.cpp), `noteTrackedFiberSaved`, and [`SpiralServiceManagerInputs.cpp`](../../apps/VC3D/SpiralServiceManagerInputs.cpp) | Object identity, geometry revision, input role, metadata and changed-record tracking; batch edits and consistent snapshots when rebuilding fit inputs. |
| Spiral crossing graph initialization | [`fit_spiral.py`](../../../spiral-fitting/fit_spiral.py), track loading near line 2294; [`track_graph.py`](../../../spiral-fitting/track_graph.py), `TrackGraph` | Crossing endpoints, source IDs, point-local indices, arclengths and clearance. The current graph deliberately retains NumPy CSR arrays and rustworkx topology instead of per-edge Python dictionaries. |
| Spiral iterative sampling | [`tracks.py`](../../../spiral-fitting/tracks.py), `_sample_crossing_csr`, `_gather_sampled_track_points`, `iter_track_losses` | Repeated adjacency and point gathers over resident arrays. Per-neighbor or per-point database calls would enter a hot fitting loop. |
| Spiral fiber-link constraints | [`fit_spiral.py`](../../../spiral-fitting/fit_spiral.py), `materialize_fiber_fit_inputs`; [`spiral_helpers.py`](../../../spiral-fitting/spiral_helpers.py), `resolve_fiber_links`/`build_link_components` | Accepted/pending policy, endpoint point identities, same-winding links and components. The store's traversal filters kinds, but cannot filter approval or arbitrary edge metadata. |
| Spiral point-to-patch attachments | `fit_spiral.py`, `_derive_point_inputs`; [`PointCollectionPatch.cpp`](../../core/src/vc_object_store/linking/PointCollectionPatch.cpp) | One attachment per point: point identity, patch identity, grid position, winding annotation, absolute/relative role and detector settings. Multiple links between the same objects must retain their separate anchors. |
| Relationship detector rerun | [`LinkCandidates.cpp`](../../core/src/vc_object_store/linking/LinkCandidates.cpp), `commitCandidatesImpl` | Existing edges, detector provenance and approvals. It currently loads **all edges of each kind** before applying stale-input scope, so a small rerun can pay dataset-wide metadata costs. |

The catalog benchmark uses typed track summaries as the existing store analogue
of the fiber collection catalog. The metadata fixture is a controlled scaling
probe, not a captured consumer trace: N tracks, N crossing links in a ring,
N links incident to one patch, and one rare-kind link. Ten percent of crossings
are approved. Objects and crossings each have 256 bytes of extra metadata; a
second run uses 4096 bytes. Geometry is absent in this fixture so it isolates
record costs; dependency-aware geometry checks are measured separately.

**Metadata and graph reads**

Median wall times in milliseconds, including Python result construction:

| Operation | 1,000 tracks | 10,000 tracks | 50,000 tracks |
| --- | ---: | ---: | ---: |
| Load all crossing records | 6.24 | 87.86 | 501.37 |
| Return first 100 approved crossings by loading/filtering records | 6.50 | 94.20 | 552.46 |
| Read all incident records on the patch | 5.29 | 66.73 | 374.83 |
| Return 200 family/length-filtered, length-sorted records | 5.90 | 74.30 | 486.99 |
| Same catalog query with `fields=['name','track']` | 3.46 | 32.89 | 223.65 |
| Select one object from a connected chain, `limit=1` | 3.50 | 37.95 | 228.21 |
| Export the chain graph with projected names | 19.20 | 234.54 | 1412.91 |

The last two rows use the separate existing workflow fixture (N objects and
N-1 edges). These costs are important for interactive selection, loading review
panels and preparing fitting inputs; they are not measured vc3d frame times.

`edges_of` and `edges_by_kind` have no projection, approval filter, limit or
cursor. `readEdgeRows` already batches endpoint/tag hydration, so this is not
simply one SQL query per endpoint. It still loads every selected edge's common
fields, dependencies and JSON; the binding converts these to Python records.
The saved cProfile captures one JSON decode per crossing even when the final
page contains only 100 records. Its native/binding time is opaque to cProfile;
the profile does not establish a C++ hotspot breakdown.

A diagnostic SQL query for 100 approved crossings, selecting endpoints and
anchor coordinates, took **45.29 ms** at 50,000 crossings with the existing
schema and **0.179 ms** with a temporary `(kind, approval_state, id)` index.
The plan changes from `edges_unique(kind=?)` plus a temporary order-by B-tree
to the proposed index without that sort. Both runs use the same SQL and data;
the index is created only in the disposable benchmark database. This is a
lower bound, not a promised public-API speedup: it omits complete records,
dependency checks and Python metadata decoding, and uses Python's SQLite.
The index's ingestion/storage tradeoff has not been benchmarked.

In [`Queries.cpp`](../../core/src/vc_object_store/Queries.cpp), `connected_to`
materializes the entire component, converts UUIDs back to internal IDs and
constructs a JSON membership list before SQL applies `LIMIT` or spatial bounds.
`limit=1` therefore cannot bound traversal work. [`Snapshots.cpp`](../../core/src/vc_object_store/Snapshots.cpp)
also hydrates complete object records before export projection and performs a
geometry-state query per exported edge. Avoid using either operation for every
mouse move, catalog page or graph step.

At 10,000 tracks, increasing the extra metadata from 256 to 4096 bytes raised
crossing-record loading from **87.86 to 140.14 ms** and the approved page from
**94.20 to 151.95 ms**. The projected catalog stayed around **33 ms**, while the
full-record catalog rose from **74.30 to 118.08 ms**. Projection removes much
of the catalog's payload cost, but full edge records have no equivalent option.
The 50,000-track metadata
worker also retains hundreds of MiB at peak; peak RSS includes fixture creation,
all operations and allocator retention, so it is not an isolated query allocation.

**Geometry and graph access inside fitting**

The existing workflow benchmark uses a 24 MiB fiber array and verifies all
returned bytes. These are equivalent point requests, reordered and restored to
caller order in the grouped variant:

| 512 point requests | Median ms |
| --- | ---: |
| Cyclic across eight 1 MiB chunks | 0.040 |
| Cyclic across nine chunks | 292.16 |
| Same nine-chunk requests grouped by offset | 5.32 |
| Random requests across the array | 188.76 |
| Same random requests grouped by offset | 14.70 |
| Read the entire array | 29.24 |

A second fresh worker reproduced the cliff: 0.042 / 294.32 / 5.42 ms for the
eight-chunk / nine-chunk / grouped-nine cases. The eight-chunk result is a warm
cache hit. [`GeometryReader.cpp`](../../core/src/vc_object_store/geometry/GeometryReader.cpp)
limits each reader to eight MiB of chunks and verifies the full SHA-256 hash on
each miss. The outer default 16 MiB reader-pool budget does not remove that
inner limit. Group requests by storage chunk, retain resident training arrays,
or add a range reader that plans requests by chunk and restores output order.
Keep content verification intact.

Single-range and batched APIs also differ in cache behavior:
`read_geometry_array` constructs a fresh uncached reader; range/rectangle APIs
can reuse verified reader data. For a 6 MiB fiber, 100 separate 24-byte reads
took **52.89 ms**, versus **0.012 ms** for one warmed batched request.
The checked-in real TIFF patch `20241113070770` (129 x 129 x 3) showed the same
effect: reading a column with 129 separate calls took **11.94 ms**, versus
**0.014 ms** batched. That test copies the fixture before ingestion and compares
all bytes; the small patch is not representative of production surface sizes.

Loading 10,000 small tracks by current-head lookup plus one array read each took
**197.69 ms**; supplying known revision UUIDs reduced it to **100.65 ms**.
For graph stepping, 1,000 sparse neighbor queries took approximately **8 ms**;
copying the same neighbor lists from a resident map took approximately **0.04 ms**
(a lower bound that excludes map construction). Bulk loading should happen at
fit initialization/reload, followed by the existing resident CSR/rustworkx path.

**Edits, reviews and synchronization**

At the largest metadata scale, 200 edge metadata updates took **256.28 ms** as
individual commits and **7.30 ms** in one write transaction. The separate baseline
measured 1,000 object updates at **1216.70 ms** with individual commits versus
**18.17 ms** in one transaction. These use default FULL durability on ext4/NVMe;
the results do not depend on weakening durability. Batch related UI/service
changes and detector outputs within a logical transaction.

Writing an identical metadata value four times still advanced generation four
times, appended four journal entries and reset an approved record to pending.
This occurred for both objects and edges. `mutateMeta` unconditionally invokes
`stampContentChange`. Avoid resubmitting unchanged fields during autosave or
reload; consider explicit no-op detection with a separately agreed semantic
contract. Otherwise harmless reloads create review churn and downstream cache
or sync work.

Geometry edits have a different scaling dimension: whole input size plus link
degree/dependency count. On 262,144 points, an unrelated edit preview took
**2.70 / 3.43 / 3.97 ms** for a fiber with **0 / 10 / 100** incident links, and
**3.88 / 17.40 / 18.14 ms** for a point collection. Point collections resolve
stable point IDs; this extra cost is relevant to vc3d point editing. A single
coordinate publication on a 2049 x 2049 patch took **120.85 ms**, even though
immutable chunks deduplicate unchanged storage. This is a whole-payload edit
API, not a point-update API. Higher-degree attachment graphs and tangent-support
dependencies still need production-shaped measurements.

The healthy paths should be preserved:

- A single spatial match stayed about **0.017–0.026 ms** while catalogs grew
  from 1,000 to 50,000 objects; hydrating and filtering the latter took **431 ms**.
  Spatial tile query plus rectangle load took about **0.007 ms** for the synthetic
  surface fixtures. These are conservative candidates and still need consumer
  filtering, display limits and coordinate-unit handling.
- A 16-node component stayed **0.042–0.045 ms** as unrelated graph size grew.
  A filtered hub-neighbor query stayed about **0.008 ms**. Rare edge-kind lookup
  also stayed small: the existing `edges_unique` index already leads with kind.
- No-change sync stayed **0.025–0.031 ms** at 1,000 and 5,000 objects. One-record
  sync took **2.55–6.22 ms** in the sampled histories, while initial metadata
  sync took **259 / 1324 ms**. This does not cover large geometry transfer or
  long-history storage growth.

**Proposed order before moving consumers**

1. Add a projected edge-selection API with endpoint sets, kind, approval policy,
   limit and stable cursor ordering. Retain per-point edge identities and
   anchors; `neighbours()` deduplicates endpoints and is not an attachment list.
   Test composite indexes against both page reads and detector ingestion.
2. Add typed family/length filtering and length/ID cursor pages for the vc3d
   catalog, including filters before the viewport display limit. An indexed tag
   can express categorical membership but cannot replace length ordering.
3. Build a snapshot adapter from selected object revisions and crossing records
   to Spiral's existing source-ID/point-ID/CSR arrays. Preserve pending-link
   policy, coordinate units, stable identities and revision consistency. Cache
   approved components or traverse the prefiltered resident graph; current
   `component()` cannot express Spiral's acceptance policy by itself.
4. Group geometry reads and batch metadata writes; suppress unchanged autosave
   fields. Move expensive load/validation work off the UI thread. Python edge
   and traversal bindings currently retain the GIL, unlike range/spatial reads;
   service responsiveness under concurrent calls remains unmeasured.
5. Scope detector reconciliation to selected endpoints before loading existing
   edge metadata. Measure high-degree linked edits, concurrent readers/writer,
   cold filesystem caches, and real production graph distributions before
   treating these calibration numbers as migration budgets.

**Reproduction and validation**

Build: Linux aarch64, GCC 13.3.0, `RelWithDebInfo`, `-O2`,
`-march=armv8.2-a`, LTO, `VC_MARCH_NATIVE=OFF`; Python 3.14.
See [environment.json](performance/2026-10-19/environment.json) for exact flags
and source hashes. All scenarios ran sequentially, in fresh worker processes
and disposable stores. No OS cache dropping or fixed-frequency calibration was
performed. New metadata and workflow timings use one warmup and three measured
repetitions; SQL diagnostics and most baseline reads use five; initial/changed
sync samples are single executions. Raw files include min/median/max and all
samples (plus means in the new metadata/workflow reports). Wall time is host
calibration, not a CI performance gate or an end-to-end application measurement.

For the main bottlenecks, the three timed samples have these summaries (ms):

| Case | Mean | Median | Min | Max |
| --- | ---: | ---: | ---: | ---: |
| 100 approved crossings, 50k crossing fixture | 552.66 | 552.46 | 551.94 | 553.57 |
| 200 family/length catalog rows, 50k fixture | 479.62 | 486.99 | 464.39 | 487.48 |
| 200 edge updates, separate commits | 249.28 | 256.28 | 234.47 | 257.09 |
| 200 edge updates, one transaction | 7.40 | 7.30 | 7.13 | 7.77 |
| 512 point reads, cyclic nine chunks | 292.34 | 292.16 | 291.76 | 293.09 |
| Same reads grouped by offset | 5.32 | 5.32 | 5.30 | 5.33 |

```sh
ninja -C build vc_object_bindings test_object_store test_object_records test_object_links
.venv/bin/python scripts/benchmark_object_store_metadata.py --output /tmp/vcobj-consumer-metadata.json
.venv/bin/python scripts/benchmark_object_store_metadata.py --sizes 10000 --payload-bytes 4096 --output /tmp/vcobj-consumer-large-metadata.json
.venv/bin/python scripts/benchmark_object_store_workflows.py --output /tmp/vcobj-consumer-workflows.json
.venv/bin/python scripts/benchmark_object_store.py --output /tmp/vcobj-consumer-baseline.json
.venv/bin/python scripts/benchmark_object_store_workflows.py --worker patch --patch core/test/data/segments/20241113070770
.venv/bin/python scripts/benchmark_object_store_workflows.py --worker sampling --n 1048576
VC_BUILD_PYTHON="$PWD/build/python" PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest python/tests/test_object_records.py python/tests/test_object_queries.py python/tests/test_object_geometry_edits.py python/tests/test_object_links.py python/tests/test_object_sync.py python/tests/test_object_spatial_regions.py -q
build/bin/test_object_store
build/bin/test_object_records
build/bin/test_object_links
```

All benchmark result checks passed. Native suites passed **5 + 13 + 11** test
cases. Selected Python tests: **74 passed, 1 failed**. The existing
`test_incremental_matches_full_snapshot_merge` fails at the copied full-snapshot
oracle's `synchronize(..., dry_run=True)` with `sync: inconsistent shared
baseline`; it also fails in isolation. The database code was unchanged for this
investigation. The cause is not established; this is a separate validation
gap to resolve before relying on synchronization migration coverage, not evidence
of a performance regression from the diagnostic script.
