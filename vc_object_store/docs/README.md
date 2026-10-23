# Object store

`ObjectStore` keeps objects, immutable geometry, metadata and relationships in one
SQLite file. Source files are needed for ingestion; subsequent queries, edits,
detection and export use stored revisions. CT remains an input to fiber/track
patch detection. See [build/CLI instructions](../README.md),
[typed geometry](typed-geometry.md) and [structural editing](structural-editing.md).
C++ callers link `vc_objects` for storage, `vc_object_geometry` for native I/O,
and `vc_object_links` for detectors. Public signatures live in
[ObjectStore.hpp](../include/vc_object_store/objects/ObjectStore.hpp).

## Ingestion and export

```python
from vc_object_store import ObjectStore

with ObjectStore.create("objects.vcobj", author="reviewer") as store:
    report = store.ingest_geometry_source("patch", "/data/patch")
    assert not report["problems"], report["problems"]
    report = store.link_patch_overlaps()
    assert not report["problems"], report["problems"]
    patches = store.objects_by_kind("patch")
    store.export_geometry(patches[0]["uuid"], "/data/exported-patch")
```

Ingestion is **one-time**. Existing UUIDs (even unchanged or retired), duplicates,
and missing/ambiguous source-link targets fail. Fiber directories and PCL
documents publish atomically; packed tracks use per-track transactions and can
report partial success. Track identity is dataset reference plus external key.
Always inspect `problems`.
`ingest_registered_geometry` loads placeholders only if they have no geometry
history, preserving unrelated metadata. Use store edits for subsequent changes.

PCL source links use document-scoped collection/point identities. The namespace
is `meta.source.origin`, the absolute lexically normalized import path; symlink
aliases and moved files are different namespaces. Import only new collections
at that path to extend a document. Only active owners resolve links; duplicate
active identities fail. Derived identity indexes are maintained through
edits, restoration and sync. Replacement can scan all affected IDs;
ordinary resolution reads only referenced chunks or metadata-snapshot buckets.

Native export selects an immutable object revision and stages output before publication;
`overwrite=False` protects existing destinations. It rejects auxiliary arrays.
Directory export currently requires Linux `renameat2`. Metadata-only
`export_json` supports filters and field projection; it is not a geometry backup.
Typed detail counts, lengths, dimensions and boxes are import summaries: use
current geometry descriptors and spatial queries after edits.

## Revisions and transactions

C++ applications can share an `ObjectStoreSession` to manage an existing writable
database's attachment, checked writes, worker cancellation, and journal events.
Consumers call the existing `ObjectStore` APIs directly inside its callbacks:

```cpp
vc::objects::ObjectStoreSession session;
vc::objects::OpenOptions options;
options.author = "reviewer";
const auto attachment = session.attach("objects.vcobj", options);
vc::objects::RevisionSelection observed;
session.read(attachment, [&](const auto& store, const auto& txn, const auto& cancel) {
    if (cancel.load()) return;
    observed[objectUuid] = store.currentObjectRevision(txn, objectUuid)->uuid;
});
session.act(attachment, observed, [&](auto& store, auto& txn) {
    store.setMeta(txn, objectUuid, "review_note", R"("checked")");
});
```

Keep all work using a snapshot inside its callback and do not reenter the
session from a callback. Replacement validates the new database first, drains
accepted writes, and cancels/waits for old readers. Include the attachment
generation in asynchronous requests and caches: even reattaching the same store
invalidates old requests. `takeEvents()` polls local and external commits for
all object kinds; call it from the application's event loop. `finalize()` waits
for accepted writes; edits are already committed by `act()`.

Fiber consumers use the same `selectObjects`, `geometryRegion`, geometry, and
relationship APIs. Update any catalog summaries used by the application in the
same checked transaction as the geometry edit, and explicitly decide how affected
relationships apply. There is no separate fiber repository or collection layer.

For native fiber transport, `fiberNetworkJson(store, txn, selection, roots)`
exports pinned geometry together with its applicable `fiber_branch` edges. Supply
explicit object revision pins for every endpoint, including retired endpoints
needed to resolve applicability. Empty roots exports all active selected fibers;
otherwise the exporter includes their complete linked components. The result is
keyed by object UUID, uses `UUID.json` references, and keeps display names separate.
It resolves stable control-point anchors to positional wire indices and emits
reciprocal ordinary/adjacent branches with approval state and edge UUIDs. Coordinate
identity and bindings accompany each object. `fiberRevisionJson` remains the
geometry/annotation-only codec for consumers that read relationships separately.

Object revisions capture metadata, geometry identity and relationship membership.
`current_object_revision`, `object_revision`, and `object_history` expose them;
`current_geometry`, `geometry_revision`, and `geometry_history` expose geometry.
Their UUIDs are distinct: metadata edits need not create geometry revisions.

Pin all modified objects and participating relationship endpoints when preparing
an edit outside the writer lock:

```python
observed = {u: store.current_object_revision(u)["uuid"] for u in (a, b)}
with store.write(expected_revisions=observed) as tx:
    tx.set_meta(a, "review_note", "checked")
    tx.set_meta(edge, "confidence", 0.9)
```

C++ uses `store.write(RevisionSelection{...})`. `None` means the object must not
exist; preselect its UUID. Stale/missing expectations and caught geometry
revision conflicts abort the entire checked action. Plain `write()` serializes
read/modify/write under its lock but cannot protect an earlier external read.
Standalone Python mutations commit their own transaction. Read transactions pin
a snapshot; writer revision reads describe the last published state. Each commit
publishes at most one revision per changed object, including shared endpoints.

`published_at_us` is UTC Unix microseconds, shared by revisions in one action and
strictly increasing between actions even if the clock goes backward.
`revision_selection_at(time)` selects the latest revisions at that time;
`revision_read_view_at(time)` returns their `objects`, `relationships`, and resolved
`selection` together. Objects created later or physically deleted by then are
excluded. Retired objects are omitted unless `include_retired=True`.
`revision_read_view({uuid: revision_or_None})` uses an explicit closed selection;
`None` resolves the latest head in that snapshot. Restore historical content by
publishing a new checked action, not by moving old heads.

## Relationships and graphs

Relationships are shared by both endpoints. Addition, removal or rebinding
revises both objects (once for a self-link). `recorded_relationships(object_revision)`
returns immutable edge versions with `a_revision`/`b_revision` object pins and
geometry pins. Resolve these pins rather than substituting current heads.
`relationships_at(selection)` reports `applicable` only when both selected
endpoints share the same version and its geometry, metadata and lifecycle
dependencies match. Missing endpoints are inapplicable.

Anchors declare point IDs/indices, `channels`, `metadata_keys` (top-level keys),
or `whole_geometry`. Editing a dependency requires an explicit decision:

| Decision | Effect |
| --- | --- |
| `keep_recorded_relationship` / `action="keep_recorded"` | Validate recorded identities and preserve historical pins; may be inapplicable now |
| `retain_relationship` / `action="retain"` | Revalidate against resulting endpoints and publish shared new pins |
| `update_edge` / `action="update"` | Publish caller-supplied anchors after validation |
| `remove_edge` / `action="remove"` | Remove the current relationship |

Geometry `edge_edits` specify `edge_uuid`, `expected_generation`, `action`,
`a_revision` and `b_revision` (geometry UUIDs or `None`); updates also supply
`a_anchor` and `b_anchor`. Include expected object revisions for all validation
endpoints in checked writes. Metadata/lifecycle decisions must match final state
at commit. Unrelated metadata/point edits leave relationships applicable; physical
or domain-specific validity remains the caller's responsibility.

**Same-winding queries must pass `adjacent=False`.** On `select_edges`,
`neighbours`, `component`, `reachable_within` and `graph_read`, it selects ordinary
`fiber_branch` edges, including branches without the flag. `True` selects
adjacent branches; `None` does not filter. Explicit filters exclude other edge
kinds and malformed flags before traversal, including parallel edges.
`applicable_only=True` additionally checks current shared membership/dependencies;
`include_retired=True` does not override applicability. Edge pagination follows
applicability filtering. Fiber imports reject contradictory ordinary/adjacent
classifications, including reciprocal declarations, atomically.

```python
with store.graph_read(revisions=selection, adjacent=False, approval_states=[]) as graph:
    component = graph.component(seed)
    records = graph.objects()
    links = graph.edges()
    packed = graph.read_geometry_arrays("points", nodes=[0, 1])
```

`graph_read` defaults to approved edges; `approval_states=[]` includes all states.
Explicit revision graphs always require applicability, retain isolated selected
nodes, and cannot combine `revisions` with current object filters or
`include_retired`. Nodes are in canonical UUID order; traversal is breadth-first
in edge-UUID order. Current queries otherwise inspect current records.
Applicability queries require a read transaction.
Graph views are thread-confined and must close before checkpointing. NumPy arrays
are read-only and retain their buffers after closure, not the database transaction.

## Retirement, restoration and synchronization

`retired` and `lifecycle_epoch` are library-managed revision fields. Use
`tx.retire_object(uuid)` or `tx.restore_object(uuid, source_revision=None)` in a
checked write, with explicit decisions for every surviving incident edge after
endpoint edits. Retirement retains geometry, metadata and history. Restoration
uses the last published active content, or an explicit active revision of that
object; it advances generation and resets approval while preserving creation
stamps. Already-retired retirement and already-active default restoration are
no-ops. Restoration does not recreate removed links or physically removed rows.

Lifecycle epochs prevent preserved links from silently reactivating after
restoration, even with identical geometry. Revalidate explicitly. Default current
catalogs, spatial queries, traversal and exports omit retired objects and dormant
edges; traversal cannot pass through retired nodes. `include_retired=True`
exposes them for inspection without asserting applicability. Direct UUID lookups,
geometry and historical reads remain available. Lifecycle-only changes reuse
payloads; restoring different geometry can rebuild spatial indexes.

Application references should be ordinary `declared_reference` edges with
explicit dependencies. Endpoint order is canonical, so put directional roles in
metadata. Arbitrary UUID strings in JSON are not tracked references.
`delete_object` retains history and still rejects managed geometry/current edge
references. There is no history/payload purge API.

Establish a shared baseline with `synchronize(peer)` before independent edits.
Conflicts require a fresh `dry_run=True` preview and explicit `local`/`remote`
resolutions with its `expected_plan` token. Handles must be idle and writable;
call `fork_replica()` on a copied database before synchronizing it with its source.
Sync transfers current records, lifecycle bindings and retained geometry history,
not the peer's complete metadata/graph timeline. Use a database copy for that.
Missing historical pins receive incomplete, untimed baselines, never fabricated
history. A fresh retired replica may lack an active revision to restore: restore
on the originating replica and sync, or copy its full database. Metadata-only
JSON import rejects lifecycle-bearing exports.

## Volume identity and coordinate scale

Geometry objects use `meta.coordinates = {"volume_uuid": ..., "scale_factor": ...}`.
Despite the field name, `volume_uuid` is the volume's `id` from the open-data
`metadata.json`, stored verbatim as a nonempty string (for example
`"20260411134726"`). It is not hashed, lowercased, or converted to an object UUID.
Paths, levels, resolution and labels never infer it. Scale is a finite
positive JSON number with direction **stored XYZ = volume voxel XYZ × scale**;
it differs from physical resolution and patch grid scale. Values are metadata
only and never rescale payloads or detector parameters. Unknown/missing members
mean `null`, not scale 1. Setting coordinates replaces the whole field and
rejects unknown keys; positive 64-bit integers stay exact.

Native object/document/collection declarations, flat `volume_uuid`/`scale_factor`
fields and `ingest_geometry_source(..., coordinate_context={...})` must agree
where known. Packed tracks read declarations from `metadata.json` as well as
explicit context. Every native exporter carries authoritative `meta.coordinates`,
including unknown values, rather than resurrecting declarations in `meta.source`.
An explicit `coordinate_binding` travels alongside them. Export→import preserves
the frame without an out-of-band context argument. Set coordinates with `set_meta`;
dependent anchors declare `metadata_keys=["coordinates"]`.

C++ consumers share normalization, conflict checking, transfer metadata and exact
volume matching through `objects/GeometryCoordinates.hpp`. A display name or
matching grid dimensions cannot override a mismatched volume ID. Existing
bindings containing a derived UUID require an explicit metadata correction;
they are not silently treated as aliases of catalog IDs.

`ingestGeometryBatch` takes decoded objects, geometry and source annotations and
inserts them and their native relationships in the caller's transaction. Duplicate,
existing and retired identities are errors; this is not a refresh operation.
`updateGeometryArrays` accepts desired typed arrays, the expected geometry revision,
optional transformed annotations, and a caller-supplied relationship decision
callback. It uses sparse coordinate edits and stable-ID sequence operations for
fibers/PCLs/tracks, and sparse channel edits for unchanged patch grids. Omitted
annotations follow stable IDs; resizing an annotated sequence without IDs requires
explicit transformed annotations. Grid/channel schema changes retain their dedicated
surface APIs. These helpers do not introduce another history or editing model.

## Arrays, surface channels and sparse edits

Generic readers support full arrays, byte ranges, patch rectangles and packed
homogeneous arrays. Packed results expose dtype, shape, data and row offsets;
trailing dimensions/dtypes must match and request order/duplicates are preserved.
Storage dtypes are `|u1`, `<u2`, `<i4`, `<i8`, `<u8`, `<f4`, `<f8`.

Patch `points` is float32 XYZ. Extra arrays are aligned channels of shape
`[rows,columns]` or `[rows,columns,components]` with positive component count.
Absent and zero-filled channels differ. Ingestion recognizes lowercase `.tif`
and `.tiff` sidecars by stem, excluding XYZ; duplicate/reserved names, unaligned,
truncated or unsupported layouts fail before publication. Tiled/stripped and
contiguous/separate sample planes preserve TIFF sample order and exact values;
no color conversion, rescaling, NaN normalization or alpha premultiplication.
Multipage/volume/non-top-left TIFFs and oversized masks are rejected as channels.

`meta.source["overlapping.json"]` retains the complete sidecar: missing means
unknown, an empty list means known empty. Lists or objects with an `overlapping`
list retain ordering, duplicates, extra fields and unresolved identities. These
are directional source claims; ingestion infers no graph edges or symmetry.

`update_surface_channels(uuid, channels, remove, expected_revision, edge_edits=[])`
adds/replaces/removes channels atomically. Supply only changed arrays and exact
little-endian bytes. Removing absent channels, duplicate names or mismatched
spatial dimensions fails; dtype/components may change. Use
[structural resizing](structural-editing.md#surface-growth-and-cropping) for shape
changes. Typed owned patches and native export reject auxiliary arrays; bounded
views and generic readers remain available.

```python
import struct

current = store.current_geometry(uid)
edits = [{"array": "points", "element_offset": 15,
          "data": struct.pack("<3f", 12, 24, 36)}]
affected = store.affected_geometry_edit_edges(uid, edits, current["uuid"])
assert not affected  # otherwise supply explicit edge_edits
revision = store.edit_geometry(uid, edits, current["uuid"])
raw = store.read_geometry_rectangle(revision, "points", 0, 0, 1, 1)
```

Sparse offsets count **scalars**, not bytes or points. Edits preserve shape,
dtype, names and attributes; overlaps apply in input order, last write wins.
Net no-ops create no revision/journal entry, but stale expectations still fail.
Rectangle edits preserve all components and require exact byte lengths.
Geometry, bounds, approvals and relationship decisions publish atomically.

Patch tiles support 32/64/128 (default 128); size 0 uses linear 1 MiB
chunks. Sparse deltas retain exact bytes, including NaN payloads/signed zero,
with checksum verification and a 256 MiB decoded limit. Current heads use a
chunk overlay; historical reads reconstruct from snapshots or reverse a shorter
current suffix. Inherited channels reuse immutable array references. Only changed
payloads are rewritten, but manifest/bounds bookkeeping can scale with chunk
count and historical reads with history length. There are no automatic periodic geometry
checkpoints or pruning. Long-held SQLite readers can retain old WAL pages.
Structural changes publish chunk snapshots and restart the delta chain; subsequent
coordinate edits continue using deltas. Structural operations can read and assemble
the full geometry. Full `put_geometry` replacements have a separate contract.

## Bounded oriented region browsing

`geometry_region(RegionSlice, RegionOptions, cancel=None)` combines conservative
spatial bounds with exact double-precision segment/slab intersections. It returns
original unclipped XYZ and zero-based segment indices in contiguous runs; the
consumer handles rendering/trimming. `geometry/SegmentClip.hpp` shares the clipper.

```python
from vc_object_store import RegionSlice, RegionOptions

view = RegionSlice()
view.origin = [100, 200, 300]
view.u, view.v, view.normal = [1, 0, 0], [0, 1, 0], [0, 0, 1]
view.local_bounds = [-50, -50, -.5, 50, 50, .5]
options = RegionOptions()
options.kind = "fiber"
options.maximum_objects = 100
result = store.geometry_region(view, options)
```

Axes must be finite/orthonormal (tolerance `1e-12`); bounds are finite and ordered,
with zero thickness allowed. Segments can intersect with both endpoints outside.
`options.kind` selects one polyline kind: `fiber` (nonempty `line_points` take
precedence over controls) or `track` (`points`, widened exactly to float64).
Other kinds are rejected. `min_points` defaults to 2.
Family/minimum length filter import annotations, not recomputed geometry.
Retired/external-only objects are excluded. UUID ordering is stable; `selected`
is attempted first when limited. Only intersecting objects consume the limit,
and each chosen object finishes unless a budget/cancellation stops it.
`maximum_objects=0` means unlimited and ignores `selected`. Runs merge across
storage boundaries without duplicate segments. Results own their data and carry
revision identity from one snapshot; each Python call starts a fresh snapshot.
`options.uuids` optionally restricts the query before selected-object priority and
limits. `None` is unrestricted; an explicit empty list matches nothing. Filter
setup participates in cancellation and work/processing budgets.

C++ `ObjectSelection::coordinates` and `RegionOptions::coordinates` accept
an exact JSON coordinate declaration (missing fields compare as null).
`objectCoordinateGroups(txn, "fiber", cancel)` aggregates current metadata without
materializing object records or decoding geometry. This lets a viewer query each
coordinate frame directly, without constructing an inclusion list for every
object in the store.

A volume target instead queries in one volume's voxels across every frame
declared in it. `ObjectSelection::volume` / `select_objects(volume=...)`,
`spatial_regions(volume=...)` and `RegionOptions::volume` take the volume ID
(the `volume_uuid` of `meta.coordinates`) and match objects declared in it with a
scale factor. Their bbox, minimum length, `length_desc` order and length cursor
use volume voxels, where volume length = stored length × (1 / scale_factor), the
factor `geometryScaleToVolume()` returns. Each frame is queried on its own index
and merged; equal volume lengths order by stored length, then UUID, so a cursor
of the last row's volume length and UUID resumes exactly. Region run points and
spatial region bounds are returned in volume voxels; records keep their stored
coordinates. `include_unbound` appends objects not bound to the volume (including
other volumes' frames) in UUID order after the bound ones, unless a positive
minimum length is set; a cursor into them carries `after_uuid` without
`after_length`. `volume` and `coordinates` are exclusive. Region queries drive the persisted R-tree and order only
spatial candidates by UUID. Geometry and intersection predicates are unchanged.

| Budget | Default | Charged work |
| --- | ---: | --- |
| `output_byte_budget` | 8 MiB | 24 bytes per appended XYZ; a new run needs 48 |
| `processing_byte_budget` | 64 MiB | Payload, verification, metadata, reconstruction and cache bytes |
| `candidate_work_budget` | 1,000,000 | Candidate/block visits and segment tests |
| `sql_step_budget` | 10,000,000 | SQLite VM instructions, including rejected probes |
| `max_unit_bytes` | 1 MiB | Largest admitted processing operation |

Zero budgets allow no work. Work is reserved before processing; whole cached
chunks are charged on every touch and verification/reconstruction separately.
Counters measure work, not unique bytes, disk traffic or memory. `stop` is
`exhausted`, `object_limit`, `output_budget`, `processing_budget`, `candidate_budget`,
`sql_budget` or `cancelled`; only `exhausted` means `complete`. Partial runs remain
valid but imply no spatial coverage. Invalid input/corruption raises.

`RegionCancellation.cancel()` may run from another Python thread; queries
release the GIL. C++ callbacks inspect stats, must not re-enter the store, and
propagate exceptions. Checks occur at SQL steps, candidate visits, admitted
units and segment tests. Hash/JSON/decompression units are indivisible; neither
SQLite instructions nor OS waits have wall-clock bounds. Budgets guarantee
neither latency nor allocator memory. There is no continuation/coverage cache.

## Tests

For an existing core build tree configured with `VC_OBJECT_STORE_BUILD_TESTS=ON`:

```sh
AGENTS_AGENT_MODE=1 ninja -C build-core
AGENTS_AGENT_MODE=1 ctest --test-dir build-core --output-on-failure
```

The C++ tests cover shared-session lifecycle, direct fiber API transactions,
and bounded UUID-filtered spatial reads. To run the Python tests with an existing
Python-enabled build, use its Python interpreter and point to its module directory:

```sh
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 VC_OBJECT_STORE_BUILD_PYTHON_PATH=build/python \
  /path/to/python -m pytest tests/python -q
```

VC3D and CLI imports share `prepareGeometrySource` and the checked batch ingest.
Fibers accept current version-4 documents and version-1 fiber collections;
branch endpoints require `target_uuid`. `branch_file` only locates a missing
peer's transport file. No older-format repair or database migration is performed.
Native export's `--revision` / Python `object_revision_uuid` pins an object
revision, including its metadata and relationship snapshot, and returns its
geometry revision UUID.

Native format clients link `vc_object_store::vc_object_geometry`. Set
`VC_OBJECT_STORE_BUILD_ADAPTERS=ON` and `VC_OBJECT_STORE_BUILD_LINKING=OFF`
for import/export without relationship detectors or their CT reader dependency.
VC3D uses this configuration; the standalone tools/Python build keeps detectors
on by default. The existing OpenCV/TIFF dependencies supply native formats.

## Large-store opening and volume-ID correction

Build and run the read-only Paris 4 timing harness from this directory:

```sh
AGENTS_AGENT_MODE=1 ninja -C build bench_fiber_open
build/bin/bench_fiber_open /path/to/paris4-fibers.vcobj
```

The harness uses the fixed source-voxel viewport centered at `(17000,15500,55000)`,
with half-extents `(1024,1024,3)`, and a 5,000-fiber cap. On Ubuntu arm64,
RelWithDebInfo (`-O2 -g`), three consecutive runs on the 931,104-fiber Paris 4
store gave these mean / min / median / max region-query times (seconds):

| Query | Mean | Min | Median | Max | Coverage |
| --- | ---: | ---: | ---: | ---: | --- |
| UUID-first scan | 3.550 | 2.325 | 2.371 | 5.952 | 50 fibers, SQL budget exhausted |
| R-tree-first candidates | 0.0583 | 0.0578 | 0.0580 | 0.0590 | All 142 intersecting fibers |

SQLite VM work fell from 10,000,000 steps (the budget) to 386,464. The previous
query returned partial coverage, so this is not a like-for-like throughput ratio.
These measurements were collected while incremental builds were also running;
no cold-cache claims are made. Metadata aggregation averaged 2.922 s after the
change, a 200-row page with revision pins 0.0089 s. VC3D performs both in background
workers. The VC3D read-only harness is documented in `docs/fiber-collections.md`
in the sibling project. No numeric representations or predicates were changed.

Older imports may contain a UUIDv5-derived volume binding. Correcting a known
binding is explicit and preserves prior revisions and all geometry. First preview:

```sh
python scripts/rebind_volume.py objects.vcobj --from-id OLD_VALUE --to-id CATALOG_ID
```

After closing editors, add `--apply` with `PYTHONPATH=build/python` to run checked,
batched metadata transactions. The tool is resumable and refuses stores with
relationships, whose coordinate dependencies need explicit review. Unknown
bindings are not inferred. Verify with `PYTHONPATH=build/python python -m pytest
tests/python/test_rebind_volume.py -q` and `PYTHONPATH=build/python python
scripts/import_afv.py --self-test`.

The approved Paris 4 correction on 2026-10-22 updated 931,104 fibers and one
collection from `2d2a9b21-9c82-5343-9443-57838a5fa5a1` to `20260411134726`.
All current bindings were checked; every scale factor remained 1. The SHA-256
fingerprint of all 931,104 sorted geometry-head pairs was unchanged, as were
1,862,208 control points and 344,875,721 line points. Historical metadata was
checked for 20 fibers and the collection. The final read-only VC3D check took
3 ms to attach, 2,931 ms to show the first page, and 2 ms to load the selected
fiber, with 293 UI heartbeats and just one resident editor. This was one
offscreen Ubuntu arm64 run, not a cross-platform rendering benchmark.

Validation passed: five targeted C++ tests, all 30 Python tests, the AFV
importer self-test, and VC3D's `--test-object-store` controller regression.
A fresh import of 1,000 real AFV fibers was also checked for byte-exact line
geometry and the verbatim catalog volume ID.
