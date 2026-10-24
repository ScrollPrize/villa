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

Native export selects an immutable revision and stages output before publication;
`overwrite=False` protects existing destinations. It rejects auxiliary arrays.
Directory export currently requires Linux `renameat2`. Metadata-only
`export_json` supports filters and field projection; it is not a geometry backup.
Typed detail counts, lengths, dimensions and boxes are import summaries: use
current geometry descriptors and spatial queries after edits.

## Revisions and transactions

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
The UUID is explicit, normalized to lowercase, and need not identify a stored
object. Paths, levels, resolution and labels never infer it. Scale is a finite
positive JSON number with direction **stored XYZ = volume voxel XYZ × scale**;
it differs from physical resolution and patch grid scale. Values are metadata
only and never rescale payloads or detector parameters. Unknown/missing members
mean `null`, not scale 1. Setting coordinates replaces the whole field and
rejects unknown keys; positive 64-bit integers stay exact.

Native object/document/collection declarations, flat `volume_uuid`/`scale_factor`
fields and `ingest_geometry_source(..., coordinate_context={...})` must agree
where known. Packed tracks use context. Set coordinates with `set_meta`; dependent
anchors declare `metadata_keys=["coordinates"]`.

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

## Bounded oriented fiber browsing

`fiber_region(FiberSlice, FiberRegionOptions, cancel=None)` combines conservative
spatial bounds with exact double-precision segment/slab intersections. It returns
original unclipped XYZ and zero-based segment indices in contiguous runs; the
consumer handles rendering/trimming. `geometry/SegmentClip.hpp` shares the clipper.

```python
from vc_object_store import FiberSlice, FiberRegionOptions

view = FiberSlice()
view.origin = [100, 200, 300]
view.u, view.v, view.normal = [1, 0, 0], [0, 1, 0], [0, 0, 1]
view.local_bounds = [-50, -50, -.5, 50, 50, .5]
options = FiberRegionOptions()
options.maximum_fibers = 100
result = store.fiber_region(view, options)
```

Axes must be finite/orthonormal (tolerance `1e-12`); bounds are finite and ordered,
with zero thickness allowed. Segments can intersect with both endpoints outside.
Nonempty `line_points` take precedence over controls; `min_points` defaults to 2.
Family/minimum length filter import annotations, not recomputed geometry.
Retired/external-only fibers are excluded. UUID ordering is stable; `selected`
is attempted first when limited. Only intersecting fibers consume the limit,
and each chosen fiber finishes unless a budget/cancellation stops it.
`maximum_fibers=0` means unlimited and ignores `selected`. Runs merge across
storage boundaries without duplicate segments. Results own their data and carry
revision identity from one snapshot; each Python call starts a fresh snapshot.

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
`exhausted`, `fiber_limit`, `output_budget`, `processing_budget`, `candidate_budget`,
`sql_budget` or `cancelled`; only `exhausted` means `complete`. Partial runs remain
valid but imply no spatial coverage. Invalid input/corruption raises.

`FiberRegionCancellation.cancel()` may run from another Python thread; queries
release the GIL. C++ callbacks inspect stats, must not re-enter the store, and
propagate exceptions. Checks occur at SQL steps, candidate visits, admitted
units and segment tests. Hash/JSON/decompression units are indivisible; neither
SQLite instructions nor OS waits have wall-clock bounds. Budgets guarantee
neither latency nor allocator memory. There is no continuation/coverage cache.

## Tests

[Build instructions](../README.md#build) include the pytest command.
