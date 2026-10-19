# Object database and relationship metadata

`ObjectStore` is the primary database: one SQLite file (`objects.vcobj`) owns
object identities, annotations, immutable geometry revisions, content-addressed
binary chunks, and relationships between objects. Edges, anchors, approvals,
and endpoint revision bindings are metadata in that same database. Objects can
exist without geometry or relationships.

The C++ database library is `vc_objects`. Headers, record types, and transactions
live under `vc/core/objects/` in `vc::objects`. Native file import/export lives
in `vc_object_geometry`; link detectors live in `vc_object_links`. These higher
layers depend on the object store. The database library itself does not depend
on CT readers or Qt. Python exposes the database and its helpers in `vc.objects`.

The database uses SQLite application ID `0x56434f42`, schema version 7.
Writable opens migrate versions 2–6 transactionally. Versions 2–4 build spatial
bounds from current geometry, reading the current payloads once; version 5 only
advances the schema marker to permit tiled patch storage. Version 6 adds the
edge approval/cursor and typed family/length query indexes. Existing geometry
manifests, revision identities, and payloads are never rewritten by migration.
Read-only opens can inspect older versions without migration. Whole-object
spatial queries require version 4; fine spatial regions require version 5.
Older binaries cannot open version 7. SQLite must include its R-tree module.
Metadata JSON exports use `vc_object_metadata` version 2. Geometry ingestion and
relationship discovery operate on the same store. VC3D/Spiral geometry readers
are separate consumers.

## One handle and one transaction

```python
from vc.objects import ObjectStore

with ObjectStore.create("objects.vcobj", author="sean") as store:
    with store.write() as tx:
        patch_id = tx.upsert_object({"kind": "patch", "name": "patch-001"})
        first = tx.put_geometry(patch_id, patch_geometry, expected_revision=None)
        other_id = tx.upsert_object({"kind": "patch", "name": "patch-002"})
        edge_id = tx.add_edge({"kind": "same-surface", "a": patch_id, "b": other_id,
                               "a_anchor": {"whole_geometry": True},
                               "b_anchor": {"whole_geometry": True}})

```

`patch_geometry` uses the array dictionary described below.
Use one `with store.write()` block for a batch of metadata updates instead of
committing each key separately. For independent bulk updates, batches of roughly
100–1,000 changes balance commit cost against writer lock duration; keep changes
that must be atomic in the same transaction. Connections reuse prepared
statements, and commits retain SQLite's `synchronous=FULL` durability.
C++ uses `ObjectStore::upsertObject`, `putGeometry`, `addEdge`, and
`repairEdgeGeometry` with the same `WriteTxn`. Transactions can commit or
roll back object records, geometry, and relationships together. There is one
relationship set: no graph membership copies, explicit adoption, or independent
graph revision selections. A stable store UUID identifies the database even
when its file moves.

## Source summaries and current geometry

The `fiber`, `patch`, and `track` detail records are editable source/import
annotations. Their counts, lengths, grid dimensions, scale, and bounding boxes
are captured during ingestion. `put_geometry()` preserves these fields, just
as it preserves names and other annotations; they are not computed views of the
current revision. Re-ingestion refreshes the geometry summaries when publishing
changed source geometry.

For current geometry, use `current_geometry()` for array shapes and patch
`attributes["scale"]`. `select_objects(bbox=...)` and `spatial_regions(...)` use
the managed geometry's spatial bounds from the same transaction snapshot,
regardless of the annotated boxes. Objects without managed geometry use their
annotated boxes. Callers that maintain updated summary annotations can publish
those metadata changes together with geometry in one write transaction.

## Ingesting geometry

Build the importer and bindings with Ninja:

```sh
ninja -C build vc_object_ingest vc_object_export vc_object_bindings
```

Ingest files referenced by an existing object store, optionally rebasing its relative
source hints to a dataset directory:

```sh
build/bin/vc_object_ingest --db /data/objects.vcobj --author sean \
    --registered --dataset-root /data/dataset
```

`--uuid UUID` restricts ingestion to selected registered objects and is repeatable.
The importer also creates a new database and registers missing objects:

```sh
build/bin/vc_object_ingest --db /data/objects.vcobj --author sean \
    --patch /data/dataset/verified_patches/patch1 \
    --fiber /data/dataset/fibers/fiber1.json \
    --point-collections /data/dataset/relative_windings.json \
    --tracks /data/tracks.dbm.vctracks
```

Source options are repeatable. Fibers, patches, and collections preserve source
UUIDs. Track identity remains `(dataset_ref, external_key)`, where external_key
is the packed source ID. A DBM path is accepted only when its current packed
mirror exists; the importer does not decode DBM/pickle itself. Existing track
dataset aliases and object UUIDs are reused.

Source files must remain stable during ingestion. Patch size/mtime checks detect
ordinary concurrent writes, and incomplete patch saves are rejected before
loading. Geometry decoding happens outside write transactions. Registered-object
batches and each fiber-directory/collection source publish atomically, including
their native relationships. Any invalid endpoint or anchor rolls back that batch.
Packed tracks publish individually and report partial success. A nonzero CLI exit
code indicates problems. A concurrent object edit aborts publication.

On first ingestion, tags and typed fiber/patch/track properties become object
fields. Other native annotations live in editable `object.meta["source"]`.
Fiber branches become `fiber_branch` edges; point links become `point_link`
edges with point-ID anchors; linked winding collections become
`collection_winding` edges. Distinct anchors within the same object are supported.
These edges start pending and bind to the ingested geometry revisions. Ingest
connected fibers together with `--fiber DIR`, or supply a branch's `target_uuid`
for a target already stored. Source filenames are resolved only during ingestion;
stored edges use UUIDs. Point and collection source IDs are scoped to their
originating JSON document.

Subsequent ingestion retains edited DB annotations and relationships, including
manual deletions, and refreshes geometry measurements. Unchanged ingestion is a
no-op. Changes to referenced points require explicit retain/update/remove
relationship decisions through `put_geometry`; ingestion refuses such replacements.
Source metadata is seeded when an object does not yet have `meta.source`.

## Discovering relationships from the store

Ingestion and discovery are separate operations on the same `ObjectStore`.
After ingestion, discovery reads current immutable geometry revisions from the
SQLite file; original JSON, TIFF and packed track files are no longer needed.
Discovery does not register objects or refresh object measurements.

```python
with ObjectStore.create("objects.vcobj", author="sean") as store:
    report = store.ingest_geometry_source("patch", "/data/patch-a")
    assert not report["problems"], report["problems"]
    report = store.ingest_geometry_source("patch", "/data/patch-b")
    assert not report["problems"], report["problems"]
    report = store.link_patch_overlaps()
    print(report)
```

| Discovery method | Optional object selection | Other input |
| --- | --- | --- |
| `link_patch_overlaps()` | `patch_uuids` | None |
| `link_point_collections_to_patches()` | `collection_uuids`, `patch_uuids` | None |
| `link_tracks_to_tracks()` | `track_uuids`, `dataset_ref` | None |
| `link_fibers_to_patches(ct_volume)` | `fiber_uuids`, `patch_uuids` | CT or `normals_manifest` |
| `link_tracks_to_patches(ct_volume)` | `track_uuids`, `patch_uuids` | CT or `normals_manifest` |

Empty UUID selections mean all objects of that kind; unknown/wrong-kind UUIDs
are errors. Objects without stored geometry are reported, with no fallback to
source paths. Track dataset references are grouping labels, not files to open.
CT/normal-field data remains external because it supplies the detector's normal
field rather than object geometry. Fiber strips can optionally use an explicit
`cache_directory`; the default computes them in memory.

The C++ equivalents in `vc/core/object_links/ObjectLinks.hpp` accept the store
and parameter structs with the same selections (`patchUuids`, etc.). CLI link
flags are now bare flags, for example:

```sh
build/bin/vc_object_links --db objects.vcobj --author "$USER" --link-patch-overlaps
```

Each pass reads one snapshot, computes candidates, then checks the observed
object generations and geometry revisions in its write transaction. Changed
inputs abort publication. New links bind to those revisions. Reruns revalidate
the detector's own pending links against the geometry it actually examined;
reviewed/manual links remain untouched. Stale pending removal is limited to
successfully processed selected endpoints. Inspect each report's `problems`.

## Representations

All arrays use contiguous C order and explicit little-endian dtype descriptors.
Coordinates are unscaled source XYZ; they are not transformed to fitting units.

| Kind | Arrays | Immutable attributes |
| --- | --- | --- |
| `patch` | `points`: `<f4`, `[rows, columns, 3]` | Two-axis `scale` (immutable); other source fields in object metadata |
| `fiber` | `control_points`, `line_points`: `<f8`, `[N, 3]` | Object metadata holds source fields and per-point tracing settings |
| `point_collection` | `points`: `<f8`, `[N, 3]`; `point_ids`: `<i8`, `[N]` | Object metadata holds collection fields and ordered winding annotations |
| `track` | `points`: `<i4`, `[N, 3]` | Source ID/family in object track detail |

Patch coordinates follow the existing QuadSurface loader, including invalid
points and applied masks. Ancillary TIFF channels such as rendered images and
generation visualizations are not ingested. Fiber point order is retained.
Collection points are ordered by numeric ID; IDs are retained, not renumbered.
Track ZYX integers are reordered to XYZ without rescaling or reducing precision.
Coordinates and point IDs are binary arrays. Immutable geometry attributes hold
format properties such as patch scale. Source annotations are editable object
metadata, not geometry revisions. Updating object
metadata does not advance geometry.

## Revisions and access

`geometry_revisions` records the object UUID, parent revision UUID, content hash,
manifest, object generation at publication, and author/time/host stamps.
`geometry_heads` selects the current revision for each object. A revision's
manifest lists named arrays, dtype, shape, byte length, physical storage layout,
and ordered chunk hashes.
`geometry_chunks` stores raw BLOBs of at most 1 MiB, keyed by SHA-256; identical
chunks are shared across objects and revisions. `geometry_revision_chunks`
provides foreign-key-backed references to their payloads. SQL triggers reject
updates/deletes of published revisions and chunks. No pruning/GC API is provided
in this version; deleting an object with retained geometry is also refused.

`putGeometry` takes an expected current revision. Null means the object must
have no geometry yet. A mismatched expectation fails, including on an otherwise
identical write. Identical content with the correct expectation returns the
current revision. A changed payload creates a new revision whose parent is the
previous head; restoring old content creates a new revision and reuses chunks.
Metadata-only edits do not copy or advance geometry.

Python mirrors the C++ API:

```python
from vc.objects import ObjectStore
import numpy as np

with ObjectStore.open('/data/objects.vcobj', read_only=True) as db:
    revision = db.current_geometry(object_uuid)
    descriptors = {a['name']: a for a in revision['arrays']}
    a = descriptors['points']
    points = np.frombuffer(
        db.read_geometry_array(revision['uuid'], 'points'),
        dtype=a['dtype'],
    ).reshape(a['shape'])
    history = db.geometry_history(object_uuid)
```

`geometry_revision(revision_uuid)` and `geometry_history(object_uuid)` return
only descriptors and attributes. `read_geometry_array` accepts optional
`byte_offset` and `byte_count` for bounded reads and verifies every touched
chunk against its hash. It works without the source files. Read transactions
see a coherent SQLite snapshot of heads, revisions, and payloads.

For many point reads from the same array, use
`read_geometry_ranges(revision_uuid, array_name, ranges)`, where each range is
`(byte_offset, byte_count)`. The result is a list of byte strings in caller order.
C++ exposes `readGeometryRanges` with an existing transaction. These calls share
one verified manifest and a bounded 8 MiB LRU chunk cache. Range batches plan
copies by physical chunk, verify each touched chunk once and restore caller
order, including overlapping, duplicated and unaligned ranges. Small explicit
single-array ranges (up to 1 MiB) also reuse the validated read cache; full-array
reads stream without retaining the full payload. Geometry edit and repair
operations also reuse readers and stable-point-ID indexes across relationship
checks. Write/internal readers remain scoped to their operation.

For patch rectangles, use
`read_geometry_rectangle(revision_uuid, array_name, row, column, rows, columns)`
(or C++ `readGeometryRectangle`). It returns densely packed, row-major float32
XYZ bytes, with exact preservation of stored bit patterns. Coordinates and
counts are in vertices; the array must be a patch float32 XYZ array. Empty
rectangles are allowed, and out-of-bounds rectangles fail. Both old linear and
new tiled storage layouts are supported.

For several rectangles from one revision/array, use
`read_geometry_rectangles(revision_uuid, array_name, rectangles)` (C++
`readGeometryRectangles`), with each rectangle `(row, column, rows, columns)`.
Results preserve caller order, including duplicates and empty rectangles. A batch
shares the manifest and verified chunks even when retention between calls is disabled.

Rectangle and batched byte-range reads retain verified readers on pooled read
connections. Each connection has a **16 MiB retention budget** by default, charging
payload bytes and conservative metadata overhead, with at most 64 revisions and
8 MiB of chunks per revision. Thus the default four-reader pool can retain up to
64 MiB of charged cache data; active reads, returned buffers, allocator overhead
and SQLite's own page cache are separate. `geometry_cache_bytes` on Python
`ObjectStore.create/open`, or C++ `OpenOptions::geometryCacheBytes`, changes the
per-connection budget; zero disables retention. Oversized readers are not retained.

A read connection pins its SQLite snapshot before checking `PRAGMA data_version`.
A change discards that connection's retained readers, including after commits by
another process or this store's writer. Concurrent snapshots have separate caches;
a snapshot cannot obtain a revision cached by a newer snapshot. Any commit currently
invalidates retention, including metadata-only changes. Write transactions never use
this cache, so rolled-back publications cannot become cached readable revisions.
Newly loaded manifests/chunks are hash-verified; cache hits reuse verified RAM bytes.
`read_geometry_array`, exports, sync and internal publication/integrity paths still
use fresh readers. No schema change or geometry migration is needed.

Spatial region queries also have a separate **16 MiB per read connection** budget,
configured with Python `spatial_cache_bytes` or C++ `OpenOptions::spatialCacheBytes`.
The first eligible query lazily loads descriptors and builds a resident BVH with
exact double bounds. Subsequent bounding-box queries, including kind/UUID filters
and ordered paging, use that index. Tag/connectivity filters and write transactions
use SQL. Setting the budget to zero disables the resident spatial index.

The complete index must fit: a scan that exceeds the charged budget is discarded,
and that connection uses SQL until its snapshot version changes. The scan stops
at the budget rather than sorting the entire database in SQL first. Charges cover
descriptor/vector capacities, index nodes and conservative allocation overhead;
temporary construction and result buffers are separate. With both defaults, four
reader connections can retain up to **128 MiB** of charged geometry and spatial
cache data combined, in addition to SQLite's page cache.

Both caches share the snapshot-pinned invalidation token. Any observed commit
discards the resident spatial index too, including external SQL commits without
journal entries. First use after invalidation pays index construction again.
This trades startup/rebuild latency and memory for faster queries on a stable
snapshot; workloads dominated by isolated queries or frequent commits can disable
the spatial cache.

New patch `points` revisions use non-overlapping **128×128-vertex storage tiles**
by default, independently of the 32×32-cell R-tree tiles. A full physical tile
contains 192 KiB; boundary tiles store only their actual vertices, without
padding. One 33×33 spatial region touches at most four physical tiles, regardless
of surface width. Logical array layout, shape, dtype, and all coordinate bytes
remain unchanged. Full-array reads and arbitrary byte-range reads reconstruct
that same logical row-major sequence. Other geometry arrays retain their original
1 MiB linear chunks.

`put_geometry(..., patch_tile_size=32|64|128)` selects a different tile size for
new patch revisions; `patch_tile_size=0` selects legacy linear storage. C++ uses
the final `GeometryStorageOptions` argument to `putGeometry`. This preference
never forces a new revision for unchanged content: no-op publication and affected
relationship checks compare against the current revision's layout. A subsequent
actual edit uses the requested layout. There is no automatic history repacking
or explicit repack API in this change.

Storage metadata lives per array in the immutable manifest, separately from the
logical geometry format version (still 1). Tiled arrays use
`storage={"version":1,"layout":"tiles","tile_size":128}`; missing storage metadata
means legacy linear chunks. Physical chunks are ordered by tile row then tile
column. Readers and sync reject unknown layouts, inconsistent sizes, missing
chunks, and hash mismatches. Sync preserves incoming layouts and chunk hashes.
After a committed database change, later rectangle/range reads invalidate their
retained cache and reverify stored bytes, including after a prior successful read.

To publish arrays directly, supply a dictionary with `kind`, optional
`attributes`, and an `arrays` list. Each array has `name`, `dtype`, `shape`, and
`data` as Python bytes. Supported dtypes are `|u1`, `<i4`, `<i8`, `<u8`, `<f4`,
and `<f8`; each object kind requires the layouts listed above. Patch attributes
must contain a finite, positive two-axis scale. Both `db.put_geometry(...)` and
`with db.write() as w: w.put_geometry(...)` accept `expected_revision` and an
`edge_edits` list of explicit relationship decisions.
The C++ equivalents accept `GeometryData` and an existing `WriteTxn`, so geometry
and other object changes can commit atomically. Publication uses an internal
savepoint: catching an error and committing other work cannot retain a partial
geometry revision.

Python ingestion methods are `ingest_geometry_source(kind, source)` and
`ingest_registered_geometry(dataset_root=None, object_uuids=[])`. They return
`ingested`, `unchanged`, `edges_imported`, and `problems`, and own their transactions.

## Exporting native files

Export the current geometry or a selected historical revision without accessing
its original source files:

```sh
build/bin/vc_object_export --db /data/objects.vcobj --uuid OBJECT_UUID \
    --output /data/export/patch
# Add --revision REVISION_UUID to select history, or --overwrite to replace output.
```

```python
with ObjectStore.open('/data/objects.vcobj', read_only=True) as db:
    exported_revision = db.export_geometry(object_uuid, destination)
    db.export_geometry(object_uuid, destination,
                       revision_uuid=old_revision_uuid, overwrite=True)
```

C++ callers use `exportGeometry` from `vc/core/objects/GeometryExport.hpp` in
`vc_object_geometry`. Both APIs return the exported revision UUID. The selected
revision must belong to the requested object. Export reads a consistent snapshot
and releases the read transaction before writing files. It does not alter store
objects, approvals, source hints, revisions, or relationships. To export an edit,
first publish it with `put_geometry`, then export that revision.

| Kind | Destination contents |
| --- | --- |
| Patch | Directory containing float32 `x.tif`, `y.tif`, `z.tif`, and `meta.json`; dimensions and bounds recalculated, scale retained |
| Fiber | `vc3d_fiber` JSON with coordinates, captured annotations, and tracing settings |
| Point collection | Version-1 point-collections JSON containing this one collection, retaining collection key, point IDs, and annotations |
| Track | One-track packed directory containing header, ZYX int32 coordinates, offsets, source ID, family, z bounds, arclength, tortuosity, and metadata |

Track exports record the object/revision UUIDs in `metadata.json`, but do
not invent a DBM source signature. They are standalone packed stores, not caches
of an external DBM file. Re-ingesting at a new dataset path uses the existing
`(dataset_ref, external_key)` identity rule and therefore registers a new track
object. No Spiral reader integration is included.

The destination's parent must already exist. Existing files/directories are
refused unless `overwrite=True` / `--overwrite` is explicit. A directory is
replaced in full; a collection JSON is replaced in full, not merged. Symlink
destinations and file/directory type changes are refused. The open database,
its sidecar paths, and directories containing it cannot be overwritten.

Exports are staged beside the destination and published with one atomic rename.
On Linux, `renameat2` provides no-replace publication and atomic exchange of
existing directories; unsupported filesystems fail without a destructive
fallback. Directory export currently requires Linux. Caught failures clean up
staging and preserve an existing destination. This guarantees atomic visibility,
not power-loss durability: a process crash can leave a hidden `.vc-export-*`
staging directory, and no export-level fsync protocol is provided.

These are reconstructed native geometry files, not byte-for-byte source archives.
Patch masks are already baked into the stored geometry; masked-out original
coordinates and ancillary TIFFs cannot be recovered. Patch invalid samples must
use `(-1,-1,-1)`; a different sample with `z <= 0` is rejected because the native
loader would discard it. Empty patch grids and nonfinite floating coordinates
are also rejected. Exports use current object names, tags, typed properties, `meta.source`
annotations, and stored native relationships. This also applies when selecting
historical geometry: annotations are current, not historical snapshots. Native
relationships must be valid for the exported revision: their bindings must match
or all declared point dependencies must be unchanged; otherwise export fails. Generic relationships remain available through metadata JSON export.
Other derived source annotations remain the caller's responsibility when editing
geometry. If point counts change, update or remove
the corresponding positional annotation lists. Export rejects mismatched
annotation counts, duplicate/negative collection point IDs, and auxiliary arrays
that native formats cannot represent, avoiding silent data loss. It does not
validate application-specific tracing semantics or update trace results.

## Point relationships and geometry edits

Each geometry endpoint declares exactly which points the relationship references.
An anchor's `points` list contains array names and point coordinates (excluding
the final XYZ axis). The first entry is the attachment point; additional entries
can describe the vertices or neighbours supporting an interpolated attachment or
direction. The producer must declare every point the relationship depends on.

| Geometry | Point reference |
| --- | --- |
| Fiber control point 5 | `{"array": "control_points", "index": [5]}` |
| Fiber line point 12 | `{"array": "line_points", "index": [12]}` |
| Patch row 20, column 145 | `{"array": "points", "index": [20, 145]}` |
| Collection stable point ID 42 | `{"array": "point_ids", "index": [42]}` |
| Track point 8 | `{"array": "points", "index": [8]}` |

Use `whole_geometry: true` for an actual object-wide relationship such as patch
overlap. It cannot be combined with point references. Geometry-backed endpoints
must declare dependencies when a relationship is created or updated. Relationships
between objects without managed geometry may omit them; introducing geometry then
requires explicitly updating those relationships or removing them.

`affected_geometry_edges(object_uuid, geometry)` previews the relationships whose
dependencies change. `put_geometry` checks them again inside the write transaction.
For every affected relationship, the caller must supply exactly one decision:

- **retain**: keep the same relationship, point references, and annotations.
- **update**: supply both complete replacement anchors in the stored edge's a/b order.
- **remove**: delete the relationship.

No decision is selected automatically. Missing, duplicate, unrelated, invalid, or
outdated decisions reject the edit. A deleted point cannot be retained. A topology
change requires decisions for positional references; stable collection IDs survive
reordering and must never be reassigned to different points. Geometry attributes
such as patch scale also participate in dependency checks.

For example, create a fiber-to-patch attachment:

```python
edge_id = db.add_edge({
    "kind": "contact", "a": fiber_id, "b": patch_id,
    "a_anchor": {"points": [{"array": "control_points", "index": [5]}]},
    "b_anchor": {"points": [{"array": "points", "index": [20, 145]}]},
})
```

When changing fiber point 5, the caller can explicitly retain that attachment:

```python
edge = db.edge(edge_id)
a = db.current_geometry(edge["a"])
b = db.current_geometry(edge["b"])
decision = {
    "edge_uuid": edge_id,
    "expected_generation": edge["generation"],
    "a_revision": a["uuid"] if a else None,
    "b_revision": b["uuid"] if b else None,
    "action": "retain",
}
new_revision = db.put_geometry(
    fiber_id, edited_geometry, expected_revision=original_revision,
    edge_edits=[decision],
)
```

The revisions in each decision are the endpoint heads observed **before** the edit.
For `update`, add `a_anchor` and `b_anchor` containing the new references and any
position/direction annotations. For `remove`, change `action` to `"remove"`.
The complete decision list must cover the affected set returned by the preview;
the example has one affected edge. C++ uses `GeometryEdgeEdit` and
`affectedGeometryEdges` with the same transaction handle.

Point references are authoritative for attachment identity. XYZ and direction
annotations are caller-owned observations; retain does not recompute them. Use
update to replace them when needed. Native fiber exports resolve point positions
from the referenced geometry rather than treating an annotation as current geometry.

Changes to unrelated points leave edge records, approvals, journals, and bindings
untouched. A binding to an older revision remains valid if all declared point
dependencies are unchanged. `edge_geometry_state` checks these dependencies.
Retain/update publishes anchors and bindings together, resets approval to pending,
and records one `geometry_repaired` journal entry. Geometry, decisions, approvals,
and journals share a savepoint: a caught failure cannot leave partial changes when
the caller commits other work. The database uses WAL with FULL synchronous writes.

For a standalone relationship correction, `repair_edge_geometry(repair)` accepts
`edge_uuid`, `expected_generation`, `a_revision`, `b_revision`, `a_anchor`, and
`b_anchor`. It checks the exact current endpoint revisions and edge generation,
publishes atomically, and never selects replacement points. Identical anchors and
bindings are a no-op after validating those expectations. C++ uses
`repairEdgeGeometry(WriteTxn&, EdgeGeometryRepair)`.

## Retention, sources, and backup

Revisions and chunks are immutable and retained. There is no pruning/GC API;
objects with retained geometry cannot be deleted. Objects referenced by edges
also cannot be deleted until those edges are removed.

Objects with geometry use `storage_kind='sqlite'`; their `data_checksum` and
`data_generation` describe the current geometry. Metadata upserts cannot replace
these managed storage fields. `storage_ref` remains a source-location hint for
explicit re-ingestion, never the authoritative geometry. `resolve_storage_path`
can locate that source by UUID. Source hints are used only for explicit re-ingestion; geometry access and
relationship discovery read the database.

Use SQLite's backup API for live copies, or stop writers, checkpoint, and copy
the database. One backup contains objects, geometry history, relationships,
approvals, revision bindings, and the journal. Copies retain the store identity;
each independently edited copy needs a separate replica identity. Call
`copy.fork_replica()` on a new copy, then synchronize it with the original **before
editing either copy** to establish their shared baseline.

## Metadata selection and export

`select_objects(bbox=[min_x, min_y, min_z, max_x, max_y, max_z])` uses one
persisted SQLite R-tree shared by all geometry kinds. It returns objects whose
axis-aligned bounds overlap the query, including boundary contact. This is a
candidate query in source XYZ coordinates, not an exact surface/line intersection.
Use `spatial_regions` below to refine these candidates without loading full arrays.

Each object has one summary entry plus entries for its current geometry regions.
Current stored geometry determines the bounds, including point collections and
both line/control points of fibers. Non-finite coordinates are ignored. Patch
indexing covers the union of VC3D's validity conventions: finite points with
X != -1 **or** Z > 0. This excludes the standard (-1,-1,-1) sentinel but preserves
geometry accepted by either raw-surface or mapped-surface consumers. Consumers
still apply their own validity rules during exact intersection.
Empty/entirely invalid geometries have no entry. Before geometry is ingested,
fiber/patch/track annotations can supply bounds. Metadata edits cannot override
the index for stored geometry; typed detail `bbox` fields remain annotations.
Publication, deletion, rollback, and sync maintain
the index transactionally; readers see bounds from their transaction snapshot.
Exact double bounds are retained and checked after R-tree candidate discovery,
so floating-point rounding in the R-tree does not change overlap results.
Query and annotated bounds must be finite and ordered.

For lightweight records, pass `fields` to `select_objects`. UUID and kind are
always returned; `fields=[]` selects just those identities. `fields=None`
(the default) returns complete records. Fields are top-level object fields,
such as `name`, `generation`, `tags`, `meta`, or the complete `track` detail.
Unknown fields are errors. Unrequested metadata, tags, and detail tables are
not loaded or decoded. Spatial, kind, tag, UUID and component filters intersect
before pagination. Component selection still traverses the entire component.

```python
visible = store.select_objects(
    bbox=[100, 200, 300, 400, 500, 310],
    object_kind="fiber", fields=["name", "generation"], limit=100,
)
```

C++ uses `ObjectSelection::bbox` and the optional fields argument to
`selectObjects`. Unrequested members of its returned `ObjectRecord`s retain
default values; callers must not upsert projected records as complete objects.
The composite `(kind, uuid)` index also supports ordered kind-filtered pages.

`select_objects` and `export_json` share filters for `object_kind`, `tag`,
`object_uuids`, and `connected_to` (the full connected component). `edge_kinds`
restricts component traversal and exported edges. Filters intersect. Empty UUID
and edge-kind lists mean unrestricted. Export contains only edges whose endpoints
are both selected, with geometry UUID/hash references and endpoint revision
bindings; binary payloads are never included.

```python
selection = dict(connected_to=object_uuid, edge_kinds=["fiber_branch"], tag="reviewed")
records = store.select_objects(**selection)
page = store.select_objects(**selection, limit=100, offset=0)
metadata_json = store.export_json(**selection, fields=["name", "tags", "meta"])
```

Selections use UUID order. `select_objects` accepts nonnegative `limit` and
`offset`; omitted `limit` returns all matches. C++ uses `ObjectSelection::limit`
and `offset`. Separate Python calls use separate snapshots, so concurrent edits
can change page boundaries; C++ callers can hold one read transaction across
pages when they need a consistent listing. Filters run in SQL before records,
tags and typed details are loaded. Neighbour and component queries probe endpoint
indexes without decoding full edge records.

For `export_json`, `fields` projects top-level record fields; UUIDs, kinds, edge endpoints and
geometry references are always retained. An empty list exports all metadata.
`import_json` accepts complete metadata-only records without managed geometry.
Projected exports are for inspection; use synchronization to transfer objects
with geometry and their revision bindings.

## Consumer queries and store-backed graph reads

`select_edges` filters kind, approval states, tag and endpoint sets before loading
records. It returns UUID-ordered pages; `after_uuid` is an exclusive cursor that
remains usable if the cursor edge is deleted. Projection always includes
`uuid`, `kind`, `a`, and `b`; `fields=[]` requests only those identities.
`endpoint_uuids=None` is unrestricted; `endpoint_uuids=[]` matches nothing.
`both_endpoints=True` restricts both endpoints to the set, otherwise either may
match. Empty `approval_states` includes every state. Parallel and self-links
retain their individual identities and anchors.

```python
page = store.select_edges(
    edge_kind="crossing", approval_states=["approved"], limit=100,
    fields=["a_anchor", "b_anchor", "meta"],
)
next_page = store.select_edges(
    edge_kind="crossing", approval_states=["approved"], limit=100,
    fields=["a_anchor", "b_anchor", "meta"], after_uuid=page[-1]["uuid"],
) if page else []
```

Typed catalog queries support `family`, `min_length` and `order_by="length_desc"`
for `object_kind="fiber"` or `"track"`. Length uses the ingested summary in source
units, not a new geometry measurement. Ties use ascending UUID order. The cursor
is the last row's `(length, uuid)`; supply both `after_length` and `after_uuid`.
Default UUID ordering also accepts `after_uuid`. Cursor and offset cannot be
combined. `spatial_regions` accepts family/minimum-length filters before its
region limit. Convert physical UI units to the source units before querying.

```python
page = store.select_objects(
    object_kind="track", family="horizontal", min_length=100,
    order_by="length_desc", limit=200, fields=["name", "track"],
)
```

C++ equivalents are `selectEdges(txn, EdgeSelection, fields)` and the additional
`ObjectSelection` fields. Python convenience queries each open a new snapshot;
use one C++ transaction when an entire sequence of pages must be consistent.
Connected selections still compute the complete component before applying the
page limit. Component membership uses internal IDs and is reused within the same
read transaction, with a bounded cache; it is never reused after a write or
across separate snapshots.

`graph_read` pins one read transaction and provides a compact in-memory view of
that store's graph. There is no exported file, sidecar or second database.
The Python default includes only approved edges; pass `approval_states=[]` to
include every state. `object_kind`, `object_uuids` and `bbox` select nodes;
`edge_kind` and `edge_tag` further restrict relationships. Both endpoints must
be selected. Nodes include isolated objects. As with `select_objects`, an empty
`object_uuids` list is unrestricted.

```python
with store.graph_read(object_kind="track", edge_kind="crossing",
                      approval_states=["approved"]) as graph:
    node_uuids = graph.object_uuids
    revisions = graph.geometry_revisions
    offsets = graph.offsets
    partners = graph.partners
    edge_indices = graph.edge_indices
    neighbours_of_first = graph.neighbours(0) if node_uuids else None
    # Optional projected records and geometry come from the same snapshot.
    crossings = graph.edges(fields=["a_anchor", "b_anchor", "meta"])
    summaries = graph.objects(fields=["track"])
    # graph.read_geometry_ranges(node_index, "points", [(byte_offset, byte_count)])
```

`offsets`, `partners`, `edge_indices`, `endpoints`, and neighbor slices are
read-only NumPy views of native uint64 arrays, not Python records. Edge rows
correspond to `edge_uuids` and `graph.edges()`; node rows correspond to
`object_uuids`, `geometry_revisions`, `generations`, and `graph.objects()`.
Every edge has two CSR incidences, including self-links; parallel links are not
deduplicated. `endpoints` has shape `[edge_count, 2]`. `journal_sequence` identifies
the journal position observed by the view. UUID-keyed endpoint metadata and
anchor dependencies preserve source identities; no coordinate conversion or
application-specific crossing interpretation is applied automatically.

Views are thread-confined and must close on their creating thread. They use a
reader-pool slot and hold a WAL snapshot, so close them after preparing the fit's
inputs, before checkpointing, or before opening more views than the reader pool
can supply. Metadata/geometry methods fail after close; previously obtained NumPy
arrays remain valid through their own memory owner. Reopen the view explicitly
when accepting changed inputs. A store cannot close while its view is active.
C++ `graphReadView(txn, objects, edges)` supplies the same topology and revision
identities using the caller's transaction and explicit `EdgeSelection` policy;
its default edge selection is unfiltered.

For fitting inputs, `graph.read_geometry_arrays(array_name, nodes=None)` loads
the selected nodes' pinned revisions into one packed native buffer. `nodes=None`
uses graph node order; an explicit list preserves its order and duplicates, and
`nodes=[]` returns an empty result. Missing geometry or arrays fail explicitly.
`store.read_geometry_arrays(revision_uuids, array_name)` provides the same loader
for known revisions in one read transaction. C++ uses `readGeometryArrays`.

```python
with store.graph_read(object_kind="track", edge_kind="crossing") as graph:
    source_uuids = graph.object_uuids
    topology_offsets, partners = graph.offsets, graph.partners
    packed = graph.read_geometry_arrays("points")
    points, track_offsets = packed.values, packed.offsets

# These arrays remain valid after the read transaction closes. Fitting gathers
# use ordinary NumPy indexing, without a database call or an export step.
first_track = points[track_offsets[0]:track_offsets[1]] if source_uuids else points
```

`PackedGeometryArray.values` is a read-only NumPy view with the stored dtype,
byte order and coordinate convention. Arrays concatenate along the first
dimension; `offsets` contains one more entry than the number of requested
revisions and indexes that dimension, not bytes. Dtypes and trailing dimensions
must match. Tiled and linear physical layouts can be combined when their logical
dimensions match. `data` exposes the same buffer as read-only bytes, and `shape`
and `dtype` describe it. Every view owns its native buffer independently of the
store/view/wrapper lifetime. No payload conversion or Python bytes copy occurs
when obtaining `values`. Empty revision lists have dtype `|u1`, shape `[0]`,
and offsets `[0]`; selected empty arrays retain their declared dtype and shape.

Packed loading verifies manifests and every selected chunk on each call, using
bounded SQL parameter batches and the same integrity/layout helpers as ordinary
reads. The buffer occupies the full selected payload size, plus offsets and
temporary layout/hash metadata. Keep the returned arrays for repeated fitting
access; calling the loader repeatedly repeats the read and verification work.
For subsets larger than available RAM, partition node lists explicitly. Sparse
range reads remain available when loading the full selected arrays is unsuitable.

`set_meta` and `remove_meta` are idempotent for unchanged JSON values or missing
keys. Missing and explicit JSON null remain different. A no-op preserves
approval, generation, timestamps and journal. Use `set_meta_many` for atomic
`(uuid, key, value)` triples or the existing `with store.write()` context for
mixed changes. A failed batch rolls back all its updates; durability remains FULL.

## Two-store three-way synchronization

```python
with ObjectStore.open("alice.vcobj", author="alice") as local, \
     ObjectStore.open("bob.vcobj", author="bob") as remote:
    preview = local.synchronize(remote, dry_run=True)
    # Inspect preview["conflicts"]: UUID, reason, base, local and remote records.
    if not preview["conflicts"]:
        result = local.synchronize(remote, expected_plan=preview["plan_token"])
    else:
        resolutions = {conflict["uuid"]: "local" for conflict in preview["conflicts"]}
        # Choose local/remote per record after reviewing, not automatically in production.
        result = local.synchronize(remote, resolutions=resolutions,
                                   expected_plan=preview["plan_token"])
```

C++ uses `ObjectStore::synchronize(peer, SyncOptions)` and returns `SyncReport`.
Both handles must be writable and have no caller-owned active writes. Python
also refuses handles in use by another operation. Preview takes consistent write
locks but makes no changes. Unresolved conflicts leave both databases unchanged.
Resolutions require the current preview token, so concurrent edits cannot be
silently overwritten after review. Successful synchronization updates both files.

The last state shared by these two replica UUIDs is the merge baseline. One-sided
additions, edits and deletions apply automatically. Different edits to the same
record (including delete-versus-edit) conflict, even when they touch different
fields. No timestamp or generation wins implicitly. On the first sync, identical
records and disjoint additions merge; differing records with the same UUID need
resolution. Object/edge UUID collisions, dangling endpoints and duplicate track
or edge identities also block publication. Duplicate edges with distinct UUIDs
can be resolved by choosing one duplicate's absent side.

Synchronization also requires valid relationship decisions. If merging a geometry
change with another replica's relationship would change its referenced points
without a matching relationship update, sync reports a conflict and makes no
changes. Resolve the relationship explicitly before retrying. Sync preserves
unaffected relationships and the caller's recorded retain/update/remove decisions.
All retained geometry histories from both replicas are copied, including
unselected branches, with original revision UUIDs, provenance and verified chunk
hashes. Record provenance is preserved; each replica keeps its own local journal
of applied changes rather than combining journal sequence numbers.

The report includes `applied`, `conflicts`, `plan_token`, replica IDs and counts
of local/remote record changes. Sync is record-level. After a shared baseline is
established, journal positions identify changed records (including approvals and
deletions), and geometry revision positions identify newly received histories.
The merge loads those records plus their endpoint and identity dependencies.
Baselines retain per-record versions and deletion tombstones instead of another
full snapshot for every edit. The first sync, or the first sync against a legacy
version-2 baseline, uses a full snapshot to establish this incremental state.
There is no server, field-level merge, pruning, or payload garbage collection.
Shared baseline histories are retained in both files. SQLite commits across two
WAL files are sequential: if interrupted after the first commit, retry sync. It
uses the last baseline present in **both** files and reconciles any subsequent
edits normally; there is no claim of distributed atomic commit. Do not discard a
replica's baseline history or journal between syncs. Mutations must go through
the store API so incremental synchronization can observe them.

## Implementation layout and validation

Implementation and target definitions are grouped in `core/src/vc_object_store/`.
`vc_objects` owns the database; `vc_object_geometry` and `vc_object_links` are
higher-level adapters, not alternative database owners.

| Location | Responsibility |
| --- | --- |
| `Database.cpp`, `Transactions.cpp` | Opening, connection lifetime, transactions, checkpointing |
| `Records.cpp`, `Metadata.cpp` | Object records, annotations, approval, and journal |
| `SourcePaths.cpp`, `Snapshots.cpp` | Source lookup and metadata JSON snapshots |
| `geometry/` | Revision/chunk storage and native geometry ingestion/export |
| `relationships/` | Edge CRUD, traversal, and geometry revision validation |
| `imports/` | Native branch, point and winding relationship ingestion |
| `sync/` | Three-way merge, conflict validation and immutable geometry transfer |
| `linking/` | Geometry detectors that propose relationship metadata |
| `formats/`, `internal/` | Shared source-format helpers and private SQLite machinery |

Native ingestion and export have public headers `GeometryIngest.hpp` and
`GeometryExport.hpp` under `vc/core/objects/`. `ObjectStore.hpp` exposes database
operations. Link discovery uses `vc/core/object_links/ObjectLinks.hpp`.

Build and run the affected tests:

```sh
ninja -C build
QT_QPA_PLATFORM=offscreen ctest --test-dir build --output-on-failure \
  -R '^(test_object_store|test_object_records|test_object_links|test_progress)$'
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 VC_BUILD_PYTHON=build/python .venv/bin/python -m pytest \
  python/tests/test_object*.py -q
```

`VC_BUILD_PYTHON` selects the built package before any installed editable copy
loads its native libraries. Plugin autoload is disabled to avoid unrelated
application plugins in the shared development environment.


## Fine spatial regions

`spatial_regions(bbox, object_kind=None, tag=None, object_uuids=[],
connected_to=None, edge_kinds=[], limit=None, offset=0)` queries the same
persisted R-tree for lightweight geometry candidates. C++ uses
`spatialRegions(txn, ObjectSelection)` with a required `bbox`. Filters intersect
as in `select_objects`; pagination counts **regions**, ordered by object UUID,
array name, and array start. Several regions can belong to one object.

Each result has `object_uuid`, `kind`, `revision_uuid`, `array_name`, `start`,
`shape`, and exact double `bbox`. Start and shape exclude the XYZ axis:

- Patches: up to 32×32 cells / 33×33 vertices, with `[row, column]` starts and
  vertex counts. Every vertex contributes to the box, including interior
  extrema; adjacent tiles share boundary vertices. Final tiles can be smaller.
  Single-row/column patches are indexed too, but have no surface cells.
- Fibers and tracks: up to 256 segments / 257 vertices, with `[point_index]`
  starts and counts. Adjacent blocks share one endpoint so segments crossing a
  boundary are covered exactly once. Fiber line and control arrays are separate.
- Point collections: up to 256 points per block, with no overlap. The same slice
  selects corresponding `point_ids` in the revision.
- External objects with annotated bounds: `revision_uuid=None`, empty
  `array_name`, `start` and `shape`. Load these through their external source.

Boxes are conservative for stored vertices, unit-grid triangles (either diagonal),
and straight polyline segments. These are **not exact hits**, nearest-point
answers, or bounds for an interpolated spline that can overshoot its vertices.
Higher-level decimation that connects vertices across tile boundaries needs its
own conservative handling. A caller can perform full-resolution intersection on
returned slices without adopting VC3D's current tiling or sampling scheme.

Region descriptors bind to immutable revisions. Use their `revision_uuid` when
loading, even if the object's head changes between query and read:

```python
for tile in store.spatial_regions(view_box, object_kind="patch"):
    if tile["revision_uuid"] is None:
        continue  # External geometry.
    row, col = tile["start"]
    height, width = tile["shape"]
    vertices = store.read_geometry_rectangle(
        tile["revision_uuid"], tile["array_name"], row, col, height, width,
    )  # Dense little-endian float32 XYZ; reshape to (height, width, 3).
```

The index stores geometry ranges and bounds, not payload copies, and is rebuilt
locally during migration/sync. Publication still scans the provided coordinates,
but unchanged regions retain their R-tree entries; only changed/new/removed
regions modify the tree. This avoids rewriting thousands of entries for a local
edit. Index updates and head publication share the same savepoint. Migration
uses verified, bounded reads ordered to match the physical layout. All region reads honor the
transaction snapshot. Historical revisions retain their payloads, but only the
current head has a spatial index.

Old revisions retain linear 1 MiB chunks and their associated wide-surface read
amplification. New tiled revisions avoid that dependency on surface width. A
single operation still verifies its entire manifest; large numbers of small
storage tiles increase manifest parsing and SQLite lookup overhead.
