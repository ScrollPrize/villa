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

The database uses SQLite application ID `0x56434f42`, schema version 3.
Writable opens migrate version-2 files transactionally; read-only opens can
inspect either version without migration. Older binaries cannot open version 3.
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
manifest lists named arrays, dtype, shape, byte length, and ordered chunk hashes.
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
one verified manifest and a bounded eight-chunk cache. Geometry edit and repair
operations also reuse readers and stable-point-ID indexes across relationship
checks. Caches end with the operation; subsequent calls reverify stored bytes.

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

`fields` projects top-level record fields; UUIDs, kinds, edge endpoints and
geometry references are always retained. An empty list exports all metadata.
`import_json` accepts complete metadata-only records without managed geometry.
Projected exports are for inspection; use synchronization to transfer objects
with geometry and their revision bindings.

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
  -R '^(test_object_store|test_object_records|test_object_links|test_progress|graph_db_controller)$'
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 VC_BUILD_PYTHON=build/python .venv/bin/python -m pytest \
  python/tests/test_objects.py python/tests/test_object_records.py python/tests/test_object_geometry.py python/tests/test_object_links.py \
  python/tests/test_object_ingestion.py python/tests/test_object_sync.py python/tests/test_object_geometry_edits.py -q
```

`VC_BUILD_PYTHON` selects the built package before any installed editable copy
loads its native libraries. Plugin autoload is disabled to avoid unrelated
application plugins in the shared development environment.
