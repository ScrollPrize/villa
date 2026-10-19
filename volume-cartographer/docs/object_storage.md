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

The database uses SQLite application ID `0x56434f42`. Geometry ingestion and
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
        edge_id = tx.add_edge({"kind": "same-surface", "a": patch_id, "b": other_id})

    second = store.put_geometry(patch_id, edited_geometry, expected_revision=first, force=True)
    assert store.edge_geometry_state(edge_id)["stale"]
    # After checking the relationship and its anchors against the new geometry:
    with store.write() as tx:
        tx.revalidate_edge_geometry(edge_id)
        tx.set_approval(edge_id, "approved")
```

`patch_geometry` and `edited_geometry` use the array dictionary described below.
C++ uses `ObjectStore::upsertObject`, `putGeometry`, `addEdge`, and
`revalidateEdgeGeometry` with the same `WriteTxn`. Transactions can commit or
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
no-op. Replacing geometry that has edges requires `force=True` (`--force` in the
CLI); it invalidates edge approvals and leaves old anchors stale. Source metadata
is seeded when an object does not yet have `meta.source`.

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

To publish arrays directly, supply a dictionary with `kind`, optional
`attributes`, and an `arrays` list. Each array has `name`, `dtype`, `shape`, and
`data` as Python bytes. Supported dtypes are `|u1`, `<i4`, `<i8`, `<u8`, `<f4`,
and `<f8`; each object kind requires the layouts listed above. Patch attributes
must contain a finite, positive two-axis scale. Both `db.put_geometry(...)` and
`with db.write() as w: w.put_geometry(...)` accept `expected_revision` and `force=False`.
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
relationships must bind to the exported revision and be non-stale; otherwise
export fails. Generic relationships remain available through metadata JSON export.
Other derived source annotations remain the caller's responsibility when editing
geometry. If point counts change, update or remove
the corresponding positional annotation lists. Export rejects mismatched
annotation counts, duplicate/negative collection point IDs, and auxiliary arrays
that native formats cannot represent, avoiding silent data loss. It does not
validate application-specific tracing semantics or update trace results.

## Relationships and approvals

New edges bind to their endpoints' current geometry revisions in normalized
endpoint order. Objects without managed geometry have null bindings.
`edge_geometry_state(edge_uuid)` compares the saved bindings with current
object heads. Historical revisions remain readable and exportable.

Publishing changed geometry with existing edges fails by default. Passing
`force=True` explicitly allows it, bumps and journals the object and all connected
edges, and resets their approvals in the same transaction. Existing endpoint
bindings remain unchanged, identifying the revisions the old anchors refer to.
This also applies when geometry is first published for an already linked object.
Unrelated edges are unchanged, and identical-content publication with the correct
expected revision is a complete no-op.

Stale edges cannot be approved. After checking or recomputing the relationship,
call `revalidate_edge_geometry(edge_uuid)` to bind current revisions, then approve
it. Revalidation resets approval and journals the change; it does not move
anchors or run a detector. Updating edge metadata alone does not revalidate it.

The database uses WAL with FULL synchronous writes. Geometry publication and
edge revalidation use savepoints: a caught failure cannot leave partial geometry,
approval, or journal changes when the enclosing transaction commits other work.

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
metadata_json = store.export_json(**selection, fields=["name", "tags", "meta"])
```

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

A geometry head change that affects existing edges requires `force=True` on
synchronization too. Force does not resolve competing edits; explicit record
choices are still required. Changed geometry invalidates edge approval, and old
bindings remain stale until revalidation. Retained geometry cannot be deleted by
merge. All retained geometry histories from both replicas are copied, including
unselected branches, with original revision UUIDs, provenance and verified chunk
hashes. Record provenance is preserved; each replica keeps its own local journal
of applied changes rather than combining journal sequence numbers.

The report includes `applied`, `conflicts`, `plan_token`, replica IDs and counts
of local/remote record changes. Sync is currently whole-store and record-level;
there is no server, field-level merge, pruning, or payload garbage collection.
Shared baseline histories are retained in both files. SQLite commits across two
WAL files are sequential: if interrupted after the first commit, retry sync. It
uses the last baseline present in **both** files and reconciles any subsequent
edits normally; there is no claim of distributed atomic commit. Do not discard a
replica's baseline history between syncs.

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
  python/tests/test_object_ingestion.py python/tests/test_object_sync.py -q
```

`VC_BUILD_PYTHON` selects the built package before any installed editable copy
loads its native libraries. Plugin autoload is disabled to avoid unrelated
application plugins in the shared development environment.
