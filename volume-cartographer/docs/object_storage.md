# Object database and relationship metadata

`ObjectStore` is the primary database: one SQLite file (`objects.vcobj`) owns
object identities, annotations, immutable geometry revisions, content-addressed
binary chunks, and relationships between objects. Edges, anchors, approvals,
and endpoint revision bindings are metadata in that same database. Objects can
exist without geometry or relationships.

The C++ database library is `vc_objects`. Headers, record types, and transactions
live under `vc/core/objects/` in `vc::objects`. Native file import/export lives
in `vc_object_geometry`; link detectors live in `vc_graph_links`. These higher
layers depend on the object store. The database library itself does not depend
on CT readers or Qt. Python exposes the database and its helpers in `vc.objects`.

This is a deliberate WIP API/schema break. The separate `GraphDb` and `RecordDb`
APIs and the `vc.graph` Python module have been removed. Object schema version 2
uses application ID `0x56434f42`. Earlier object stores and unified or split graph
databases are rejected without migration; create a new file and re-ingest sources.
Geometry ingestion does not run link detectors or integrate stored arrays into
VC3D/Spiral readers.

## One handle and one transaction

```python
from vc.objects import ObjectStore

with ObjectStore.create("objects.vcobj", author="sean") as store:
    with store.write() as tx:
        patch_id = tx.upsert_object({"kind": "patch", "name": "patch-001"})
        first = tx.put_geometry(patch_id, patch_geometry, expected_revision=None)
        other_id = tx.upsert_object({"kind": "patch", "name": "patch-002"})
        edge_id = tx.add_edge({"kind": "same-surface", "a": patch_id, "b": other_id})

    second = store.put_geometry(patch_id, edited_geometry, expected_revision=first)
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
loading. Geometry decoding happens outside write transactions. Each object
publishes in one transaction; a malformed object leaves its previous state
intact. The batch continues and reports failures, with a nonzero CLI exit code.
A multi-collection JSON is parsed/validated before any of its collections publish.
Re-running an unchanged ingestion does not create revisions, journal entries,
or approval changes. A concurrent object edit causes that object's publication
to fail rather than overwriting newer state.

## Representations

All arrays use contiguous C order and explicit little-endian dtype descriptors.
Coordinates are unscaled source XYZ; they are not transformed to fitting units.

| Kind | Arrays | Immutable attributes |
| --- | --- | --- |
| `patch` | `points`: `<f4`, `[rows, columns, 3]` | Source meta.json, including two-axis `scale` |
| `fiber` | `control_points`, `line_points`: `<f8`, `[N, 3]` | Source fields and per-point extra attributes, including tracing settings |
| `point_collection` | `points`: `<f8`, `[N, 3]`; `point_ids`: `<i8`, `[N]` | Collection fields and ordered per-point annotations, including winding values |
| `track` | `points`: `<i4`, `[N, 3]` | Source ID and family |

Patch coordinates follow the existing QuadSurface loader, including invalid
points and applied masks. Ancillary TIFF channels such as rendered images and
generation visualizations are not ingested. Fiber point order is retained.
Collection points are ordered by numeric ID; IDs are retained, not renumbered.
Track ZYX integers are reordered to XYZ without rescaling or reducing precision.
Source annotations are retained as immutable JSON attributes; geometry coordinate
arrays are binary, not JSON. Object tags and caller metadata remain on the object.
Changes to captured source attributes also create a geometry revision, even if
its coordinate payload chunks can all be reused.

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
`with db.write() as w: w.put_geometry(...)` accept `expected_revision`.
The C++ equivalents accept `GeometryData` and an existing `WriteTxn`, so geometry
and other object changes can commit atomically. Publication uses an internal
savepoint: catching an error and committing other work cannot retain a partial
geometry revision.

Python ingestion methods are `ingest_geometry_source(kind, source)` and
`ingest_registered_geometry(dataset_root=None, object_uuids=[])`. They return
`ingested`, `unchanged`, and `problems` and own their per-object transactions.

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
are also rejected. Captured source attributes are exported rather than current
object tags/metadata; other derived source annotations are the caller's
responsibility when editing geometry. If point counts change, update or remove
the corresponding positional annotation lists. Export rejects mismatched
annotation counts, duplicate/negative collection point IDs, and auxiliary arrays
that native formats cannot represent, avoiding silent data loss. It does not
validate application-specific tracing semantics or update trace results.

## Relationships and approvals

New edges bind to their endpoints' current geometry revisions in normalized
endpoint order. Objects without managed geometry have null bindings.
`edge_geometry_state(edge_uuid)` compares the saved bindings with current
object heads. Historical revisions remain readable and exportable.

Publishing changed geometry bumps and journals the object and all connected
edges, resetting their approvals in the same transaction. Existing endpoint
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
can locate that source by UUID. `sweep_external_data` excludes managed objects
because absent source files do not make retained geometry unavailable.

Use SQLite's backup API for live copies, or stop writers, checkpoint, and copy
the database. One backup contains objects, geometry history, relationships,
approvals, revision bindings, and the journal. Copies retain the store identity;
use one logical writable store per identity rather than editing divergent copies.

JSON export/import remains a metadata snapshot format. Export rejects selected
objects with managed geometry instead of discarding payloads and revision bindings.
Use a complete database backup for those objects.

## Implementation layout and validation

Implementation and target definitions are grouped in `core/src/vc_object_store/`.
`vc_objects` owns the database; `vc_object_geometry` and `vc_graph_links` are
higher-level adapters, not alternative database owners.

| Location | Responsibility |
| --- | --- |
| `Database.cpp`, `Transactions.cpp` | Opening, connection lifetime, transactions, checkpointing |
| `Records.cpp`, `Metadata.cpp` | Object records, annotations, approval, and journal |
| `SourcePaths.cpp`, `Snapshots.cpp` | Source lookup and metadata JSON snapshots |
| `geometry/` | Revision/chunk storage and native geometry ingestion/export |
| `relationships/` | Edge CRUD, traversal, and geometry revision validation |
| `imports/` | Separate fiber, patch, and track metadata registration |
| `linking/` | Geometry detectors that propose relationship metadata |
| `formats/`, `internal/` | Shared source-format helpers and private SQLite machinery |

The source import APIs have explicit public headers under
`vc/core/objects/import/`: `FiberImport.hpp`, `PatchImport.hpp`, and
`TrackImport.hpp`. Including `ObjectStore.hpp` alone exposes database operations,
not filesystem import entry points.

Build and run the affected tests:

```sh
ninja -C build
QT_QPA_PLATFORM=offscreen ctest --test-dir build --output-on-failure \
  -R '^(test_object_store|test_graph_db|test_graph_links|test_progress|graph_db_controller)$'
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 VC_BUILD_PYTHON=build/python .venv/bin/python -m pytest \
  python/tests/test_objects.py python/tests/test_graph.py python/tests/test_graph_objects.py -q
```

`VC_BUILD_PYTHON` selects the built package before any installed editable copy
loads its native libraries. Plugin autoload is disabled to avoid unrelated
application plugins in the shared development environment.
