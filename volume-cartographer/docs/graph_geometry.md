# Geometry in the graph database

Schema v3 adds immutable geometry to the same SQLite database as objects and
relationships. No separate geometry database or service is required. The
standalone `vc_graph` library handles typed binary arrays and revisions;
`vc_graph_geometry` adds readers and writers for the existing source formats using `vc_core`.
The store uses OpenSSL Crypto for SHA-256. Ingestion does not run link detectors
or integrate the stored arrays into VC3D/Spiral readers yet.

## Ingesting geometry

Build the importer and bindings with Ninja:

```sh
ninja -C build vc_graph_ingest vc_graph_export vc_graph_bindings
```

Ingest files referenced by an existing graph, optionally rebasing its relative
source hints to a dataset directory:

```sh
build/bin/vc_graph_ingest --db /data/graph.vcgr --author sean \
    --registered --dataset-root /data/dataset
```

`--uuid UUID` restricts ingestion to selected registered objects and is repeatable.
The importer also creates a new database and registers missing objects:

```sh
build/bin/vc_graph_ingest --db /data/graph.vcgr --author sean \
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
or approval changes. A concurrent graph edit causes that object's publication
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
arrays are binary, not JSON. Graph tags and caller metadata remain on the object.
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
from vc.graph import GraphDb
import numpy as np

with GraphDb.open('/data/graph.vcgr', read_only=True) as db:
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
and other graph changes can commit atomically. Publication uses an internal
savepoint: catching an error and committing other work cannot retain a partial
geometry revision.

Python ingestion methods are `ingest_geometry_source(kind, source)` and
`ingest_registered_geometry(dataset_root=None, object_uuids=[])`. They return
`ingested`, `unchanged`, and `problems` and own their per-object transactions.

## Exporting native files

Export the current geometry or a selected historical revision without accessing
its original source files:

```sh
build/bin/vc_graph_export --db /data/graph.vcgr --uuid OBJECT_UUID \
    --output /data/export/patch
# Add --revision REVISION_UUID to select history, or --overwrite to replace output.
```

```python
with GraphDb.open('/data/graph.vcgr', read_only=True) as db:
    exported_revision = db.export_geometry(object_uuid, destination)
    db.export_geometry(object_uuid, destination,
                       revision_uuid=old_revision_uuid, overwrite=True)
```

C++ callers use `exportGeometry` from `vc/core/graph/GeometryExport.hpp` in
`vc_graph_geometry`. Both APIs return the exported revision UUID. The selected
revision must belong to the requested object. Export reads a consistent snapshot
and releases the read transaction before writing files. It does not alter graph
objects, approvals, source hints, revisions, or relationships. To export an edit,
first publish it with `put_geometry`, then export that revision.

| Kind | Destination contents |
| --- | --- |
| Patch | Directory containing float32 `x.tif`, `y.tif`, `z.tif`, and `meta.json`; dimensions and bounds recalculated, scale retained |
| Fiber | `vc3d_fiber` JSON with coordinates, captured annotations, and tracing settings |
| Point collection | Version-1 point-collections JSON containing this one collection, retaining collection key, point IDs, and annotations |
| Track | One-track packed directory containing header, ZYX int32 coordinates, offsets, source ID, family, z bounds, arclength, tortuosity, and metadata |

Track exports record the graph object/revision UUIDs in `metadata.json`, but do
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
graph tags/metadata; other derived source annotations are the caller's
responsibility when editing geometry. If point counts change, update or remove
the corresponding positional annotation lists. Export rejects mismatched
annotation counts, duplicate/negative collection point IDs, and auxiliary arrays
that native formats cannot represent, avoiding silent data loss. It does not
validate application-specific tracing semantics or update trace results.

## Relationships and approvals

New edges bind to the current geometry revisions of their endpoints, in the
same normalized endpoint order as the edge. `edge_geometry_state(edge_uuid)`
reports those revision UUIDs and whether they differ from the current heads.
Geometry publication bumps the object's generation and resets its approval;
it also bumps/journals connected edges and resets their approvals. Their old
revision bindings remain available, so their anchors have explicit context.
This is conservative: all connected relationships are invalidated, including
relationships which a future application might classify as geometry-independent.

Existing metadata-only edges start with unknown revision bindings. Initial
ingestion makes those edges stale and pending; it does not assert that old
anchors were computed against the newly captured bytes. Stale edges cannot be
approved. After explicitly checking/recomputing the relationship, call
`revalidate_edge_geometry(edge_uuid)` (also available on a write transaction)
to bind it to current revisions, then approve normally. Revalidation does not
move anchors or run a detector. This is an explicit caller assertion.

Geometry revisions are historical geometry, not full historical graph snapshots.
The graph journal still records mutation/approval stamps rather than complete
previous object and edge records. Historical metadata/relationship checkout and
branch merging remain outside this implementation.

## Migration, sources, and backup

Writable opens migrate v1/v2 databases to v3 atomically; read-only opens require
migration first. Migration alone changes no object geometry, generations, or
approval state. Older binaries reject the new schema for writing.

Managed objects have `storage_kind='sqlite'`, a current geometry content hash
in `data_checksum`, and its publication generation in `data_generation`.
`storage_ref` remains a source-location hint for explicit re-ingestion; it is
not the authoritative geometry. `resolve_storage_path` can still locate that
source by UUID. `sweep_external_data` excludes managed objects because absent
source files do not make stored geometry unavailable. Metadata upserts cannot
replace the managed storage kind, checksum, or data generation.

Use SQLite's backup API for a live self-contained copy, or close all writers
and copy the fully checkpointed database. The geometry and retained revisions
are in the same database. Existing JSON export remains a metadata-only format:
it refuses a selection containing managed geometry, rather than silently losing
payloads or revision/edge bindings. Metadata-only selections still export as
before. A portable geometry archive/JSON interchange format is not added here.
