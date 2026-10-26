# vc_object_store

A SQLite object database for geometry, revisioned metadata, and graph-like relationships for geometry data (meshes, pointlines, points) within the Vesuvius Challenge. The purpose of this database is to replace the current use of files for most vc workflows and centralize geometry object storage and metadata (including graph edges) and address the issue of carrying "provenence" or history of a given object. Allowing database views at timestamps provides the ability to simply timestamp when something such as a spiral fit is created and having later access to exactly the state of the object store at that time, removing any need for copied data or revision history stored at separate metadata files.  

The object store imports geometry files (tifxyz, pcl, fiber, track) and stores them as binary chunks or tiles using sqlite blob storage. Once objects are imported, the object store contains tooling to discover geometric relationships between these objects and stores this edge information as traversable object metadata. The primary types of edges indicate that the linked objects lie on the same winding: 

- Surface to Surface
- Fiber to Surface 
- Fiber to Fiber
- Track to Surface 
- Track to Track

PCL to Surface edges can be absolute winding, relative winding, or same-winding (which is the same as relative with a 0 delta along `wind_a`)

Modifications to geometry are stored as the delta to the previous iteration and contain a timestamp. If the object requires modifications to its shape a "checkpoint" is created, and any later deltas for history refer to this checkpoint. "Point-in-time" database views can be recovered by simply providing the timestamp 

```python
view = store.revision_read_view_at(timestamp_us)

view["objects"]       # Historical object records and geometry revision references
view["relationships"] # Recorded relationships, with applicability flags
view["selection"]     # Object UUID → selected revision UUID
```

The object store can be used in place of tifxyz files, fiber jsons, track crossings/dbm files, and pcl jsons in any workflow where these are used and edit history is desired. The store contains the information necessary for any tool which works with these files to instead use the db. The object store can export these files after revision if this is needed. 

The db has fast spatial queries through a persisted r-tree backed index with optional in-memory views for repeated queries. 

The C++ `ObjectStoreSession` accepts writable or read-only attachments. Read-only
sessions reject actions before invoking their callbacks. VC3D calls the database's
object, geometry, relationship and retirement APIs directly inside `act`, with
explicit revision expectations. A split creates two objects and retires the
original in one transaction. `updateGeometryArrays` shares sparse array/sequence
updates across geometry kinds; `ingestGeometryBatch` shares one-time ingestion
of decoded geometry and native topology. Both run inside the caller’s transaction
and retain the database’s existing revision and relationship checks.

Fiber JSON version 4 is the transfer format; typed database geometry exports
in it, including control-only fibers. File import also reads VC3D version 1
and 3 fibers and stores them as version 4: a level-0 coordinate space becomes
the volume binding, `branch_file` links resolve to the named file's UUID, and a
link declared from both ends is imported once. Version 1 control points are
bare XYZ arrays, and a missing `optimization_mode` reads as `lasagna`. A fiber
file without a `uuid` is given a name-based (version 5) UUID derived from its
file name, so separate imports of the same file agree on its identity.

## Use

```python
from vc_object_store import ObjectStore

with ObjectStore.create("objects.vcobj", author="reviewer") as store:
    report = store.ingest_geometry_source("surface", "/data/surface")
```

`vc_object_ingest` creates the store when `--db` does not exist yet. To start
a store from a folder of fiber JSONs:

```sh
build/bin/vc_object_ingest --db objects.vcobj --author "$USER" --fiber /data/fibers
```

VC3D can also create an empty store (File → Object Store → New Object Store...).

```sh
build/bin/vc_object_ingest --db objects.vcobj --author "$USER" --surface /data/surface
build/bin/vc_object_ingest --db objects.vcobj --author "$USER" \
  --surface-channel SURFACE_UUID ink /data/ink_prediction.tif
build/bin/vc_object_links --db objects.vcobj --author "$USER" --link-surface-overlaps
build/bin/vc_object_links --db objects.vcobj --author "$USER" \
  --link-fiber-surfaces --ct /data/ct.zarr
```

A team can share a store as a hub file on a network share or on any SSH server.
Initialize the hub from a local store that nothing else has open, then copy it
into place (to a server, use `scp`):

```sh
vc_object_hub init objects.vcobj
cp objects.vcobj /mnt/share/hub.vcobj
```

Nobody opens the hub directly. Each client clones it into a local replica,
works there, and syncs; editing an object that already exists on the hub needs
that object's lease (`vc_object_hub lease acquire REPLICA UUID`):

```sh
vc_object_hub clone /mnt/share/hub.vcobj ~/replica.vcobj
vc_object_hub clone sftp://user@host/path/hub.vcobj ~/replica.vcobj   # no mount needed
vc_object_hub sync ~/replica.vcobj
```

In VC3D, File → Object Store → Clone Hosted Object Store... clones and attaches
a replica, and Sync Hosted Object Store Now syncs it; initializing a hub is
command-line only. See [hosted hub and replicas](docs/README.md#hosted-hub-and-replicas).

See the [API guide](docs/README.md) for checked revisions, geometry reads and
edits, graph snapshots, spatial queries, and synchronization. Detector distances
and scale options use working voxels. CT normals use uint8 Zarr data with complete
109³ support. Native CT opening supports local and anonymous HTTP/S3 arrays and
pyramid levels; credentials and disk-cache configuration are not exposed by this
adapter. Codec support follows vc_reader.

`nearest_surfaces(point, tolerance, uuids=None, coordinates=None, volume=None)`
(C++ `ObjectStore::nearestSurfaces`) returns the nearest point of each surface
within tolerance, ordered by distance: object and geometry revision, fractional
grid `[column, row]`, the surface point and distance. It reads only the persisted
spatial tiles near the point and tests their exact valid triangles (the same
split as surface regions); relationship discovery shares the triangle routine.
Bounds are not quantized.

Tests are self-contained and do not require the volume-cartographer checkout or
its fixture data.

## Transformed geometry views

Canonical geometry stays in its recorded volume. Request another coordinate frame
explicitly; the object store resolves the registration and applies it while reading:

```cpp
auto snapshot = store.geometrySnapshot();
auto surface = snapshot.surface(uuid).viewTransformedAt(targetVolume);
auto tile = surface.readTile(row, column, rows, columns);
// Equivalently, for any geometry kind:
auto view = snapshot.viewTransformedAt(uuid, targetVolume);
```

```python
with store.geometry_snapshot() as snapshot:
    view = snapshot.surface(uuid).view_transformed_at(target_volume)
    tile = view.read_tile(row=0, column=0, rows=64, columns=64)
    # Fiber, track, point_collection and umbilicus accessors support the same call.
    fiber = snapshot.fiber(fiber_uuid).view_transformed_at(target_volume)
    controls = fiber.points("control_points", start=0, count=2)
```

Views retain the canonical revision and snapshot lifetime. Surface validity is
separate from transformed coordinates; track views return floating coordinates.
`coordinateEdits` and `surfaceEdits` map changed positions back to canonical
storage, preserving untouched values. Track edits round to nearest integers,
with ties away from zero. Call `validateForWrite` in the checked write transaction
before applying edits; changed geometry or registrations invalidate the view.
Alternatively, `writeCoordinates` and `writeSurface` validate and apply an edit in
one call. Python exposes these as `store.edit_transformed_coordinates(view, ...)`
and `store.edit_transformed_surface(view, ...)`.

`availableTransforms(txn, uuid)` and each geometry view’s `availableTransforms()`
(`available_transforms` in Python) list reachable
volumes and ambiguity. Incompatible or ambiguous view requests fail explicitly.
Registrations and path choices are revisioned store objects. Open Data registrations
can be supplied through `OpenOptions::catalogTransforms`, or `catalog=catalog`
when opening a Python store with a `vc_reader.Catalog`. Catalog parsing, affine
composition and path resolution are implemented in vc_reader's native library,
shared by its Python and C++ consumers. Target-volume length filters measure the
transformed polyline, including nonuniform scaling and shear.
