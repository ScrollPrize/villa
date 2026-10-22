# vc_object_store

A SQLite object database for geometry, revisioned metadata, and graph-like relationships for geometry data (meshes, pointlines, points) within the Vesuvius Challenge. The purpose of this database is to replace the current use of files for most vc workflows and centralize geometry object storage and metadata (including graph edges) and address the issue of carrying "provenence" or history of a given object. Allowing database views at timestamps provides the ability to simply timestamp when something such as a spiral fit is created and having later access to exactly the state of the object store at that time, removing any need for copied data or revision history stored at separate metadata files.  

The object store imports geometry files (tifxyz, pcl, fiber, track) and stores them as binary chunks or tiles using sqlite blob storage. Once objects are imported, the object store contains tooling to discover geometric relationships between these objects and stores this edge information as traversable object metadata. The primary types of edges indicate that the linked objects lie on the same winding: 

- Patch to Patch
- Fiber to Patch 
- Fiber to Fiber
- Track to Patch 
- Track to Track

PCL to Patch edges can be absolute winding, relative winding, or same-winding (which is the same as relative with a 0 delta along `wind_a`)

Modifications to geometry are stored as the delta to the previous iteration and contain a timestamp. If the object requires modifications to its shape a "checkpoint" is created, and any later deltas for history refer to this checkpoint. "Point-in-time" database views can be recovered by simply providing the timestamp 

```python
view = store.revision_read_view_at(timestamp_us)

view["objects"]       # Historical object records and geometry revision references
view["relationships"] # Recorded relationships, with applicability flags
view["selection"]     # Object UUID → selected revision UUID
```

The object store can be used in place of tifxyz files, fiber jsons, track crossings/dbm files, and pcl jsons in any workflow where these are used and edit history is desired. The store contains the information necessary for any tool which works with these files to instead use the db. The object store can export these files after revision if this is needed. 

The db has fast spatial queries through a persisted r-tree backed index with optional in-memory views for repeated queries. 

## Use

```python
from vc_object_store import ObjectStore

with ObjectStore.create("objects.vcobj", author="reviewer") as store:
    report = store.ingest_geometry_source("patch", "/data/patch")
```

```sh
build/bin/vc_object_ingest --db objects.vcobj --author "$USER" --patch /data/patch
build/bin/vc_object_links --db objects.vcobj --author "$USER" --link-patch-overlaps
build/bin/vc_object_links --db objects.vcobj --author "$USER" \
  --link-fiber-patches --ct /data/ct.zarr
```

See the [API guide](docs/README.md) for checked revisions, geometry reads and
edits, graph snapshots, spatial queries, and synchronization. Detector distances
and scale options use working voxels. CT normals use uint8 Zarr data with complete
109³ support. Native CT opening supports local and anonymous HTTP/S3 arrays and
pyramid levels; credentials and disk-cache configuration are not exposed by this
adapter. Codec support follows vc_reader.

Surface proximity queries use the store's resident bounds index and exact
triangles. Bounds are not quantized.

Tests are self-contained and do not require the volume-cartographer checkout or
its fixture data.
