# Shared geometry API

The package separates geometry, stored objects and file formats:

| Layer | Public API | Responsibility |
| --- | --- | --- |
| Owned geometry | `geometry/Geometry.hpp`, `vc::geometry` | Point collections, fibers, tracks and patches; validation and geometry operations |
| Stored geometry | `store/GeometryViews.hpp`, `vc::objects` | Snapshot views, bounded reads and checked sparse edits |
| Object store | `objects/ObjectStore.hpp` | Identity, metadata, history, transactions, relationships and spatial queries |
| Native formats | `objects/GeometryIngest.hpp`, `objects/GeometryExport.hpp` | Import/export orchestration; decoders and encoders live in `src/formats/` |

Link `vc_geometry` for owned geometry alone, or `vc_objects` for stored geometry.
Installed CMake targets have the `vc_object_store::` prefix. Owned geometry has no
SQLite, OpenCV, Qt, volume-cartographer or spiral-fitting dependency.
All four types and the snapshot/edit API are also available in Python.

## Owned geometry

Owned values contain geometry only. Names, UUIDs, tags, source annotations and
relationships belong to stored objects. Values can be created without opening a
database. Public fields allow construction and editing; `validate()` and storage
conversion check invariants.

| Type | Fields | Operations |
| --- | --- | --- |
| `PointCollection` | int64 `ids`, float64 XYZ `points` | `point(id)`, `indexOf(id)`, validation |
| `Fiber` | float64 XYZ `controlPoints`, `linePoints`; int64 `controlPointIds` | Polyline length, validation |
| `Track` | int32 XYZ `points` | Polyline length, validation |
| `Patch` | `rows`, `columns`, column/row `scale`, float32 XYZ `points` | Vertex access, validity, validation |

Coordinates remain source XYZ voxels; there is no implicit rescaling, unit
conversion, or axis reversal. Native packed tracks use ZYX; the format adapter
performs that conversion at the file boundary. Fiber length uses sampled line
points if present, otherwise control points. A patch is row-major; scale is grid
steps per source voxel, and invalid vertices use the existing `(-1,-1,-1)`
sentinel. Stable point IDs are nonnegative and unique and need not be sorted.

```cpp
#include <vc_object_store/geometry/Geometry.hpp>
#include <vc_object_store/objects/ObjectStore.hpp>

vc::geometry::PointCollection points{
    {42, 7},
    {{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}}
};
points.validate();
auto position = points.point(42); // optional XYZ, not row 42

auto write = store.write();
vc::objects::ObjectRecord record;
record.kind = "point_collection";
auto id = store.upsertObject(write, record);
store.putGeometry(write, id, vc::geometry::encode(points));
write.commit();
```

`encode()` converts the typed variant to existing `GeometryData` arrays with
explicit little-endian encoding. `decode()` performs the reverse conversion.
Point order, IDs and scalar precision are preserved. Owned conversion rejects
auxiliary arrays and additional immutable geometry attributes instead of silently
dropping them. The generic array API remains available for those payloads.

Python `put_geometry()` accepts either the existing array dictionary or one of
the four owned types, including inside `store.write()` transactions:

```python
from vc_object_store import ObjectStore, PointCollection

with ObjectStore.create("points.vcobj", author="reviewer") as store:
    with store.write() as write:
        uid = write.upsert_object({"kind": "point_collection"})
        write.put_geometry(uid, PointCollection([42, 7], [[1, 2, 3], [4, 5, 6]]))
```

Python fields are converted to ordinary Python lists. Assign a modified list
back to a field; mutating a list returned by a field getter does not modify the
native value.

## Snapshot views and bounded reads

```cpp
#include <vc_object_store/store/GeometryViews.hpp>

auto snapshot = store.geometrySnapshot();
auto collection = snapshot.pointCollection(collectionUuid);
auto xyz = collection.point(42);
auto patch = snapshot.patch(patchUuid);
auto tile = patch.readTile(30, 31, 4, 4); // row, column, rows, columns
```

```python
with store.geometry_snapshot() as snapshot:
    collection = snapshot.point_collection(collection_uuid)
    ids = collection.ids(0, len(collection))
    xyz = collection.point(42)
    patch = snapshot.patch(patch_uuid)
    tile = patch.read_tile(30, 31, 4, 4)
```

`snapshot.geometry(uuid)` returns the corresponding view (a variant in C++).
Typed factories reject kind mismatches and objects without stored geometry.
Counts and patch dimensions come from the pinned geometry revision, not stored
import summaries.

Views read only requested coordinate ranges or rectangles, through the existing
verified chunk reader. Point lookup builds an ID index once per view, loading
the ID array but not the coordinate array. Full materialization is explicit:
`view.load()` returns the owned type. Large patches should use `readTile()` /
`read_tile()` instead.

Snapshots pin a consistent database state at creation. Views keep
that snapshot alive even if the snapshot variable goes out of scope. Explicit
`snapshot.close()` (also Python context-manager exit) invalidates all its views
and releases the reader. Snapshots are thread-confined and consume
one slot in the store's reader pool; close them promptly.

A C++ snapshot borrowing `ObjectStore&` must finish before the store is moved or
destroyed. The `GeometrySnapshot(std::shared_ptr<ObjectStore>)` overload owns a
store lease; Python uses this overload and rejects closing a store with active
snapshots. Returned owned geometry remains valid after snapshot closure.

## Checked edits and relationships

Views are read-only. Use the existing checked store transaction for mutations;
there is no separate typed editor or commit path.

```python
import struct
with store.geometry_snapshot() as snapshot:
    collection = snapshot.point_collection(collection_uuid)
    with store.write(expected_revisions={collection_uuid: collection.object_revision}) as tx:
        new_revision = tx.edit_geometry(collection_uuid, [dict(
            array="points", element_offset=0, data=struct.pack("<3d", 10, 20, 30)
        )], collection.revision["uuid"])
    # collection still reads the original revision.
```

The transaction checks the pinned object and geometry revisions, including for
no-ops. Affected relationships require explicit retain, update, remove or keep
recorded decisions through `edit_geometry(edge_edits=...)`; include the affected
endpoints in the transaction's expected revisions. No relationship meaning is
inferred by the geometry layer.

C++ clients applying full desired arrays can use `updateGeometryArrays`: it
uses these same sparse and structural transactions, preserves auxiliary
channels and exact scalar bytes, and accepts explicit annotation transforms.

For adding/removing/reordering points or changing dimensions, use
[structural edits](structural-editing.md). Full `put_geometry` replacement remains
available before structural management begins; `affected_geometry_edges()` previews
its relationship dependencies. Multi-object actions use checked write transactions.

## Validation

Import-summary counts/lengths remain historical annotations; view counts and owned
measurements describe current geometry. Generic arrays retain their stored names,
dtypes and byte order. See the [build/test commands](../README.md#build).

Native directory export currently requires Linux `renameat2`.

Sequence geometry (fibers, point collections, tracks) is created once with
`putGeometry` / `put_geometry`. Initial per-point annotations supplied in source
metadata are consumed into that geometry revision. Read them with
`geometryPointAnnotations` / `geometry_point_annotations`; edit them through
sequence operations, alongside the points they describe. Fiber controls and
point collections have stable IDs from creation. Fiber JSON transfer preserves
`control_point_ids` and allocates IDs when a new input omits them.

C++ callers with owned arrays can use `updateGeometryArrays` for all geometry
kinds. It delegates to sparse and structural edits in one checked transaction.
Directional reorderings require explicitly transformed annotations. A failed
update rolls back all its arrays and prevents the enclosing action from committing.
