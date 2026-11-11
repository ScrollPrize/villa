# Typed geometry

Geometry comes in two forms: **owned values** (`vc::geometry`, no database
needed) and **snapshot views** over stored revisions (bounded reads of exactly
the ranges asked for). Owned values hold geometry only — names, UUIDs, tags,
annotations, and relationships belong to stored objects. This page is the
type reference; day-to-day use is in [reading](reading.md),
[writing](writing.md#creating-objects-and-geometry), and
[editing](editing.md).

| Layer | Public API | Responsibility |
| --- | --- | --- |
| Owned geometry | [geometry/Geometry.hpp](../include/vc_object_store/geometry/Geometry.hpp), `vc::geometry` | The five types below; validation and geometry operations |
| Stored geometry | [store/GeometryViews.hpp](../include/vc_object_store/store/GeometryViews.hpp), `vc::objects` | Snapshot views, bounded reads, checked sparse edits |
| Object store | [objects/ObjectStore.hpp](../include/vc_object_store/objects/ObjectStore.hpp) | Identity, metadata, history, transactions, relationships, spatial queries |
| Native formats | [objects/GeometryIngest.hpp](../include/vc_object_store/objects/GeometryIngest.hpp), [objects/GeometryExport.hpp](../include/vc_object_store/objects/GeometryExport.hpp) | Import/export orchestration; codecs live in `src/formats/` |

Link `vc_object_store::vc_geometry` for owned geometry alone (no SQLite,
OpenCV, Qt, or volume-cartographer dependency), or `vc_objects` for stored
geometry. All five types and the snapshot/edit API are also in Python.

## Owned types

Public fields allow direct construction and editing; `validate()` (and
storage conversion) checks invariants. Coordinates are source XYZ voxels —
no implicit rescaling, unit conversion, or axis reversal. (Native packed
tracks are ZYX on disk; the format adapter converts at the file boundary.)
Stable point IDs are nonnegative, unique, and need not be sorted.

| Type | Fields | Notes |
| --- | --- | --- |
| `PointCollection` | int64 `ids`, float64 XYZ `points` | `point(id)` looks up by ID, not row; `indexOf(id)` |
| `Fiber` | float64 XYZ `controlPoints`, `linePoints`; int64 `controlPointIds` | Length uses line points when present, else controls |
| `Track` | int32 XYZ `points` | Polyline length |
| `Surface` | `rows`, `columns`, column/row `scale`, float32 XYZ `points`, `validity`, `channels` | Row-major; `scale` is grid steps per source voxel; missing vertices are `(-1,-1,-1)`. `validity` is the `valid` channel (empty, or one byte per vertex); `valid(row, column)` requires coordinates **and** validity. `channels` holds the other aligned channels |
| `Umbilicus` | float64 XYZ `points` (ordered control polyline, at least one) | A scroll's centre line; native file is `vc_umbilicus` JSON |

```cpp
#include <vc_object_store/geometry/Geometry.hpp>
#include <vc_object_store/objects/ObjectStore.hpp>

vc::geometry::PointCollection points{{42, 7}, {{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}}};
points.validate();
auto position = points.point(42);   // optional XYZ, not row 42

auto write = store.write();
vc::objects::ObjectRecord record;
record.kind = "point_collection";
auto id = store.upsertObject(write, record);
store.putGeometry(write, id, vc::geometry::encode(points));
write.commit();
```

```python
from vc_object_store import ObjectStore, PointCollection

with ObjectStore.create("points.vcobj", author="reviewer") as store:
    with store.write() as write:
        uid = write.upsert_object({"kind": "point_collection"})
        write.put_geometry(uid, PointCollection([42, 7], [[1, 2, 3], [4, 5, 6]]))
```

`encode()` / `decode()` convert between owned values and `GeometryData`
arrays with explicit little-endian encoding, preserving point order, IDs,
scalar precision, and surface channels. Owned conversion **rejects** other
auxiliary arrays and extra immutable geometry attributes rather than dropping
them — use the generic array API for those payloads
([raw array reads](reading.md#raw-array-reads)). In Python, fields convert to
ordinary lists: assign a modified list back to the field; mutating a returned
list does not modify the native value.

## Snapshot views

```cpp
#include <vc_object_store/store/GeometryViews.hpp>

auto snapshot = store.geometrySnapshot();
auto collection = snapshot.pointCollection(collectionUuid);
auto xyz = collection.point(42);
auto tile = snapshot.surface(surfaceUuid).readTile(30, 31, 4, 4);  // row, column, rows, columns
```

`snapshot.geometry(uuid)` returns the matching view (a variant in C++);
typed factories reject kind mismatches and objects without geometry. Counts
and dimensions come from the pinned revision, not import summaries. Views
read through the verified chunk reader; `point(id)` builds an ID index once
per view (loading IDs, not coordinates). `view.load()` materializes the owned
value; large surfaces should use `readTile` instead, and read other channels
with `readGeometryRectangle`.

Lifetime: a snapshot pins one consistent state at creation, views keep it
alive, and `snapshot.close()` (or Python context-manager exit) invalidates
all views. Snapshots are thread-confined and hold one reader slot. A C++
snapshot borrowing `ObjectStore&` must finish before the store moves or dies;
the `GeometrySnapshot(std::shared_ptr<ObjectStore>)` overload owns a lease
(Python uses it and rejects closing a store with active snapshots). Loaded
owned geometry outlives the snapshot. Views also work over any
`StoreReader`, hosted included — see [hosted](hosted.md).

## Edits go through transactions

Views are read-only; there is no separate typed editor or commit path.
Mutate through the checked store transaction, expecting the version the view
read — the view keeps reading its pinned revision afterwards:

```python
import struct
from vc_object_store import object_reference

selected = store.current_object_revision(collection_uuid)
reference = object_reference(selected["record"])
with store.geometry_snapshot() as snapshot:
    collection = snapshot.point_collection(collection_uuid, selected["geometry_revision"])
    with store.write(expected_versions={collection_uuid: reference}) as tx:
        revision = tx.edit_geometry(collection_uuid, [dict(
            array="points", element_offset=0, data=struct.pack("<3d", 10, 20, 30)
        )], reference["version"])
```

The version check applies even to no-ops. Affected relationships need
explicit decisions via `edge_edits` ([graph edges](graph-edges.md#decisions-when-geometry-changes));
the geometry layer infers none. For whole desired arrays use
[`updateGeometryArrays`](editing.md#whole-array-updates-updategeometryarrays);
for adding/removing rows or resizing grids, the
[structural edits](editing.md#structural-sequence-edits).

Sequence geometry (fibers, point collections, tracks) is created once with
`putGeometry` / `put_geometry`. Source annotations supplied at creation are
consumed into that revision, read with `geometry_point_annotations`, and
edited through sequence operations alongside the points they describe.
Import-summary counts and lengths stay historical annotations; view counts
and owned measurements describe current geometry.
