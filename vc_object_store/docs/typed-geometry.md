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
| `Fiber` | float64 XYZ `controlPoints`, `linePoints` | Polyline length, validation |
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

Snapshots pin a consistent database state at creation. Views and editors keep
that snapshot alive even if the snapshot variable goes out of scope. Explicit
`snapshot.close()` (also Python context-manager exit) invalidates all its views
and editors and releases the reader. Snapshots are thread-confined and consume
one slot in the store's reader pool; close them promptly.

A C++ snapshot borrowing `ObjectStore&` must finish before the store is moved or
destroyed. The `GeometrySnapshot(std::shared_ptr<ObjectStore>)` overload owns a
store lease; Python uses this overload and rejects closing a store with active
snapshots. Returned owned geometry remains valid after snapshot closure.

## Checked edits and relationships

```python
with store.geometry_snapshot() as snapshot:
    collection = snapshot.point_collection(collection_uuid)
    edit = collection.edit()
    edit.move_point(42, [10, 20, 30])
    affected = edit.affected_relationships()
    # This example applies when no relationships depend on the edited point.
    assert not affected
    new_revision = edit.commit()
    # collection still reads the original revision.
```

Editors provide `movePoint`, `moveControlPoint`, `moveLinePoint`,
`moveTrackPoint` and `moveVertex` (snake_case in Python). Calls check the geometry
kind, indices, IDs and values before recording a sparse change. No database
mutation occurs until `commit()`. Repeated changes apply in call order.

Commit checks both the object revision and geometry revision observed by the
view. Concurrent geometry, metadata or relationship changes therefore conflict,
including on proposed no-ops. Successful edits preserve the existing sparse
storage, history and numerical behavior. An editor can commit once; a rejected
relationship decision leaves the database unchanged and can be corrected before
retrying.

Affected relationships require the existing explicit `GeometryEdgeEdit`
decisions: retain, update, remove or keep recorded. `commit(edge_edits=...)` in
Python accepts the same decision dictionaries as `edit_geometry()`. Other
relationship endpoints are also checked against the editor's snapshot.
No relationship meaning is inferred by the geometry layer.

Structural changes (adding/removing/reordering points or changing dimensions)
use an owned value and `putGeometry` / `put_geometry` with the expected geometry
revision and explicit relationship decisions. Python `affected_geometry_edges()`
also accepts owned values to preview a replacement, on both the store and a
write transaction. For actions involving multiple
object replacements or metadata changes, use the existing checked write
transaction API.

## Compatibility and implementation

The database schema, hashes, array names/dtypes, file formats and existing APIs
are unchanged. Generic array reads and writes remain supported. Historical
`FiberDetail`, `PatchDetail` and `TrackDetail` counts and lengths remain import
annotations for compatibility; use view counts and owned geometry measurements
for current values.

`src/formats/{Fiber,Patch,PointCollection,Track}Input.cpp` decode sources.
`GeometryOutput.cpp` encodes native files. These adapters do not access the
database: ingestion resolves identities and imports relationships; export
resolves revision metadata and publishes staged files. Shared array encoding and
layout validation serve both typed geometry and the existing storage paths.

The tests run without application checkouts or real fixture files:

```sh
AGENTS_AGENT_MODE=1 cmake --build build -j 8
AGENTS_AGENT_MODE=1 ctest --test-dir build --output-on-failure
AGENTS_AGENT_MODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  VC_OBJECT_STORE_BUILD_PYTHON_PATH=build/python \
  /path/to/python -m pytest -q tests/python
```

For real data, the validation harness creates temporary databases, checks typed
values against source JSON, compares every stored array byte-for-byte after
typed decoding/encoding, and verifies native export. Patch/track exports are
also re-ingested and their arrays compared. Inputs must contain only the native
geometry arrays: auxiliary surface channels are not representable by owned
typed geometry or native export. Sources are read-only; temporary
databases and exports are deleted on exit. Flags are repeatable:

```sh
AGENTS_AGENT_MODE=1 VC_OBJECT_STORE_BUILD_PYTHON_PATH=build/python \
  /path/to/python scripts/validate_typed_geometry.py \
  --point-collections /data/abs_winding.json \
  --fiber /data/fiber.json \
  --patch /data/patch
```

Existing native directory
export still requires Linux `renameat2`; this change does not alter that limit.
