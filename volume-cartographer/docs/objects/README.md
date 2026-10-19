# Object store

`ObjectStore` owns objects, immutable geometry, and relationship metadata in a
single SQLite file. Ingest native data, discover relationships from stored
geometry, then query, annotate, export, or synchronize using that same handle.

[Storage and synchronization API](../object_storage.md) ·
[Link detector reference](../object_links.md)

The [query APIs](../object_storage.md#consumer-queries-and-store-backed-graph-reads)
provide bounded pages, graph views, and packed geometry arrays for resident
working data. Existing application consumers remain separate from the store.

## Build and command line

```sh
ninja -C build vc_object_ingest vc_object_export vc_object_links_cli vc_object_bindings
build/bin/vc_object_ingest --db objects.vcobj --author "$USER" \
  --patch /data/patch-a --patch /data/patch-b --fiber /data/fibers
build/bin/vc_object_links --db objects.vcobj --author "$USER" --link-patch-overlaps
build/bin/vc_object_links --db objects.vcobj --author "$USER" \
  --link-fiber-patches --ct /data/ct.zarr
```

`vc_object_links_cli` is the executable's CMake target; `vc_object_links` is the
link library target and the installed executable name. Use `--help` for detector
options. Original geometry files are needed only for ingestion. CT or normal
volumes remain inputs to fiber/track-to-patch detection.

## Python

```python
from vc.objects import ObjectStore

with ObjectStore.create("objects.vcobj", author="reviewer") as store:
    for source in ("/data/patch-a", "/data/patch-b"):
        report = store.ingest_geometry_source("patch", source)
        assert not report["problems"], report["problems"]
    report = store.link_patch_overlaps()
    assert not report["problems"], report["problems"]

    patches = store.objects_by_kind("patch")
    seed = patches[0]["uuid"]
    neighbours = store.neighbours(seed, edge_kind="same-surface")
    component = store.component(seed, edge_kinds=["same-surface"])
    for edge in store.edges_of(seed):
        store.set_meta(edge["uuid"], "review_note", "inspect this contact")

    metadata = store.export_json(connected_to=seed, fields=["name", "meta", "tags"])
```

Names, tags, typed properties and native annotations belong to object metadata.
Typed point counts, lengths, grid dimensions and boxes are source/import
summaries; direct geometry edits preserve them. Read current array shapes and
patch scale from `current_geometry()`, and current bounds through spatial queries.
Edges hold relationship types, anchors, provenance, approvals, and endpoint
geometry bindings. Source paths are ingestion hints; UUIDs identify records.
Geometry-backed endpoints declare their exact point dependencies. Changing a
referenced point requires an explicit retain/update/remove decision for each affected
relationship in `put_geometry(..., edge_edits=[...])`. Unrelated point edits leave
relationships untouched. See the [edit contract](../object_storage.md#point-relationships-and-geometry-edits).

## C++

Link `vc_objects` for storage, `vc_object_geometry` for native ingestion/export,
and `vc_object_links` for relationship detection.

```cpp
#include "vc/core/objects/ObjectStore.hpp"
#include "vc/core/objects/GeometryIngest.hpp"
#include "vc/core/object_links/ObjectLinks.hpp"

vc::objects::OpenOptions options;
options.author = "reviewer";
auto store = vc::objects::ObjectStore::open("objects.vcobj", options);
const auto report = vc::objects::ingestGeometrySource(store, "patch", "/data/patch-a");
if (!report.problems.empty()) throw std::runtime_error(report.problems.front());
const auto links = vc::object_links::linkPatchOverlaps(store);

auto read = store.read();
vc::objects::ObjectSelection selection;
selection.kind = "patch";
const auto patches = store.selectObjects(read, selection);
vc::objects::JsonExportOptions exportOptions;
exportOptions.selection = selection;
exportOptions.fields = {"name", "tags"};
const auto metadata = store.exportJson(read, exportOptions);
```

Use a write transaction to commit metadata and geometry together. Each read
transaction sees a consistent snapshot. For independent copies, establish a
shared baseline with `synchronize` before editing, then review conflicting
records and choose a side explicitly. See the
[synchronization contract](../object_storage.md#two-store-three-way-synchronization)
for geometry transfer and interruption recovery.
