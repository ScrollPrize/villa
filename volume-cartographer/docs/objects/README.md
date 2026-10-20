# Object store

`ObjectStore` owns objects, immutable geometry, and relationship metadata in a
single SQLite file. Ingest native data, discover relationships from stored
geometry, then query, annotate, export, or synchronize using that same handle.

The query APIs provide bounded pages, graph views, and packed geometry arrays for resident
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
relationships untouched.

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
records and choose a side explicitly.

## Sparse geometry edits

`edit_geometry` / `ObjectStore::editGeometry` replaces scalar ranges in existing
geometry arrays. It is independent of application tools and works with every
supported array dtype and geometry kind, including auxiliary arrays. Offsets
count **scalar elements in the flattened C-order array**, not bytes or points:
XYZ point 5 begins at element 15. Supply exact little-endian bytes in the existing
dtype. An edit cannot change array shape, dtype, names, or geometry attributes;
use `put_geometry` for ingestion and explicit full/structural replacements.

```python
import struct

current = store.current_geometry(object_uuid)
edits = [
    {"array": "points", "element_offset": 15,
     "data": struct.pack("<3f", 12.0, 24.0, 36.0)},  # float32 example
]
affected = store.affected_geometry_edit_edges(
    object_uuid, edits, expected_revision=current["uuid"])
# If affected is nonempty, supply explicit edge_edits decisions, as for put_geometry.
with store.write() as tx:
    revision = tx.edit_geometry(
        object_uuid, edits, expected_revision=current["uuid"], edge_edits=[])
```

One call creates at most one revision in the caller's transaction. Standalone
Python calls commit their own transaction. Overlapping ranges are applied in
input order (last write wins), original bytes come from the expected revision,
and net no-ops produce no revision or journal entry. Stale revisions fail even
for empty/no-op edits. Publication is savepoint-protected: a failed edit leaves
the caller's transaction usable. Geometry, current storage, spatial bounds,
relationship decisions, and approval changes publish atomically.

Schema v8 retains initial/full snapshots and stores subsequent sparse revisions
as immutable deltas. Within each changed physical chunk, the versioned binary
encoding stores byte-range gaps/lengths as varints, original bytes, and exact
bitwise XOR bytes. Each revision's payload uses Zstandard level 1 when smaller,
otherwise raw bytes. Payload and manifest checksums, original/result chunk
hashes, and a 256 MiB decoded-delta limit protect reconstruction. This is
lossless for integers and all floating-point bit patterns, including signed
zero and NaN payloads. It does not replay numerical operations.

The current geometry is a replaceable chunk overlay on its full snapshot,
with one complete current manifest. Delta revisions retain compact array
descriptors and changed-chunk hashes, not full tile-reference lists. Only
touched payload chunks and spatial regions are updated; manifest bookkeeping
and aggregate spatial bounds still scale with the object's chunk/region count.
`content_hash` continues to identify the complete resolved geometry manifest,
independent of whether its revision uses a snapshot or delta.

There are **no automatic checkpoints or recovery saves**. Existing revision,
range, rectangle, packed-array, export, and synchronization APIs accept either
representation. Historical reads verify the manifest chain back to the last
explicit snapshot, then reconstruct touched chunks from that snapshot or reverse
a shorter suffix from the current materialization when the target is its
ancestor. Current reads do not replay history. Older historical reads can become
slower as history grows; there is no pruning API. Metadata/relationship history
is not reconstructed by geometry deltas.
SQLite WAL checkpointing is separate from geometry checkpoints: long-lived read
transactions can retain older current-tile pages in the WAL, temporarily using
more disk space than the retained delta history.

Writable opens upgrade older databases without rewriting their retained
snapshots. Legacy snapshot databases remain readable in read-only mode.
`put_geometry` retains its full-snapshot behavior; callers opt into sparse edits
through the new API. VC3D's tools, undo stack, and file-save path are unchanged.

## Tests

Run the focused suite against the existing build and Python environment:

```sh
ninja -C build
ctest --test-dir build --output-on-failure \
  -R '^(test_object_(records|store|links)|test_progress|test_surface_patch_index|test_volume_static_read|test_volume_chunk_errors)$'
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 VC_BUILD_PYTHON=build/python \
  .venv/bin/python -m pytest python/tests/test_object*.py -q
```

Python tests cover public storage workflows, including ingestion/export of the
checked-in PHerc 0172 segments. Native tests cover transaction snapshots,
concurrency, detector kernels, and C++ validation. Keep each regression at the
layer that exercises it directly; avoid duplicating CRUD tests, checking SQLite
itself, or recreating every intermediate development schema. One oldest-supported
schema upgrade smoke test covers compatibility.
