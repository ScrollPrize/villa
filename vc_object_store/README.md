# vc_object_store

A SQLite object database for Vesuvius Challenge geometry. It replaces loose
geometry files (tifxyz surfaces, fiber JSON, point-collection JSON, packed
tracks, umbilici) with one `.vcobj` file holding objects, immutable revisioned
geometry, revisioned metadata, and graph edges between objects — so provenance
is stored once, and the state of every object at any past time can be read back
without copies.

## What it does

- **Stores five geometry kinds** as immutable, chunked binary revisions:
  `surface` (grid + aligned channels such as `valid`/ink), `fiber`,
  `track`, `point_collection`, and `umbilicus`.
- **Versions everything.** Every object and relationship has a stable UUID and
  a numbered version, starting at 1, advancing once per transaction that
  changes it. A historical reference is
  `(store, generation, object, version)`; with history enabled, retained revisions stay readable and restore is a new checked edit, never a moved head.
- **Checked writes.** A write names the versions it read. If another writer
  moved them first, the write conflicts and commits nothing.
- **Graph edges.** Relationships (same-winding/same-surface links, fiber
  branches, track crossings, point–surface links, attachments, declared
  references) are first-class revisioned records with anchors into endpoints'
  geometry, approval state, and applicability: whether an edge still applies
  to the endpoints' current geometry is computed, not assumed.
- **Relationship discovery.** Detectors find surface overlaps and line–surface,
  track–track, and point-collection–surface links over stored geometry and
  save them as pending edges.
- **Spatial queries.** A persisted R-tree index backs bounded region browsing
  (`geometry_region`) and nearest-surface lookup (`nearest_surfaces`), reading
  only the tiles near the query.
- **Coordinate frames.** Geometry stays in the volume it was recorded in
  (`meta.coordinates`). Volume-to-volume registrations, themselves revisioned
  records, let reads and edits happen in another volume's voxels.
- **Files stay available.** Ingestion is one-time; afterwards any revision
  exports back to its native file format. Non-geometry files (fitted models,
  logs) attach to an object version as content-addressed artifacts.
- **Local or hosted.** One writer can serve a store with `vc_object_server`;
  C++ (through `ObjectStoreSession`) and Python clients then use the same API
  without opening the database file.

## Immutable and large-scale stores

Schema 21 supports current-only storage (`history_enabled=False`), optional spatial indexing, sealing, [store_references, overlays and independent shards](docs/collections.md), and optional [packed SQLite containers](docs/packed-stores.md). Current record versions and the store commit sequence remain available with history disabled; requests for unavailable history fail explicitly. A lineage UUID is not a numeric change counter. Spatial queries require the spatial index.

For large immutable datasets, the C++ [initial batch construction API](docs/initial-import.md) accepts objects, geometry and relationships from a shared parallel streaming source. See the [format and policy reference](docs/store-format.md) and [schema integration contract](docs/schema-integration.md). The redesigned lazy/delta history layout remains deferred.

## Examples

Python (`links` return a report dict; always inspect `problems`):

```python
from vc_object_store import ObjectStore

with ObjectStore.create("objects.vcobj", author="reviewer") as store:
    report = store.ingest_geometry_source("surface", "/data/surface")
    assert not report["problems"], report["problems"]
    store.link_surface_overlaps()

    surfaces = store.objects_by_kind("surface")
    with store.geometry_snapshot() as snapshot:
        tile = snapshot.surface(surfaces[0]["uuid"]).read_tile(0, 0, 64, 64)
    store.export_geometry(surfaces[0]["uuid"], "/data/exported-surface")
```

A point-in-time read takes the versions live at a time and pins them:

```python
refs = store.versions_at(when_us)                 # every object live at that time
records = store.objects_at_versions(refs)          # their records then
edges = store.relationships_at_versions(refs)      # edges with applicable flags
with store.graph_read(versions=refs, adjacent=False, approval_states=[]) as graph:
    component = graph.component(seed_uuid)
```

CLI (creating the store when `--db` does not exist yet):

```sh
vc_object_ingest --db objects.vcobj --author "$USER" --surface /data/surface
vc_object_ingest --db objects.vcobj --author "$USER" --fiber /data/fibers
vc_object_ingest --db objects.vcobj --author "$USER" \
  --surface-channel SURFACE_UUID ink /data/ink_prediction.tif
vc_object_links --db objects.vcobj --author "$USER" --link-surface-overlaps
vc_object_links --db objects.vcobj --author "$USER" \
  --link-fiber-surfaces --ct /data/ct.zarr        # or --normal-x/--normal-y, --normal-pool, or none
vc_object_export --db objects.vcobj --uuid SURFACE_UUID --output /data/out
vc_object_server objects.vcobj --port 8750   # serve to clients (see docs/hosted.md)
```

VC3D runs the same operations on its attached store: File → Object Store →
New Object Store..., Import from Folder..., and Stores → Attach Hosted
Store... for a served store.

## API docs

Start at the [docs index](docs/README.md). In short:

- [Ingestion](docs/ingestion.md) — one-time import of files and placeholders.
- [Reading](docs/reading.md) — listing, snapshots, bounded array reads,
  history, spatial queries, selection records.
- [Writing](docs/writing.md) — creating objects and geometry, the checked
  transaction model, export, artifacts, projects.
- [Editing](docs/editing.md) — sparse and structural geometry edits, surface
  channels, metadata, retire/restore, editor drafts.
- [Graph edges](docs/graph-edges.md) — relationship kinds, anchors,
  applicability, traversal, and the link detectors.
- [Transforms](docs/transforms.md) — volume identity and scale, registrations,
  transformed views and edits.
- [Typed geometry](docs/typed-geometry.md) — owned geometry values and
  snapshot views, in C++ and Python.
- [Hosted stores and sessions](docs/hosted.md) — `vc_object_server`,
  hosted clients, `ObjectStoreSession`, change events, durable drafts.

C++ callers link `vc_object_store::vc_objects` for storage,
`vc_object_store::vc_object_geometry` for native file I/O, and
`vc_object_store::vc_object_links` for detectors. Public signatures live in
[include/vc_object_store/objects/ObjectStore.hpp](include/vc_object_store/objects/ObjectStore.hpp).

## Build and test

Tests are self-contained; they do not need a volume-cartographer checkout or
its fixture data. A standalone build and regression run:

```sh
cmake -S vc_object_store -B /tmp/vc-object-store-build -G Ninja \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo -DVC_OBJECT_STORE_BUILD_ADAPTERS=ON \
  -DVC_OBJECT_STORE_BUILD_LINKING=OFF -DVC_OBJECT_STORE_BUILD_TESTS=ON
cmake --build /tmp/vc-object-store-build -j 4
ctest --test-dir /tmp/vc-object-store-build --output-on-failure
```

Python tests, against a Python-enabled build (`-DVC_OBJECT_STORE_BUILD_PYTHON=ON`,
module under `<build>/python`):

```sh
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 VC_OBJECT_STORE_BUILD_PYTHON_PATH=build/python \
  /path/to/python -m pytest vc_object_store/tests/python -q
```

Older upstream stores (schema 16–21; the runtime opens schema 22) convert with
`vc_object_convert old.vcobj new.vcobj`, which also writes a revision mapping
and a report beside the destination. Native directory export currently
requires Linux `renameat2` or macOS `renamex_np`.
