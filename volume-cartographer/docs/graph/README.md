# Object graph

Store and query relationships between patches, fibers, and Spiral tracks in a
`.vcobj` SQLite database. ObjectStore owns objects and their relationship metadata. Geometry can
remain in source files or be ingested into the same database; these examples
register external sources. Imported patches and fibers retain their source UUIDs, so
names and paths are labels and location hints, not identity.

[Setup](#setup) · [Command line](#command-line) · [Python](#python) ·
[C++](#c) · [Scales and caching](#scales-and-caching) ·
[Detailed linking reference](../graph_linking_reference.md)

## Setup

From an already configured Volume Cartographer checkout:

```bash
ninja -C build vc_graph_import vc_object_bindings
```

For Python, install the [native prerequisites](../../README.md#python-source-builds)
and run `uv sync` in `volume-cartographer/`. Run the Python examples with
`uv run python`. An existing environment must load bindings from this checkout,
including when testing a module built under `build/python`.

Inputs are patch directories containing tifxyz geometry and a UUID in
`meta.json`, and fiber JSON files containing a UUID and line/control points.
Examples below use this layout; replace the paths with your own:

```text
/data/dataset/
├── objects.vcobj                 # created by the examples
├── verified_patches/<name>/   # meta.json, x.tif, y.tif, z.tif
└── fibers/<name>.json
```

## Command line

Import objects and detect patch overlaps (creates the DB if needed):

```bash
build/bin/vc_graph_import --db /data/dataset/objects.vcobj --author "$USER" \
  --fibers /data/dataset/fibers \
  --segments /data/dataset/verified_patches \
  --link-patch-overlaps /data/dataset/verified_patches
```

Omit `--link-patch-overlaps` to import metadata only. Patch import does not infer
links from `overlapping.json`; fiber import also reads existing branch links.
Use `--help` for all options, or see the Python/C++ examples below to run the
same operations in code.

## Python

### Import and link objects

```python
from pathlib import Path
from vc.objects import ObjectStore

root = Path("/data/dataset")
path = str(root / "objects.vcobj")
open_db = ObjectStore.open if Path(path).exists() else ObjectStore.create
with open_db(path, author="reviewer") as db:
    for report in (
        db.import_patches(str(root / "verified_patches")),
        db.import_fibers(str(root / "fibers")),
    ):
        if report["problems"]:
            raise RuntimeError(report["problems"])

    report = db.link_patch_overlaps(str(root / "verified_patches"), workers=4)
    if report["problems"]:
        raise RuntimeError(report["problems"])
    print("new patch links:", report["edges_created"])

    report = db.link_fibers_to_patches(
        str(root / "fibers"), str(root / "verified_patches"), "/data/ct.zarr",
        ct_level=0, fiber_scale=0.25, patch_scale=1,
        working_to_base_scale=4, ct_to_base_scale=4,
    )
    if report["problems"]:
        raise RuntimeError(report["problems"])
    print("new fiber–patch links:", report["edges_created"])
```

These scales assume full-resolution fibers and quarter-resolution patches/CT;
[adjust them for your data](#scales-and-caching). Each Python import or detector
call commits independently. Reports can include skipped inputs; inspect
`problems` even when no exception is raised.

### Find overlaps, attached fibers, and those fibers' neighbours

```python
from vc.objects import ObjectStore

with ObjectStore.open("/data/dataset/objects.vcobj", read_only=True) as db:
    patches = db.objects_by_kind("patch")
    if not patches:
        raise RuntimeError("No patches have been imported")
    patch = patches[0]  # Or db.object("<known-patch-uuid>").
    print("selected patch:", patch["uuid"], patch["name"])

    overlaps = db.neighbours(
        patch["uuid"], edge_kind="same-surface", neighbour_kind="patch"
    )
    for uuid in overlaps:
        print("overlap:", uuid, db.object(uuid)["name"])

    for fiber in db.neighbours(patch["uuid"], neighbour_kind="fiber"):
        print("attached fiber:", fiber)
        for other in db.neighbours(fiber, neighbour_kind="fiber"):
            print("  linked fiber:", other)

    for edge in db.edges_of(patch["uuid"]):
        print(edge["kind"], edge["approval_state"], edge["meta"])

    path = db.resolve_storage_path(patch["uuid"], dataset_root="/data/dataset")
    print("patch directory:", path)
```

`neighbours` returns direct neighbours as UUIDs, including pending and rejected
edges. Inspect `edges_of` to filter by approval state. A fiber–fiber link is a
stored relationship; sharing a patch alone does not create one. Use
`db.component(uuid)` for the connected component, or `db.reachable_within(uuid, 2)`
for objects at most two edges away.

To approve a reviewed edge, open a writable handle and call
`db.set_approval(edge_uuid, "approved")`. Use `with db.write() as txn:` to group
manual edits such as `txn.set_approval(...)` and `txn.set_meta(...)` atomically.

## C++

Link against `vc_objects` for storage and queries, or `vc_object_links` for geometry
detectors. Both require C++20. In a CMake build that includes Volume Cartographer,
add targets for the examples below:

```cmake
add_executable(graph_query graph_query.cpp)
target_link_libraries(graph_query PRIVATE vc_objects)
add_executable(graph_build graph_build.cpp)
target_link_libraries(graph_build PRIVATE vc_object_links)
```

Build with `ninja -C build graph_query graph_build`. In the Volume Cartographer
build tree, the executables appear in `build/bin/`.

### Query an existing graph

Save as `graph_query.cpp`. Pass the graph file and a patch UUID; optionally pass
the dataset root if it differs from the graph's directory.

```bash
build/bin/graph_query /data/dataset/objects.vcobj "<patch-uuid>" /data/dataset
```

```cpp
#include <iostream>
#include "vc/core/objects/ObjectStore.hpp"

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: graph_query objects.vcobj patch-uuid [dataset-root]\n";
        return 1;
    }
    vc::objects::OpenOptions options;
    options.readOnly = true;
    auto db = vc::objects::ObjectStore::open(argv[1], options);
    auto txn = db.read();
    auto patch = db.objectByUuid(txn, argv[2]);
    if (!patch || patch->kind != "patch") return 1;
    std::cout << "selected patch: " << patch->uuid << " " << patch->name << '\n';

    for (const auto& uuid : db.neighbours(txn, patch->uuid, "same-surface", "patch"))
        std::cout << "overlap: " << uuid << '\n';
    for (const auto& fiber : db.neighbours(txn, patch->uuid, std::nullopt, "fiber")) {
        std::cout << "attached fiber: " << fiber << '\n';
        for (const auto& other : db.neighbours(txn, fiber, std::nullopt, "fiber"))
            std::cout << "  linked fiber: " << other << '\n';
    }
    for (const auto& edge : db.edgesOf(txn, patch->uuid))
        std::cout << edge.kind << " " << edge.approvalState << '\n';

    std::optional<std::filesystem::path> root;
    if (argc > 3) root = argv[3];
    if (auto path = db.resolveStoragePath(txn, patch->uuid, root))
        std::cout << "patch directory: " << *path << '\n';
}
```

Use `db.objectsByKind(txn, "patch")` to discover UUIDs,
`db.componentOf(txn, uuid)` for a connected component, and
`db.reachableWithin(txn, uuid, 2)` for traversal limited to two edges.

### Import and link objects

Save as `graph_build.cpp`. Run with a dataset directory and CT Zarr path.
It uses the same coordinate scales as the Python example.

```bash
build/bin/graph_build /data/dataset /data/ct.zarr
```

```cpp
#include <iostream>
#include <stdexcept>
#include "vc/core/objects/ObjectStore.hpp"
#include "vc/core/objects/import/FiberImport.hpp"
#include "vc/core/objects/import/PatchImport.hpp"
#include "vc/core/object_links/ObjectLinks.hpp"

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "Usage: graph_build dataset-dir ct.zarr\n";
        return 1;
    }
    const std::filesystem::path root = argv[1], path = root / "objects.vcobj";
    vc::objects::OpenOptions options;
    options.author = "reviewer";
    auto db = std::filesystem::exists(path)
        ? vc::objects::ObjectStore::open(path, options)
        : vc::objects::ObjectStore::create(path, options);
    const auto check = [](const auto& report) {
        for (const auto& problem : report.problems) std::cerr << problem << '\n';
        if (!report.problems.empty()) throw std::runtime_error("Check input problems");
    };
    {
        auto txn = db.write();
        check(vc::objects::importPatches(db, txn, root / "verified_patches"));
        check(vc::objects::importFibers(db, txn, root / "fibers"));
        txn.commit();
    }
    // Detectors manage their own transactions: call outside the write scope.
    vc::object_links::PatchOverlapParams overlaps;
    overlaps.workers = 4;
    check(vc::object_links::linkPatchOverlaps(db, root / "verified_patches", overlaps));

    vc::object_links::FiberPatchParams params;
    params.fiberScale = 0.25;
    params.patchScale = 1;
    params.workingToBaseScale = 4;
    params.ctToBaseScale = 4;
    params.ctLevel = 0;
    check(vc::object_links::linkFibersToPatches(
        db, root / "fibers", root / "verified_patches", argv[2], params));
}
```

C++ writes require an explicit transaction and `commit()`; destruction without
commit rolls back. Keep read transactions short: each holds a snapshot. For a
manual review update, use `db.setApproval(txn, edgeUuid, "approved")` inside a
write transaction and commit it.

## Scales and caching

- `fiber_scale` / `patch_scale` convert source XYZ to common working voxels.
  `working_to_base_scale` / `ct_to_base_scale` give working and selected-level
  CT voxel sizes in common base units. All default to 1. Changing `ct_level`
  does not automatically adjust these scales.
- Fiber strips are saved automatically in `fibers/.strip-cache/<uuid>.cbor`.
  Changed geometry/settings invalidate them. After changing CT voxel data in
  place without changing metadata, use Python `rebuild_strips=True`, C++
  `params.rebuildStrips = true`, or CLI `--rebuild-fiber-strips`.
- Detectors propose pending links and preserve reviewed/manual edges on reruns.
  Patch and fiber–patch links default to `same-surface`; imported fiber branch
  links use `fiber_branch`; track crossings use `crossing`.
- File resolution checks the saved path's UUID, then searches the dataset root.
  It returns `None` / `std::nullopt` if missing and errors on ambiguous fallback
  matches. Existing graphs need no rebuild. See [storage and relocation](../graph_storage.md).

For Spiral tracks, pass a DBM and its adjacent `.vctracks` store, or a packed
store directly. See [track-to-track](../graph_linking_reference.md#track-to-track-graph-links)
and [track-to-patch](../graph_linking_reference.md#track-to-patch-graph-links)
examples and their Python/C++ entry points. Algorithm details, cache rules, and
historical benchmarks live in the [linking reference](../graph_linking_reference.md).
