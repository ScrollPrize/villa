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

## Edit persistence benchmark

`bench_object_edits` compares `editGeometry()` plus durable transaction commit
with the existing `QuadSurface::saveOverwrite()` autosave worker call on copies
of real tifxyz segments. It never writes to its source. The output reports mean,
p50, p95, p99, and maximum milliseconds, separating update collection, delta
packing, edit processing, and commit. It also reports first/last windows, database
growth after SQLite WAL checkpointing, peak WAL file size, and full-array reads
of current, previous, middle, and initial revisions. All four versions and the
saved tifxyz are verified byte-for-byte.

```sh
AGENTS_AGENT_MODE=1 ninja -C build bench_object_edits
AGENTS_AGENT_MODE=1 .venv/bin/python scripts/bench_object_edits.py \
  --segment /path/to/real/segment \
  --radius 5.75 --radius 16 --radius 64 \
  --edits 1000 --file-saves 50 --repeats 3 \
  --output /tmp/edit-results.jsonl
```

The runner executes cases serially, using a fresh temporary database for each
case. Supply `--segment` repeatedly for more inputs. Output files must not exist.
Use `--per-point` to test one API range per vertex instead of contiguous runs.
Temporary copies and databases are removed after each successful native run.

The workloads select valid points in circular grid footprints at 64 deterministic
locations, including tile boundaries, and change all XYZ components. They model
the persistence workload, not the actual push/pull deformation or rendering.
The UI default radius is 5.75 grid steps (sigma 2), while 16 and 64 are larger
footprint stress cases. World-distance filtering, curvature, invalid points,
and movement during a stroke change the actual number of edited vertices.

Current VC3D geometry autosave queues vertices and runs the file save through
`QtConcurrent`; its timer interval is 10 seconds. The benchmark deliberately
measures each database action as a separate durable commit. It uses the default
SQLite WAL/`synchronous=FULL` settings, with no geometry checkpoints. File-save
timings use the existing write/close/rename implementation, which does not
explicitly fsync each save, so the durability guarantees differ. The first file
save includes the normal initial recovery snapshot; warm statistics exclude it.
Input loading and initial DB ingestion are outside the timed edit loops.

These measurements do not establish interactive frame rates: they exclude
rendering, deformation computation, relationship repairs, simultaneous writers,
long-held read transactions, and approval-mask edits. Future integration should
submit immutable edit data to a worker, preserving asynchronous persistence.

### Measurements, 2026-10-20

[Recorded results](benchmarks/edit-persistence-2026-10-20.json) use real Scroll 1
9.6 µm segments from `spiral_dataset_graph_uuid/verified_patches`, on this ARM
Cortex-X925/A725 machine with a Samsung NVMe/ext4 filesystem and a
`RelWithDebInfo` build. The selected bounding boxes span 11.7–38.0 cm³; the point
count and storage tiling, rather than physical bounding-box volume alone,
determine persistence cost. These are 9.6 µm inputs, not exactly 8 µm.

Each main case has three independent runs of 1,000 edits and 50 file saves.
Values below are medians of the three per-run statistics, not pooled percentiles.
Default-radius footprints contain up to 101 valid points.

| Segment | Grid points | DB mean | DB p50 | DB p95 | File-save p50 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `0000_top_vt_strip_1` | 78,581 | 2.52 ms | 2.38 ms | 4.18 ms | 0.80 ms |
| `0000_low_band_final` | 454,740 | 2.89 ms | 2.69 ms | 3.98 ms | 5.03 ms |
| `0000_top_band` | 3,173,488 | 3.20 ms | 2.99 ms | 4.51 ms | 64.94 ms |

On the largest surface, radius 16 has a median footprint of 797 points and
commits in 3.63 ms p50 / 6.01 ms p95. Radius 64 has a median footprint of 9,177
valid points (up to 12,814) and takes 8.93 ms p50 / 15.26 ms p95. The worst
observed default-radius commit was 17.42 ms; the radius-64 maximum was 18.52 ms.
Thus synchronous persistence on the UI thread cannot guarantee a 16.7 ms budget.
Submitting one range per vertex instead of contiguous runs increased the large
radius-64 case to 11.66 ms p50 / 18.10 ms p95 in a separate single run.

Packing default-size edits took approximately 0.003 ms p95. At radius 64 on the
largest surface, contiguous packing took 0.111 ms p95. On that surface, default
commits spent approximately 1.81 ms in edit processing and 1.39 ms in commit
(mean times). These are instrumented call-boundary timings; native sampling
could not run because `perf_event_paranoid=4` and sandbox ptrace restrictions
blocked perf and GDB respectively.

There was no large progressive slowdown over 1,000 writes. Default first/last
100-edit medians were 2.39/2.68 ms, 2.19/2.71 ms, and 3.09/2.99 ms for the three
surfaces. Historical reconstruction remains a separate limitation: on the
largest surface after 1,000 default edits, full current reads took about 50 ms,
previous-version reads 138 ms, and middle-version reads 219 ms. A middle-version
read after the radius-64 stress workload took approximately 4.85 seconds.
Those are single reads per run, with medians across three runs; no cold-cache
or history-navigation latency guarantee is implied.

The evidence supports a background commit per completed action for these
workloads. It does not demonstrate zero impact on interactive rendering, nor
fast arbitrary history navigation. No VC3D integration or production DB changes
were made for this benchmark.

Reproduce the recorded inputs by passing these three `--segment` arguments to
the command above:

```sh
--segment /home/sean/Desktop/spiral_dataset_graph_uuid/verified_patches/0000_top_vt_strip_1 \
--segment /home/sean/Desktop/spiral_dataset_graph_uuid/verified_patches/0000_low_band_final \
--segment /home/sean/Desktop/spiral_dataset_graph_uuid/verified_patches/0000_top_band
```

Validation for the benchmark addition:

```sh
AGENTS_AGENT_MODE=1 ninja -C build bench_object_edits test_object_store \
  test_quadsurface_save_roundtrip test_quadsurface_snapshot
AGENTS_AGENT_MODE=1 ctest --test-dir build --output-on-failure \
  -R '^(test_object_store|test_quadsurface_save_roundtrip|test_quadsurface_snapshot)$'
AGENTS_AGENT_MODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 VC_BUILD_PYTHON=build/python \
  .venv/bin/python -m pytest python/tests/test_object_geometry_edits.py -q
```

All three native test executables and all 20 Python edit tests passed. The 28
recorded benchmark runs also passed their exact geometry and file verification.
