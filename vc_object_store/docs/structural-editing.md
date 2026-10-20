# Structural geometry editing

Structural edits publish ordinary immutable geometry and object revisions.
Use a checked writer with **object** revision UUIDs, and pass the expected
**geometry** revision UUID to each mutation. The Python API exposes structural
mutations on the checked writer; C++ uses `ObjectStore::editGeometrySequence()`
and `ObjectStore::resizeSurface()`.

## Sequences

`edit_geometry_sequence(object_uuid, edit, expected_revision, edge_edits=[])`
splices rows of `points` (point collections/tracks), `control_points`, or
`line_points` (fibers). `start` and `erase_count` select the removed span.
`parts` supplies the replacement, in order:

- New rows: `count`, little-endian XYZ `data`, and, for collections/control
  points, explicit nonnegative `ids`. Optional `annotations` contains one JSON
  object or null per new row.
- Existing rows: `revision`, `start` (default zero), `count`, and optional
  `reverse`. These copy coordinates, IDs, and annotations together. References
  can select another object of the same kind, including historical revisions.
  Include every source object in `expected_revisions`.

```python
import struct

object_head = db.current_object_revision(uid)['uuid']
geometry_head = db.current_geometry(uid)['uuid']
with db.write(expected_revisions={uid: object_head}) as tx:
    result = tx.edit_geometry_sequence(uid, {
        'array': 'points', 'start': 2, 'erase_count': 0,
        'parts': [{'count': 1, 'ids': [42],
                   'data': struct.pack('<3d', 10, 20, 30),
                   'annotations': [{'label': 'new point'}]}],
    }, geometry_head)
```

Append uses `start=count, erase_count=0`; insertion uses `erase_count=0`;
deletion uses no parts. Span replacement combines removal and new/reference
parts. Reverse the complete sequence by replacing it with a reversed reference
to its current revision. Arbitrary reorderings use references to the desired
runs. A duplicate resulting ID is rejected, including ambiguous merges.

Split by filling previously created empty result objects with disjoint source
slices. Merge by inserting source slices into the destination. Retire source
objects and update relationships in the **same checked transaction**. The store
does not choose which objects to retire or how to reconnect domain topology.

Point-collection IDs survive reordering. Fibers may supply `control_point_ids`
at creation; otherwise their first structural control edit establishes IDs
`0..N-1` for the existing controls. Later new controls require explicit IDs.
`Fiber.control_point_ids` preserves these IDs through typed loads/encoding.
Sampled `line_points` and tracks have no stable point IDs.

A point deleted from an identity-managed sequence cannot be recreated as new
rows using its old ID. An explicit slice of its historical revision can restore
it. The allocation ledger also survives selection of an older geometry through
lifecycle restoration. Before the first structural edit, legacy IDs/annotations
form the baseline; this does not reconstruct identities lost in earlier legacy
full replacements. IDs are unique within a sequence; merging colliding source
ID namespaces requires an application decision.

Annotations are stored by ID in immutable indexed trees, separately from array
order; each geometry manifest retains the roots. `geometry_point_annotations(revision_uuid, array_name)` returns
annotations aligned with that revision's sequence. The first structural edit
moves the corresponding legacy `meta.source` positional annotation field into
this representation. Existing native PCL export resolves annotations through
IDs, preventing the same-length label-swap problem. Low-level ID edits/full
replacements of annotated legacy collections are rejected; after structural
management begins, full `put_geometry` replacement is rejected. Use structural
splices or the existing sparse coordinate edit API.

`segment_to_next`, fields containing `direction` or `tangent`, and nested fields
with those names require `transformed_annotations`: a complete annotation list
for the resulting order. This applies to insertion/deletion as well as reversal.
The application explicitly relocates span ownership and transforms directions;
the store does not infer domain meaning or negate vectors automatically. Other
application-specific directional/span fields must also be transformed by the
caller. Omission of a required transformation aborts the checked action, even if
the application catches the immediate exception.

## Surface growth and cropping

`resize_surface(object_uuid, resize, expected_revision, edge_edits=[])` accepts:

- `row`, `column`: the old logical coordinate of the new top-left corner.
- `rows`, `columns`: the resulting shape, including zero-sized grids.
- `fill`: one typed pixel for **every** existing array, including `points`, in
  the usual `{name, dtype, shape, data}` format. Shape is `[1,1]` for a scalar or
  `[1,1,components]` for a component array. Fills are required even for cropping.

New `(r,c)` maps to old `(r+row,c+column)`. Negative offsets grow top/left;
positive offsets crop them. The requested shape independently grows/crops the
bottom/right. XYZ and all channels resize together, preserving exact scalar
bytes, component counts, NaN payloads and signed zero. Newly exposed cells use
the supplied fills, including cells that had been cropped away earlier.

Geometry revision descriptors expose `grid_origin=[row,column]`. The tile grid
stays fixed in these coordinates across non-aligned left/top changes. Logical
anchor coordinates map as `new_index = old_index - resize_offset`. The caller
must explicitly update, remove, retain, or preserve affected relationships.
Origin changes invalidate positional applicability even if coordinates happen
to compare equal. Historical links keep their original revision pins.

Existing rectangle edits and channel add/replace/remove operations work after
resizing. Reads still return contiguous arrays, byte ranges, packed arrays, and
rectangles. Consumers do not need to interpret shared storage.

## Storage, history and cost limits

Sequence arrays are immutable AVL ropes with subtree byte counts and coordinate
bounds. A manifest contains one root per array instead of every block reference.
Splicing path-copies the affected branches and retains whole unchanged subtrees.
Leaves refer to exact byte slices of payloads of at most 64 KiB, aligned to
complete points. Newly adjacent fragments whose combined size is at most 4 KiB
are compacted locally, including boundaries exposed by slicing. Thus an array
of B bytes has at most `2 * (B // 4096 + 1)` leaves. Compaction never rewrites the
whole array and its work is included in measurements. Reads traverse tree
weights, not a revision chain: bounded reads cost tree descent plus the requested
leaves; full reads cost proportionally to output and its bounded fragmentation.

Stable-ID indexes use persistent ordered trees with at most 128 IDs per bucket.
Each live identity owns its coordinate bytes and annotation independently of
sequence position. Insertions/deletions update only affected buckets and tree
paths. Allocation ledgers retain deleted IDs, including after lifecycle undo;
restoring divergent allocation roots unions them without rewriting history.
Annotations larger than 1 KiB use separate chunked immutable blobs, keeping
identity buckets bounded. Sample annotations use a separate row-counted tree.
Span/direction summaries travel with tree roots, so detecting a required caller
transformation does not scan all annotations. A complete caller-supplied
transformation still costs proportionally to its input.

Surface arrays use persistent ordered tile maps keyed by stable signed grid
coordinates. Resizing visits changed cells and changed edge boundaries, retaining
interior subtrees. For example, right growth of a one-row, very wide grid does
not scan its unchanged width. Cropping removes outside cells and rewrites partial
boundary tiles, so subsequent growth uses fills instead of resurrecting old
pixels. Tile bounds include neighbouring tile bounds for conservative triangle
coverage; only affected neighbourhoods are updated.

SQL spatial indexes retain one coarse row per coordinate array. Fine spatial
queries descend immutable subtree bounds, return bounded leaf regions and line
seams, and apply pagination after refinement. Both SQL and resident spatial-cache
queries use this path. Whole-object bounds come from root summaries. No positional
suffix index is rebuilt after insertion. Broad-phase regions are conservative;
exact coordinates and numerical validity conventions are preserved.

The relationship dependency index maps stable point IDs to incident edges.
Sequence publication checks changed IDs plus conservative whole-geometry,
channel and positional dependencies; unchanged ID links need no scan. Surface
shape/origin changes still require decisions for applicable positional links.
Stable-ID validation uses the persistent identity index instead of reading the
full ID array. Work remains proportional to the number of affected relationships.

The **first structural edit converts legacy layouts and annotations**, with
work proportional to the source geometry. This one-time cost is measured
separately. Subsequent append/insertion is bounded by new data, split/compaction
boundaries and tree paths; growth is bounded by changed tiles/neighbourhoods and
tree paths. Reversal, arbitrary reordering, complete retracing, explicit complete
annotation transforms and full contiguous reads may process their full input.
There is no background compaction, reduced precision, approximation, or revision
replay hidden in the incremental path. The operation-local tree cache is bounded
by an 8 MiB conservative charge and is discarded between operations.

Schema **11** gates the indexed representation, adds the derived relationship
lookup table/triggers and a geometry-generation index. Writable opens migrate
atomically; legacy payloads and revisions remain untouched until conversion.
Older databases remain readable with read-only opens. Older binaries reject
schema 11 instead of attempting to interpret its storage. The public logical
geometry format and array APIs remain unchanged.

Synchronization recursively transfers new immutable tree nodes, payloads and
annotation blobs, reuses existing subtrees, and validates hashes/collisions.
Retained historical geometry stays readable after reopening and synchronization.
Existing synchronization limitations on complete object metadata/relationship
timelines still apply; use a database copy when that entire timeline is required.
No physical purge is introduced.

## Validation and measurement

From the package directory, using the existing build and environment:

```sh
AGENTS_AGENT_MODE=1 ninja -C build -j 4
AGENTS_AGENT_MODE=1 ctest --test-dir build --output-on-failure
AGENTS_AGENT_MODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  VC_OBJECT_STORE_BUILD_PYTHON_PATH=build/python \
  /home/sean/villa/volume-cartographer/.venv/bin/python -m pytest -q tests/python
AGENTS_AGENT_MODE=1 build/bin/bench_structural_edits 5 > /tmp/structural-bench.jsonl
```

The C++ regressions instrument complete checked publication, including
100,000-point annotated collections with 128 unchanged ID relationships,
a 500,000-point append and 128/16,384-column one-row grids. They bound identity,
manifest, tree-node, relationship and spatial work in addition to payload bytes.
Python tests cover sequence operations/labels, explicit span transforms,
all-side growth/cropping, typed channels, historical/pinned reads, rollback,
split/merge retirement, random long edit histories, AVL/fragmentation invariants,
sparse edits, large annotation blobs, spatial seams/pagination under both cache
modes, synchronization, and v10 migration without unintended history changes.

The benchmark compares full replacement with structural append/insertion and
right growth on deterministic synthetic inputs, with zero/eight relationships.
Initialization is reported separately. Each edit includes expected-head reads,
checked history publication, commit/durability, and any local compaction. The
full-replacement baseline includes caller array preparation. Inputs are
1,000/10,000/100,000-point collections and 128/512/1,024-square surfaces, with an
additional 80-append history. These are scaling fixtures, not real-scroll viewer
workloads. No macOS/amd64 execution or universal interactive-latency claim is made.

Reports include mean/min/median/max latency, payload and tree bytes
read/hashed/written, manifest bytes, identity/relationship/spatial work,
compactions and SQLite VM steps. Counters measure logical processing, not disk
traffic. Metadata bytes count dirty current JSON once rather than every copy.
Opt-in scoped timings report exclusive wall time in instrumented functions
(nested phases subtracted), including SQLite waits. SQLite PROFILE durations
are statement lifetimes, not exclusive CPU times. Kernel `perf` sampling was
unavailable (`perf_event_paranoid=4`); no system setting was changed.

## Recorded measurements (2026-10-20)

Linux aarch64, GCC 13, **RelWithDebInfo**, five isolated timed edits per case.
Each cell reports mean milliseconds and `(min / median / max)`. The
[raw indexed report](benchmarks/structural-edits-indexed-2026-10-20.json)
contains every sample, counter, initialization measurement and profile.
The [earlier flat-layout report](benchmarks/structural-edits-2026-10-20.json)
is retained for comparison; it described the incomplete first implementation.

| Operation | Size | Links | Full replacement, ms | Indexed structural, ms |
|---|---:|---:|---:|---:|
| append | 1000 | 0 | 4.41 (3.98 / 4.49 / 4.78) | 5.38 (4.99 / 5.29 / 5.81) |
| append | 10000 | 0 | 5.36 (5.11 / 5.21 / 5.70) | 6.54 (4.33 / 5.41 / 12.49) |
| append | 100000 | 0 | 11.38 (5.68 / 7.49 / 23.41) | 3.24 (2.92 / 3.10 / 3.89) |
| insert | 1000 | 0 | 4.34 (4.02 / 4.30 / 4.78) | 5.36 (4.69 / 5.48 / 5.81) |
| insert | 10000 | 0 | 4.97 (4.51 / 4.71 / 5.91) | 6.83 (4.78 / 5.52 / 12.98) |
| insert | 100000 | 0 | 16.52 (10.89 / 18.09 / 24.13) | 4.00 (3.70 / 3.82 / 4.67) |
| right_growth | 128 | 0 | 4.64 (4.06 / 4.82 / 4.98) | 5.82 (5.45 / 5.96 / 6.12) |
| right_growth | 512 | 0 | 12.88 (12.28 / 12.82 / 13.66) | 18.34 (15.69 / 16.86 / 24.66) |
| right_growth | 1024 | 0 | 49.35 (45.67 / 46.69 / 55.38) | 37.29 (33.98 / 35.16 / 45.52) |
| append | 1000 | 8 | 4.57 (3.98 / 4.67 / 5.01) | 4.91 (4.39 / 4.86 / 5.39) |
| append | 10000 | 8 | 5.24 (4.55 / 5.11 / 6.00) | 6.30 (4.31 / 4.41 / 13.10) |
| append | 100000 | 8 | 12.81 (8.53 / 10.00 / 24.12) | 3.64 (3.00 / 3.89 / 4.09) |
| insert | 1000 | 8 | 4.62 (4.31 / 4.55 / 4.97) | 5.18 (4.73 / 4.83 / 6.13) |
| insert | 10000 | 8 | 5.97 (5.44 / 6.07 / 6.33) | 6.70 (4.89 / 5.11 / 13.28) |
| insert | 100000 | 8 | 22.02 (14.90 / 19.21 / 29.00) | 4.00 (3.70 / 3.89 / 4.30) |
| right_growth | 128 | 8 | 7.45 (6.61 / 7.73 / 8.09) | 9.18 (8.89 / 9.02 / 9.58) |
| right_growth | 512 | 8 | 25.39 (25.14 / 25.23 / 26.09) | 21.74 (19.69 / 19.99 / 27.78) |
| right_growth | 1024 | 8 | 82.26 (76.26 / 81.66 / 87.79) | 40.97 (38.28 / 39.03 / 48.53) |

Small objects can be slower with indexed storage because tree management and
commit overhead outweigh avoided bulk work. The work counters, rather than
latency alone, establish bounded publication. Repeated synthetic surface values
allow considerable payload deduplication; these are not representative viewer
latency guarantees.

First-iteration indexed work (including tree metadata in `@tree`):

| Operation | Size | Links | ID lookups | Relationship candidates | Coarse spatial rows | Node/root reference operations | Tree bytes read/hashed | Manifest bytes read/written |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| append | 1000 | 0 | 2 | 0 | 1 | 14 | 37,071 / 37,071 | 4,396 / 1,102 |
| append | 10000 | 0 | 2 | 0 | 1 | 23 | 25,897 / 25,897 | 4,440 / 1,113 |
| append | 100000 | 0 | 2 | 0 | 1 | 38 | 47,052 / 47,052 | 4,492 / 1,126 |
| insert | 100000 | 8 | 2 | 0 | 1 | 59 | 69,156 / 69,156 | 4,492 / 1,124 |
| right_growth | 128 | 8 | 0 | 8 | 1 | 42 | 41,330 / 41,330 | 22,881 / 789 |
| right_growth | 512 | 8 | 0 | 8 | 1 | 290 | 326,570 / 326,570 | 23,031 / 795 |
| right_growth | 1024 | 8 | 0 | 8 | 1 | 706 | 889,341 / 889,341 | 23,287 / 803 |

ID lookups count requested identities; their bounded bucket/node processing is
included in tree bytes and hashing. Spatial rows count SQL publication; subtree
and halo updates are included in tree-node work. All tested append sizes perform
two identity lookups and publish one coarse spatial row. Unchanged stable-ID
links produce zero relationship candidates. In the earlier 10,000-point flat
implementation, a single append scanned 20,000 identity rows. The earlier
512-square/eight-link growth published 272 spatial rows and read about 2.15 MB
of manifests. The indexed path removes those whole-object passes.

Legacy conversion is a separate, explicit proportional cost. Independent
zero-link conversions recorded during this run:

| Conversion | Size | Samples | Mean ms | Min / median / max ms |
|---|---:|---:|---:|---:|
| sequence_initialize | 1000 | 5 | 8.28 | 7.10 / 7.71 / 11.52 |
| sequence_initialize | 10000 | 4 | 44.72 | 39.68 / 41.69 / 55.83 |
| sequence_initialize | 100000 | 4 | 434.49 | 420.25 / 427.67 / 462.37 |
| surface_initialize | 128 | 2 | 5.05 | 4.60 / 5.05 / 5.50 |
| surface_initialize | 512 | 2 | 35.47 | 30.91 / 35.47 / 40.02 |
| surface_initialize | 1024 | 2 | 192.50 | 187.29 / 192.50 / 197.72 |

The 80-append history runs start with 1,000 points and no relationships. Local
compaction is inside every timed transaction:

- Full replacement: 2.49 (1.20 / 1.79 / 9.53) ms; 0 local compactions.
- Indexed structural: 4.38 (2.40 / 3.54 / 13.10) ms; 158 local compactions.

The exclusive phase profile identifies these largest measured function costs
(mean ms per edit; SQLite commit outside these phases also contributes):

- append, 100000, 0 links: `treeNode` 0.676 ms; `identityPut` 0.243 ms; `ObjectStore::editGeometrySequence` 0.208 ms.
- right_growth, 1024, 8 links: `treeNode` 16.440 ms; `ObjectStore::resizeSurface` 12.540 ms; `storeSharedPayload` 2.877 ms.

Tree-node handling is now the main indexed-storage overhead; coordinate
processing no longer grows with the untouched sequence. Further constant-factor
tuning is possible without changing the completed structural contract.

Final validation: **319 Python tests and all 9 C++ suites passed**, including
20 Python structural cases and four C++ structural/instrumentation cases.
`git diff --check` passed. Build/test commands are listed above; no dependencies
were installed and no consumer integration or real-scroll validation is claimed.
