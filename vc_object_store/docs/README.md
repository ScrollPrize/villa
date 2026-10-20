# Object store

`ObjectStore` owns objects, immutable geometry, and relationship metadata in a
single SQLite file. Ingest native data, discover relationships from stored
geometry, then query, annotate, export, or synchronize using that same handle.

The query APIs provide bounded pages, graph views, and packed geometry arrays for resident
working data. Existing application consumers remain separate from the store.

The [shared geometry API](typed-geometry.md) provides owned point collections,
fibers, tracks and patches, typed snapshot views, bounded reads and checked
edits in C++ and Python. It preserves the generic array API and stored formats.

## Build and command line

Run these commands from `villa/vc_object_store`, after configuring with
`VC_OBJECT_STORE_BUILD_PYTHON=ON` as described in the [build guide](../README.md).

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
options. Original geometry files are needed only for ingestion. CT
volumes remain inputs to fiber/track-to-patch detection.

## Python

```python
from vc_object_store import ObjectStore

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

Link `vc_geometry` for owned geometry, `vc_objects` for storage and typed views,
`vc_object_geometry` for native ingestion/export,
and `vc_object_links` for relationship detection.

```cpp
#include "vc_object_store/objects/ObjectStore.hpp"
#include "vc_object_store/objects/GeometryIngest.hpp"
#include "vc_object_store/object_links/ObjectLinks.hpp"

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

## Object revisions and checked actions

An immutable object revision identifies its geometry payload, complete metadata,
and relationship membership. `current_object_revision`, `object_revision`, and
`object_history` return these identities and metadata. Use the returned
`geometry_revision` with the existing array/range/rectangle readers. Object and
geometry revision UUIDs are distinct; annotations can change without creating a
geometry revision.

For an edit prepared outside a writer transaction, supply the object revisions
you observed, including every participating relationship endpoint:

```python
observed = {uuid: store.current_object_revision(uuid)["uuid"]
            for uuid in (a_uuid, b_uuid)}
with store.write(expected_revisions=observed) as tx:
    tx.set_meta(a_uuid, "review_note", "checked")
    tx.set_meta(edge_uuid, "confidence", 0.9)
```

C++ uses `store.write(RevisionSelection{...})` and the existing transaction
mutators. Expectations are checked before entering the action. Commit checks
coverage of all modified objects and endpoints used in relationship validation.
Missing expectations, stale heads, and caught geometry revision conflicts roll
back the entire checked action, including its geometry payloads and journal
entries. A `None` expectation means the object must not yet exist; choose its UUID
before creating it in a checked action. The original `write()` remains available
for serialized ingestion and read/modify/write operations performed under its
writer lock. It does not protect edits prepared from earlier external reads.

Every object revision also stores `published_at_us`, a UTC Unix timestamp in
microseconds. All revisions from one action receive the same timestamp; the
writer guarantees strictly increasing publication times across actions, including
clock ties or backward clock adjustments. There is no separate commit-marker
table. Save a revision's publication time to recover the entire store at that
point:

```python
at = store.current_object_revision(object_uuid)["published_at_us"]
# Later, even after geometry, annotations and links have changed:
past = store.revision_read_view_at(at)
past_objects = past["objects"]          # metadata plus immutable geometry UUIDs
past_links = past["relationships"]      # exact recorded links, with applicability
past_selection = past["selection"]      # object UUID -> revision UUID
```

`revision_selection_at(at)` (C++ `revisionSelectionAt`) picks the latest published
revision at or before `at` for each object. Physical deletions leave history
events and are excluded after their publication time. These events do not change
the existing restrictions on deleting objects with retained geometry. Objects
created after `at` are also excluded. Read referenced geometry with the usual
array readers; all returned geometry and metadata correspond to the same
selection. A recorded link's pinned target may intentionally be older than that
selection; `applicable` distinguishes those historical references from usable
connections at the selected time. Restoring into the live store is a new checked
action rather than moving the historical heads.

One commit publishes at most one revision per changed object. Revision reads
inside an uncommitted writer still describe the last published state. Metadata
uses JSON field deltas; geometry uses the existing immutable chunks and sparse
deltas. Unchanged relationship versions and memberships are inherited, with
indexed materializations for current metadata and membership reads.

Relationships are shared records owned by both endpoints. Addition, modification,
removal, or explicit rebinding revises both endpoint objects in the same action
(once for a self-link). `recorded_relationships(object_revision_uuid)` reads the
exact immutable versions belonging to that object revision. Each result contains
the edge record, `a_revision`/`b_revision` object pins, and the geometry bindings
used for validation. Follow those object pins to resolve recorded references;
never substitute current heads.

`relationships_at({object_uuid: revision_uuid, ...})` evaluates a closed revision
selection. Here `None` selects the latest head in the query's read snapshot.
Results include `applicable`: both selected endpoints must contain the same
relationship version and its declared dependencies must match. Missing endpoints
or incompatible versions are reported as inapplicable. Explicit selections let
applications query the inputs actually applied to a fit.
`revision_read_view(selection)` returns `objects`, `relationships`, and the
resolved `selection` together from one snapshot, including when selecting latest
heads. C++ callers use these reads within one `ReadTxn`. The existing `edges_of`,
traversal, and graph-view APIs remain current-record queries; they do not replace
this revision-aware applicability check.

There are two distinct preservation decisions:

- `keep_recorded_relationship(edge_uuid)`, or a geometry edge decision with
  `action="keep_recorded"`, checks the resulting anchor identities but preserves
  the historical pins and shared relationship version. The edited object inherits
  the reference; other objects need no new revision. The relationship can be
  inapplicable to the new selected geometry.
- `retain_relationship(edge_uuid)` explicitly validates and binds the shared
  relationship to the resulting endpoint state. Geometry decisions named
  `retain` and `update` retain this existing rebinding behavior. Rebinding is a
  shared relationship change and revises both endpoints.

An anchor can declare `metadata_keys`, the top-level ordinary metadata keys that
affect its meaning, such as `["winding"]`. Changes to these require an explicit
keep, revalidate, update, or remove decision. Decisions are checked against final
metadata at commit. Other metadata edits leave geometric applicability unchanged.
Restoring old metadata or geometry is a new checked action through the same
mutators and requires the same relationship decisions. Domain-specific validity
remains the application's decision; the store validates declared dependencies
and anchor identities.

Writable opens migrate older schemas atomically, including the v9 history and
v10 lifecycle tables. Pre-v9 annotations were mutable,
so migration cannot recover their original historical values. Migration baselines
are marked `metadata_complete=False`, including detached baselines needed to
preserve stale geometry bindings. Those detached baselines have no publication
timestamp and are never selected as past live heads. New revisions record complete publication
state. Open an older store writable once before using these history APIs.
Publication times describe this store's timeline. The existing `synchronize`
operation merges current records and geometry history; it does not replicate a
peer's historical metadata/graph timeline. An incoming binding to older geometry
without matching local object history receives the same explicitly incomplete,
untimed baseline. Use a database copy when the complete original timeline must
be transported.

## Retirement, restoration, and retained references

Schema v10 adds library-managed `retired` and `lifecycle_epoch` fields to object
records. They are ordinary immutable revision content, backed by an indexed
current lifecycle table. They cannot be changed through `upsert_object` or
ordinary `meta` fields. Writable opens migrate older stores atomically without
rewriting geometry or old history. Read-only access to older schemas retains its
existing behavior; retirement requires a writable migrated store.

Use a checked action for lifecycle changes, including every modified object and
all endpoints used for relationship decisions:

```python
observed = {u: store.current_object_revision(u)["uuid"] for u in (source, target)}
with store.write(expected_revisions=observed) as tx:
    tx.retire_object(source)
    tx.keep_recorded_relationship(edge_uuid)

observed = {u: store.current_object_revision(u)["uuid"] for u in (source, target)}
with store.write(expected_revisions=observed) as tx:
    tx.restore_object(source)
    tx.keep_recorded_relationship(edge_uuid)  # preserve the old pins, still inactive
    # Instead, retain_relationship(edge_uuid) explicitly validates a current link.
```

C++ provides `retireObject(WriteTxn&, uuid)` and
`restoreObject(WriteTxn&, uuid, optionalSourceRevision)`. Retirement preserves
identity, geometry, metadata, and recorded links. Restoration publishes new
history with the metadata and geometry of the last active revision, or of the
explicit active `source_revision` belonging to that object. Creation stamps stay
unchanged; content generation advances and approval resets normally. Changes
made while retired remain in history; default restoration returns to the last
active content. Already-retired retirement and already-active default restoration
are no-ops. Direct UUID lookup, geometry reads, and historical reads remain
available. Restoring a physically removed live row is not supported.

Each transition requires explicit decisions for all remaining incident edges,
including already inactive ones. Call `keep_recorded_relationship`,
`retain_relationship`, or `update_edge` after endpoint edits, or `remove_edge`
within the action. Commit checks decisions against the final state. Removed
edges stay removed; selecting historical object content does not resurrect a
shared edge or overwrite another endpoint's history. Recreate desired links
explicitly with `add_edge`, or validate surviving links. Stale expectations,
missing participants, and missing decisions roll back the whole action, including
new objects and geometry. Source retirement can therefore accompany a merge or
split assembled with existing geometry mutators; this adds no structural-edit API.

A lifecycle epoch changes on retirement/restoration. Preserved relationships keep
their pinned epochs, so restoration does not silently reactivate them even when
geometry is identical. An explicit revalidation publishes a shared relationship
revision and a journal-visible change. Historical selections before retirement
continue to resolve and evaluate their original relationships. Retired selected
endpoints are always inapplicable to current use. Lifecycle-only changes reuse
geometry references without decoding, copying, or hashing payloads. Restoring
different historical geometry reuses immutable storage but rebuilds the current
spatial index, which can read the selected geometry.

Current catalogs, counts, kind/tag selections, spatial regions (SQL and resident
cache), neighbours, connected selections, traversal, edge queries, graph views,
and metadata exports exclude retired objects by default. Traversal cannot pass
through a retired node. Edges with retired endpoints or incompatible lifecycle
epochs are excluded too. Use `include_retired=True` for explicit inspection;
on edge/traversal/graph queries this also exposes dormant recorded links, **not**
a claim of applicability. C++ selections have `includeRetired`; convenience
queries take a trailing boolean. `object(uuid)`, `edge(uuid)`, and C++ direct UUID
lookups always return existing records regardless of lifecycle. The current
record APIs still do not replace `relationships_at` for all geometry and metadata
dependency checks.

`revision_selection_at` and `revision_read_view_at` exclude objects retired at
the requested time; `include_retired=True` includes their selected retired
revisions. Explicit `revision_read_view(selection)` includes exactly the selected
objects. Recorded references always resolve their pinned revisions regardless
of today's visibility.

Represent a reference stored conceptually in an application metadata field by
registering an ordinary edge with kind `declared_reference`, endpoints identifying
the referencing and referenced objects, and `meta` describing the application
field and role. Use `whole_geometry=True` on each anchor for an object-wide
reference, or normal point/channel dependencies for a narrower reference. These
are shared edges with normal revision pins and explicit lifecycle decisions;
endpoint order is canonicalized, so record any directional role explicitly in
edge metadata. Graph-like objects register each referenced object as an endpoint
of one of these edges. No separate untracked JSON-reference format is promised.
Arbitrary UUID strings in metadata are **not** declared references and do not
block deletion or create graph membership. Current and retained historical
`declared_reference` edges remain readable through the ordinary history APIs.

`delete_object` still only removes an eligible live row while preserving history;
managed geometry and current edge foreign-key restrictions remain unchanged.
It is not a purge and is not made more permissive here. There is no history or
payload purge API: lack of current edges does not authorize removing the object's
own revisions, historical references, undo content, or shared payloads. A future
retention operation must account for all retained relationship versions,
including declared-reference edges, before any physical purge.

Synchronization carries current lifecycle state and lifecycle relationship
bindings, preserving dormant links on existing and fresh replicas. It still does
not transport the complete peer timeline. Missing historical lifecycle pins get
explicitly incomplete, untimed baselines, as older geometry pins do. Default
restoration uses only published active revisions, never detached reference
baselines. A fresh replica of a retired object can lack its last active revision;
restore on the originating replica and synchronize, or copy its full database.
Metadata-only JSON import rejects lifecycle-bearing exports: use synchronization for state
transfer or a database copy for a complete timeline.

## Volume identity and coordinate scale

Patches, fibers, tracks, and point collections share one object metadata field:

```json
{"coordinates": {"volume_uuid": "abcdefab-1111-4111-8111-111111111111", "scale_factor": 4}}
```

This is `ObjectRecord::meta` in C++ and `store.object(uuid)["meta"]` in Python.
Each object owns its values independently. `volume_uuid` is an explicit volume
identifier, represented as 32 hexadecimal digits with hyphens in the
`8-4-4-4-12` layout; uppercase input is normalized to lowercase. It does not
require a corresponding object in this database. Paths, coordinate-space labels,
and object UUIDs never supply an inferred volume identity.

`scale_factor` is a finite positive JSON number. Its direction is
`stored_xyz = volume_voxel_xyz * scale_factor`: it describes the geometry's
annotation frame relative to the identified source volume's voxel frame.
For example, factor 4 describes coordinates on an annotation grid four times
finer than that volume grid. It is distinct from physical voxel resolution and
from a patch's two-dimensional grid `scale`. **It is metadata only: ingestion,
metadata edits, and backfill never rescale XYZ or change geometry payloads.**
Consumers decide whether and how to use it; detectors do not automatically
change their scale parameters.

Both fields default to JSON `null` on new geometry objects. Missing fields in
older records likewise mean unknown, including a missing `coordinates` object;
unknown scale does not mean 1. Setting `coordinates` replaces the whole field,
normalizing omitted members to `null`. Unknown keys and malformed UUIDs/scales
are rejected. Positive 64-bit integers retain their exact JSON integer values.

Native import accepts `coordinates` on a source object and, for point
collections, on the document or individual collection. Flat `volume_uuid` and
`scale_factor` declarations are also accepted as input. Python's
`ingest_geometry_source(..., coordinate_context={...})` and the final
`coordinateContext` JSON argument of C++ `ingestGeometrySource` supply the same
two fields for any geometry kind. Packed tracks have no native declarations;
use context or ordinary metadata writes. Known declarations must agree;
missing/null declarations may be filled by another source. Collection documents
publish atomically; packed tracks retain their existing per-track publication.

Legacy `vc_open_data_source_coordinate_scale_factor` is accepted only as an
input mapping to `scale_factor`. All five legacy `vc_open_data_*` coordinate
fields are removed from newly imported source metadata; no legacy coordinate
API is retained. A level, resolution, path, or label alone never generates a
scale or volume UUID. Discarded legacy fields are not interpreted or validated.

```python
uid = store.upsert_object({
    "kind": "point_collection",
    "meta": {"coordinates": {"volume_uuid": volume_uuid, "scale_factor": 4}},
})
head = store.current_object_revision(uid)["uuid"]
with store.write(expected_revisions={uid: head}) as tx:
    tx.set_meta(uid, "coordinates", {"volume_uuid": volume_uuid, "scale_factor": 2})
assert store.object_revision(head)["record"]["meta"]["coordinates"]["scale_factor"] == 4
```

Coordinate edits use ordinary object revisions and reuse the same geometry
revision. Relationships that depend on the frame should declare
`metadata_keys=["coordinates"]` on the affected endpoint anchors. A coordinate
edit then requires the usual explicit keep-recorded, retain, update, or remove
decision, with expected revisions for participating endpoints. This is a
dependency declaration; the store does not establish physical registration or
validate a domain-specific geometric connection.

Ordinary re-ingestion still seeds source metadata once and preserves database
annotations. Use explicit checked backfill to repair older objects:

```python
from vc_object_store import read_geometry_source_coordinates

# Pin before reading source metadata; no geometry arrays are decoded.
head = store.current_object_revision(uid)["uuid"]
declarations = read_geometry_source_coordinates("point_collection", source_path, uid)
with store.write(expected_revisions={uid: head}) as tx:
    tx.backfill_geometry_coordinates(uid, declarations)
```

`read_geometry_source_coordinates` also supports a fiber JSON file or patch
directory and accepts `coordinate_context`. It checks the source object UUID.
Alternatively pass the two declarations directly, or omit them to use any
legacy scale already in `meta.source`. Backfill fills unknown fields, refuses to
overwrite known conflicting values, preserves other annotations, and removes
legacy coordinate fields from current source metadata. A conflict, even if
caught inside the writer, aborts the checked action. Multiple backfills and
relationship decisions can share one transaction. Without available information,
values stay unknown. Historical revisions remain unchanged, including any legacy
metadata they already contained.

The C++ equivalents are `readGeometrySourceCoordinates(kind, path, uuid,
coordinateContext)` from `<vc_object_store/objects/GeometryIngest.hpp>` and
`store.backfillGeometryCoordinates(writeTxn, uuid, declarationsJson)` from
`ObjectStore.hpp`. Use `store.write(RevisionSelection{{uuid, expectedRevision}})`;
unchecked writers are rejected for backfill. Ordinary `setMeta` and
`upsertObject` use the same coordinate validator. No schema migration, shared
collection-set object, or native-export change is involved.

## Surface channels and overlap sidecars

Every extra named array on a patch is a grid-aligned channel. `points` remains
reserved for float32 XYZ. Channels have a name, native dtype, full C-order shape,
and component count: `[rows, columns]` is scalar; `[rows, columns, components]`
has a positive component count. Both spatial dimensions must match `points`.
`GeometryRevision::arrays` exposes the descriptors without reading payloads;
Python's `geometry_revision` / `current_geometry` descriptors also include
`components`. Supported storage dtypes are `|u1`, `<u2`, `<i4`, `<i8`, `<u8`,
`<f4`, and `<f8`. Absence is distinct from a present, zero-filled channel.

Patch ingestion recognizes every regular lowercase `*.tif` or `*.tiff` file
beside `meta.json`, except the XYZ planes. The filename stem is the channel name
(e.g. `approval`, `generations`, `winding`, `mask`, or an additional custom name).
Duplicate stems and the reserved `points` stem fail ingestion. All recognized
channels must be aligned; unsupported dtypes/layouts, unreadable/truncated data,
multipage/volume TIFFs and non-top-left orientation fail explicitly, before
publication. Both stripped/tiled and contiguous/separate sample planes are
supported. Scalar gray and multicomponent RGB sample values are preserved in
**TIFF sample order**; no RGB/BGR conversion, scaling, resampling, zero/NaN
normalization, or alpha premultiplication occurs. The native sample reader is
shared with the existing XYZ loader, whose invalid-point/mask behavior remains
unchanged. An oversized legacy mask is reported as unaligned instead of being
silently resized for channel storage.

The complete `overlapping.json` value is stored at
`object["meta"]["source"]["overlapping.json"]`, in ordinary object history.
A missing key means unknown overlaps; a present empty list (or
`{"overlapping": []}`) means known empty. The sidecar may be a list of source
identity strings or an object with an `overlapping` list. Order, duplicates,
additional object fields and unresolved identity strings are retained. These
are directional source claims: ingestion creates no overlap graph edges,
resolves no target revision, and infers neither symmetry nor completeness.
This avoids inventing identity mappings or dropping not-yet-imported targets.
As with other source annotations, ingestion seeds this metadata once; file
refresh/reconciliation is the separate audit item 6.

Use `update_surface_channels` / `ObjectStore::updateSurfaceChannels` to add or
replace supplied channels and remove named channels in one publication. Supply
only the changed channels, with little-endian bytes. An empty `remove` list
removes nothing; removing an absent channel, duplicate names, or mismatched
layouts raises an error. Replacement may change dtype and components but not
the spatial dimensions. Growth/cropping and coordinated explicit fills belong
to audit item 5; these APIs deliberately provide no separate resizing mechanism.

```python
import numpy as np

current = store.current_geometry(surface_uuid)
points = next(a for a in current["arrays"] if a["name"] == "points")
rows, columns = points["shape"][:2]
generations = np.zeros((rows, columns), dtype="<u2")
head = store.current_object_revision(surface_uuid)["uuid"]
with store.write(expected_revisions={surface_uuid: head}) as tx:
    revision = tx.update_surface_channels(
        surface_uuid,
        channels=[dict(name="generations", dtype="<u2", shape=[rows, columns],
                       data=generations.tobytes())],
        remove=[], expected_revision=current["uuid"])
    tx.set_meta(surface_uuid, "review_pass", 1)

# Read or edit one grid cell without loading the full channel or XYZ.
raw = store.read_geometry_rectangle(revision, "generations", 0, 0, 1, 1)
value = np.frombuffer(raw, dtype="<u2")[0]
revision = store.edit_geometry_rectangle(
    surface_uuid, "generations", 0, 0, 1, 1, b"\x01\x00", revision)
revision = store.update_surface_channels(surface_uuid, [], ["generations"], revision)
```

The C++ equivalents accept `GeometryArray` values and return geometry revision
UUIDs; `editGeometryRectangle` takes row, column, height, width and a byte vector.
Existing array/range/packed readers and `editGeometry` scalar edits also work
with channels. Rectangle reads/edits preserve the existing dtype and all
components; bounds and exact byte lengths are checked. New patch arrays share
the configured 32/64/128 tile size (default 128); tile size 0 retains linear
storage, whose small reads may require a full 1 MiB chunk. Legacy linear arrays
remain readable/editable. Historical revisions retain channel absence,
descriptors and exact values, including after reopen and geometry sync.

Lifecycle publication writes only supplied channel payloads. Unchanged arrays
carry immutable `source_revision` references, including references to earlier
sparse-delta revisions, instead of copying or hashing their payloads. References
are flattened when possible. Sparse edits keep the existing byte-delta format
and touched-tile overlay. Manifest/reference bookkeeping still scales with the
array/chunk descriptors; historical inherited/delta reads can traverse history.
This is storage reuse, not a claim of constant-time publication or measured
interactive latency. Native export still rejects auxiliary arrays and remains
outside this channel contract. Owned typed `Patch` values also still contain XYZ
only; loading a complete channel-bearing patch into that owned type is rejected,
while bounded patch views and the named-array APIs remain available. Older binaries do not understand inherited
array descriptors; use an updated reader for databases written with this API.

Channel operations participate in the same checked object transaction as
metadata and relationships, with expected geometry revisions as an additional
check. Point anchors depend on their declared XYZ points; changing an unrelated
channel does not invalidate them. An anchor can additionally declare
`channels=["approval", "winding"]` (C++ `Anchor::channels`); changes to presence,
layout or values of those channels require an explicit edge decision. A whole
geometry/unknown anchor remains conservative and depends on all arrays.
Declared channel absence is also meaningful; retaining a relationship after
removal is an explicit caller decision. Domain-specific validity remains the
caller’s responsibility. Supply `edge_edits` and expected object revisions for
all validation endpoints when such decisions are needed, as in the checked-action
examples above.

Channel regression tests are `tests/python/test_surface_channels.py` and
`tests/cpp/test_surface_channels.cpp`. TIFF fixture tests additionally use
`tifffile` in the existing Python test environment; the production package has
no new Python dependency. The C++ test's internal `GeometryWorkScope` counts
per-array payload bytes read and hashed, independently of chunk deduplication.
Run the normal build and tests from the package README; the focused commands are:

```sh
ctest --test-dir build --output-on-failure -R '^test_surface_channels$'
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 VC_OBJECT_STORE_BUILD_PYTHON_PATH=build/python \
  /path/to/python -m pytest -q tests/python/test_surface_channels.py
```

## Sparse geometry edits

`edit_geometry` / `ObjectStore::editGeometry` replaces scalar ranges in existing
geometry arrays. It is independent of application tools and works with every
supported array dtype and geometry kind, including auxiliary arrays. Offsets
count **scalar elements in the flattened C-order array**, not bytes or points:
XYZ point 5 begins at element 15. Supply exact little-endian bytes in the existing
dtype. An edit cannot change array shape, dtype, names, or geometry attributes;
use `update_surface_channels` for channel lifecycle operations and `put_geometry`
for ingestion and explicit full/structural replacements.

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
ancestor. Current sparse-head reads use their overlay; inherited arrays may still follow
earlier revision references. Older historical reads can become
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
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 VC_OBJECT_STORE_BUILD_PYTHON_PATH=build/python \
  .venv/bin/python -m pytest tests/python/test_object*.py -q
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

Historical validation for the benchmark addition, before extraction into this
package (the QuadSurface checks belong to volume-cartographer):

```sh
AGENTS_AGENT_MODE=1 ninja -C build bench_object_edits test_object_store \
  test_quadsurface_save_roundtrip test_quadsurface_snapshot
AGENTS_AGENT_MODE=1 ctest --test-dir build --output-on-failure \
  -R '^(test_object_store|test_quadsurface_save_roundtrip|test_quadsurface_snapshot)$'
AGENTS_AGENT_MODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 VC_OBJECT_STORE_BUILD_PYTHON_PATH=build/python \
  .venv/bin/python -m pytest tests/python/test_object_geometry_edits.py -q
```

All three native test executables and all 20 Python edit tests passed. The 28
recorded benchmark runs also passed their exact geometry and file verification.

Structural sequence edits and coordinated surface growth/cropping are described
in [Structural geometry editing](structural-editing.md), including checked
transactions, identity/span semantics, storage reuse, and current cost limits.
