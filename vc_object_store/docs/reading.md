# Reading

Reads take one of three shapes: record lookups (current or at a numbered
version), geometry reads through a snapshot (bounded, typed),
and spatial queries over the persisted index. Everything here works on a
read-only open. Retired objects are omitted from listings, traversal, and
spatial results unless `include_retired=True`; direct UUID and version lookups
always find them.

## Opening a store

```python
from vc_object_store import ObjectStore

store = ObjectStore.create("objects.vcobj", author="reviewer")   # new store
store = ObjectStore.open("objects.vcobj", author="reader", read_only=True)
```

| Form | Notes |
| --- | --- |
| Python | `ObjectStore.create(path, author=..., catalog=None)`, `ObjectStore.open(path, author=None, read_only=False, catalog=None)`; `catalog` is a `vc_reader.Catalog` (or its dict form) supplying registrations the store has not recorded — see [transforms](transforms.md#registrations-and-route-choices) |
| C++ | `ObjectStore::create` / `::open` with `OpenOptions` (`author`, `readOnly`, `catalogTransforms`, `serviceOwner`) |
| Hosted | Python `HostedStore(endpoint)`; C++ `vc::objects::HostedStore` or a session attachment — see [hosted](hosted.md) |

A writable open takes a host-local ownership lock on `<path>-owner`, shared
between standalone writers and exclusive for the server's service owner, so a
store served by `vc_object_server` rejects other writable opens (read-only
ones still work). A store with an unsupported schema is rejected untouched;
convert it with `vc_object_convert` (see [Converting old stores](#converting-old-stores)).

Closing a store with live snapshots is an error in Python. Use the store as a
context manager, or call `store.close()` after closing snapshots and graphs.

## Listing and selecting records

```python
store.object(uuid)                       # current record, or None
store.objects_by_kind("surface")         # live records of a kind
store.objects_by_tag("initial_import")
store.object_count("fiber", include_retired=True)
store.object_tags()                      # distinct tags of live objects
rows = store.select_objects(object_kind="track", fields=["name", "track"])
```

`select_objects` is the general query: kind, tag, bounding box, length
filters, coordinate declaration, volume target, connection, limits and
cursors, with optional `fields` projection. Each row carries `uuid`, `kind`,
and the `version`, `version_generation` and `version_store` it was read at, so a listing pins
exactly what it returned (read snapshots only). `select_edges` does the same
for relationships. On `HostedStore` these are paged; `select_all_objects` /
`select_all_edges` return the complete result exact as of one commit and can
raise `StoreBusy` under sustained concurrent writes. In C++ the same queries
take an `ObjectSelection` / `EdgeSelection` in a read transaction.

Use `ObjectRecord::reference()` or `ObjectRevision::reference()` in C++, and `vc_object_store.object_reference(record)` in Python, to preserve the complete `(store, generation, object, version)` identity. The attached store is a lookup scope, not necessarily the owner of a returned record. After an overlay edit, the new version belongs to the committing store; use the transaction receipt's store/generation for changed versions. Unchanged inputs retain their original references.

For repinning after a write, use `TransactionReceipt::objectReference(uuid)` or Python `receipt_object_reference(receipt, uuid)`. Receipts retain complete identities for every live expected/changed object, including unchanged inherited inputs. Do not combine the receipt's store/generation with its numeric `objects` entries: those entries also include unchanged referenced objects. Saved receipts retain their original references on replay. Older receipts without `object_references` remain readable but cannot safely repin an editor; the helper requires reloading rather than guessing ownership. Durable drafts preserve their data and record that reload requirement.

`object_history(uuid)` follows the winning record even when it is a deletion marker in a referenced store. It does not fall through that deletion to older live records. Ingestion also reserves deleted identities across the same scope: re-import is not a restore operation.

Consumers that use history only when it is retained can request `object_history(uuid, if_retained=True)` (C++ `objectHistory(uuid, true)` on `StoreReader`). This returns no entries when that record's owner has history disabled; explicit history reads still reject unavailable history. Do not infer a referenced record's retention policy from the attached root's `history_enabled` setting.

Volume-target listings (`volume=...`, the volume's `id`) match every frame
declared in that volume, and their bbox, length filters, and length ordering
use **volume voxels**; `include_unbound=True` appends the rest in UUID order.
`volume` and an exact `coordinates` declaration are mutually exclusive. See
[transforms](transforms.md#volume-target-queries).

Length filtering and ordering measure the selected stored polyline, never an ingestion summary. This applies to native and volume-target selections, spatial regions and viewport regions; registering an unrelated transform does not change their semantics. Equal measured lengths sort by UUID in every frame. Length-ordered rows expose their measured cursor value as `volume_length` (C++ `volumeLength`, the existing field name), in native units without a volume target and target voxels otherwise; persisted fiber/track detail metadata is not rewritten by a query. Records without stored geometry have no measurable length and are omitted from length-filtered/ordered selections, but remain in ordinary unfiltered listings. An unbound tail explicitly requested for a volume query remains UUID-ordered without a volume length.

## Geometry: snapshots and bounded reads

A geometry snapshot pins one consistent store state. Views read only the
ranges or rectangles asked for; full materialization is explicit (`load()`).

```python
with store.geometry_snapshot() as snapshot:
    surface = snapshot.surface(uuid)
    tile = surface.read_tile(row=30, column=31, rows=4, columns=4)
    collection = snapshot.point_collection(pcl_uuid)
    xyz = collection.point(42)                    # by stable point ID
    ids = collection.ids(0, len(collection))
    fiber = snapshot.fiber(fiber_uuid)
    controls = fiber.control_points(start=0, count=2)
    owned = snapshot.track(track_uuid).load()     # -> vc_object_store.Track
```

- Factories: `snapshot.surface/fiber/track/point_collection/umbilicus(uuid)`,
  or `snapshot.geometry(uuid)` for the matching view. A kind mismatch or an
  object without stored geometry raises.
- Views expose counts and dimensions from the pinned geometry revision (not
  import summaries), plus `view.version` — the numbered object version read,
  for [checked edits](editing.md).
- A tile holds `points` and validity; other surface channels are read with
  `read_geometry_rectangle`. Point lookup loads the ID array once per view,
  not the coordinates.
- Views keep their snapshot alive; closing the snapshot invalidates them.
  Snapshots are thread-confined and hold one reader slot — close them
  promptly. `load()` results stay valid after closure.
- Explicit geometry revisions can be read even if their object is no longer visible. The revision must still belong to the requested object. For exact surface metadata and geometry together, use `readSurfaceGrid(reader, ObjectVersion, volume)` or Python `surfaces.read_surface(location, uuid, volume=..., reference=ref)`. This works through overlays and after current deletion as long as the selected source version remains retained. The older `version`/`generation` selectors remain available and select the current owner's history; they cannot identify a different source owner.
- C++ consumers that already selected an `ObjectRevision` must pass that state to `viewTransformedAt(selected, volume)` and `readSurfaceGrid(reader, selected, transformedView)`, not look up its current record again. For a geometry-only selector, `StoreReader::objectAtGeometry` resolves the latest retained metadata naming that geometry through the shared reference scope. This is not an exact metadata-version selector: several object versions can share one geometry. Geometry transform listings use the same retained metadata, including through deleted overlays and history-off dependencies.
- Region `VisibleRun` results carry `objectReference` and `objectToVolume` (Python/wire: `object_reference`, `object_to_volume`), identifying the selected object and the affine used to draw its points. Carry those into editing and picking; do not pair the displayed geometry with a fresh current lookup. Hosted transform registrations and route choices are fetched together in one server read transaction.
- C++ `ResidentSurface` queries an immutable, already-positioned `geometry::Surface` through the same tile index, validity and intersection kernel as stored/session-draft surfaces. Use this for a displayed historical or edited grid; never substitute current geometry merely to obtain an intersection. It does not install a draft or mutate a store.
- `StoreReader::graphReadViewAround(selectedReferences, edgeSelection)` expands recorded endpoint evidence for off-screen neighbors without reading moving heads. Explicitly selected references always win. For unselected neighbors, the newest referenced version within one owner/lineage wins; competing owners/lineages or unavailable endpoint references fail explicitly. Final edges still pass the ordinary selected-graph applicability checks. This is a recorded-evidence closure, not a reconstructed global historical snapshot.
- Pinned overlap materialization takes `a_reference` and `b_reference` alongside the overlap UUID, pose metadata and whole-geometry flag. Each reference is `{store, generation, object, version}`; metadata, geometry and coordinate transforms are resolved together. Its cache distinguishes complete selected states, including metadata-only coordinate edits sharing the same geometry.
- Native export accepts the same complete reference (`export_geometry(..., reference=ref)`; CLI `--store UUID --generation UUID --version N`). Historical restore accepts `source_reference=ref`. Number-only convenience selectors still select the current owner's history; use full references for retained inherited states after an override or deletion. Native collection/fiber export uses `StoreReader::versionsAt(anchor)` for the selected owner's snapshot, not the overlay's current topology.
- Any view can be read in another volume's frame with
  `view.view_transformed_at(target_volume)` — see
  [transforms](transforms.md#transformed-views).

C++ mirrors this with `store.geometrySnapshot()`
([store/GeometryViews.hpp](../include/vc_object_store/store/GeometryViews.hpp)).

### Raw array reads

For byte-level access by geometry revision UUID (`store.current_geometry(uuid)`
returns the current descriptor including its `uuid`):

```python
revision = store.current_geometry(uuid)["uuid"]
raw   = store.read_geometry_array(revision, "points")
piece = store.read_geometry_ranges(revision, "points", [[24, 24]])
rect  = store.read_geometry_rectangle(revision, "points", 0, 0, 64, 64)
packed = store.read_geometry_arrays([rev_a, rev_b], "points")  # .values/.offsets
notes = store.geometry_point_annotations(revision, "points")
```

Packed results expose dtype, shape, data, and row offsets; trailing
dimensions and dtypes must match, and request order/duplicates are preserved.
Storage dtypes are `|u1`, `<u2`, `<i4`, `<i8`, `<u8`, `<f4`, `<f8`. Reads are
checksum-verified: damaged payloads raise instead of returning bytes.
`geometry_point_annotations` returns per-point annotations aligned with that
revision's sequence (by stable ID, not array order).

## History reads

```python
store.current_version(uuid)              # numbered head version
store.current_object_revision(uuid)      # record + selected geometry revision
store.object_history(uuid)               # every published object revision
store.geometry_history(uuid)             # every geometry revision
store.geometry_revision(revision_uuid)   # one geometry descriptor
refs = store.versions_at(when_us)        # [{store, generation, object, version}]
records = store.objects_at_versions(refs)
```

Object revisions capture metadata, geometry identity, and relationship
membership; a metadata-only edit creates no geometry revision.
`versions_at` returns the numbered version of every object live at
`when_us` (objects created later, or physically deleted by then, excluded;
retired objects omitted unless `include_retired=True`). Restoring historical
content publishes a **new** checked action — see
[retire and restore](editing.md#retire-restore-delete). Relationships at a
selection are read with `relationships_at_versions(refs)`, and the whole graph
with `graph_read` — see [graph edges](graph-edges.md#traversal-and-graph-reads).

## Spatial queries

Both queries use the persisted R-tree/tile index; neither quantizes bounds.

`nearest_surfaces(point, tolerance, uuids=None, coordinates=None, volume=None)`
(C++ `ObjectStore::nearestSurfaces`) returns the nearest point of each surface
within `tolerance`, ordered by distance: object and geometry revision,
fractional grid `[column, row]`, surface point, distance. It reads only tiles
near the point and tests their exact valid triangles — invalid vertices are
holes. `uuids=[]` matches nothing; `volume=` poses the query in that volume's
voxels.

`geometry_region(slice, options)` browses an oriented slab. It returns
original unclipped XYZ and zero-based segment indices in contiguous runs for
one polyline kind (`fiber`, `track`, or `surface` — a surface contributes the
cut of its exact triangle mesh by the slab's mid-plane):

```python
from vc_object_store import RegionSlice, RegionOptions

view = RegionSlice()
view.origin = [100, 200, 300]
view.u, view.v, view.normal = [1, 0, 0], [0, 1, 0], [0, 0, 1]
view.local_bounds = [-50, -50, -0.5, 50, 50, 0.5]   # u0, v0, n0, u1, v1, n1
options = RegionOptions()
options.kind = "fiber"
options.maximum_objects = 100          # 0 = unlimited
result = store.geometry_region(view, options)
```

Axes must be finite and orthonormal (tolerance `1e-12`); zero slab thickness is
allowed. A segment counts when it crosses the slab, even with both endpoints
outside. Fibers use `line_points` when nonempty, else controls; tracks widen
exactly to float64. `options.uuids` restricts candidates (`None` unrestricted,
empty = nothing), `selected` is attempted first under a limit, and only
intersecting objects consume the limit. Family and minimum-length filters use
import annotations, not recomputed geometry; surfaces reject them. With
`options.volume`, run points come back in that volume's voxels.

Region work is budgeted; a stopped result is valid but incomplete. Only
`stop == "exhausted"` means `complete`:

| Budget | Default | Charged work |
| --- | ---: | --- |
| `output_byte_budget` | 8 MiB | 24 bytes per appended XYZ |
| `processing_byte_budget` | 64 MiB | Payload, verification, metadata, reconstruction |
| `candidate_work_budget` | 1,000,000 | Candidate visits and segment tests |
| `sql_step_budget` | 10,000,000 | SQLite VM instructions |
| `max_unit_bytes` | 1 MiB | Largest single processing unit |

`RegionCancellation.cancel()` may run from another Python thread (the query
releases the GIL). C++ callbacks inspect stats, must not re-enter the store,
and propagate exceptions. Budgets bound work, not wall-clock latency or
allocator memory; there is no continuation cache — rerun with a bigger budget.

`store.spatial_regions(...)` returns the stored bounds themselves (per object
or per tile), without geometry reconstruction.

## Selection records

A selection record freezes the exact result of an object selection at one
commit, so a long job or training set can read its inputs back later at the
same versions while the store moves on. It holds the query, store, generation,
commit sequence, every match in listing order at its numbered version, and —
for volume/coordinate selections — the versions of the recorded transforms in
force. `record_selection(query)` captures one (hosted stores too);
`encode_selection_record` / `decode_selection_record` give the binary form
(a JSON header, then 24 bytes per object), meant to be kept as an
[artifact](writing.md#artifacts). Read the objects back with
`objects_at_versions`. Explicit transforms and limits are refused.

Selection encoding version 2 retains each object's owning store and publication lineage through a compact origin dictionary; entries remain 24 bytes. Transforms carry the same complete identity. Existing version-1 single-store records remain readable. This is an artifact format update, not a database schema change. Number-only version selectors use the current record's owner and lineage by default; a full `ObjectVersion` is required to select a state from another owner or branch explicitly.

Current listings include `version_store` alongside `version` and `version_generation`, including projected and hosted listings. This is read-result metadata derived from the owning database, not a stored per-record column. Hosted clients require this identity when creating exact references; they do not substitute the server attachment's identity when it is missing. Per-object history follows the visible owner's retained history, and historical exports use that owner's timeline (including its shards), not the overlay's publication timestamps. This does not define a combined historical timeline for independently published linked stores.

Editable attached collections use `AttachedCollectionsSnapshot`: retain the returned commit sequence, owner reference, collection references and loaded coordinate mappings alongside editable UI data. `replaceAttachedCollections` requires this loaded baseline, rejects concurrent edits/removals/additions or changed mappings, and repins it from the successful receipt. Do not reconstruct a replacement baseline from current heads when saving JSON or other detached editor data. Membership uses the existing store-commit guard, conservatively rejecting unrelated commits too; an inherited owner's identity need not change when a local attachment is added. No separate membership directory is maintained.

## Selected export and graphs

Native geometry export accepts a `StoreReader`, so local and hosted sessions use the same selected metadata, geometry and topology path. Explicit exact references do not require a live current object. Topology-bearing exports select the owning publication snapshot (or the immutable current-only owner's state); they do not invent a cross-store historical timeline. Local destination validation protects every database/container and artifact sidecar in the dependency closure, including aliases and ancestor paths. Hosted exports write in the client's filesystem namespace.

`graphReadViewAt` constructs a graph from complete selected references through the same relationship reader used by local and hosted sessions. History policy is checked on the selected owner, never on the attachment. A current-only owner cannot retain a replaced local endpoint; keeping that relationship must fail or be explicitly revalidated, rather than pretending an archive exists.

## Converting old stores

The runtime opens schema 21 only. Upstream schemas 16–20 require conversion; unpublished experimental layouts are not supported. Stop every client, then convert into a new
file (the source is never changed):

```sh
vc_object_convert old.vcobj new.vcobj   # writes new.vcobj.mapping.json and new.vcobj.report.json
```

The mapping gives every old object/relationship revision UUID its numbered
reference, for converting pins in exports and job configs. The report lists
gaps rather than guessing (untimed baselines, geometry revisions selecting no
object version, metadata still holding old revision UUIDs). Converted commits
have author `unrecorded`. Referenced artifact blobs are copied and verified;
a missing or corrupt blob fails conversion and removes the partial
destination.
