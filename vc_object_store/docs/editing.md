# Editing

Edits publish new immutable revisions; nothing is modified in place. Every
geometry mutation takes the object's **expected numbered version** (the one
you read, e.g. `view.version`) and returns the new geometry revision UUID.
A stale version conflicts and writes nothing — including for no-op edits.
A net no-op creates no revision. And every edit must decide what happens to
the relationships it affects: preview them with
`store.affected_geometry_edit_edges(uuid, edits, version)`, then pass explicit
`edge_edits` (`retain` / `update` / `remove` / `keep_recorded` — see
[graph edges](graph-edges.md#decisions-when-geometry-changes)) with **all**
endpoints in the transaction's `expected_versions`. Geometry, bounds,
approvals, and relationship decisions publish atomically in one commit.

## Sparse coordinate edits

Shape-, dtype-, and name-preserving patches into an array:

```python
import struct

version = store.current_version(uid)
edits = [{"array": "points", "element_offset": 15,
          "data": struct.pack("<3f", 12, 24, 36)}]
assert not store.affected_geometry_edit_edges(uid, edits, version)
revision = store.edit_geometry(uid, edits, version)
```

- `element_offset` counts **scalars**, not bytes or points. Overlapping edits
  apply in input order; last write wins. Exact bytes are preserved, including
  NaN payloads and signed zero.
- `edit_geometry_rectangle(uuid, array, row, column, rows, columns, bytes,
  expected_version)` patches a rectangle; all components are written, the
  byte length must be exact, and a channel's rectangle uses that channel's own
  pixel coordinates.
- C++ equivalents: `ObjectStore::editGeometry` / `editGeometryRectangle`,
  which take the expected **geometry revision** —
  `ObjectStore::geometryAtVersion(txn, uuid, version)` resolves it from a
  numbered version.
- Edits in another volume's frame go through transformed views
  (`store.edit_transformed_coordinates`, `store.edit_transformed_surface`),
  which map positions back to canonical storage — see
  [transforms](transforms.md#editing-through-a-transform).

## Whole-array updates: `updateGeometryArrays`

C++ callers holding complete desired arrays call
`updateGeometryArrays(txn, uuid, data, expectedGeometryRevision, ...)`
(typed request: `UpdateGeometryArraysOp`). It makes the current arrays equal
the desired ones using sparse deltas — surface arrays are compared in 1 MiB
windows, so surfaces of any size update without reading the whole revision —
sequences keep their stable point identities, auxiliary channels are
preserved exactly, and optional replacement per-point annotations ride along.
Every affected relationship is decided one way: `affected="retain"` or
`affected="keep_recorded"`. A failure rolls back all of its arrays and blocks
the enclosing commit. Grid/channel schema changes keep their dedicated APIs
below.

## Surface channels

```python
revision = store.update_surface_channels(
    uid,
    channels=[{"name": "ink", "dtype": "<f4", "shape": [700, 740], "data": ink_bytes}],
    remove=["review"],
    expected_version=store.current_version(uid),
)
```

Adds, replaces, or removes surface channels atomically (`points` is not a
channel). Supply only changed arrays, as exact little-endian bytes; dtype and
component count may change. A channel's rows and columns must each be a whole
multiple of the points grid. Removing an absent channel or a duplicate name
fails. Absent and zero-filled channels are different states.

`valid` is the per-vertex validity channel: `|u1` at grid resolution, nonzero
means valid (ingestion writes 255 from `mask.tif`). A vertex is valid when
its coordinates are not the `(-1,-1,-1)` sentinel **and** `valid` is nonzero.
Typed views, relationship discovery, and `nearest_surfaces` apply it; spatial
bounds still cover every vertex with coordinates. `mask`, `x`, `y`, and `z`
are not channel names. Importing one channel from a TIFF into a stored
surface is the same edit: `vc_object_ingest --surface-channel UUID NAME TIFF`
(C++ `importSurfaceChannel`).

## Structural sequence edits

`edit_geometry_sequence` splices rows of `points` (point collections, tracks),
`control_points`, or `line_points` (fibers). `start` and `erase_count` select
the span removed; `parts` supplies the replacement, in order — new rows
(`count`, little-endian XYZ `data`, explicit nonnegative `ids` for
collections/controls, optional per-row `annotations`) or existing rows
(`revision`, `start`, `count`, optional `reverse`), which copy coordinates,
IDs, and annotations together, possibly from another object of the same kind
or a historical revision (expect every source object in `expected_versions`):

```python
import struct

version = store.current_version(uid)
with store.write(expected_versions={uid: version}) as tx:
    result = tx.edit_geometry_sequence(uid, {
        "array": "points", "start": 2, "erase_count": 0,
        "parts": [{"count": 1, "ids": [42],
                   "data": struct.pack("<3d", 10, 20, 30),
                   "annotations": [{"label": "new point"}]}],
    }, version)
```

Append is `start=count, erase_count=0`; deletion is no parts; reversing is
replacing the whole sequence with a reversed reference to itself. Split by
filling previously created empty objects with disjoint source slices, merge
by inserting slices into the destination, and retire sources plus fix
relationships **in the same checked transaction** — the store does not choose
what to retire or how domain topology reconnects. (Through a session, a split
is one typed request that creates both results and retires the original.)

Stable-ID rules:

- Point-collection IDs survive reordering. Fiber controls may supply
  `control_point_ids` at creation; otherwise the first structural control edit
  assigns `0..N-1`, and later new controls need explicit IDs. Sampled
  `line_points` and tracks have no stable IDs.
- A point deleted from an identity-managed sequence cannot be recreated as
  new rows under its old ID — restore it with an explicit slice of its
  historical revision instead. The allocation ledger survives lifecycle
  restoration. Duplicate resulting IDs are rejected.
- Annotations are stored by ID, separately from array order, so a splice
  cannot swap labels. `segment_to_next` and any field containing `direction`
  or `tangent` (including nested) require `transformed_annotations`: a
  complete annotation list for the resulting order. The store never negates
  vectors or relocates span ownership for you; omitting a required
  transformation aborts the action.
- After structural management begins, low-level ID edits and full
  `put_geometry` replacements of annotated collections are rejected.

## Surface growth and cropping

```python
with store.write(expected_versions={uid: version}) as tx:
    grown = tx.resize_surface(uid, {
        "row": -1, "column": -1, "rows": 37, "columns": 39, "fill": fills,
    }, version)
```

`row`/`column` give the old logical coordinate of the new top-left corner
(negative grows top/left, positive crops); `rows`/`columns` the resulting
shape (zero-sized grids allowed). `fill` is one typed pixel per existing
array, `points` included, required even when only cropping. New `(r, c)` maps
to old `(r + row, c + column)`. XYZ and all channels resize together; a
finer-than-grid channel moves by the same rectangle scaled to its own pixels,
and emptying a grid axis while such a channel exists is refused. Exact scalar
bytes are preserved.

Geometry descriptors expose `grid_origin=[row, column]`, which tracks the
offset across resizes. Logical anchor coordinates map as
`new_index = old_index - offset`; update/remove/retain/keep those
relationships explicitly — an origin change invalidates positional
applicability even if coordinates compare equal. Historical links keep their
original pins.

## Metadata edits

```python
store.set_meta(uuid, "review_note", "checked")
store.set_meta(uuid, "coordinates", {"volume_uuid": "20260411134726", "scale_factor": 2})
store.remove_meta(uuid, "review_note")
store.set_approval(uuid, "approved")        # objects and edges alike
```

Standalone calls commit their own transaction; `tx.set_meta(...)`,
`tx.set_meta_many`, `tx.remove_meta`, `tx.set_approval` batch them into a
checked one. Setting `coordinates` replaces the whole field and rejects
unknown keys (see [volume identity](transforms.md#volume-identity-and-coordinate-scale)).
A metadata edit that changes a key an anchor depends on (`metadata_keys=...`)
needs the same explicit relationship decision as a geometry edit; unrelated
metadata edits leave relationships applicable. `retired` and
`lifecycle_epoch` are library-managed — use the lifecycle calls below, not
`set_meta`.

## Retire, restore, delete

```python
with store.write(expected_versions={a: store.current_version(a),
                                    b: store.current_version(b)}) as tx:
    tx.retire_object(a)
    tx.keep_recorded_relationships_of(a)

with store.write(expected_versions=...) as tx:
    tx.restore_object(a)                                  # last active content
    tx.restore_object(a, source_version=3)                # an explicit active version
```

Retirement keeps geometry, metadata, and history; retired objects vanish
from default catalogs, spatial queries, traversal, and exports, and traversal
cannot pass through them. Restoration republishes the last active content (or
an explicit active `source_version`, optionally `source_generation`), advances
the lifecycle epoch, and resets approval while preserving creation stamps.
Retiring an already-retired object and default-restoring an already-active
one are no-ops. Restoration does **not** recreate removed links or physically
removed rows, and the new epoch keeps preserved links from silently
reactivating even with identical geometry — revalidate explicitly. Surviving
incident edges need explicit decisions in the same transaction.

`delete_object(uuid)` retains history and is refused while managed geometry
or current edges reference the object. There is no history or payload purge
API.

## Editor drafts

A C++ `ObjectStoreSession` also holds **surface drafts**: uncommitted surface
grids, in memory, that stand in for their objects (or objects not created yet)
in that session's `geometryRegion` and `nearestSurfaces`. Drafts are cut and
searched by the same tile and triangle rules as stored surfaces, so an editor
needs no spatial index of its own. Committing is an ordinary checked geometry
edit, after which the editor clears the draft. `requireCurrent(pin)` checks
that the version on screen is still current before editing starts. Drafts can
be made crash-durable and saved with `submitDurably` — see
[durable drafts](hosted.md#durable-drafts).

## Storage cost notes

Sparse edits store compressed byte deltas against the last structural
checkpoint (surface tiles 32/64/128, default 128; 0 uses linear 1 MiB chunks),
verified by checksum with a 256 MiB decoded limit. Only changed payloads are
rewritten, but manifest/bounds bookkeeping can scale with chunk count and
historical reads with history length. Structural edits publish a complete
chunk snapshot (unchanged chunks are reused) and read/assemble the full
result, so their time and peak memory scale with geometry size — favor
frequent coordinate edits and occasional structural changes. There are no
automatic periodic checkpoints or pruning, and long-held readers can retain
old WAL pages.
