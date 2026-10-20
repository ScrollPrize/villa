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

## Storage and cost

The first structural edit converts legacy layouts/annotations in work proportional
to the geometry. Later sequence splices path-copy immutable AVL ropes with bounded
leaves; surface edits update persistent tile maps in stable grid coordinates.
Unchanged subtrees and payloads are shared. Reads descend trees rather than replaying
revision chains. Identity indexes retain deleted-ID allocation ledgers through
restoration; annotations remain associated with IDs independently of row order.

Work scales with new data, changed boundaries, tree paths and affected relationships.
Reversal, complete annotation transforms, retracing and full reads can process their
entire inputs. Spatial queries refine coarse array bounds through tree bounds and
paginate afterward; seams conservatively cover crossing segments. The per-operation
tree cache is bounded by an 8 MiB charge. There is no background compaction or
numerical approximation. Large annotations use separately chunked immutable blobs.

Schema 11 gates this representation. Writable migration preserves old revisions
and payloads until conversion; older binaries reject it. Synchronization transfers
and verifies new nodes/payloads while reusing existing subtrees. It does not copy the
complete object metadata/relationship timeline; use a database copy for that.

## Validation and measurement

See the [build/test commands](../README.md#build). Native regressions check bounded
payload, identity, relationship and spatial work for large appends, annotated
insertions and narrow/wide surface growth. Run `build/bin/bench_structural_edits 5`
for the synthetic scaling benchmark: full replacement versus append/insertion and
right growth, with zero/eight relationships and initialization reported separately.

Each measured edit includes expected-head reads, checked publication, durable
commit and local compaction; full replacement includes caller preparation. Inputs
span 1,000–100,000 points, 128–1,024-square surfaces and an 80-append history.
Reports include mean/min/median/max latency, payload/tree/manifest work, SQLite
steps and exclusive scoped phase times. Counters measure logical work, not disk
traffic; SQLite PROFILE times are statement lifetimes. These fixtures do not
establish real-scroll viewer latency.
