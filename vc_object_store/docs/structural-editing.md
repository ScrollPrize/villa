# Structural geometry editing

Structural edits publish ordinary immutable geometry and object revisions.
Use a checked writer with **object** revision UUIDs, and pass the expected
**geometry** revision UUID to each mutation: `ObjectStore::editGeometrySequence()`.

## Sequences

`editGeometrySequence(txn, objectUuid, GeometrySequenceEdit, expectedRevision, edgeEdits)`
splices rows of `points` (point collections/tracks), `control_points`, or
`line_points` (fibers). `start` and `eraseCount` select the removed span.
`parts` supplies the replacement, in order:

- New rows: `count`, little-endian XYZ `bytes`, and, for collections/control
  points, explicit nonnegative `ids`. Optional `annotations` is a JSON list with
  one object or null per new row.
- Existing rows: `revision`, `start` (default zero), `count`, and optional
  `reverse`. These copy coordinates, IDs, and annotations together. References
  can select another object of the same kind, including historical revisions.
  Include every source object in the write's expected revisions.

```cpp
vc::objects::GeometrySequencePart part;
part.count = 1;
part.ids = {42};
const double xyz[3] = {10, 20, 30};
part.bytes.assign(reinterpret_cast<const uint8_t*>(xyz), reinterpret_cast<const uint8_t*>(xyz) + sizeof xyz);
part.annotations = R"([{"label":"new point"}])";
vc::objects::GeometrySequenceEdit insert;
insert.array = "points"; insert.start = 2; insert.parts = {part};
auto txn = db.write({{uid, objectHead}});
const auto result = db.editGeometrySequence(txn, uid, insert, geometryHead);
txn.commit();
```

Append uses `start = count, eraseCount = 0`; insertion uses `eraseCount = 0`;
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
`Fiber::controlPointIds` preserves these IDs through typed loads/encoding.
Sampled `line_points` and tracks have no stable point IDs.

A point deleted from an identity-managed sequence cannot be recreated as new
rows using its old ID. An explicit slice of its historical revision can restore
it. The allocation ledger also survives selection of an older geometry through
lifecycle restoration. The first structural edit establishes the ID/annotation
baseline from the current geometry. IDs are unique within a sequence; merging
colliding source ID namespaces requires an application decision.

Annotations are stored by ID in immutable metadata snapshots, separately from array
order; geometry manifests reference those snapshots. `geometryPointAnnotations(txn, revisionUuid, arrayName)` returns
annotations aligned with that revision's sequence. The first structural edit
moves the corresponding `meta.source` positional annotation field into
this representation. Existing native PCL export resolves annotations through
IDs, preventing the same-length label-swap problem. Low-level ID edits/full
replacements of annotated collections are rejected; after structural
management begins, full `putGeometry` replacement is rejected. Use structural
splices or the existing sparse coordinate edit API.

`segment_to_next`, fields containing `direction` or `tangent`, and nested fields
with those names require `transformedAnnotations`: a complete annotation list
for the resulting order. This applies to insertion/deletion as well as reversal.
The application explicitly relocates span ownership and transforms directions;
the store does not infer domain meaning or negate vectors automatically. Other
application-specific directional/span fields must also be transformed by the
caller. Omission of a required transformation aborts the checked action, even if
the application catches the immediate exception.

## Storage and cost

Every structural edit publishes a complete chunk snapshot of the resulting
geometry. Unchanged content-addressed chunks are reused. Subsequent coordinate
edits store compressed byte deltas against that snapshot and update a replaceable
current-state cache. They do not retain a full changed tile per revision.

Structural edits read and assemble the resulting arrays, so their time and peak
memory scale with geometry size. Sequence insertions can change many chunk
boundaries and reduce chunk reuse. This favors
frequent coordinate edits and occasional structural changes.

Point-ID row mappings, annotations and allocation ledgers use flat metadata
snapshots with 128-entry buckets. Structural changes rebuild their indexes;
coordinate deltas share them unchanged and resolve coordinates from the selected
revision. Large annotations use separately chunked immutable blobs. There is no
persistent geometry or metadata tree.

Historical reads reconstruct deltas within the selected snapshot's chain.
Structural checkpoints start new chains without removing earlier history. There
are no automatic periodic checkpoints or pruning. Spatial queries use ordinary
chunk/region bounds, including crossing segments. Hub sync transfers and
verifies snapshots, deltas and referenced metadata payloads.
