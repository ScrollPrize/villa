# Transforms and coordinate frames

Geometry is stored in the frame it was recorded in and never rescaled in
place. Each geometry object declares that frame in `meta.coordinates` — a
volume identity and a scale. **Registrations** (affine volume-to-volume
transforms) and **route choices** between volumes are themselves revisioned
store records. From them the store answers three things: read this object in
another volume's voxels (a transformed view), edit it through that view, and
run listings and spatial queries in one volume's voxels across every frame
declared in it.

## Volume identity and coordinate scale

```json
"meta": {"coordinates": {"volume_uuid": "20260411134726", "scale_factor": 2}}
```

- Despite the field name, `volume_uuid` is the volume's `id` from the
  open-data `metadata.json`, stored verbatim as a nonempty string — not
  hashed, lowercased, or converted to an object UUID. Paths, pyramid levels,
  resolution, display names, and matching grid dimensions never infer or
  override it.
- `scale_factor` is a finite positive number with direction
  **stored XYZ = volume voxel XYZ × scale_factor**. It differs from physical
  resolution and from a surface's grid `scale`. Values are metadata only:
  payloads and detector parameters are never rescaled by them.
- Missing members read as `null`, **not** scale 1. Setting coordinates
  replaces the whole field and rejects unknown keys; positive 64-bit integers
  stay exact.
- Source declarations (native documents, flat `volume_uuid`/`scale_factor`
  fields, `ingest_geometry_source(..., coordinate_context={...})`,
  packed-track `metadata.json`) must agree where known, or ingestion fails.
  Exporters emit the authoritative `meta.coordinates`, never stale source
  declarations. Setting coordinates after import is `set_meta`; anchors that
  depend on it declare `metadata_keys=["coordinates"]`.
- C++ consumers share normalization and conflict checking through
  [objects/GeometryCoordinates.hpp](../include/vc_object_store/objects/GeometryCoordinates.hpp).
  An existing binding holding a derived UUID needs an explicit metadata
  correction — it is not silently aliased to a catalog ID.

## Registrations and route choices

```python
store.register_volume_transform("source", "target",
    [[2, 0.5, 0, 10], [0, 1, 0, 0], [0, 0, 1, -4]], name="d05 registration")
available = store.available_transforms(uid)[0]
# -> {"volume": "target", "available": ..., "ambiguous": ..., "candidates": [{"path": [...]}, ...]}
store.choose_volume_transform("source", "target", available["candidates"][0]["path"])
```

- `register_volume_transform(source, target, matrix, name=...)` stores a
  3×4 affine (source voxels → target voxels) as a `volume_transform` record
  and returns its UUID. `choose_volume_transform(source, target, path)` stores
  a `volume_transform_choice` selecting one route (a chain of registrations)
  for that volume pair. A choice settles **both directions**: the reverse
  walks the chosen route backwards unless it has an explicit choice of its
  own.
- `available_transforms(uuid)` (and each view's `available_transforms`) lists
  the volumes reachable from the object's frame, with every distinct route,
  the stored choice, and whether resolution is `available`, `ambiguous`, or
  unavailable. Two different unchosen routes to the same volume are
  ambiguous: requests fail explicitly until a choice is recorded.
- C++: `registerVolumeTransform` / `chooseVolumeTransform` (checked requests
  through a session; a choice's request depends on the recorded transforms
  its path was resolved against), `volumeTransformRecord` /
  `volumeTransformChoiceRecord` to build the records, and
  `VolumeTransforms::resolve` / `related` for the graph itself. The graph,
  affine composition, and path resolution are `vc_reader`'s native library,
  shared by its Python and C++ consumers.
- **Catalog registrations.** Open a store with Open Data registrations the
  store has not recorded: Python `ObjectStore.open(..., catalog=catalog)`
  (a `vc_reader.Catalog` or its dict), C++ `OpenOptions::catalogTransforms`.
  These resolve client-side only. To share them, publish them as records with
  `recordCatalogTransforms` (typed op `RecordCatalogTransformsOp`); every
  client then resolves them from the store. Hosted stores always resolve from
  recorded transforms — selections never carry client transforms, and a
  request that sets them is refused (see
  [hosted](hosted.md#what-differs-on-a-hosted-attachment)).

## Transformed views

Canonical geometry stays in its recorded volume. Ask for another frame
explicitly; the store resolves the registration and applies it while reading,
on any geometry kind:

```python
with store.geometry_snapshot() as snapshot:
    view = snapshot.surface(uuid).view_transformed_at(target_volume)
    tile = view.read_tile(row=0, column=0, rows=64, columns=64)
    fiber = snapshot.fiber(fiber_uuid).view_transformed_at(target_volume)
    controls = fiber.points("control_points", start=0, count=2)
    # or, for any kind: snapshot.view_transformed_at(uuid, target_volume)
```

```cpp
auto snapshot = store.geometrySnapshot();
auto surface = snapshot.surface(uuid).viewTransformedAt(targetVolume);
auto tile = surface.readTile(row, column, rows, columns);
```

- Views keep the canonical revision and snapshot lifetime. Surface validity
  stays separate from transformed coordinates (a transformed `-1` component
  is not a missing vertex); track views return floating coordinates.
- Incompatible or ambiguous requests raise (`RuntimeError` in Python, which
  view do you mean is never guessed).
- `surfaces.read_surface(location, uuid, volume=..., version=None,
  target_scale=1.0)` reads a whole surface grid in a volume's voxels through
  the same mapping, on a file or hosted location.
- `store.geometry_length_in_volume(uuid, target_volume)` measures the
  transformed polyline (nonuniform scaling and shear included).

## Editing through a transform

`coordinate_edits` / `surface_edits` on a transformed view map changed
positions back to canonical storage and leave untouched values untouched.
Track edits round to the nearest integer (ties away from zero). The one-call
Python forms validate and apply:

```python
revision = store.edit_transformed_coordinates(view, "points", start, [[x, y, z], ...])
revision = store.edit_transformed_surface(view, row, column, rows, columns, points)
```

Validation reruns **inside the write transaction**: if the geometry or the
registration moved since the view was read, the edit conflicts
("registration changed" / `TransactionConflict`). In C++, call the view's
`validateForWrite` in the checked transaction, or `writeCoordinates` /
`writeSurface` for validate-and-apply in one call. In typed requests, put
the view's `checkMappingOperation()` (`CheckVolumeMappingOp`, wire name
`check_volume_mapping`) before its inverse edits: the writer resolves the
mapping again and conflicts if it is unavailable, ambiguous, or changed —
including registrations or route choices added after the read. The request
must still expect the object's numbered version. (`transformDependencies`
alone is weaker: it guards existing registration records, not additions.)

## Volume-target queries

Listings and spatial queries can target one volume across every frame
declared in it:

- `select_objects(volume=...)`, `spatial_regions(volume=...)`,
  `RegionOptions.volume`, `nearest_surfaces(..., volume=...)`.
- Matches are objects declared in that volume with a `scale_factor`. Their
  bounding boxes, minimum lengths, `length_desc` order, and length cursors
  use volume voxels, where volume length = stored length / scale_factor
  (the factor is `geometry_scale_to_volume()` in Python,
  `geometryScaleToVolume()` in C++).
- Each frame is queried on its own index and merged; equal volume lengths
  order by stored length then UUID, so a cursor of the last row's volume
  length and UUID resumes exactly. Region run points and spatial bounds come
  back in volume voxels; records keep their stored coordinates.
- `include_unbound=True` appends objects not bound to the volume (including
  other volumes' frames) in UUID order after the bound ones — unless a
  positive minimum length is set. `unbound_object_count(kind, volume)` counts
  that tail. `volume` and an exact `coordinates` declaration are exclusive.
- A selection recorded against a volume also records the versions of the
  transforms and route choices in force — see
  [selection records](reading.md#selection-records) — and the local form
  refuses a volume selection that would need catalog registrations the store
  has not recorded.

## Umbilici per volume

Each volume has at most one current umbilicus (its centre line). C++:
`importVolumeUmbilicus` / `setVolumeUmbilicusRequest`
([objects/VolumeUmbilicus.hpp](../include/vc_object_store/objects/VolumeUmbilicus.hpp))
replaces the volume's umbilici in one checked request that guards the commit
sequence before selecting what to replace — including when none exist — so an
intervening commit conflicts and the request must be rebuilt from a fresh
read. Importing an umbilicus **file** as ordinary geometry is
[ingestion](ingestion.md#from-files-ingest_geometry_source).
