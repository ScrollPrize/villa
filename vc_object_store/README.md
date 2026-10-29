# vc_object_store

A SQLite object database for geometry, revisioned metadata, and graph-like relationships for geometry data (meshes, pointlines, points) within the Vesuvius Challenge. The purpose of this database is to replace the current use of files for most vc workflows and centralize geometry object storage and metadata (including graph edges) and address the issue of carrying "provenence" or history of a given object. Allowing database views at timestamps provides the ability to simply timestamp when something such as a spiral fit is created and having later access to exactly the state of the object store at that time, removing any need for copied data or revision history stored at separate metadata files.  

The object store imports geometry files (tifxyz, pcl, fiber, track) and stores them as binary chunks or tiles using sqlite blob storage. Once objects are imported, the object store contains tooling to discover geometric relationships between these objects and stores this edge information as traversable object metadata. The primary types of edges indicate that the linked objects lie on the same winding: 

- Surface to Surface
- Fiber to Surface 
- Fiber to Fiber
- Track to Surface 
- Track to Track

PCL to Surface edges can be absolute winding, relative winding, or same-winding (which is the same as relative with a 0 delta along `wind_a`)

Modifications to geometry are stored as the delta to the previous iteration and contain a timestamp. If the object requires modifications to its shape a "checkpoint" is created, and any later deltas for history refer to this checkpoint. "Point-in-time" database views can be recovered by simply providing the timestamp 

```python
view = store.revision_read_view_at(timestamp_us)

view["objects"]       # Historical object records and geometry revision references
view["relationships"] # Recorded relationships, with applicability flags
view["selection"]     # Object UUID → selected revision UUID
```

The object store can be used in place of tifxyz files, fiber jsons, track crossings/dbm files, and pcl jsons in any workflow where these are used and edit history is desired. The store contains the information necessary for any tool which works with these files to instead use the db. The object store can export these files after revision if this is needed. 

The db has fast spatial queries through a persisted r-tree backed index with optional in-memory views for repeated queries. 

The C++ `ObjectStoreSession` accepts writable or read-only attachments. Read-only
sessions reject actions before invoking their callbacks. VC3D calls the database's
object, geometry, relationship and retirement APIs directly inside `act`, with
explicit revision expectations. A split creates two objects and retires the
original in one transaction. `updateGeometryArrays` shares sparse array/sequence
updates across geometry kinds; `ingestGeometryBatch` shares one-time ingestion
of decoded geometry and native topology. Both run inside the caller’s transaction
and retain the database’s existing revision and relationship checks.

A session also holds surface drafts: uncommitted surface grids, in memory
only, that stand in for their objects (or for objects not created yet) in that
session's `geometryRegion` and `nearestSurfaces`. They are cut and searched by
the same tile and triangle rules as stored surfaces, so an editor needs no
spatial index of its own; committing is still an ordinary geometry edit,
after which the editor clears the draft.

Fiber JSON version 4 is the transfer format; typed database geometry exports
in it, including control-only fibers. File import also reads VC3D version 1
and 3 fibers and stores them as version 4: a level-0 coordinate space becomes
the volume binding, `branch_file` links resolve to the named file's UUID, and a
link declared from both ends is imported once. Version 1 control points are
bare XYZ arrays, and a missing `optimization_mode` reads as `lasagna`. A fiber
file without a `uuid` is given a name-based (version 5) UUID derived from its
file name, so separate imports of the same file agree on its identity.

## Use

```python
from vc_object_store import ObjectStore

with ObjectStore.create("objects.vcobj", author="reviewer") as store:
    report = store.ingest_geometry_source("surface", "/data/surface")
```

`vc_object_ingest` creates the store when `--db` does not exist yet. To start
a store from a folder of fiber JSONs:

```sh
build/bin/vc_object_ingest --db objects.vcobj --author "$USER" --fiber /data/fibers
```

VC3D can also create an empty store (File → Object Store → New Object Store...).

`--surface` takes one tifxyz directory or a folder of them; each surface in a
folder commits on its own, so a broken one is reported without stopping the
rest. VC3D runs the same ingestion into its attached store from File → Object
Store → Import from Folder..., optionally binding the imports to the active
volume. `--tag NAME` (repeatable; in VC3D the Tags field, which defaults to the
folder's name) adds tags to every imported object, alongside any the source declares.

```sh
build/bin/vc_object_ingest --db objects.vcobj --author "$USER" --surface /data/surface
build/bin/vc_object_ingest --db objects.vcobj --author "$USER" \
  --surface-channel SURFACE_UUID ink /data/ink_prediction.tif
build/bin/vc_object_links --db objects.vcobj --author "$USER" --link-surface-overlaps
build/bin/vc_object_links --db objects.vcobj --author "$USER" \
  --link-fiber-surfaces --ct /data/ct.zarr
```

A team can share a store as a hub file on a network share or on any SSH server.
Initialize the hub from a local store that nothing else has open, then copy it
into place (to a server, use `scp`):

```sh
vc_object_hub init objects.vcobj
cp objects.vcobj /mnt/share/hub.vcobj
```

Nobody opens the hub directly. Each client clones it into a local replica,
works there, and syncs; editing an object that already exists on the hub needs
that object's lease (`vc_object_hub lease acquire REPLICA UUID`):

```sh
vc_object_hub clone /mnt/share/hub.vcobj ~/replica.vcobj
vc_object_hub clone sftp://user@host/path/hub.vcobj ~/replica.vcobj   # no mount needed
vc_object_hub sync ~/replica.vcobj
```

In VC3D, File → Object Store → Clone Hosted Object Store... clones and attaches
a replica, and Sync Hosted Object Store Now syncs it; initializing a hub is
command-line only. See [hosted hub and replicas](docs/README.md#hosted-hub-and-replicas).

See the [API guide](docs/README.md) for checked revisions, geometry reads and
edits, graph snapshots, spatial queries, and hosted hubs. Detector distances
and scale options use working voxels.

Fiber- and track-surface linking widens each centerline into a strip that follows
the sheet normal. The normal source is one of:

- CT (`--ct`, `ct_volume`): structure-tensor normals from uint8 Zarr data with
  complete 109³ support.
- Lasagna normals (`--normal-x/--normal-y`, `normal_x_volume`/`normal_y_volume`):
  paired uint8 nx/ny stores, read at `<store>/<ct_level>` and decoded as the
  spiral fitter does: `(v - 128) / 127`, `nz = sqrt(1 - nx² - ny²)`, with
  `nx = ny = 0` meaning no data. Nearest-voxel lookup; `ct_to_base_scale` is the
  store's downsampling (the spiral's `lasagna_scale`).
- Neither: no strip is built. Anchors compare the centerline points themselves
  with each surface, by distance only (same sheet within `same_sheet`, adjacent
  within `adjacent_low..adjacent_high`), under the same strip gate ratio/margin.

Edges record the choice in `meta.normal_method` (`ct-…`, `lasagna-…` or
`centerline`) and `meta.normal_source`. Native CT opening supports local and anonymous HTTP/S3 arrays and
pyramid levels; credentials and disk-cache configuration are not exposed by this
adapter. Codec support follows vc_reader.

`nearest_surfaces(point, tolerance, uuids=None, coordinates=None, volume=None)`
(C++ `ObjectStore::nearestSurfaces`) returns the nearest point of each surface
within tolerance, ordered by distance: object and geometry revision, fractional
grid `[column, row]`, the surface point and distance. It reads only the persisted
spatial tiles near the point and tests their exact valid triangles (the same
split as surface regions); relationship discovery shares the triangle routine.
Bounds are not quantized.

Tests are self-contained and do not require the volume-cartographer checkout or
its fixture data.

### Detector outcomes

Detector reports (`DetectorReport`; a dict in Python) separate failures from
ordinary results. `issues` lists each outcome as `{severity, stage,
object_uuids, reason, count}` in Python, and `problems` provides their messages:

- `failure`: a selected object could not be loaded or validated (`load`),
  linking an object raised an error (`discover`), the run was cancelled
  (`cancel`), or a whole-run error was caught (`run`). A failed object is left
  out of the run and of stale-edge removal, so its existing edges are unchanged;
  independent objects still commit after successful frame preflight.
- `diagnostic`: an ordinary result, such as rejected overlap pairs (`count`
  pairs per `reason`), an object too degenerate to link, or an unavailable strip
  cache. Diagnostics alone do not fail a run.

`failed` is true when any issue is a failure, and `committed` says whether the
reported edge counts were committed. Invalid object frames are collected as
`validate` failures across all selected input kinds; any such failure skips
discovery and writes for that detector. Invalid caller parameters still raise.
The native report stores readable messages on issues; Python derives the
compatibility `problems` list from those messages.
`vc_object_links` exits nonzero only for failures, printing them with `!` (to
stderr) and diagnostics with `-`. Each detector commits on its own; a failure in
one does not roll back another, and a rerun reconciles pending edges.

### Coordinate-aware detectors

The Python and C++ PCL-surface, surface-overlap, fiber-surface and stored-track-surface
APIs accept an optional `volume_uuid` (`CoordinateParams::volumeUuid` in C++).
With this option, every selected object's `meta.coordinates` must declare that
volume and a finite positive `scale_factor`, where **stored XYZ = volume XYZ ×
scale_factor**. Detectors resolve scales individually within one discovery pass;
all tolerances, strip distances and overlap UV poses then use
reference-volume voxels (surface areas use squared voxels). The selection and stale-edge scope are unchanged.

```python
report = store.link_point_collections_to_surfaces(
    volume_uuid="20260411134726", collection_uuids=collections, surface_uuids=surfaces)
report = store.link_surface_overlaps(volume_uuid="20260411134726", surface_uuids=surfaces)
report = store.link_fibers_to_surfaces(
    "/data/ct.zarr", volume_uuid="20260411134726",
    fiber_uuids=fibers, surface_uuids=surfaces, ct_level=1, ct_to_base_scale=2)
```

In this mode, leave `point_scale`, `surface_scale`, `fiber_scale`, `track_scale`
and `working_to_base_scale` at their identity defaults (1). Nonidentity values
are rejected rather than overriding metadata or applying scaling twice.
`ct_to_base_scale` is the selected CT level's downsampling relative to the declared
volume; `ct_level` selects data and does not add another scaling factor. The caller
is responsible for binding the CT path to that volume. Without `volume_uuid`,
existing explicit-scale behavior is preserved; unknown metadata remains allowed.

Frames are declared on the objects, never per detector run. Source files that do
not declare one (for example the Spiral dataset's tifxyz patches, fibers and PCL
JSONs) get it at ingest: `vc_object_ingest --volume-uuid V --scale-factor s`, or
`coordinate_context={"volume_uuid": V, "scale_factor": s}` on
`ingest_geometry_source` / `ingest_registered_geometry`. The context fills in
missing fields for every object in that run; a conflicting source declaration is
an ingest error. Objects already in a store are declared with
`set_meta(uuid, "coordinates", {...})`, as VC3D's Bind Fiber Coordinates does.
Generated edges record each endpoint's resolved frame in `meta.coordinate_frame`.
Explicit-scale reruns set this field to null, clearing any previous metadata-mode
declaration. Frame validation runs before geometry processing or relationship
writes. Existing geometry/load failure reporting remains unchanged.

Stored arrays and source anchors remain unchanged. Grid indices do not scale;
working surface grid steps per voxel are divided by the XYZ conversion multiplier.
Invalid surface vertices retain their sentinels. Overlap edges record
`meta.fit.pose` (row-major `[r00, r01, tx, r10, r11, ty]`), mapping metric UV of
surface `pose_from` to `pose_to`, with `pose_units = "volume_voxels"` and per-surface
`grid_scales` (`{uuid: [u, v]}`, grid steps per voxel), and the
`agreement_radius` (`tolerance + uv_inlier_tolerance`, volume voxels) its dense
agreement check used. Direction and scales are keyed by UUID because stored
edges order their endpoints independently of the detector.

### Overlap correspondences

`materializeOverlapCorrespondences` (Python `overlap_correspondences`) turns
accepted overlap edges into concrete point pairs, applying each edge's stored
pose to the current endpoint geometry rather than repeating the detector's
matching:

```python
for result in store.overlap_correspondences(edge_uuids, volume_uuid,
                                            spacing=8, max_pairs=256):
    if result["error"]:
        ...  # report; never drop silently
    for pair in result["pairs"]:
        pair["a_grid"], pair["b_grid"]  # (row, column) in each stored grid
        pair["a_xyz"], pair["b_xyz"]    # volume voxels; "distance" between them
```

Surface a's valid vertices are sampled every `spacing` voxels in row-major order
and mapped to surface b's metric UV; a pair is kept when b's containing quad is
valid and the points lie within the edge's recorded `agreement_radius`, the same
bound detection accepted it with; callers wanting a tighter bound filter on
`distance`. Agreeing pairs above `max_pairs` are thinned
evenly in sample order, so results are deterministic. Pairs follow the stored
edge's a/b order. Each edge reports an `error` instead of pairs when it has no
volume-frame pose (including edges detected before the direction, scales and
radius were recorded), its geometry is stale, an endpoint's grid scale has changed
since detection, or no pair agrees. Approval filtering is the caller's choice.

Python exposes the existing native `geometry_coordinates`,
`merge_geometry_coordinates` and `geometry_scale_to_volume` helpers, plus strict
`resolve_geometry_coordinates` and `geometry_scale_to_frame`. The volume helper
returns `None` for unknown or mismatched frames. The strict helpers raise instead.
To convert stored coordinates to volume voxels:

```python
from vc_object_store import geometry_scale_to_frame
multiplier = geometry_scale_to_frame(metadata, V)
volume_xyz = stored_xyz * multiplier  # stored_xyz / scale_factor
```

Consumers must preserve invalid vertex sentinels when applying the multiplier to arrays.

Run regression tests against an existing native build and Python environment:

```sh
cmake --build /tmp/vc-object-coordinate-build -j 6
ctest --test-dir /tmp/vc-object-coordinate-build --output-on-failure
VC_OBJECT_STORE_BUILD_PYTHON_PATH=/tmp/vc-object-coordinate-build/python \
  python -m pytest -q vc_object_store/tests/python
PYTHONPATH=/tmp/vc-object-coordinate-build/python \
  python vc_object_store/tests/smoke/coordinate_frames.py \
  ../spiral/dataset /tmp/coordinate-real-smoke
```

The real-data smoke command requires a new output directory. It ingests one patch
and five PCLs (assigning UUIDs in temporary import wrappers, since the source
files lack them), declares a local frame at ingest, and compares
explicit scaling with mixed source scales (patch 2; PCLs 0.25 and 4). It checks exact
attachment/surface-grid agreement and idempotent reruns; it does not train a fitter.

Validated on Ubuntu amd64, GCC 13.3, Release, Python 3.14.5. The local build used
existing cached nanobind and vc-delta3d sources, with downloads disabled:

```sh
export AGENTS_AGENT_MODE=1
coordinate_build_python=/home/paul/.cache/uv/builds-v0/.tmp3Merin/bin/python
coordinate_test_python=/mnt/bcache/projects/vesuvius-scrolls/villa-spiral-simplification/spiral-fitting/.venv/bin/python
coordinate_delta_source=/tmp/claude-1000/-mnt-bcache-projects-vesuvius-scrolls-villa/aa6c6e86-9c08-4d44-a30b-e17a92fd9821/scratchpad/build-tests/_deps/vc_delta3d-src
cmake -S vc_object_store -B /tmp/vc-object-coordinate-build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DVC_OBJECT_STORE_BUILD_PYTHON=ON \
  -DCMAKE_MAKE_PROGRAM=/home/paul/.cache/uv/builds-v0/.tmp3Merin/bin/ninja \
  -DPython_EXECUTABLE="$coordinate_build_python" \
  -DFETCHCONTENT_SOURCE_DIR_VC_DELTA3D="$coordinate_delta_source" \
  -DFETCHCONTENT_FULLY_DISCONNECTED=ON
cmake --build /tmp/vc-object-coordinate-build -j 6
ctest --test-dir /tmp/vc-object-coordinate-build --output-on-failure
VC_OBJECT_STORE_BUILD_PYTHON_PATH=/tmp/vc-object-coordinate-build/python \
  "$coordinate_test_python" -m pytest -q vc_object_store/tests/python
PYTHONPATH=/tmp/vc-object-coordinate-build/python \
  "$coordinate_test_python" vc_object_store/tests/smoke/coordinate_frames.py \
  ../spiral/dataset /tmp/vc-object-coordinate-real-smoke-final
```

The real subset produced 32 identical attachments in both coordinate modes, with
zero duplicates, updates or removals on repeat discovery. macOS and arm64 were not
run locally. This validates store coordinate conversion only; the spiral CLI port
and training acceptance remain separate stages.

## Transformed geometry views

Canonical geometry stays in its recorded volume. Request another coordinate frame
explicitly; the object store resolves the registration and applies it while reading:

```cpp
auto snapshot = store.geometrySnapshot();
auto surface = snapshot.surface(uuid).viewTransformedAt(targetVolume);
auto tile = surface.readTile(row, column, rows, columns);
// Equivalently, for any geometry kind:
auto view = snapshot.viewTransformedAt(uuid, targetVolume);
```

```python
with store.geometry_snapshot() as snapshot:
    view = snapshot.surface(uuid).view_transformed_at(target_volume)
    tile = view.read_tile(row=0, column=0, rows=64, columns=64)
    # Fiber, track, point_collection and umbilicus accessors support the same call.
    fiber = snapshot.fiber(fiber_uuid).view_transformed_at(target_volume)
    controls = fiber.points("control_points", start=0, count=2)
```

Views retain the canonical revision and snapshot lifetime. Surface validity is
separate from transformed coordinates; track views return floating coordinates.
`coordinateEdits` and `surfaceEdits` map changed positions back to canonical
storage, preserving untouched values. Track edits round to nearest integers,
with ties away from zero. Call `validateForWrite` in the checked write transaction
before applying edits; changed geometry or registrations invalidate the view.
Alternatively, `writeCoordinates` and `writeSurface` validate and apply an edit in
one call. Python exposes these as `store.edit_transformed_coordinates(view, ...)`
and `store.edit_transformed_surface(view, ...)`.

`availableTransforms(txn, uuid)` and each geometry view’s `availableTransforms()`
(`available_transforms` in Python) list reachable
volumes and ambiguity. Incompatible or ambiguous view requests fail explicitly.
Registrations and path choices are revisioned store objects. Open Data registrations
can be supplied through `OpenOptions::catalogTransforms`, or `catalog=catalog`
when opening a Python store with a `vc_reader.Catalog`. Catalog parsing, affine
composition and path resolution are implemented in vc_reader's native library,
shared by its Python and C++ consumers. Target-volume length filters measure the
transformed polyline, including nonuniform scaling and shear.
