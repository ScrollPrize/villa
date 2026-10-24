# vc_object_store

A SQLite object database for geometry, revisioned metadata, and graph-like relationships for geometry data (meshes, pointlines, points) within the Vesuvius Challenge. The purpose of this database is to replace the current use of files for most vc workflows and centralize geometry object storage and metadata (including graph edges) and address the issue of carrying "provenence" or history of a given object. Allowing database views at timestamps provides the ability to simply timestamp when something such as a spiral fit is created and having later access to exactly the state of the object store at that time, removing any need for copied data or revision history stored at separate metadata files.  

The object store imports geometry files (tifxyz, pcl, fiber, track) and stores them as binary chunks or tiles using sqlite blob storage. Once objects are imported, the object store contains tooling to discover geometric relationships between these objects and stores this edge information as traversable object metadata. The primary types of edges indicate that the linked objects lie on the same winding: 

- Patch to Patch
- Fiber to Patch 
- Fiber to Fiber
- Track to Patch 
- Track to Track

PCL to Patch edges can be absolute winding, relative winding, or same-winding (which is the same as relative with a 0 delta along `wind_a`)

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

Fiber JSON is an explicit transfer format: readers require version 4 and UUIDs.
Typed database geometry exports in that format, including control-only fibers.
There is no old-format migration or compatibility reader.

## Use

```python
from vc_object_store import ObjectStore

with ObjectStore.create("objects.vcobj", author="reviewer") as store:
    report = store.ingest_geometry_source("patch", "/data/patch")
```

```sh
build/bin/vc_object_ingest --db objects.vcobj --author "$USER" --patch /data/patch
build/bin/vc_object_links --db objects.vcobj --author "$USER" --link-patch-overlaps
build/bin/vc_object_links --db objects.vcobj --author "$USER" \
  --link-fiber-patches --ct /data/ct.zarr
```

See the [API guide](docs/README.md) for checked revisions, geometry reads and
edits, graph snapshots, spatial queries, and synchronization. Detector distances
and scale options use working voxels. CT normals use uint8 Zarr data with complete
109³ support. Native CT opening supports local and anonymous HTTP/S3 arrays and
pyramid levels; credentials and disk-cache configuration are not exposed by this
adapter. Codec support follows vc_reader.

Surface proximity queries use the store's resident bounds index and exact
triangles. Bounds are not quantized.

Tests are self-contained and do not require the volume-cartographer checkout or
its fixture data.

### Coordinate-aware detectors

The Python and C++ PCL-patch, patch-overlap, fiber-patch and stored-track-patch
APIs accept an optional `volume_uuid` (`CoordinateParams::volumeUuid` in C++).
With this option, every selected object's `meta.coordinates` must declare that
volume and a finite positive `scale_factor`, where **stored XYZ = volume XYZ ×
scale_factor**. Detectors resolve scales individually within one discovery pass;
all tolerances, strip distances and overlap UV poses then use
reference-volume voxels (patch areas use squared voxels). The selection and stale-edge scope are unchanged.

```python
report = store.link_point_collections_to_patches(
    volume_uuid="20260411134726", collection_uuids=collections, patch_uuids=patches)
report = store.link_patch_overlaps(volume_uuid="20260411134726", patch_uuids=patches)
report = store.link_fibers_to_patches(
    "/data/ct.zarr", volume_uuid="20260411134726",
    fiber_uuids=fibers, patch_uuids=patches, ct_level=1, ct_to_base_scale=2)
```

In this mode, leave `point_scale`, `patch_scale`, `fiber_scale`, `track_scale`
and `working_to_base_scale` at their identity defaults (1). Nonidentity values
are rejected rather than overriding metadata or applying scaling twice.
`ct_to_base_scale` is the selected CT level's downsampling relative to the declared
volume; `ct_level` selects data and does not add another scaling factor. The caller
is responsible for binding the CT path to that volume. Without `volume_uuid`,
existing explicit-scale behavior is preserved; unknown metadata remains allowed.

For legacy objects, `legacy_coordinates={object_uuid: {"volume_uuid": V,
"scale_factor": s}}` can fill missing declarations for that invocation. Keys must
be canonical object UUIDs. C++ uses `CoordinateParams::legacyCoordinates`, mapping
UUIDs to JSON strings. A declaration cannot override conflicting known metadata,
and requires `volume_uuid`. Generated edges record both resolved frames and legacy
declarations in `meta.coordinate_frame`; object metadata is not modified. Explicit-scale
reruns set this field to null, clearing any previous metadata-mode declaration. Frame
validation runs before geometry processing or relationship writes. Existing
geometry/load failure reporting remains unchanged.

Stored arrays and source anchors remain unchanged. Grid indices do not scale;
working patch grid steps per voxel are divided by the XYZ conversion multiplier.
Invalid patch vertices retain their sentinels. Overlap edges record their working
UV grid scales and pose units, so consumers can interpret metric poses correctly.

Python exposes the existing native `geometry_coordinates`,
`merge_geometry_coordinates` and `geometry_scale_to_volume` helpers, plus strict
`resolve_geometry_coordinates` and `geometry_scale_to_frame`. The volume helper
returns `None` for unknown or mismatched frames. The strict helpers raise instead.
For fitting downsampling N, use:

```python
from vc_object_store import geometry_scale_to_frame
multiplier = geometry_scale_to_frame(metadata, V, downsample_factor=N)
fit_xyz = stored_xyz * multiplier  # stored_xyz / scale_factor / N
```

The strict helper accepts an optional `legacy_coordinates` declaration and checks
representability both before and after downsampling. Consumers must preserve
invalid vertex sentinels when applying the multiplier to arrays.

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
and five PCLs, assigns UUIDs in temporary legacy import wrappers, and compares
explicit scaling with mixed source scales (patch 2; PCLs 0.25 and 4). It checks exact
attachment/patch-grid agreement and idempotent reruns; it does not train a fitter.

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
