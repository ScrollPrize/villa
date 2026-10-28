# Automated Sheet Volumes (.asv)

An **Automated Sheet Volume** is a single read-only SQLite file holding a large
collection of fitted sheet meshes, such as the per-window output of HercuViT sheet
instance segmentation: a grid-implicit quantised quad mesh per sheet, a bounding box,
and a spatial index over the boxes. It is the sheet-mesh sibling of the fibre
[Automated Fiber Volume](fiber-collections.md) (`.afv`); it is written outside VC3D
(by `hercunet.meshes.asv`) and VC3D displays it next to the project without importing
it. Opening a volume reads its metadata; each planar view then draws only the sheets
whose bounding box is near its slice, so a file with millions of sheets renders slice
by slice.

## Open

```sh
VC3D --volpkg scroll.volpkg.json --sheet-collection sheets.asv
```

or **File → Open Automated Sheet Volume…**, or the **Open .asv…** button in the dock.
The dock is under **View → Automated Sheet Volume**.

Unlike the fibre viewer, the sheet viewer does **not** gate on the volume's coordinate
identity. The meshes are stored in native L0 voxel coordinates, so the overlay simply
draws them on whatever volume is open; it is the caller's responsibility to open the
`.asv` on the volume it was traced from. (A mismatched file will draw in the wrong
place rather than be refused.)

## Viewer

The overlay is drawn on the planar CT views (axis-aligned and the oblique
segmentation cuts), as the curves where each nearby sheet's quad mesh meets the slice
plane. The dock controls, all remembered:

* **Show volume · read only** toggles the overlay.
* **Rainbow colors** (on by default) gives each sheet a stable colour from its id;
  off draws every sheet in one colour.
* **Max displayed sheets** (default 2000) limits how many sheets are *drawn* per
  viewport, among those crossing the slice, largest first. The fetch is not limited by
  this, so the cap never starves the view of small sheets.
* **Min sheet size (mm²)** hides sheets whose surface area is below this, computed as
  `vert_count × grid_step²` from the file's grid spacing.
* **Slice distance ± (L0 vx)** is the half-thickness of the band fetched around the
  slice (the drawn curve is the exact plane crossing).
* **Line width (px)**.

## Performance

The expensive step — the SQLite query and the per-sheet zlib decode — runs on a
worker thread with its own read-only connection, and its result is cached per viewport
over a region a margin larger than the slice. Each frame only intersects the cached
meshes with the current plane (an AABB pre-check skips sheets that do not cross it) and
culls to the visible rectangle, so panning and zooming within the cached region touch
no SQLite. The cache is shaped for the interaction priority **zoom > pan > scroll**: a
generous in-plane margin for panning, a thin band along the slice normal (a crossing
sheet always straddles the plane, so a thin band still captures every visible sheet),
and a re-query when the view leaves the region or the zoom changes. Reads are
debounced, so moving around only fires a query once the view settles.

## Coordinates

Coordinates are native L0 voxel XYZ (x, y, z — matching `.afv`), so sheets from every
window live in one global space. The overlay draws them directly; when the volume is
viewed at a downsampled pyramid level, the scale comes from the open volume, not the
file.

## Format version 1

A single SQLite database with RTree support:

```sql
CREATE TABLE metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE sheets(
  id INTEGER PRIMARY KEY, name TEXT NOT NULL, window TEXT NOT NULL DEFAULT '',
  family TEXT NOT NULL DEFAULT '', nu INTEGER NOT NULL, nv INTEGER NOT NULL,
  vert_count INTEGER NOT NULL CHECK(vert_count >= 3),
  min_x REAL NOT NULL, max_x REAL NOT NULL, min_y REAL NOT NULL, max_y REAL NOT NULL,
  min_z REAL NOT NULL, max_z REAL NOT NULL, payload BLOB NOT NULL,
  annotation TEXT NOT NULL DEFAULT '{}');
CREATE VIRTUAL TABLE sheet_bounds USING rtree(id, min_x, max_x, min_y, max_y, min_z, max_z);
```

`metadata` holds `format_version`, `complete` (`true` once fully written; other files
are refused), `uuid`, `frame`, `root`, `voxel_um`, `grid_step_um`, `codec` and counts.

* **sheets** has one row per sheet. `family` is `R`, `V` or empty. The bounds cover all
  its vertices. `payload` is the mesh: a codec byte (`1` zlib, `2` zstd) followed by the
  compressed record `"ASV1"`, `nu`,`nv` (little-endian uint16), the float32 bbox corners
  `lo`/`hi`, the `nu×nv` footprint bitmap (numpy big bit order), then the footprint-only
  vertices as uint16 quantised inside `[lo, hi]`. Triangles are reconstructed exactly
  from the footprint, so they are not stored.
* **sheet_bounds** holds each sheet's bounds under its id, for the view query.

## Reader

`vc/core/types/SheetCollection.hpp` (CMake target `vc_sheet_collection`, depending only
on SQLite3 + zlib, and zstd when present) provides the metadata, the size-sorted
catalog, `region()` (decoded meshes whose box intersects a query box), `viewRegion()`
(their intersection with a slice as polylines), per-sheet geometry and annotations. Each
instance owns one read-only connection and must be used by one thread at a time; the
viewer gives each worker query its own instance.

## Build and test

SQLite development headers with RTree support are required (`libsqlite3-dev` on
Debian/Ubuntu). With `-DVC_TESTING=ON`:

```sh
cmake --build build --target VC3D test_sheet_collection
ctest --test-dir build -R sheet_collection --output-on-failure
```

`test_sheet_collection` writes its own `.asv` in the format above and checks decode,
the region query and the plane intersection.
