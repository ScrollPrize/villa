# Object linking reference

[Object store](objects/README.md) · [Volume Cartographer](../README.md)

Detector behavior and cache rules. All detector APIs read ingested geometry
from `ObjectStore`; see [the workflow](object_storage.md#discovering-relationships-from-the-store).

## Fiber-to-patch object links

`vc_object_links` reports progress on stderr by default for every import and
linking operation. It shows the current stage, completed/total counts and a bar
when the total is known, elapsed time, and an estimated remaining time once
progress is available. A heartbeat continues during individual slow loads,
spatial-index builds, and database commits. Redirected output uses periodic
plain-text lines; `--no-progress` disables reporting. Final reports stay on stdout.
Rapid stage transitions are coalesced at the normal refresh interval instead of
flashing a new bar for every subtask. Fiber/track linking likewise keeps one
overall line counter, with normal sampling and contact checks shown as details.

C++ callers can provide `vc::util::ProgressCallback` through detector
`params.progress`. Ingestion accepts its own progress callback.
The library itself does not print progress. `vc::util::TerminalProgress` is the
shared CLI renderer, also used by `vc_zarr_download_region`; finish it before
printing an operation's summary. Callback string views must be copied if retained.

`vc_object_links` proposes pending fiber-to-patch links using CT sheet normals:

```sh
build/bin/vc_object_links --db objects.vcobj --author "$USER" \
  --link-fiber-patches \
  --ct /path/to/ct.zarr --ct-level 0
```

Inputs are stored fiber and patch geometry plus a uint8 CT volume. The detector
finds anchors through
long fiber contact and strip agreement, then grows through same-sheet patch
overlap. The strip has the centerline plus **one row on each side at 20 working
voxels**. `--strip-rows` and `--anchor-contact` control its width and
minimum anchor contact (default 75 working voxels).

Fiber contacts and anchors can use every dense line point between the projected
first and last control points, inclusive; extensions outside that span are
excluded. Without controls, the full line is eligible. The saved anchor is the
median contact's original `line_points` index and source XYZ position, without
snapping to a control point. Edge metadata marks `anchor_index_space` as
`line_points`. Reviewed anchors are preserved; detector-owned pending anchors
are refreshed on rerun.

Ingestion populates object measurements. Discovery reads geometry revisions and
writes only relationship metadata, leaving object lengths and approvals alone.

Normals use the selected CT-only estimator: a 2/24 isotropic seed, followed by
three 1/24/4 anisotropic tensor refinements with gradient magnitude weighting.
Widths are selected CT voxels. The current sampler integrates on a global
three-voxel lattice within radius 48, using full-resolution Gaussian derivatives
and a complete six-voxel derivative halo. It reuses gradient tiles across nearby
queries and evaluates weights at the original query position. Flat or ambiguous
CT has no valid normal. The default CT path needs no Lasagna manifest and does
not use `pred_dt` snapping. 

`--fiber-scale` and `--patch-scale` multiply source XYZ into common working
voxels (defaults 1). `--working-to-base-scale` and `--ct-to-base-scale` specify
working and **selected-level CT** voxel sizes in common base units (defaults 1).
CT coordinates are working XYZ times their ratio; the level adds no extra scaling.
For full-resolution fibers, quarter-resolution patches and quarter-resolution CT:
`--fiber-scale 0.25 --patch-scale 1 --working-to-base-scale 4 --ct-to-base-scale 4`.
All link thresholds and strip spacing use working voxels. Scaling is in memory;
files, anchor positions, and registered geometry retain source coordinates.

For a controlled comparison, `--normals-manifest /path/to/normals.lasagna.json`
selects Lasagna normals through the **same** strip marching, contact, anchor and
growth code. `--ct` is then optional. Lasagna uses `--working-to-base-scale`;
CT level/scaling options apply only to CT. Both sources leave `pred_dt` snapping
disabled. Python accepts the equivalent `normals_manifest` keyword. This option
allows measuring normal-source effects without changing strip width or gates.

Edges record measurements, rounds, coordinate scales, the actual normal source,
estimator ID, and detector identity. Reruns preserve reviewed/manual edges; stale pending links
are removed only for successfully processed inputs. Failed inputs are excluded.

C++ exposes `vc::object_links::linkFibersToPatches`. Python exposes
`ObjectStore.link_fibers_to_patches(ct_volume, **kwargs)`;
snake_case keywords match `FiberPatchParams`, including `ct_level` and
`ct_to_base_scale`. `vc.objects.CtSheetNormalSampler` exposes the same estimator
for numerical verification and direct normal queries.

## Cached fiber strips

Use `--strip-cache /path/to/cache` (Python `cache_directory`, C++
`cacheDirectory`) to save disposable `<fiber-uuid>.cbor` strips. Matching entries
skip normal sampling and marching. The default keeps work in memory.

Fiber geometry, controls, coordinate scales, strip settings, normal-source
location/method, and local CT metadata or Lasagna manifest must match.
Changing patch-link acceptance thresholds reuses the geometry. Missing,
stale or malformed caches rebuild automatically. Cache writes are atomic;
a write failure is reported without discarding computed links.

CT/normal voxel chunks are assumed immutable; caching does not scan the entire
volume. After editing voxel data in place without changing its metadata, use
`--rebuild-fiber-strips` (Python: `rebuild_strips=True`) or remove `.strip-cache/`.
The normal source must still be accessible when rerunning the detector.

## Track-to-track object links

`vc::object_links::linkTracksToTracks` implements Spiral's exact-voxel crossing
relation in `vc_object_links`. Tracks must share an integer voxel, belong to
opposite known families (`horizontal` and `vertical`), and have an unsigned
local tangent angle of at least 30°. Tangents extend 12 source voxels from the
crossing in each direction, measured by Euclidean distance. Repeated visits to
a voxel use the first local point index. Events are clustered when consecutive
indices differ by at most four on both tracks; each cluster contributes its
upper median. One crossing per pair survives: greatest minimum arclength to
the four endpoints, then greatest first/second arclength and point indices.
Tracks are ordered by source ID to make this tie-breaking deterministic.

Ingest tracks first with `ingest_geometry_source("track", path)` or
`vc_object_ingest --db objects.vcobj --author NAME --tracks path`. The ingestion adapter accepts
packed `.vctracks` geometry or a DBM with a current packed mirror. Discovery
only reads stored revisions; `crossings.npz` is neither read nor written.

```sh
ninja -C build vc_object_links_cli vc_object_bindings test_object_links
build/bin/vc_object_links --db objects.vcobj --author "$USER" \
  --link-track-tracks --track-workers 4
```

```python
with ObjectStore.open("objects.vcobj", author="reviewer") as db:
    report = db.link_tracks_to_tracks(workers=4)
    assert not report["problems"], report["problems"]
```

Select stored objects with `track_uuids` / repeated `--track-uuid`, optionally
restricted by `dataset_ref` / `--track-dataset`. Dataset references group tracks;
they are never opened as paths. Family comes from immutable geometry attributes.
Only ingestion updates object measurements. Track crossings materialize the
selected dataset coordinates for the existing crossing kernel.

Other Python keywords are `min_angle_degrees` (30),
`tangent_radius_voxels` (12), `cluster_index_radius` (4), `edge_kind`
(`crossing`), and `replace_stale_pending` (true). Workers default to one.
Crossings are computed **within each dataset**; separate datasets are not
assumed to share a coordinate system. Anchors store raw local point indices
and XYZ positions/directions in **source dataset voxel units**, reversing the
packed ZYX coordinates. Edge metadata records the parameters, clearance and
both source IDs/arclengths, keyed by object UUID to survive endpoint ordering.

New edges are pending. Reruns deduplicate existing edges and remove only this
detector's stale pending edges whose endpoints are both selected. Reviewed and
manual edges remain. A load/validation failure or cancellation before commit
leaves all edges unchanged; inspect `report["problems"]`. The CLI exits nonzero
on detector failure. Invalid parameter values raise an exception.

The independent `vc_track_crossings` kernel is also used by Spiral's native
Python adapter. Spiral's CMake defaults to the sibling `volume-cartographer`
checkout; `VC_TRACK_CROSSINGS_ROOT` can point at another checkout. The kernel
retains the native radix sort, parallel event scan and CSR consolidation, with
a serial fallback when compiled without OpenMP. No Python dependency is added
to the C++ object-link library.

Regression checks:

```sh
build/bin/test_object_links
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH="$PWD/build/python" \
  python -m pytest python/tests/test_object_links.py -q
```

Use a Python interpreter matching the compiled bindings, with NumPy and pytest
available; it must load `vc` from `build/python`, not an older installed build.

## Point-collection-to-patch object links

Regular PCL linking uses the rules from Spiral's `fit_spiral.py`: each point
chooses the largest patch within tolerance, then the closest on area ties.
Area is the valid quad count divided by both grid scale factors, matching
`tifxyz.Patch.area`. Collections named `between_patches__A__B` choose the
nearest hit within that named pair. If either patch is absent, they use the
general rule. Remaining ties use sorted object UUID order.

```sh
build/bin/vc_object_links --db objects.vcobj --author "$USER" \
  --link-pcl-patches --pcl-tolerance 2.5
```

No CT volume is required. `--point-scale` and `--patch-scale` independently
convert source XYZ coordinates to working coordinates; both default to 1.
Tolerance is in working voxels. The default 2.5 matches Spiral's training
configuration. Patch metadata's grid scale is separate from this XYZ scaling.

```python
with ObjectStore.open("objects.vcobj", author="reviewer") as db:
    report = db.link_point_collections_to_patches(
        tolerance=2.5, point_scale=1, patch_scale=1,
        window_points=1, window_min_points=1,
    )
```

C++ exposes `linkPointCollectionsToPatches` and `PointCollectionPatchParams`.
The optional neighbor gate uses numeric point-ID order: the point itself must
hit a patch, and at least `window_min_points` members of its centered
`window_points` window must hit that same patch. Even window sizes round up to
the next odd size; the requirement is clipped at collection ends. CLI flags
are `--pcl-window-points` and `--pcl-window-min-points`; both default to 1.

Ingestion records the `abs_winding.json` role as a `winding_is_absolute`
field in `object.meta["source"]`. Absolute collections bypass the neighbor gate. Python
`absolute_winding=True/False` or CLI `--pcl-absolute` / `--pcl-relative` overrides
that stored role for the selected collections.

Each stored `point_collection` retains its source UUID. Every attached point creates a pending `point-patch` edge. Its collection
anchor stores the source point ID and source XYZ; its patch anchor stores the
projected XYZ in source patch units. Metadata records grid `[row, column]`,
distance in working units, the original winding annotation (null if unknown),
absolute/relative role, selection policy and detector settings. A collection
can span several windings; its attachments retain their individual annotations.
Python also accepts `edge_kind` and `replace_stale_pending`.

Discovery reads stored point arrays, point IDs and annotations. It writes links
in one transaction, aborting if inputs change during computation. Reruns remove
only this detector's stale pending links within the successfully processed
collection/patch scope. Reviewed and manual links survive. The original source
files can be removed or relocated after ingestion.

## Track-to-patch object links

Track-to-patch linking shares the fiber-to-patch implementation: dense-point
proximity, CT sheet-normal strips, long-contact anchors, strip agreement, and
same-sheet growth all use the same rules and defaults.

```sh
build/bin/vc_object_links --db objects.vcobj --author "$USER" \
  --link-track-patches \
  --ct /path/to/ct.zarr --track-scale 1 --patch-scale 1
```

Select stored tracks and patches with repeated `--track-uuid` and
`--patch-uuid` flags. By default all objects of those kinds are considered.

`--track-scale` converts source track XYZ to working coordinates. Patch and CT
scaling follow the fiber-to-patch rules above. For tracks, patches and CT all
at quarter resolution, use `--track-scale 1 --patch-scale 1
--working-to-base-scale 4 --ct-to-base-scale 4`. `--strip-rows` and
`--anchor-contact` control the shared strip width and minimum contact length.
CT normals are the default; `--normals-manifest` selects the same optional
Lasagna comparison path.

C++ exposes `vc::object_links::linkTracksToPatches` and `TrackPatchParams`.
Python exposes:

```python
with ObjectStore.open("objects.vcobj", author="reviewer") as db:
    report = db.link_tracks_to_patches(
        "/path/to/ct.zarr",
        track_scale=1, patch_scale=1,
        working_to_base_scale=4, ct_to_base_scale=4,
        track_uuids=selected_track_uuids,
    )
    assert not report["problems"], report["problems"]
```

Python accepts all shared fiber-link keywords, with `track_scale` replacing
`fiber_scale`, plus optional `track_uuids` and `patch_uuids` selections. Every raw
track point is eligible. Anchors save the median contact's raw local index and
source XYZ position, with `anchor_index_space: "line_points"`. Edge provenance
records detector `track-patch`, source identity, normal method, scales,
and contact/strip/growth measurements.

Links commit together after validating input revisions. Reviewed/manual edges are preserved;
stale pending links are removed only for successfully processed tracks and
patches in the current selection. Unreadable geometry is reported and skipped,
and its existing links remain. Cancellation commits no changes.

## Patch-to-patch object linking

Patch overlap linking queries all eligible target patches together for each
seed sample, instead of scanning the same seed separately for every target.
Searches use up to eight workers automatically; small patches use fewer.
`--overlap-workers 1` forces serial search, and a positive value requests that
many workers. Python accepts `db.link_patch_overlaps(workers=4)`; C++ uses
`PatchOverlapParams::workers`. The default is `0` (automatic).

Workers retain bounded correspondence sets and merge in source order. Nearest
triangle selection, distance ties, thinning, correspondence caps, RANSAC seeds,
fitting and acceptance thresholds are deterministic. Index queries
borrow immutable raw filter sets, avoid allocations for empty results, and
share one index lock across each batch of 512 samples. Progress callbacks run
on the calling thread, including updates during correspondence searches.
The pair-checking bar keeps one overall counter throughout the search; the
current patch, search percentage and fitting activity appear only in its detail text.

Setting `VC_PATCH_OVERLAP_PROFILE` emits one `PATCH_OVERLAP_PROFILE` JSON line
on stderr with phase timings and query counts. Profiling is off by default.

## CT normal sampling

The default method is **`ct-anisotropic-1-24-4-grid3-r48-v4`**. Its radius-48
support contains 33³ quadrature samples. A 1024-entry indexed LRU caches
48-voxel gradient tiles; missing tiles share a bounding CT read. Independent
strip columns march in reverse order to reuse tiles from the centerline pass.
Retained sampler storage is about **102 MiB**, excluding the shared CT chunk
cache. The selected CT level defaults to 0; changing levels does not rescale
the estimator's widths automatically.
