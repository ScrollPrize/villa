# Graph linking reference

[Graph README](graph/README.md) · [Volume Cartographer](../README.md)

Detailed detector behavior, cache rules, and recorded validation results.
Benchmark paths and timings below describe historical experiments; those inputs
and temporary harnesses are not distributed with the repository.


## Fiber-to-patch graph links

`vc_graph_import` reports progress on stderr by default for every import and
linking operation. It shows the current stage, completed/total counts and a bar
when the total is known, elapsed time, and an estimated remaining time once
progress is available. A heartbeat continues during individual slow loads,
spatial-index builds, and database commits. Redirected output uses periodic
plain-text lines; `--no-progress` disables reporting. Final reports stay on stdout.
Rapid stage transitions are coalesced at the normal refresh interval instead of
flashing a new bar for every subtask. Fiber/track linking likewise keeps one
overall line counter, with normal sampling and contact checks shown as details.

C++ callers can provide `vc::util::ProgressCallback` through detector
`params.progress` or the optional last argument of the metadata import functions.
The library itself does not print progress. `vc::util::TerminalProgress` is the
shared CLI renderer, also used by `vc_zarr_download_region`; finish it before
printing an operation's summary. Callback string views must be copied if retained.

`vc_graph_import` proposes pending fiber-to-patch links using CT sheet normals:

```sh
build/bin/vc_graph_import --db graph.vcgr --author "$USER" \
  --link-fiber-patches /path/to/fibers --segments /path/to/patches \
  --ct /path/to/ct.zarr --ct-level 0
```

Inputs are fiber JSON files, tifxyz patch directories with valid UUIDs, and a
uint8 CT volume. The detector registers missing objects, finds anchors through
long fiber contact and strip agreement, then grows through same-sheet patch
overlap. The strip has the centerline plus **one row on each side at 20 working
voxels**. `--fiber-strip-rows` and `--fiber-anchor-contact` control its width and
minimum anchor contact (default 75 working voxels).

Fiber contacts and anchors can use every dense line point between the projected
first and last control points, inclusive; extensions outside that span are
excluded. Without controls, the full line is eligible. The saved anchor is the
median contact's original `line_points` index and source XYZ position, without
snapping to a control point. Edge metadata marks `anchor_index_space` as
`line_points`. Reviewed legacy anchors are preserved; pending anchors with the
old index convention are replaced on rerun.

Both fiber import and fiber-to-patch linking populate `fiber.length` from the
full dense polyline in source voxels, including extensions beyond the control
points. The importer uses the control-point polyline when there are no dense
points. Linking scales do not affect the stored length. Existing lengths are
refreshed on processing, including re-imports of unchanged files. A changed
length bumps the object generation, updates its change stamps, and resets its
approval to pending. Identical lengths leave both generation and approval alone.

Normals use the selected CT-only estimator: a 2/24 isotropic seed, followed by
three 1/24/4 anisotropic tensor refinements with gradient magnitude weighting.
Widths are selected CT voxels. The current sampler integrates on a global
three-voxel lattice within radius 48, using full-resolution Gaussian derivatives
and a complete six-voxel derivative halo. It reuses gradient tiles across nearby
queries and evaluates weights at the original query position. Flat or ambiguous
CT has no valid normal. The default CT path needs no Lasagna manifest and does
not use `pred_dt` snapping. Measured patch-normal errors and workflow timings
are recorded under [CT normal validation](#ct-normal-validation). Historical seven-row link accuracy has not been established
for this variant.

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
estimator ID, and detector version 3. Reruns preserve reviewed/manual edges; stale pending links
are removed only for successfully processed inputs. Failed inputs are excluded.

C++ exposes `vc::graph_links::linkFibersToPatches`. Python exposes
`GraphDb.link_fibers_to_patches(fibers_dir, patches_dir, ct_volume, **kwargs)`;
snake_case keywords match `FiberPatchParams`, including `ct_level` and
`ct_to_base_scale`. `vc.graph.CtSheetNormalSampler` exposes the same estimator
for numerical verification and direct normal queries.

## Cached fiber strips

`vc_graph_import --link-fiber-patches /path/to/fibers` automatically saves each
built strip in `/path/to/fibers/.strip-cache/<fiber-uuid>.cbor`. Later runs
discover matching entries and skip normal sampling and strip marching. No
additional flag is needed. The cache also applies to Python's
`GraphDb.link_fibers_to_patches`.

Fiber geometry, controls, coordinate scales, strip settings, normal-source
location/method, and local CT metadata or Lasagna manifest must match.
Changing patch-link acceptance thresholds reuses the geometry. Missing,
stale or malformed caches rebuild automatically. Cache writes are atomic;
a write failure is reported without discarding computed links.

CT/normal voxel chunks are assumed immutable; caching does not scan the entire
volume. After editing voxel data in place without changing its metadata, use
`--rebuild-fiber-strips` (Python: `rebuild_strips=True`) or remove `.strip-cache/`.
The normal source must still be accessible when rerunning the detector.

## Track-to-track graph links

`vc::graph_links::linkTracksToTracks` implements Spiral's exact-voxel crossing
relation in `vc_graph_links`. Tracks must share an integer voxel, belong to
opposite known families (`horizontal` and `vertical`), and have an unsigned
local tangent angle of at least 30°. Tangents extend 12 source voxels from the
crossing in each direction, measured by Euclidean distance. Repeated visits to
a voxel use the first local point index. Events are clustered when consecutive
indices differ by at most four on both tracks; each cluster contributes its
upper median. One crossing per pair survives: greatest minimum arclength to
the four endpoints, then greatest first/second arclength and point indices.
Tracks are ordered by source ID to make this tie-breaking deterministic.

Pass the tracks DBM directly. The linker discovers its packed geometry,
registers the tracks automatically, computes crossings in C++, and writes
pending edges to SQLite. There is no selected-tracks JSON step and
`crossings.npz` is neither read nor written.

```sh
ninja -C build vc_graph_import vc_graph_bindings test_graph_links
build/bin/vc_graph_import --db graph.vcgr --author "$USER" \
  --link-track-tracks /mnt/bigpc/spiral_dataset_working/tracks/2um_ds2_ps256_surf_v2.dbm \
  --track-workers 4
```

Python uses the same entry point:

```python
from vc.graph import GraphDb

with GraphDb.open("graph.vcgr", author="reviewer") as db:
    report = db.link_tracks_to_tracks("/data/tracks/example.dbm", workers=4)
    assert not report["problems"], report["problems"]
```

The DBM logical path and its `.dbm.db` backing filename are both accepted.
Geometry comes from the adjacent `<dbm>.vctracks` directory; this already exists
for the example above. `extract_surface_tracks.py` writes the DBM during
extraction and creates this packed directory when finished by default
(`--no-packed-store` disables that final step). The linker checks its source
fingerprint against the DBM backing files before linking. For older DBMs
without a current packed geometry store, create one using Spiral's
`convert_track_store.py` (use `--force` to refresh a stale store).
A `.vctracks` directory can also be passed directly.
The packed files must remain immutable during a run. Full-dataset runs keep
coordinates memory mapped rather than copying the entire coordinate array.

All tracks in the input are registered by default, including isolated tracks.
For a small run, optionally select source IDs with repeated
`--track-source-id <id>` flags, or `source_ids=[...]` in Python. Source IDs are
stable `source_ids.u64` values, not row numbers. Registration and edge updates
commit together; failures leave neither partial registration nor partial edges.
Reruns reuse existing UUIDs. Object approvals and generations are preserved
when the stored measurements are unchanged.
Both track linkers populate `track.length` with polyline arclength: the sum of
Euclidean distances between consecutive raw points, in source dataset voxels.
Repeated points contribute zero. Track-to-patch coordinate scaling does not
change this stored length. Reruns refresh lengths for processed existing tracks
with normal content-change semantics: changed lengths bump generation and reset
approval; unchanged lengths are a no-op.
Track family is read from the packed family code: `horizontal`, `vertical`, or
empty for unknown. Both linkers fill missing families on existing registrations;
filling a missing family also bumps generation and resets approval, even if the
length is unchanged. A conflicting nonempty family is reported as an input error.

The advanced already-registered-track API is still available: omit the input
path and optionally restrict `dataset_ref` / `track_uuids` in Python, or use a
bare `--link-track-tracks` with `--track-dataset` / repeated `--track-uuid`.
New registrations use the canonical packed path as `dataset_ref`; an existing
relative-path identity is reused when it resolves to the same dataset.

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
to the C++ graph library.

Validation used 24 real tracks (6,192 points) from
`2um_ds2_ps256_surf_v2.dbm.vctracks`: all 18 crossings, raw indices,
arclengths and clearances matched both Spiral's Python fallback and its
previous native kernel exactly. The graph linker produced the same 18 edges
and an idempotent rerun. The DBM CLI entry point also registered those 24 tracks
from `/mnt/bigpc/spiral_dataset_working/tracks` and reproduced the same 18 edges;
rerunning with the `.dbm.db` backing filename reused all identities and links.
Regression tests cover missing and deliberately invalid `crossings.npz` files:
neither is read, and no sidecar is created. No full-collection crossing test was run.

Regression checks:

```sh
build/bin/test_graph_links
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH="$PWD/build/python" \
  python -m pytest python/tests/test_graph.py -q
```

Use a Python interpreter matching the compiled bindings, with NumPy and pytest
available; it must load `vc` from `build/python`, not an older installed build.

## Point-collection-to-patch graph links

Regular PCL linking uses the rules from Spiral's `fit_spiral.py`: each point
chooses the largest patch within tolerance, then the closest on area ties.
Area is the valid quad count divided by both grid scale factors, matching
`tifxyz.Patch.area`. Collections named `between_patches__A__B` choose the
nearest hit within that named pair. If either patch is absent, they use the
general rule. Remaining ties use sorted patch-directory order.

```sh
build/bin/vc_graph_import --db graph.vcgr --author "$USER" \
  --link-pcl-patches /data/relative_windings.json \
  --link-pcl-patches /data/abs_winding.json \
  --segments /data/verified_patches --pcl-tolerance 2.5
```

No CT volume is required. `--point-scale` and `--patch-scale` independently
convert source XYZ coordinates to working coordinates; both default to 1.
Tolerance is in working voxels. The default 2.5 matches Spiral's training
configuration. Patch metadata's grid scale is separate from this XYZ scaling.

```python
with GraphDb.open("graph.vcgr", author="reviewer") as db:
    report = db.link_point_collections_to_patches(
        "/data/relative_windings.json", "/data/verified_patches",
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

As in `fit_spiral.py`, `abs_winding.json` denotes absolute winding annotations
and bypasses the neighbor gate. Stored `metadata.winding_is_absolute` does not
override the file role. Use Python `absolute_winding=True/False` or CLI
`--pcl-absolute` / `--pcl-relative` to override the filename rule explicitly.
The CLI override applies to every PCL file in that invocation.

Each collection becomes a `point_collection` object, preserving its source
UUID. Every attached point creates a pending `point-patch` edge. Its collection
anchor stores the source point ID and source XYZ; its patch anchor stores the
projected XYZ in source patch units. Metadata records grid `[row, column]`,
distance in working units, the original winding annotation (null if unknown),
absolute/relative role, selection policy and detector settings. A collection
can span several windings; its attachments retain their individual annotations.
Python also accepts `edge_kind` and `replace_stale_pending`.

The source JSON and TIFF files remain unchanged. Collections and patches must
already carry valid source UUIDs; re-save legacy collections in VC3D first.
Registration and links commit together for each input file. Malformed input or
cancellation raises an exception and rolls back that run. Reruns deduplicate
links and remove this detector's stale pending links only within the processed
collection/patch scope. Reviewed and manual links survive. Geometry remains in
the external files; `resolve_storage_path` locates a collection by UUID even
after its JSON file is renamed.

Validation used 29 real collections (259 points) and 24 real patches from
`/home/sean/Desktop/spiral_dataset_graph_uuid`. All 17 selected attachments and
winding annotations matched Spiral's Python fallback, with grid-coordinate
differences below 0.000001 cells and distance differences below 0.0006 voxels.
A three-point/two-hit neighbor gate rejected all 17 in both implementations;
an absolute-role override restored them. CLI results and idempotent reruns
matched. Regression tests also cover named pairs, anisotropic grid scales,
independent XYZ scales, reviewed-edge preservation and cancellation rollback.
No full-dataset run was performed.

## Track-to-patch graph links

Track-to-patch linking shares the fiber-to-patch implementation: dense-point
proximity, CT sheet-normal strips, long-contact anchors, strip agreement, and
same-sheet growth all use the same rules and defaults.

```sh
build/bin/vc_graph_import --db graph.vcgr --author "$USER" \
  --link-track-patches /path/to/tracks.dbm --segments /path/to/patches \
  --ct /path/to/ct.zarr --track-scale 1 --patch-scale 1
```

The DBM resolves to its adjacent `.vctracks` geometry, with the same freshness
checks and object identities as track-to-track linking. A packed directory is
also accepted. No selected-tracks JSON or `crossings.npz` is needed. By default
all input tracks are considered; use repeated `--track-source-id <id>` or
`--track-uuid <uuid>` flags to restrict the run.

`--track-scale` converts source track XYZ to working coordinates. Patch and CT
scaling follow the fiber-to-patch rules above. For tracks, patches and CT all
at quarter resolution, use `--track-scale 1 --patch-scale 1
--working-to-base-scale 4 --ct-to-base-scale 4`. `--strip-rows` and
`--anchor-contact` control the shared strip width and minimum contact length.
CT normals are the default; `--normals-manifest` selects the same optional
Lasagna comparison path.

C++ exposes `vc::graph_links::linkTracksToPatches` and `TrackPatchParams`.
Python exposes:

```python
with GraphDb.open("graph.vcgr", author="reviewer") as db:
    report = db.link_tracks_to_patches(
        "/path/to/tracks.dbm", "/path/to/patches", "/path/to/ct.zarr",
        track_scale=1, patch_scale=1,
        working_to_base_scale=4, ct_to_base_scale=4,
        source_ids=[6317896897264, 6317896894401],
    )
    assert not report["problems"], report["problems"]
```

Python accepts all shared fiber-link keywords, with `track_scale` replacing
`fiber_scale`, plus optional `source_ids` and `track_uuids` selections. Every raw
track point is eligible. Anchors save the median contact's raw local index and
source XYZ position, with `anchor_index_space: "line_points"`. Edge provenance
records detector `track-patch` version 1, source identity, normal method, scales,
and contact/strip/growth measurements.

Registration and edges commit together. Reviewed/manual edges are preserved;
stale pending links are removed only for successfully processed tracks and
patches in the current selection. Unreadable geometry is reported and skipped,
and its existing links remain. Cancellation commits no changes.

Validation used four real tracks (977 points) and three real patches: three
links matched equivalent fiber inputs exactly, including anchors and all
contact/strip/growth measurements. Rerunning skipped all three existing links;
the CLI produced the same result. No full-collection test was run. Synthetic
regressions also cover coordinate scaling, growth, wrong-sheet rejection,
fiber control-point bounds, identity reuse, and preservation of reviewed edges.

## Patch-to-patch graph linking performance

Patch overlap linking queries all eligible target patches together for each
seed sample, instead of scanning the same seed separately for every target.
Searches use up to eight workers automatically; small patches use fewer.
`--overlap-workers 1` forces serial search, and a positive value requests that
many workers. Python accepts `db.link_patch_overlaps(path, workers=4)`; C++ uses
`PatchOverlapParams::workers`. The default is `0` (automatic).

Workers retain bounded correspondence sets and merge in source order. Nearest
triangle selection, distance ties, thinning, correspondence caps, RANSAC seeds,
fitting and acceptance thresholds retain the original behavior. Index queries
borrow immutable raw filter sets, avoid allocations for empty results, and
share one index lock across each batch of 512 samples. Progress callbacks run
on the calling thread, including updates during correspondence searches.
The pair-checking bar keeps one overall counter throughout the search; the
current patch, search percentage and fitting activity appear only in its detail text.

Measurements on Linux arm64 with GCC 13, `RelWithDebInfo` and LTO, three runs
per version with a fresh graph each time (seconds, min / median / max):

| Input | Original total | Optimized total | Median speedup |
| --- | --- | --- | --- |
| 8 real patches | 0.710 / 0.721 / 0.729 | 0.225 / 0.277 / 0.323 | 2.6x |
| 24 real patches | 1.563 / 1.565 / 1.569 | 0.298 / 0.317 / 0.328 | 4.9x |
| 3 dense synthetic patches | 2.768 / 2.791 / 2.860 | 0.704 / 0.723 / 0.840 | 3.9x |

Real inputs were small symlink subsets of
`/home/sean/Desktop/spiral_dataset_graph_uuid/verified_patches`, staged in
`/tmp/vc-overlap-perf/patches` and `patches24`. The 24-patch subset was the first
24 sorted directories with at most 6 MB of TIFF geometry, giving 434,673 seed
samples and 50 bounding-box candidate pairs. The synthetic input contained
three 321x321 grids with curvature, small displacements, reflection and a hole,
using sample spacing 1 to exercise the 4,096-correspondence cap. No full
collection was tested; these timings do not predict full-collection runtime.

Phase profiling identified correspondence collection as the bottleneck:
on 24 patches its median fell from 1.389 s to 0.144 s (9.6x), and spatial
queries fell from 3,902,813 to 408,683. Loading, indexing, fitting and agreement
checks accounted for much less time. All accepted links (1, 9 and 3 respectively)
and their complete fit metadata matched the original exactly across all runs.
Regression tests additionally compare serial/parallel capped fits, cancellation,
callback thread identity and batched/individual nearest queries.

Build with `ninja -C build vc_graph_import vc_graph_bindings test_graph_links
test_surface_patch_index`. The benchmark harness in `/tmp/vc-overlap-perf/bench.py`
times the command below, with a new output basename for each run. The original
executable and core library were saved in that temporary directory for comparison;
benchmark inputs and output graphs are not repository files.

```sh
VC_PATCH_OVERLAP_PROFILE=1 build/bin/vc_graph_import \
  --db /tmp/vc-overlap-perf/example.vcgr --author overlap-perf \
  --link-patch-overlaps /tmp/vc-overlap-perf/patches24 --no-progress \
  --export-json /tmp/vc-overlap-perf/example.json
```

Setting `VC_PATCH_OVERLAP_PROFILE` emits one `PATCH_OVERLAP_PROFILE` JSON line
on stderr with phase timings and query counts. Profiling is off by default.

## CT normal validation

The default method is **`ct-anisotropic-1-24-4-grid3-r48-v4`**. Its radius-48
support contains 33³ quadrature samples. A 1024-entry indexed LRU caches
48-voxel gradient tiles; missing tiles share a bounding CT read. Independent
strip columns march in reverse order to reuse tiles from the centerline pass.
Retained sampler storage is about **102 MiB**, excluding the shared CT chunk
cache. The selected CT level defaults to 0; changing levels does not rescale
the estimator's widths automatically.

Validation on 535 saved regression probes (508 patch points and 27 fiber
points) produced valid normals for both CT and Lasagna at every point. Angular
errors are unsigned normal-axis errors against patch geometry:

| Probe set / method | Mean error | Median error | p95 error | Maximum error |
| --- | ---: | ---: | ---: | ---: |
| Patch points, CT | 2.70° | 2.26° | 5.95° | 24.70° |
| Patch points, Lasagna | 3.38° | 2.92° | 7.15° | 18.00° |
| Fiber points, CT | 3.20° | 2.71° | 7.29° | 11.21° |
| Fiber points, Lasagna | 4.24° | 4.01° | 9.41° | 10.37° |

CT has better mean/median/p95 error on these probes, but worse maximum error.
These reused regression probes are not a new unseen holdout or a measurement
of link precision.

Small-workflow timing on Linux/aarch64, GCC 13, RelWithDebInfo, using level 0 of
`s1_ds2.zarr`, separate processes and warm OS caches:

| Workflow | CT total | Lasagna total | CT normal sampling |
| --- | ---: | ---: | ---: |
| Two fibers, 849 candidate patches | 4.75 s | 1.16 s | 3.43 s |
| One fiber, three patches | 0.83 s | 0.12 s | 0.67 s |

The two-fiber input made 5,589 valid normal calls and produced no links. The
selected positive case made 1,242 valid calls and retained the same three links
and two anchors with both normal sources. No full-collection workflow was run.
The standalone benchmark and experiment scripts were removed after this work;
these numbers record the completed validation, not a bundled benchmark suite.
