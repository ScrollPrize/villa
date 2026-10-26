# Object-store input for command-line spiral fitting

## Objective and decisions

Replace filesystem ingestion in `fit_spiral.py` for patches, fibers, point
collections (PCLs), and their relationships with reads from a `vc_object_store`
database.
The store's automatic detectors are authoritative. The fitter materializes their
results into its existing runtime constraints and uses the existing losses.
It does not repeat proximity linking, fiber side filtering, or model-dependent
relinking. Object-store input is the only supported ingestion path for these
objects; remove the replaced filesystem loaders and all then-unused code.
Only the `fit_spiral.py` CLI must work in this port. Breaking the service/workspace
codepath is acceptable; do not retain obsolete code solely for its compatibility.
CT, umbilicus, shell, external supervision and packed tracks remain external
inputs under their existing contracts unless explicitly changed below.

Accepted behavior changes include fewer representative fiber attachment points,
different attachment choices, and corresponding changes in constraint sampling.
Preserving loss semantics and physical paths matters; numerical equivalence to
the old linking pipeline is not a requirement.

Decisions:

- Refresh automatic relationships by default, writing them back to the input DB.
- Collect refresh failures, report them together, then exit nonzero before fitting.
- Provide a config option to consume approved relationships only.
- Determine absolute/relative winding roles from annotation metadata, not paths.
- Express every geometric input in reference-volume voxels; infer the reference
  volume from selected objects when unambiguous.
- Freeze the selected geometry and relationships after refresh for the fit.
- Remove the file-based patch/fiber/PCL ingestion path and obsolete linking,
  relinking, configuration, tests and documentation. Preserve reusable runtime
  geometry and loss helpers that the new adapter still needs.
- Treat the DB as immutable except for the single startup refresh phase. External
  edits and other concurrent refresh jobs are unsupported during startup and fit.

## Existing interfaces and gaps

The initial contracts were reviewed on `origin/fibers-to-obj-store` at
`193e6d38783e`. This branch now integrates upstream through
`affe83f75`; use the integrated checkout's APIs
for the remaining fitter work.

- Stored geometry is available through Python typed/snapshot APIs: patch grids,
  fiber control/line points, and PCL stable point IDs and XYZ arrays.
- Geometry objects declare `meta.coordinates = {volume_uuid, scale_factor}`.
  The documented convention is **stored XYZ = reference-volume XYZ × scale_factor**.
  Missing/null values mean unknown. The UUID need not identify a stored object;
  this is not yet a complete volume registry or CT location resolver.
- The current PCL detector reads `meta.source.winding_is_absolute` and point
  `wind_a` attributes. Generated edge evidence also records `winding_is_absolute`.
- PCL-patch edges retain point IDs and patch `ij` coordinates.
- Fiber-patch edges retain a representative raw line-point index, source XYZ,
  and `fiber_patch.patch_grid` in column/row order, plus aggregate evidence.
- Fiber branches anchor stable `control_point_ids`, not positional indices.
  `fiberBranchControlIndex()` resolves an anchor to its index in the pinned
  geometry. Ordinary and adjacent branches are distinct classifications of
  `fiber_branch` relationships.
- Patch-overlap edges currently anchor whole geometries and retain a fitted UV
  pose and statistics, not the actual inlier correspondence list. Step 4 added
  `overlap_correspondences()` to materialize deterministic pairs using the
  detector's shared agreement traversal, recorded direction, grid scales and
  agreement radius. Metadata-mode detection is required for these pose units.
- Metadata-mode detectors now resolve scales per object when `volume_uuid` is
  supplied. Existing explicit-scale APIs remain available for other callers;
  their multipliers must be identity in metadata mode.
- Detector reports distinguish structured `issues` from ordinary diagnostics;
  use `failed`, not the compatibility `problems` list, to decide whether a run
  failed. Invalid object frames are collected across selected kinds at stage
  `validate`, then the detector returns without discovery or writes. Invalid
  caller parameters and some whole-run errors still raise; startup must collect
  these exceptions alongside returned failures across independent detectors.
- Dataset files predate object identities: a tifxyz `meta.json` `uuid` is its
  directory name, and PCL collections and version-3 fibers have no UUID. Store
  ingest admits them with an explicit namespace (`vc_object_ingest
  --derive-uuids`, see the store's `docs/README.md`), giving each one
  `uuid5(namespace, "<kind>:<key>")`, so reingest maps files to the same objects.
  Version-3 fiber ingest (file-named branch targets, reciprocal declarations)
  originated in `origin/fibers-to-obj-store` `09a64c275`. The integrated importer
  resolves filenames once, then uses the shared relationship importer for
  reciprocal validation, including ordinary/adjacent, approval and automatic
  status. Coordinate declarations use the shared conflict checks.
- Upstream `fiberBatchEndpoints()` identifies out-of-batch fiber endpoints for
  checked imports. Reuse it where import pinning is needed. Pending ordinary
  branches preserve `auto` metadata through import/export; this does not approve
  them or change the fitter's adjacent-branch exclusion.
- `../spiral/dataset` has two defects that block ingest independently of
  identity. 47 of 247 version-3 branch declarations name fiber files absent from
  `fibers/`, and fiber ingest is atomic. `same_windings.json` collections 175-376
  number points from 0, but PCL point IDs must be unique within a document.

Relevant code: `fit_spiral.py::_derive_point_inputs`,
`materialize_fiber_fit_inputs`, `spiral_helpers.py::ComponentChain`,
`losses.py::get_patch_rel_winding_loss`, and the store's
`src/linking/{PointCollectionPatch,LinePatch,PatchOverlap,LinkCandidates}.cpp`.

## Proposed CLI/config contract

Names below are proposed additions, not existing options:

| Config | Meaning |
| --- | --- |
| `object_store_path` | Input DB; required for CLI patch/fiber/PCL ingestion |
| `object_store_author` | Author for automatic refresh writes |
| `object_store_refresh_links = true` | Refresh PCL-patch, fiber-patch, and patch-overlap links |
| `object_store_approved_links_only = false` | If true, consume only currently approved, applicable relationships |
| `fit_volume_uuid` | Optional reference volume identity; inferred from selected objects when unambiguous, otherwise required |
| `track_scale_factor` | Required when attaching a track store: stored track coordinates = reference-volume coordinates × this finite positive factor |
| `normal_ct` | Optional CT location, level, reference-volume UUID and selected-level downsampling; fiber-patch strip normals when the fit has no lasagna normals |
| Object selection | Explicit UUID/tag selectors for patches, fibers and PCLs; defaults documented and stable |

No normal input is required for fiber-patch refresh. The CLI passes the store
the first available of:

1. The fit's lasagna normals, when its normals input is enabled: the `normal_x`
   and `normal_y` stores as `normal_x_volume`/`normal_y_volume`,
   `normal_zarr_group` as `ct_level` (an integer group) and `lasagna_scale` as
   `ct_to_base_scale`. The store decodes them exactly as the fitter does.
2. `normal_ct`, giving CT structure-tensor normals.
3. Neither: centerline-only linking, which gates anchors on the fiber polyline's
   own same- versus adjacent-sheet distances to each patch.

Print the chosen source; edges record it in `normal_method`/`normal_source`.
Lasagna and CT are mutually exclusive in one refresh, so a configured `normal_ct`
is ignored, with a message, when lasagna normals are available.

Select active geometry with usable revisions; do not infer patch eligibility
from directory names. Keep object-selection/review policy distinct from the
relationship approval filter. Print the selected object and edge counts.

Normal mode consumes applicable pending and approved relationships, excluding
rejected ones. Approved-only applies to both automatic and explicit relationships.
It does not approve detector output. Refresh still runs in approved-only mode;
new or changed pending links will therefore be excluded. Report exclusions and
warn clearly if an expected constraint family becomes empty.

Read annotations and relationships from current, aligned revision APIs. Imported
`meta.source` provides the existing winding-role convention, but do not rely on
stale import-array positions after point edits. Resolve annotations by stable
point ID. If a source format has another absolute-role spelling, normalize it
once at ingestion rather than adding filename heuristics to the fitter.

## One coordinate frame

The fitter always uses reference-volume voxels; there is no separate fitting
downsampling option. Resolve the selected object UUIDs before inferring the
reference volume V. If their known volume declarations identify exactly one
volume, use it when `fit_volume_uuid` is omitted. If no volume can be inferred
(including an empty selection), require `fit_volume_uuid`. Multiple declared
volumes are an error: an explicit value cannot override conflicts or silently
remove selected objects. Resolve selection independently of volume inference.

An inferred or explicit fit volume does not fill missing object metadata by
itself. Every selected object must declare its own frame in `meta.coordinates`,
set at ingest (`vc_object_ingest --volume-uuid/--scale-factor` or
`coordinate_context`) or later with `set_meta`; a selected object without one is
an error. Record the resolved volume and whether it was inferred or explicitly
configured in provenance.

For an object with scale `s` relative to volume V:

```text
fit_XYZ = stored_XYZ / s
fit_ZYX = reverse_axes(fit_XYZ)
```

For CT downsampled by D relative to V:

```text
selected_CT_XYZ = fit_XYZ / D
```

Reuse and extend `objects/GeometryCoordinates.hpp`, including
`geometryCoordinates()`, `mergeGeometryCoordinates()` and
`geometryScaleToVolume()`, so refresh and the fitter share the existing
normalization, conflict checking and stored-to-volume conversion. Expose the
needed helpers to Python rather than implementing a second resolver. Retain the
helper's representability checks. Validate finite positive scales and matching
volume identities. Unknown frames or different volume identities are errors;
do not silently assume scale 1 or infer registration from equal dimensions.
There are no per-run frame declarations: missing metadata is fixed on the object
itself (at ingest or with `set_meta`), so the store's history records it.

Resolve scales per object. Extend the relevant detector paths to support mixed
scales in a single selected population; splitting refresh into independent scale
batches risks incorrect stale-edge removal and patch selection. Preserve existing
explicit-scale APIs for other callers, with unambiguous precedence/validation.

Keep source arrays and DB anchors in source coordinates. Convert runtime geometry
and anchor positions exactly once. Patch row/column indices do not scale; preserve
raw-to-runtime grid mappings if atlas loading crops or subsamples grids. Adjust
grid-steps-per-voxel metadata consistently with XYZ scaling. Preserve invalid
vertex sentinels rather than scaling them into apparently valid coordinates.

Declare the frame for umbilicus, shell and external supervision as well. Convert
their geometry and distance units to the same fitting frame. Loss thresholds,
ROI bounds, spacing and radial offsets use reference-volume voxels. CT or lasagna
level selection does not implicitly apply coordinate scaling a second time.
`lasagna_scale` is the lasagna group's downsampling relative to V.

Track stores do not carry coordinate metadata. When attaching one, require an
explicit `track_scale_factor` s; assume it belongs to the fit's volume V, without
requiring a separate volume UUID. Convert stored track ZYX to fitting ZYX by
dividing by s. Apply ROI filtering in the corresponding frame, and convert
crossing arclengths and other distance-valued track data consistently. Preserve
source indices and native exact-voxel crossing discovery; coordinate conversion
does not change the track-store format or migrate tracks into the object store.
Record the declared track scale in the input manifest and checkpoint, and verify
it on resume. Never infer it from dimensions or silently assume scale 1.

Record resolved V, reference-volume fitting units, object scales and the
fiber-patch normal source (lasagna group/scale, CT level/D, or none) in logs and
checkpoints. The CT location
binding is explicit initially because `volume_uuid` alone does not resolve it.

## Startup and refresh lifecycle

1. Parse selection and frame declarations; validate available geometry, metadata,
   winding annotations and normal-source inputs before expensive work. Resolve the fit
   volume after object selection, as described above, and validate every selected
   frame before running detectors.
2. Resolve object selectors into explicit UUID sets before refresh. Distinguish
   an omitted selector (use the documented default selection) from a selector
   that matches zero objects (select nothing). Run each detector only when all
   object kinds it requires have nonempty selections; otherwise skip it and
   report why. Never pass an empty resolved UUID set to an API where empty means
   unrestricted selection. Restrict refresh writes and stale-link removal to
   relationships whose endpoints are both in the resolved selection.
   Preserve the store's reviewed-edge and detector-ownership protections.
   In a distributed CLI run, rank zero owns this one refresh phase; other ranks
   wait for its success/failure result before reading inputs. Propagate failures
   to every rank before exiting. After success, all ranks read the now-immutable
   DB and verify matching input/frame manifests before training.
3. Collect structured failures across detectors and objects. Continue independent
   refresh work where safe; do not abort at the first reported input failure.
4. Print/write a consolidated report with detector, UUIDs, stage, reason and
   committed counts. Any real refresh failure prevents training and exits nonzero.
   Ordinary candidate rejection is a diagnostic, not a refresh failure. Introduce
   structured error/diagnostic separation in the store rather than parsing text.
5. Successful detector transactions may already have committed when another
   detector fails. Report this explicitly; cross-detector rollback is not promised.
   A rerun reconciles pending results using the store's existing idempotent logic.
6. After successful refresh, capture one consistent geometry/relationship read
   snapshot. Geometry, metadata and object selection remain unchanged throughout
   startup under the immutable-input contract; only refresh writes relationships.
   Preserve the store's existing detector commit validation and relationship
   applicability checks. Supporting concurrent external edits or retrying them
   is not required by this port.
7. Filter applicability, rejection/approval state and edge semantics. Materialize
   constraints in stable UUID/point-ID order, then release the DB read transaction
   once all required arrays and provenance are resident.

No-refresh mode applies the same validation and filtering; it never falls back
to the fitter's old linker. Checkpoint resume verifies frame and input revisions;
changed inputs require an explicit new-input/resume policy rather than silently
combining old runtime indices with new relationships.

## Runtime constraint adapter

Add a dedicated module, e.g. `object_store_inputs.py`, with three stages:
snapshot extraction, frame conversion, and runtime materialization. Extract
reusable existing materialization helpers rather than copying their code.

### Patches and PCLs

- Build the existing atlas inputs from stored patch grids, preserving validity.
- Build PCL points in stable ID order with aligned winding annotations.
- Use metadata `winding_is_absolute` for role; reject malformed/conflicting roles.
  Unannotated collections retain the current zero-relative-winding semantics.
  Absolute constraints require valid annotations; do not default missing values
  to winding zero. Preserve existing relative mixed-annotation handling explicitly.
- Convert each accepted PCL-patch edge into an attachment using its point ID and
  patch coordinates. Normalize edge endpoint order by object kind, not UUID order.
- Reuse cross-patch grouping, absolute anchors and relative/absolute losses.

### Fibers and branches

- Preserve full source line/control index mappings while trimming/decimating.
  Force-retain all consumed patch-contact and branch endpoints.
- Use each fiber-patch edge's representative contact directly. Do not reconstruct
  its contact set using proximity or apply H/V side filters.
- Separate attachment records from physical chain nodes: one physical point may
  attach to multiple patches. Attachment views share the same chain/theta node.
  Adapt `points_by_patch` and chain lookup so this does not duplicate path segments.
- Resolve ordinary fiber branches by UUID and stable control-point ID against
  the pinned geometry: stable ID -> control array index -> runtime chain node.
  Reuse `fiberBranchControlIndex()` (exposing it where needed) rather than treating
  IDs as indices. Preserve control-to-line mappings through trimming/decimation;
  fail on missing anchored IDs. Reuse connected components and `ComponentChain`.
  Exclude adjacent branches explicitly.
- Keep H/V metadata for existing radial-offset losses, although it no longer
  affects attachment selection.
- Retain strip supervision for every eligible fiber with a usable in-ROI path,
  including linked fibers. This accepted change avoids deriving coverage from a
  single representative contact. Preserve current ROI trimming, spacing and
  branch-walk rules. Regular PCL strip classification can retain its point-level
  attached/unattached rules because those edges are point-specific.

### Patch-overlap relationships

- Use the store-side deterministic materialization added in step 4 for valid
  correspondence pairs from an accepted overlap edge's stored UV pose.
  `overlap_correspondences()` reuses the detector's agreement traversal; no
  additional matching implementation or persisted-pair format is needed here.
  It reads current geometry; verify the returned revisions against the frozen
  snapshot and interpret the returned endpoint order and units as in step 9.
- Produce small synthetic same-surface collections with concrete points on both
  patches, zero relative annotation and valid chain nodes. Feed the same relative
  winding loss and seam machinery; this introduces a new constraint source.
- Failure to materialize a consumed relationship must be reported, not silently
  dropped or repaired by running the old linker.

### Sampling and provenance

Preserve collection-based sampling, fiber-component member weights and physical
chain ordering. Define a separate sampling group/weight for direct patch overlaps
so a dense overlap graph cannot accidentally dominate the loss. Prefer explicit
group metadata, with deterministic kind-based defaults; do not derive weights
from local filenames. Report counts by group and constraint family.

Keep source UUIDs, revisions, edge UUIDs and anchors on derived records for
diagnostics. Rebuild theta nodes/crossing maps from the materialized geometry;
the relationship graph alone cannot supply seam corrections.

## Implementation sequence

The following 12 steps were originally planned as one commit each, based on
`origin/fibers-to-obj-store` at `259a3548ab5b4813e40def010a816d392d674559`.
Steps 1–10 are implemented; steps 11–12 remain. Upstream
through `affe83f75` is now integrated. Keep store
changes separate from fitter changes, and include focused regression tests
with each functional commit. Use the integrated volume-targeted selection and
snapshot APIs rather than rebuilding them in the fitter.

1. **Store: unify coordinate conversion.** Extend and expose existing coordinate
   helpers in Python; support per-object detector scales and strict frame
   validation from object metadata. Frames for sources that lack them are declared
   at ingest (`--volume-uuid/--scale-factor`, `coordinate_context`). Preserve
   existing explicit-scale APIs with unambiguous precedence and conflict checks.
2. **Store: ingest dataset files without UUIDs.** Derive identities from a
   per-dataset namespace at ingest (`--derive-uuids`), keeping canonical source
   UUIDs, and accept version-3 fibers. Decide and implement handling for the two
   dataset defects above: dangling branch targets (supply the missing fibers, or
   an explicit option that drops and reports them) and per-collection PCL point
   IDs (fix the document, or renumber at ingest while keeping the source IDs).
   Do not import `patch-overlap-pcls.json`; the store's overlap detector replaces it.
3. **Store: structure refresh outcomes.** Separate failures from ordinary candidate
   rejections; report affected objects and committed counts while preserving
   reviewed edges, detector ownership and idempotency.
4. **Store: materialize overlap correspondences.** Reuse detector geometry helpers
   to produce deterministic, validated point pairs with correct pose direction
   and units. Persist representative pairs if needed to avoid repeated matching.
5. **Fitter: extract reusable materialization helpers.** Separate runtime geometry
   construction from filesystem ingestion without changing behavior. Capture
   fixed-seed reference constraints and losses before removing the old path;
   reuse the extracted helpers in the adapter rather than copying them.
   Keep attachment and strip-eligibility behavior unchanged in this extraction;
   introduce the accepted behavior changes in the later materialization steps.
6. **Fitter: add snapshot input adapter.** Implement UUID/tag selection, aligned
   annotations, relationship filtering, frame conversion and provenance in
   `object_store_inputs.py`. Expose an inspectable constraints-only report without
   training, extending it as the following constraint families are implemented.
   Resolve selectors before applying any volume filter: filtering by the requested
   volume first would hide conflicting declarations that must fail validation.
   Use `revision_read_view` for the closed selection and aligned annotation APIs.
7. **Fitter: materialize patches and PCL constraints.** Build atlas inputs and
   stable-ID attachments; enforce metadata-based winding roles and preserve
   absolute/relative loss semantics, validity and raw-to-runtime grid mappings.
8. **Fitter: materialize fibers and branches.** Preserve source indices and anchored
   endpoints; share physical nodes across attachments, resolve stable branch IDs,
   exclude adjacent branches, and retain strip supervision, physical chain
   ordering and theta seam behavior.
   Establish raw line/control-to-runtime mappings and consumed anchors before
   trimming or decimation; the current file loader discards source points too
   early for this adapter. Replace the single `on_patch` assumption with attachment
   views sharing physical chain/theta nodes. The existing C++
   `fiberBranchControlIndex()` helper is now exposed as
   `fiber_branch_control_index()` and used by the adapter.
9. **Fitter: integrate overlap constraints and sampling.** Call the existing
   `store.overlap_correspondences(edge_uuids, volume_uuid)` for the snapshot's
   accepted overlap edges; do not implement pose matching or discovery again.
   Results use stored edge endpoint order, grid (row, column), and XYZ already
   in reference-volume voxels. Match endpoints by UUID, verify returned geometry
   revisions against the snapshot, and reverse XYZ to ZYX without rescaling.
   Report per-edge materialization errors. Filter pairs through the shared
   retained sampling mask, reporting pairs/edges lost to unfitted patches or
   unsampled quads as for the other attachment families.
   Turn retained pairs into zero-relative-winding constraints using the existing
   loss and seam machinery. Preserve each pair's correspondence and physical
   chain; do not invent a path through an unordered list of independent pairs.
   Add an explicit overlap group to the existing PCL sampling machinery, keeping
   an edge's weight independent of how many pairs it produces. Extend the shared
   constraints report and provenance. Test endpoint reversal, non-unit scales,
   retained masks, multiple pairs, seam corrections and group weighting.
10. **CLI: implement startup and frozen-input lifecycle.** Add default refresh,
   aggregated failure exit, approved-only/no-refresh modes, rank-zero coordination,
   manifest agreement, external-input frame conversion and resume validation.
   Select the fiber-patch normal source (lasagna, else `normal_ct`, else none).
   Wire the adapter into `fit_spiral.py` as the supported ingestion path and
   record complete input/frame provenance in logs and checkpoints.
   Reuse `InputRequest`, selector/frame validation, `read_snapshot`,
   `materialize`, `derive_point_views`, the store's detector APIs, and the
   existing distributed/checkpoint plumbing. Resolve selection once, skip
   detectors with empty required selections, collect both exceptions and
   reports with `failed`, and capture the final snapshot after refresh. Extend
   the existing manifest with external-input frames and materialization settings
   rather than adding a second provenance format; reject incompatible resumes.
   Disable the old per-step model-direction relinker and any file reload or
   proximity re-derivation reachable from the CLI in this step: frozen inputs
   must already hold before step 11 deletes the obsolete implementations.
   Test two-rank success/failure and manifest agreement, no-refresh read-only
   operation, approved-only filtering, non-unit external/track scales, and
   resume mismatch rejection. Include a real-data startup smoke test rather than
   treating the synthetic adapter tests as proof that dataset refresh works.
11. **Fitter: remove obsolete ingestion and linking.** Delete filesystem
    patch/fiber/PCL loaders, proximity linking, side filtering, relinking paths,
    and all then-unused helpers, options, tests and documentation. Retain no
    legacy mode or service/workspace compatibility shims. Audit remaining
    references while preserving reusable runtime helpers and external inputs.
    This includes `FitContext._regular_views_from_catalog_entries()` and the
    other live relink/re-derivation paths, which duplicate
    `pcl_views.regular_pcl_views()`: delete them rather than keep a second
    PCL classification.
12. **Docs and acceptance: validate the complete port.** Document configuration,
    the one-time dataset ingest (exact `vc_object_ingest` command with
    `--derive-uuids`, `--volume-uuid` and `--scale-factor`, and which sources it
    covers), frame conventions and units. Run deterministic and
    idempotent fixtures, two-rank startup/failure tests and a small real-data CLI
    smoke fit. Record exact commands, changed constraint counts, loss behavior
    and complete checkpoint provenance; confirm the smoke fit needs no patch
    directories, fiber JSONs or PCL JSONs.

Notes from steps 5–8 that the remaining steps build on:

- `FitContext(object_store_inputs=...)` takes `object_store_inputs.FitInputs`
  and skips file loading and linking; step 10 builds it from config and makes
  it the only path. `materialize()` takes the z window and margin, default erosion and
  `pcl_fiber_min_point_spacing` explicitly.
- Default PCL sampling groups are `abs_winding`, `relative_windings` and
  `same_windings` by role and annotation, unless `meta.sampling_group` is set;
  fibers stay in `fibers`. Step 9 adds the overlap group.
- A PCL point consumed by two point-patch edges is an error. Fiber points may
  attach to several patches through attachment views.
- A fiber contact just outside the nearest-line control span widens the span.
- Store attachments were detected on whole, un-eroded grids. One whose quad
  the patch losses do not sample (`spiral_helpers.patch_sampling_quad_mask`:
  eroded, outside the z window, or off the largest connected piece) is
  dropped (its point becomes unattached) and reported by edge UUID.
- The constraints report and `FitContext` share
  `object_store_inputs.derive_point_views`, so the report applies the fit's
  validation and counts the actual views.
- `pcl_use_fiber_links` still gates branch links; `pcl_use_pending_fiber_links`
  no longer applies to store links (approval follows the snapshot policy).
- Step 9: each retained overlap pair is a two-point collection (one point on
  each patch, stored endpoint order, zero relative winding) whose chain is just
  that pair, so the relative-winding loss never routes between pairs. Pairs
  carry `sampling_member_weight` 1/(retained pairs of their edge), honoured by
  `FitContext._rebuild_pcl_sampling_strata`; the default group is
  `patch_overlaps` (`meta.sampling_group` on the edge overrides it), so an
  explicit `pcl_sampling_weights` must name it. `pcl_patch_overlap_pair_spacing`
  and `pcl_patch_overlap_max_pairs` pass through to `overlap_correspondences`;
  `materialize_from_config()` applies every materialization setting of a fit
  config.
- Step 10: `object_store_inputs.load_fit_inputs()` is the startup lifecycle;
  `FitContext.load_host_inputs` calls it when `object_store_path` is set and
  the CLI refuses to run without one. Config names differ from the proposed
  table where the flat config schema required it: selectors are
  `object_store_{patch,pcl,fiber}_{uuids,tags}` (a new nullable `strings`
  type) and `normal_ct` is `object_store_normal_ct_{path,level,downsampling,
  volume_uuid}`; `object_store_allow_changed_inputs` is the explicit resume
  policy. Input toggles off select no patches/fibers; the PCL role toggles
  must agree, since roles are read from metadata. Detector parameters are
  the store's defaults except the PCL linker's tolerance/window, taken from
  `pcl_link_*`. The manifest (`input_manifest()`) is the snapshot manifest
  plus `materialization`, `external_frames` and `link_refresh`; its digest
  ignores the store path and `link_refresh`. Checkpoints carry it as
  `input_manifest.object_store[_digest]`, runs write
  `object_store_inputs.json`, and `inspect_checkpoint` refuses a resume whose
  manifest differs (naming the changed sections). `apply_config` refuses
  settings that would relink or reload frozen inputs, and the per-step
  fiber relinker is a no-op for them; the code itself is deleted in step 11.
- External frames: tracks convert (positions, ROI bounds, arclengths,
  crossing positions/clearances) by `track_scale_factor`, required for an
  object-store fit with tracks. A non-unit scale with crossings needs the
  stored-frame crossing cache and `track_exclusion_radius` 0, because exact
  crossing discovery runs only on stored integer coordinates. The umbilicus
  (scroll-spec coordinate scale) and lasagna (group downsampling) keep their
  existing declarations; the shell, fiber directions and winding inference
  are recorded as fit-volume voxels under their existing contracts, without
  a conversion option yet.
- Real-data startup smoke: `OBJECT_STORE_SMOKE_STORE=<scratch copy of an
  ingested store> [OBJECT_STORE_SMOKE_Z=9000:9100] pytest -s
  tests/test_object_store_startup.py -k real_data`. Run on 2026-10-26 with
  181 verified patches crossing z 9000-9100 in x 5500-5650, y 3300-3450,
  the three PCL documents and all fibers (ingest flags `--derive-uuids
  --renumber-point-ids --drop-dangling-branches --volume-uuid s1-reference
  --scale-factor 1`): refresh created 553 PCL-patch and 447 overlap links
  and no fiber-patch links (centerline-only); 173 patches, 28 PCL
  attachments and 809 overlap pairs were materialized, most attachments
  lying outside the 100-slice window's sampled quads; ~18 s.
- `tests/data/object_store_reference.json` is the file loader's output for the
  synthetic fixture in `tests/object_store_reference.py`. Delete
  `tests/test_object_store_reference.py` with the file loader in step 11; the
  adapter tests keep comparing against the reference file.

Use existing environments and test entrypoints throughout; no bootstrap or
installation is part of this sequence. Broader training runs validate the
completed port rather than delaying or weakening input validation.

## Verification and acceptance criteria

Use existing environments and project test entrypoints; no bootstrap/install is
part of this task. Add small temporary DB fixtures and focused regression tests.

- Coordinate round trips at different object scales and CT downsampling D,
  with fitting coordinates always in reference-volume voxels;
  mismatched/unknown UUIDs; invalid sentinels; patch grid/anchor conversion.
- Volume inference succeeds for a single declared volume; explicit matching
  declarations agree. Missing volume information requires an explicit fit volume,
  and conflicting volumes fail even when one is explicitly configured. A selected
  object without a declared frame fails. Empty selections do not
  trigger unrestricted selection during inference.
- Track attachment requires a finite positive scale, assumes the fit volume,
  and converts positions, ROI filtering and crossing distances consistently at
  non-unit s; resume detects a changed declared track scale.
- A selector matching zero objects skips the affected detectors without modifying
  any relationships. A nonempty subset refresh leaves relationships with either
  endpoint outside the selection unchanged.
- Noncontiguous PCL IDs and edited annotations; metadata-based absolute roles
  regardless of source filename; missing/contradictory annotations.
- Pending/approved/rejected and inapplicable relationships; reviewed-edge
  preservation; approved-only behavior after refresh.
- Fiber-patch refresh uses lasagna normals when the fit loads them, `normal_ct`
  only without them, and centerline-only linking with neither; no configuration
  requires a CT volume.
- Multiple independent refresh failures are all reported and training never
  starts; ordinary rejected candidates do not cause fatal exit.
- A two-rank startup refreshes only on rank zero, propagates refresh failure to
  both ranks, and verifies matching manifests before either rank trains.
- Representative fiber contacts survive trimming and decimation; control-to-line
  branch mappings remain exact; multiple attachments share one physical node.
- Reordered, noncontiguous control-point IDs resolve branch endpoints to the
  correct physical nodes; missing anchored IDs fail explicitly.
- Fiber-component chains and theta seam corrections, including a path crossing
  the spiral seam; adjacent branches never enter same-winding components.
- Patch-overlap correspondence orientation, valid domains and weighting.
- An adapter fixture that represents identical attachments to the old pipeline
  yields matching runtime constraints and losses with fixed seeds. Capture the
  reference before deleting the old path; do not retain it solely for this test.
- A small real-data CLI smoke fit using refreshed links produces finite losses,
  reports expected family counts and saves complete input/frame provenance.
  Compare old/new constraint counts and loss behavior; differences from changed
  link semantics are expected and must be described, not asserted away.
- Repeated refresh produces no duplicate relationships; frozen materialization
  is deterministic; resume detects changed inputs/frames.
- Dataset ingest with a namespace gives patches, PCL collections and fibers the
  documented uuid5 identities (equal to Python's `uuid.uuid5`), resolves
  version-3 branches to them, keeps canonical source UUIDs, and refuses a repeat
  ingest as existing identities. Without a namespace such sources are rejected.
- CLI smoke inputs require no patch directories, fiber JSONs or PCL JSONs.
  Audit remaining references to removed ingestion code and options; service/
  workspace compatibility is not an acceptance criterion.

Document the exact focused test and smoke commands once test files and fixtures
exist. Broader training runs are validation of the completed port, not a reason
to relax frame/relationship validation.

## Not in scope

- Service/workspace integration or compatibility, interactive edits, live relinking
  or publication. Breakage of these consumers is accepted for this CLI-only port.
- Replacing loss mathematics, the spiral model, or optimizer behavior.
- Migrating packed-track loading/crossing graphs and track-specific supervision.
- Signed winding constraints from adjacent fiber branches or arbitrary graph edges.
- Importing CT, lasagna, shell or other volumetric supervision into the DB.
- Automatic registration, translations, rotations or anisotropic frame transforms.
- Automatically approving detector output or overwriting reviewed decisions.
- A single atomic transaction spanning all detector refreshes.
- Concurrent external DB writers during startup or fitting.

## Later work

- Persist richer fiber contact sets/coverage if representative anchors limit
  training; use these to refine strip-supervision eligibility and sampling.
- Add explicit signed winding-offset relationships and losses for adjacent links.
- Migrate tracks and their crossing/patch relationships through the same adapter.
- Add a resolvable volume registry, stable content identity and declared frame
  transforms, extending beyond the initial UUID plus isotropic scale contract.
- Integrate service/workspace lifecycle using the snapshot adapter and revision
  provenance established here.
- Move winding-graph diagnostics to store-backed inputs and expose edge-to-loss
  provenance for inspection of inconsistent constraints.
- Revisit sampling balance using real fits.
