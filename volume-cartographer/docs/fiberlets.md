# Fiberlet anchor extraction

`vc_fiberlets anchors` is the first stage of anchor-seeded fiberlet
over-segmentation. It turns the canonical `presence/nx/ny` channels of a Fiber
Lasagna manifest into zero, one, or two local unoriented line anchors per
coarse prediction-grid cell. Connection search, path optimization, path
filtering, deduplication, and extension are not implemented by this stage.

## Input and coordinates

The command accepts a local Lasagna manifest, a local manifest with the
existing `lasagna-remote.json` sidecar, or a direct HTTP/S3 manifest. Direct
remote manifests require `--remote-cache-dir`; manifest and Zarr reads use the
shared cache-first Lasagna implementation. `--cache-gib` controls the decoded
chunk cache separately.

Only the exact canonical `presence`, `nx`, and `ny` channels are observations.
Extra manifest groups, including prefixed prediction triplets used by older
tracer configurations, do not enter the fit. The channels must be 3D `uint8`
arrays with equal ZYX shapes and effective spacing, but their chunk shapes may
differ. The manifest must provide a positive numeric `source_to_base`.

Integer stored-prediction indices are voxel centres. CLI spatial coordinates
and all JSON/OBJ spatial positions are always expressed in base voxels. A cell
has a globally fixed origin and owns the half-open ZYX range
`[cell * size, min((cell + 1) * size, shape))`. A crop selects every global
cell it intersects and samples that cell in full. It does not move or truncate
interior cells. Prediction-grid coordinates remain private solver values.

## Fit

The existing deterministic two-mode non-orthogonal PCA fit supplies initial
directions. Both anchor positions start at the center of the clipped owned-cell
voxel range. They are then jointly refined using halo observations.

For a current anchor position `p_k`, direction `u_k`, observation position
`x_i`, direction `d_i`, and presence `q_i`, its transverse weight is

```text
g_ik = exp(-|(I - u_k u_k^T)(x_i - p_k)|^2 / (2 sigma^2)).
```

The Gaussian is truncated at three sigma. A symmetric finite axial slab is
measured from the plane through the fixed cell pivot normal to `u_k`. Assignment
uses the larger positive `q_i (d_i dot u_k)^2` among components whose kernels
contain the sample; unusable and zero-evidence samples remain unassigned. This
keeps nearby non-orthogonal modes from being assigned solely by which moving
Gaussian is closer.

Each direction update is the principal eigenvector of its assigned
`g_ik q_i d_i d_i^T`. The aligned-evidence centroid is projected onto the plane
through the cell pivot normal to that updated direction and clamped to the
local transverse window. The constraint plane rotates as the direction is
refined, while continuing to pass through the fixed pivot. The next iteration
recomputes the falloff around the new anchor position.

Only a strict normalized-objective improvement is accepted, with deterministic
backtracking. Refinement therefore stops at a bounded local response maximum
instead of migrating across a weak-response valley to a stronger distant
fiber. Support denominators include all lattice sites in each finite kernel,
including invalid, zero-presence, and unassigned sites. Empty, degenerate, and
below-threshold components are discarded independently. The output is zero,
one, or two anchors per cell before duplicate suppression.

After fitting, two valid components are merged when their unoriented angle is
at most `--merge-angle-deg` (10 degrees by default) and replacing them with a
joint PCA loses at most
`max(--merge-abs-loss, --merge-rel-loss * joint_objective)`. The defaults are
0.01 absolute and 0.05 relative normalized objective loss. The comparison is
inclusive. A merge refits support, direction, and position through the same
halo-backed refinement; the joint anchor can still fail the normal
minimum-support check. Exact
single-direction cells, where the second fit is empty, are not counted as
merges.

Finally, local-maximum NMS suppresses cross-cell copies of the same anchor.
Candidates must agree in unoriented direction and be within both the transverse
radius of 2 prediction voxels and longitudinal radius of 1 prediction voxel
around their sign-aligned average axis.
They are ranked by support, coherence, then stable cell/component identity.
NMS compares every candidate with the original candidate set, preserving
crossings and longitudinally separated anchors. Crops evaluate only the exact
external cells that can directly suppress a selected anchor; those context
cells are not written to the output.

## Command

```bash
volume-cartographer/build/bin/vc_fiberlets anchors \
  /path/to/fiber.lasagna.json /tmp/fiber-anchors \
  --crop 13568,20224,18112,256,256,256
```

Cell size is restricted to 2 through 8 stored prediction voxels. `--falloff`
sets the transverse Gaussian sigma in base voxels and defaults to half the cell
side. `--window` sets only the transverse refinement radius in base voxels and
defaults to one cell side. The axial slab defaults to 1.5 cell sides. NMS uses
independent fixed defaults of 2 prediction voxels transversely and 1 prediction
voxel longitudinally; CLI reporting converts both to base voxels. The NMS angle
defaults to the merge angle.
The presence floor and aligned support threshold are inclusive.
`--base-voxel-size-um` adds optional physical reporting metadata but never
changes the solve.

`--crop` is the half-open base-volume box `X,Y,Z,W,H,D`. Because stored
prediction indices are point centres, both boundaries map with
`ceil(base/prediction_to_base_scale)`, with numerical snapping at exact lattice
boundaries. The resulting interval selects complete global anchor cells. A crop
outside the prediction grid or containing no stored prediction sample fails.

The command prints prediction/base scale, cell side, falloff, window, cell
diagonal, retained/NMS counts, and elapsed time. It writes:

- `anchors.json`: versioned machine input for later fiberlet stages. It stores
  a credential-free manifest locator and content hash, coordinate contract,
  parameters, aggregate rejection counts, and only non-empty cells.
- `anchors.obj`: all diagnostic base-coordinate line glyphs.
- `anchors_0.obj`: only component slot zero from every non-empty cell.
- `anchors_1.obj`: only component slot one from two-anchor cells.
- `anchor_cells.obj`: one point for every selected cell center and a line from
  that center to each anchor retained after fitting and NMS.
- `stages/initialized.json`, `stages/refined.json`, `stages/support.json`,
  `stages/selection.json`, and `stages/nms.json`: strict diagnostic snapshots of
  candidate initialization, local refinement, support filtering, optional
  selection filtering, and duplicate suppression. These are diagnostics only;
  `anchors.json` remains the sole path-stage input.

Each selected cell has two initialized attempt records, including explicit null
geometry for empty or degenerate attempts. Stable per-cell candidate IDs survive
component compaction and sorting. A same-direction merge creates candidate ID 2
with both attempted IDs as parents. Later records retain the fitted base-XYZ
geometry and stage-appropriate metrics. Transitions record the exact rejection
reason, support or selection threshold/value where applicable, and the actual
NMS suppressor (including whether it came from external crop context). Thus a
missing final anchor can be located without inferring history from component
array slots.

Manual napari mode automatically loads and cross-validates the complete sibling
`stages/` directory when `--anchors` points to its `anchors.obj`. Replay mode
obtains the same stage files from its hashed artifact table.

The component slots are deterministic within each cell after support-based
sorting. They are visualization layers, not global H/V or winding classes, and
none of the OBJ files is an input to later stages.

All artifacts use same-directory temporary files followed by atomic replacement.
Timing, worker count, and processing-block size are not stored because they are
operational values; identical inputs and numerical parameters produce
byte-identical artifacts across worker counts and block sizes.

Anchor artifacts are experimental and parsed strictly. They include every
effective refinement/NMS parameter, per-anchor refinement score/iteration
fields, and aggregate suppression diagnostics. Retained positions must satisfy
the prediction-grid, rotating pivot-plane, and local-window invariants.
Regenerate an older `anchors.json`; no repair or compatibility path exists.

## Calibration

Choose the largest cell for which distinct sustained parallel fibers cannot
share a cell. Compare both the cell side and cube diagonal in base voxels with
the minimum sustained sheet/fiber separation, and inspect the OBJ over the
source volume. A representative crop should be run at cell sizes 2, 4, and 8
before selecting production thresholds.

## Integer path stage

`vc_fiberlets paths` consumes the authoritative `anchors.json`, the same fiber
manifest used to create it, and a separate regular Lasagna manifest providing
surface normals:

```bash
volume-cartographer/build/bin/vc_fiberlets paths \
  /path/to/fiber.lasagna.json \
  /tmp/fiber-anchors/anchors.json \
  /tmp/fiberlet-paths \
  --normal-manifest /path/to/lasagna.lasagna.json \
  --stats
```

The command verifies the anchor artifact's manifest hash, prediction shape and
prediction-to-base scale. A content-identical manifest may be relocated. Direct
remote fiber or normal manifests use the same required `--remote-cache-dir`
and cache-first behavior as the anchor command. Fiber directions and regular
Lasagna normals are distinct: the fiber manifest's `nx/ny` are never treated as
surface normals.

Target cells are selected from the filled integer neighborhood
`0 < length(cell_offset) < radius+margin`; the initial radius is four and the
margin is 0.5.
Endpoint axes must agree with their chord within 45 degrees. Every surviving
pair is solved independently, so an anchor can currently participate in many
paths.

Candidate generation finishes before any path search. A fixed worker pool then
prepares every searchable candidate exactly once: one Hermite domain, its
corridor-filtered checked 32-bit local keys, and mapped `float32` interior-node
positions. Worker-local native-corner sets are derived directly from those
positions, sorted, and
merged into one deterministic stored-ZYX ordered global union. Required corners
remain included even when a replay tube excludes the corner itself.

`--batch` is a coordinate-call limit, 65536 by default. It partitions only
consecutive ranges of the global unique union. The path stage completes all
prediction calls, then all normal calls, materializes every prepared endpoint
and node score by deriving corners and weights from its position again,
releases sampling storage, and finally runs every DP
candidate in parallel from its retained geometry. Increasing `--batch` changes
sampler call count only; it cannot decrease the unique request population or
change path/graph artifacts. Each volume call and each parallel stage may use
`--threads`; candidate results remain in their canonical slots.

Runtime reports unique sampled voxels, coordinate batch/call counts, peak call
size, prepared-geometry bytes, worst concurrent DP state/hash scratch, an
owned-payload peak-memory estimate, evaluated DP nodes, and separate
candidate-generation, preparation, corner-merge, prediction-read, normal-read,
materialization, and DP wall/CPU times. Effective cores are process CPU seconds
divided by wall seconds. Progress is phase-labelled and operational only.

`--corridor-radius` is measured in base voxels. If omitted, it defaults to one
anchor-cell width. Cell radius and neighborhood margin remain dimensionless
cell-lattice parameters.

Each candidate has its own curved coordinate domain. A cubic Hermite centerline
uses the exact anchors, both chord-oriented fitted directions, and derivative
magnitudes equal to anchor distance. Deterministic arclength inversion places
planes about 2 prediction voxels apart and inserts the exact endpoint. Each
plane is normal to the Hermite derivative. A transverse frame starts from the
least-aligned world axis and is propagated by minimal-rotation parallel
transport. Integer transverse coordinates map through that frame at 0.5
prediction-voxel spacing.

The exact start and target anchors are source and sink. Interior transitions
advance exactly one curved plane and change either transverse index by at most
one. Their directions and lengths are computed from the resulting floating XYZ
positions, so they are not restricted to world axes or 26 quantized directions.
The layered graph is acyclic. DP state retains the incoming transition because
alignment and curvature depend on the previous physical step.

Each interior node occupies 24 bytes: one checked row-major `uint32` key,
three `float32` prediction coordinates, compact two-byte fiber and normal axes,
one byte of presence, and validity flags. Fiber/normal axes use the same +Z
hemisphere `nx/ny` encoding as Lasagna and presence uses the native byte scale.
No per-node reason string or interpolation-address object is retained. This
re-quantization is intentional for the experimental fiberlet objective; exact
anchor endpoints remain double precision.

Presence is trilinearly interpolated. Fiber directions are unoriented: the
positive-weight native corner axes are normalized and accumulated as weighted
outer products, then the shared deterministic symmetric eigensolver resolves a
unique principal axis. This preserves antipodal axes without sign cancellation;
an invalid required corner or ambiguous tensor invalidates the destination.
Normals use the same interpolation, but invalid normal data keeps the existing
isotropic curvature fallback rather than rejecting the path.

Every interior mapped move must have an unoriented angle strictly below 25
degrees to the dense fiber-prediction axis interpolated at its destination.
Virtual endpoint transitions remain governed by the endpoint-axis constraint.
Invalid fiber predictions cannot admit interior edges. This hard gate is
independent of the Lasagna surface normal, which remains used only for
curvature.

Valid-data scoring uses the regular native tracer's multiplicative local
alignment loss. It multiplies presence by six positive-clamped dots among the
incoming and outgoing steps and the sign-aligned current and next prediction
axes, then charges `1-score`. This jointly penalizes trajectory turns,
prediction discontinuities, and trajectory/prediction disagreement. The DP
multiplies that loss by the actual mapped prediction-voxel edge length. There
is no separate presence/direction weight or local direction quantization floor.

Source and sink transitions use the fitted endpoint axes as proxy endpoint
predictions; sink presence is one. Curvature uses the native tracer's shared
Lasagna-normal tangent-plane/normal-tilt split, with isotropic fallback for an
invalid normal. The former 45-degree lattice dead zone is removed: the curved
floating domain uses the greedy tracer's zero-degree free-angle default.
Cumulative history smoothness remains excluded from the DP state.

The command writes:

- `fiberlets.json`: every neighborhood pair, rejection/failure reason, objective
  breakdown, and successful base-coordinate polyline, plus per-successful-path
  length and loss/quality visualization metadata. It records the longitudinal
  2-voxel and transverse 0.5-voxel nominal spacings in prediction and base
  coordinates.
- `fiberlets.obj`: one named successful polyline per group in base coordinates,
  with strict loss-density and relative-quality comments for napari.
- `fiber_presence_xy.{obj,mtl,png}`, `fiber_presence_xz.{obj,mtl,png}`, and
  `fiber_presence_yz.{obj,mtl,png}`: three independently loadable textured
  quads through the center of the complete selected anchor-cell region. Each
  grayscale PNG contains one pixel per stored prediction voxel; black is zero
  presence and white is one.

The central slice OBJ bundles remain standalone diagnostic artifacts; napari
loads the dense presence Zarr directly and does not consume them. Each slice
OBJ references only its matching MTL and PNG by basename. For an even crop
extent, the lower of the two central prediction voxels is selected
deterministically. The quad
lies on that voxel center and extends half a prediction voxel beyond the first
and last texture samples along both varying axes. Slices use presence directly
and remain defined where a decoded fiber direction is invalid.

Fiberlet quality compares loss per traced prediction voxel, not raw integrated
loss, so a longer trace is not automatically shown as worse. For successful
path `j`:

```text
density_j = total_loss_j / polyline_length_in_prediction_voxels
quality_j = (maximum_density - density_j) /
            (maximum_density - minimum_density)
```

The normalization population is all successful scored paths in that artifact.
Equal densities all receive quality one. Quality is therefore relative to the
current output, not model confidence and not an acceptance threshold. The OBJ
comments and JSON retain raw total, density, bounds, formula, and quality;
display color is selected in napari. Smoothness terms remain part of total loss
even though their existing per-turn integration differs from the edge-
integrated alignment term.

This density is useful for ranking paths generated with the same parameters,
but it is not a calibrated correctness score. Its components have configured
weights and different integration rules, and it changes with prediction and
normal availability, lattice discretization, and the search corridor. The
artifact-relative min-max quality is additionally sensitive to outliers and to
which other paths happen to be present, so values and colors are not directly
comparable across runs. Since every feasible path is currently accepted, a
high displayed quality does not establish that a fiberlet is correct.

For interactive 3D inspection of the dense presence prediction, the napari
viewer accepts either the OME-Zarr pyramid root or a specific array level and a
base-coordinate crop:

```bash
vesuvius.view_fiber_presence \
  /path/to/fiber_presence.ome.zarr \
  --crop 13568,20224,18112,512,512,1000 \
  --anchors /path/to/anchors.obj \
  --paths /path/to/fiberlets.obj
```

The first OME dataset is the default finest level; use `--level 4` to select a
different one. The viewer passes a lazy Dask crop to napari, displays it in base
coordinates using the OME scale and translation, and initially uses napari's
`HiLo` colormap with attenuated maximum-intensity projection. It supports local
OME-Zarr v2 stores only. Install the `vesuvius[gui]` optional dependencies to
run it. The optional anchor and path OBJs are parsed as the line-only artifacts
written by `vc_fiberlets`, converted from base XYZ to napari ZYX, filtered by
group bounding-box intersection with the crop, and added as independently
toggleable line layers. Fiberlet paths expose `trace_loss_total`,
`loss_per_prediction_voxel`, and `relative_quality` as per-shape features. A
`Path colormap` selector maps quality over fixed `[0,1]`; it defaults to red-
yellow-green and offers napari's available colormaps. Changing it affects only
display state. The same dock provides six base-coordinate
sliders and numeric inputs, one for each minimum and maximum crop-box face. The
resulting six GPU clipping planes are always applied to the presence volume,
anchors, and paths without changing or reloading the underlying Zarr crop. It
also provides runtime anchor and path width controls; slider changes are
applied on release because changing shape width retriangulates the line meshes.

Slice output is enabled by default for crop inspection. Use `--no-slices` for
large or production path runs; this removes all three OBJ/MTL/PNG bundles from
the output directory. The default export rejects more than one million total
texture pixels instead of producing unexpectedly large files.

`--stats` prints retained anchors, candidate pairs, pre-DP rejections,
searched-but-unscored failures, scored paths, accepted fiberlets,
min/mean/max integrated total loss for all scored and accepted paths, and
min/mean/max accepted loss per prediction voxel. Rejected
endpoint pairs and failed searches have no path score and are counted as
unscored, never as zero. Empty score ranges print `n/a`. There is currently no
quality cutoff, so every scored path is accepted and the two score ranges are
expected to match. It also reports whether slice output is enabled and the
number of emitted slice pixels.

Each fiberlet OBJ group writes every adjacent path edge as an explicit two-
vertex `l a b` element. Path MTL/material output is not supported; rewriting an
output directory deletes a stale `fiberlets.mtl`. JSON and OBJ replacements are
individually atomic, but the pair is not transactional.

Fiberlet path visualization artifacts are experimental and parsed strictly by
the napari viewer. Regenerate older `fiberlets.obj` output that lacks the
quality report comments or per-group metrics; there is no repair or
compatibility path. Obsolete `mtllib` or `usemtl` records are rejected.

## Dense-fiber failure replay

`fiberlet-replay` builds anchors and one canonical fiberlet graph in a tube
around the selected reference interval, starting at the first control point.
It then runs two independent evaluators over exactly that interval: the regular
native 3D greedy tracer and the fiberlet graph tracer. The interval reaches the
reference end by default; `--length N` limits it to `N` base voxels and clamps
an oversized request at the reference end.

```bash
volume-cartographer/build/bin/vc_fiberlets fiberlet-replay \
  /path/to/fiber.lasagna.json \
  /path/to/reference-fiber.json \
  /tmp/fiberlet-replay \
  --normal-manifest /path/to/lasagna.lasagna.json \
  --beam 16 \
  --lookahead 3 \
  --length 4096
```

Both evaluators use the same monotone exact reference matcher. `--fail 20`
means a 20-base-voxel radius along the local Lasagna surface normal and an
80-base-voxel radius in its full 2D tangent plane; this tangent plane is not the
learned fiber direction. For error vector `d` and normalized local normal `n`,
the evaluator computes
`dn = abs(dot(d,n))`, `dt = sqrt(length(d)^2-dn^2)`, and
`threshold_error = sqrt(dn^2+(dt/4)^2)`. It fails only when
`threshold_error > --fail`; exact equality is accepted. The existing Euclidean
nearest-reference match remains authoritative.

The normal is sampled at that matched reference point after converting base
coordinates to the normal manifest's working scale. Invalid, missing,
non-finite, or zero-length normals use the old Euclidean threshold without the
4x relaxation. Fiberlet reseeding uses Euclidean `4T` only as a broad phase and
then applies the exact same evaluator, so accepted seeds and route samples have
one consistent region.

A failure ends only that evaluator's current segment. It restarts from the
authoritative reference point and fitted forward tangent at a strictly advanced
arc, then continues to the selected end. Native
termination, graph exhaustion, and absence of an admissible graph seed are
typed failures rather than silent completion. Fiberlet failures complete the
containing graph edge before reseeding. If the selected end lies inside an edge,
only samples through that bound are retained and the segment is explicitly
marked `terminal_partial_edge`, with no terminal anchor. Reset jumps are stored
as separate segments and are never drawn as trace geometry.

After graph construction, greedy and graph evaluation run concurrently over
immutable shared reference/graph state. Each command-line failure record
contains tracer, local index, reason, reference arc and interval fraction,
Euclidean/normal/tangential/threshold errors, threshold ratio, local-normal
validity, and both current failure counts. Matches and failures persist these
same explicit fields; records without an evaluated point use nulls. The strict
writer recomputes their geometric and numeric consistency before publication.
Console arrival order is
diagnostic only. Publication sorts visualization identity by reference arc,
tracer, and tracer-local index.

The graph retains every successful fiberlet over the selected reference tube;
failure-local graphs are not used for evaluation. Fiberlet curve volume samples
use the complete globally deduplicated coordinate union and `--batch` only
limits coordinates per sampler call, as described above. The final summary
reports both evaluator failure counts and confirms reference fraction one for
both.

The root manifest contains one `threshold` descriptor with the normal radius,
fixed factor and tangential radius, strict comparison, and invalid-normal
policy. Greedy and fiberlet descriptors are generated from it. The earlier
unpublished ambiguous replay keys `error_base_voxels` and `error_ratio` are not
supported.

By default the command publishes only the strict version-2 whole-run bundle.
`--vis` additionally extracts a local tube for every failure and requires
`--volume /path/to/ct.ome.zarr/2`. The path must name the concrete uint8 3D
Zarr array/group to render, not the OME-Zarr pyramid root. Choose a group whose
chunks are fully stored locally. The producer finds that group in its parent
OME-Zarr `multiscales` metadata and uses the declared coordinate transform to
map base-volume trace coordinates into group voxels. The source group is opened
and validated before replay extraction, so no visualization is published when
it is missing, is not advertised by the parent OME-Zarr, or has the wrong type.
`--along 128` then selects the reference arclength before and after that failure
and `--radius 64` selects the Euclidean tube radius. These controls affect only
visualization extraction; the evaluators always traverse the selected
comparison interval. Every local visualization contains its own anchors,
anchor stages, fiberlets, graph, cropped evaluator segments, reference, and
failure marker. It also contains three self-contained sheet-aligned textured
strip triples: `replay/reference_strip.{obj,mtl,tif}`,
`replay/greedy_strip.{obj,mtl,tif}`, and
`replay/fiberlet_strip.{obj,mtl,tif}`. They are built directly by the existing
`buildLineViewSurfaces()` default path, with the exact trace points as the
longitudinal samples and the standard 21-row line surface. Trace resets remain
separate surface components with no faces between them. Each component is
rendered by the same fine-to-coarse helper used by `vc_lasagna_line_probe`.
The sampling grid is selected automatically from each surface's arc extent in
the chosen Zarr group's voxel coordinates: endpoints are retained and adjacent
texels are at most one group voxel apart. Disconnected images are packed into
one padded grayscale atlas per trace type by transforming the existing
textured-mesh UVs; each tile's replicated one-pixel border prevents
interpolation into a neighboring component.
An empty trace type gets an empty OBJ, MTL, and 1x1 uncompressed TIFF. The local
manifest records the selected group path, its actual base-to-group scale/offset
transform and shape, native-grid contract, and source provenance. No replay-specific
renderer, mask, uint16 conversion, or PNG compatibility path is retained.

For example, generate replay visualizations with:

```bash
volume-cartographer/build/bin/vc_fiberlets fiberlet-replay \
  /path/to/fiber.lasagna.json \
  /path/to/reference-fiber.json \
  /tmp/fiberlet-replay \
  --normal-manifest /path/to/lasagna.lasagna.json \
  --vis \
  --volume /path/to/ct.ome.zarr/2
```

For the current sparse Paris4 CT store, group `/2` is the first fully stored
scale and must be passed directly as shown above.

The same `--vis` run also writes `fiber_replay.jpg` for immediate inspection,
even when neither evaluator fails. It contains the selected reference fiber's
existing VC3D top strip and side strip, with the reference centerline in
yellow, the regular greedy trace in red, and the fiberlet trace in cyan. Trace
reset segments are disconnected. The image uses the same concrete CT group,
default line-view surfaces, and shared fine-to-coarse renderer as the
per-failure strips, but requests an 8x render scale from that renderer for
detailed inspection. This does not resize or change the native-resolution
per-failure OBJ/MTL/TIFF artifacts. Greedy failures are marked by three-pixel
vertical red bands at the pre-reset error arc; fiberlet failures use cyan, and
coincident bands are magenta. The later reset seed is intentionally not marked.
An explicit `--length N` limits this JPEG
to that same selected `N`-base-voxel interval; without `--length`, it covers the
remaining reference fiber. Long 8x strips are split at equal reference-arc
fractions into at most 32,000-column ranges and stacked as labeled panels in
the same JPEG; the top and side ranges are mapped independently so neither is
resampled or loses columns.

The immutable copy is `runs/<content-hash>/replay/full_strip.jpg`; the root
`fiber_replay.json` records its hash, selected arcs, reference-point count, CT
group transform, full unwrapped top/side dimensions, 8x scale, marker
semantics, exact panel ranges, layout, and colors. The stable
`fiber_replay.jpg` is direct-inspection output, not a napari replay manifest.
The separate per-failure OBJ/MTL/TIFF artifacts and their direct manifests are
unchanged.

Each run is published under `runs/<content-hash>/`; only after all requested
generations exist is `fiber_replay.json` atomically replaced. The root stores
the two scale bindings, requested and forced-effective trace configuration,
requested/effective interval metadata, the exact selected reference geometry,
complete segmented greedy and fiberlet results, failure arrays/counts, and an
ordered visualization report with paths and hashes. For every failure, `--vis`
also atomically publishes a stable directly openable alias named
`fiber_replay_visualization.<tracer>.<failure-index>.json` and prints its
absolute path. These aliases point into the current immutable generation and
are the viewer inputs. The root does not store the external presence-Zarr path.
Strict version-1 single-visualization replay files remain directly loadable.

Load the bundle and the independently selected presence Zarr with:

```bash
python -m vesuvius.scripts.view_fiber_presence \
  /path/to/fiber-presence.ome.zarr \
  --replay /tmp/fiber-replay/fiber_replay_visualization.greedy.000000.json
```

Replay mode accepts one direct visualization manifest and has no index
argument. Passing the aggregate root reports a directly usable manifest path,
or requests regeneration with `--vis` if none exist. It rejects manual
crop/anchor/path arguments and verifies the external presence-Zarr shape/scale,
artifact paths, hashes, strict geometry/UV/material bindings, stored CT texture values, and manifest
identity. Reference, segmented greedy trace, segmented fiberlet trace, failure
marker, anchors, stages, fiberlets, and presence are separate toggleable layers.
The established clipping, radius, width, size, and path-quality controls apply
to the selected local generation.

The three hidden grayscale `reference CT strip`, `greedy CT strip`, and
`fiberlet CT strip` Surface layers read their values from the hashed TIFF atlases
referenced by their hashed OBJ/MTL artifacts and share a p1/p99 display range.
Napari has no UV-texture surface path, so the adapter bilinearly tessellates the
validated OBJ surface to the stored native texture grid and displays every
texel once. The viewer neither accepts a CT
volume argument nor opens the provenance path in the manifest. Older direct
visualization manifests without the all-three strip extension still open and
do not synthesize these layers. The unpublished geometry-only and vertex-RGB
strip formats are rejected and must be regenerated.

`Reload artifacts` rereads the same stable direct manifest, which is atomically
updated by a later replay publication. It does not reload the presence Zarr and
preserves display state. Replacement strip geometry, faces, UVs, and stored values
are applied from the new artifacts while preserving the three Surface layers'
visibility and other display settings. Incompatible or malformed replacement
output is rejected before replacing any layer.

This remains an overcomplete diagnostic collection. There is no path-quality
cutoff, degree selection, overlap deduplication, extension, H/V, or winding
assignment.

## Extraction benchmark

The benchmark runs the same local tube anchor and on-demand fiberlet extraction
used by replay, without writing artifacts. The interval starts at the first
control point and extends to the end of the reference by default. An explicit
`--along` limits the interval to that many base voxels:

```bash
volume-cartographer/build/bin/vc_fiberlets benchmark \
  /path/to/fiber.lasagna.json \
  /path/to/reference-fiber.json \
  --normal-manifest /path/to/lasagna.lasagna.json \
  --threads 32
```

Its final `fiberlet_extraction_benchmark` row reports the exact reference arc
interval and tube radius; cell, anchor, candidate, and successful-path counts;
anchor and path stage rates; aggregate/mean/peak sampled voxels; estimated peak
batch bytes; evaluated DP-node rate; and separate candidate-generation,
sampling, search, and total wall times. Use identical manifests, fiber, options,
build type, and interval for before/after performance comparisons.
