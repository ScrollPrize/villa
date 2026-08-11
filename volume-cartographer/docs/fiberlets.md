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
window and half-cell longitudinal limit around their sign-aligned average axis.
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
side. `--window` sets the transverse refinement/NMS radius in base voxels and
defaults to one cell side. The axial slab defaults to 1.5 cell sides, the NMS
longitudinal limit to half a cell side, and the NMS angle to the merge angle.
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

Target cells are selected from the integer shell
`radius-0.5 <= length(cell_offset) < radius+0.5`; the initial radius is four.
Endpoint axes must agree with their chord within 45 degrees. Every surviving
pair is solved independently, so an anchor can currently participate in many
paths.

Candidate generation finishes before any path search. The path stage computes
the rectangular enclosing ZYX box of every searchable Hermite corridor and its
virtual endpoint attachment voxels, clips that box to the prediction grid, and
materializes every stored prediction and Lasagna normal in the box exactly
once. Candidate workers then read this immutable dense scoring volume; they do
not decode chunks, interpolate normals, or launch nested sampling workers.
Corridors may leave the selected anchor-cell box, so the preload box can extend
beyond the original crop. This intentionally favors speed for the current
small test crops and can consume substantial memory on large stored-prediction
regions.

Replay mode instead enumerates the union of admissible corridor bounding boxes,
intersects every integer prediction voxel with the replay tube, sorts the voxel
keys, and samples each key once into an immutable sparse lookup. The standalone
`paths` command retains the dense rectangular preload described above.

`--threads` controls both the one-time preload batch and the subsequent fixed
candidate worker pool. Candidate results are written into their original
deterministic slots, and timing/worker values are not serialized. Runtime output
reports the dense voxel count, estimated peak preload working bytes, worker
count, and candidate-generation/preload/search times.

During candidate solving, `vc_fiberlets` writes a monotonic progress line to
stderr about once per second. It reports completed/total searches, percentage,
search-only elapsed seconds, candidates per second, and ETA seconds. A final
100-percent line with zero ETA is always emitted; a run with no searchable
candidates reports `0/0`, 100 percent, and zero ETA. Progress is operational
output and does not affect stdout or artifacts.

`--corridor-radius` is measured in base voxels. If omitted, it defaults to one
anchor-cell width. Cell radius and shell width remain dimensionless cell-lattice
parameters.

The path graph contains only integer stored-prediction voxels. Exact sub-voxel
anchors are virtual endpoints connected through nearby integer voxels. A
cubic-Hermite reference bounds the corridor, and 26-neighbour moves must have
strictly positive chord progress. DP state retains the incoming move, allowing
one-step curvature without a cumulative history state.

Valid-data scoring uses the regular native tracer's multiplicative local
alignment loss. It multiplies presence by six positive-clamped dots among the
incoming and outgoing steps and the sign-aligned current and next prediction
axes, then charges `1-score`. This jointly penalizes trajectory turns,
prediction discontinuities, and trajectory/prediction disagreement. The DP
multiplies that loss by lattice-edge length so axial and diagonal integration
remain comparable. There is no separate presence/direction weight or local
direction quantization floor.

Source and sink virtual edges use the fitted endpoint axes as their endpoint
predictions; sink presence is one. Curvature uses the native tracer's shared
Lasagna-normal tangent-plane/normal-tilt split, with isotropic fallback for an
invalid normal. Its 45-degree free angle remains the integer-lattice adaptation.
Cumulative history smoothness remains excluded from the DP state. Invalid
destination predictions pay only the finite default cost of 4 per prediction
voxel plus curvature, allowing short gaps to be bridged. On leaving a gap, the
incoming step substitutes for the unavailable current prediction.

The command writes:

- `fiberlets.json`: every shell pair, rejection/failure reason, objective
  breakdown, and successful base-coordinate polyline, plus per-successful-path
  length and loss/quality visualization metadata.
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

`fiber-replay` starts the regular native 3D tracer at the first control point
of a strict VC3D fiber JSON. It forces the native tracer to greedy mode and
compares every committed trace point with the dense stored fiber line:

```bash
volume-cartographer/build/bin/vc_fiberlets fiber-replay \
  /path/to/fiber.lasagna.json \
  /path/to/reference-fiber.json \
  /tmp/fiber-replay \
  --normal-manifest /path/to/lasagna.lasagna.json
```

The command reports start/completion and elapsed time for trace setup, tracing,
tube selection, anchor extraction, fiberlet tracing, and publication. Anchor
fitting reports selected-cell and NMS-context progress with ETA once per second;
fiberlet DP reports candidate-search progress and ETA with the same cadence.

All distances are base voxels. The correspondence cursor advances monotonically.
Each step predicts one nominal trace step along the reference, then computes the
exact closest point in the bounded forward interval. `--match-refine 1` permits
zero through two nominal steps of forward reference advance. The first distance
strictly above `--fail 20` is the failure point; `--after 100` retains that many
additional native greedy steps. Exhaustive statuses distinguish a complete or
truncated failure, reference completion without failure, and native termination
before failure.

For failures, `--along 512` selects that much dense-reference arclength on each
side and `--radius 128` defines an exact Euclidean tube including endpoint caps.
Anchor cells are selected when their prediction-sample footprint intersects the
tube. Refined anchors are rejected outside the tube before NMS. Fiberlet virtual
endpoints and integer DP nodes are also tube constrained, and their scoring data
uses the sparse replay preload. No central textured slice OBJ is produced.

Each run is published under `runs/<content-hash>/`; only after the complete
generation exists is `fiber_replay.json` atomically replaced. The bundle stores
the two independent trace/canonical scale bindings, requested and forced-effective
trace configurations, matching diagnostics, reference/trace/failure geometry,
tube cells, crop, relative artifact paths, and content hashes. It deliberately
does not store the external presence-Zarr path.

Load the bundle and the independently selected presence Zarr with:

```bash
python -m vesuvius.scripts.view_fiber_presence \
  /path/to/fiber-presence.ome.zarr \
  --replay /tmp/fiber-replay/fiber_replay.json
```

Replay mode rejects manual crop/anchor/path arguments and verifies the external
Zarr shape and scale. It strictly validates bundle paths, containment, hashes,
status-specific artifacts, and OBJ equality with authoritative JSON geometry.
Reference, greedy trace, failure, anchors, fiberlets, and presence are separate
toggleable layers. Cell centers and their retained-anchor displacement lines are
also separate layers loaded from `anchor_cells.obj`, so zero-anchor cells remain
visible. The five anchor-stage JSON files are validated as one lineage chain and
shown as distinct colored line layers. The NMS layer is visible initially;
earlier stages and the duplicate final-anchor OBJ layer start hidden. Layer
names show candidate/rejection counts, and per-shape features expose lineage,
support/coherence, transition reason, tested threshold, and NMS suppressor.
The six crop controls clip every layer and the dock provides width or
size controls for the diagnostic geometry. For failure replay,
the viewer rasterizes both the reference and complete greedy replay trace at the
displayed Zarr level and computes one base-voxel distance transform to their
union. The `Presence radius` slider applies a hard lazy mask to that distance
field and defaults to the extraction-tube radius used for anchors and fiberlets.
Changing the slider threshold does not recompute the EDT.

The independent `Anchor radius` slider filters the final anchors, all five
anchor-stage layers, cell centers, and center-to-anchor refinement offsets. It
uses exact base-voxel distance to the reference/greedy-trace union and also
defaults to the extraction-tube radius. Anchor glyphs use their center, while a
refinement offset uses its anchor endpoint. Items at exactly the selected radius
remain visible. The filter changes only line alpha or point visibility, so
layer geometry, stage features, ordering, clipping, widths/sizes, layer
visibility toggles, and the full counts in layer names remain unchanged.
Fiberlet paths and the presence mask are independent of this control.

Use `Reload artifacts` after regenerating the replay output. The viewer rereads
the original root `fiber_replay.json`, so atomic publication automatically
selects the newest hashed generation. It updates reference, greedy trace,
failure marker, final anchors, every anchor stage, cell centers, refinement
offsets, and fiberlets in their existing layers. Empty layers are retained and
can become populated, or vice versa, during reload.

Reload does not reopen or reload the presence Zarr. It retains the exact lazy
source crop and recomputes only the reference-dependent EDT, lazy presence mask,
and anchor distances. Current radii, clipping, visibility, widths/sizes, path
colormap, layer order, and volume rendering remain unchanged. The replacement
must retain the fiber-prediction manifest hash, prediction shape/scale, crop,
extraction radius, and five-stage contract. A mismatch or malformed/hash-racing
publication is reported without replacing the current display; change the Zarr
or crop by restarting the viewer.

This is an overcomplete diagnostic collection. There is no path-quality cutoff,
degree selection, overlap deduplication, extension, H/V or winding assignment,
or final graph construction yet. Inspect the OBJ on a small crop before using
the output for later graph work.
