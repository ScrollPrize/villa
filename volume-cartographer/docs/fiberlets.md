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

Anchor and fiberlet extraction use float32 end to end: observations,
configuration used by solver math, component and peak state, retained anchors,
fiberlet domain/frame/interpolation geometry, path costs, DP state, candidate
points, graph geometry, diagnostics, and serialized numeric values. Integer
lattice/index/count state remains integer. External reference fibers, replay
distance calculations, cold volume-scale metadata, and elapsed/process timing
remain double precision and convert only at subsystem boundaries. Version-1
and version-2 JSON artifacts carry untyped JSON numbers and load into this
float representation without a schema change. The shared prediction and normal
sampler interfaces remain double-valued for other consumers; extraction
normalizes, range-checks, and narrows their results once at the sampling
boundary and retains only float tiles afterward.
Prediction-grid extents must not exceed `2^24` voxels on any axis so every
integer voxel coordinate remains exact in float32. Artifact and OBJ writers
also reject base-coordinate scaling that would overflow float32.
On the canonical 32-thread, 5,000-base-voxel Paris4 replay, the complete
float32 representation reduced median command wall time from 9.22 to 8.96
seconds, total CPU from 202.89 to 199.47 seconds, and peak RSS from 2,121,656
to 2,007,020 KiB. Replay failures were unchanged and repeated float32 runs
produced identical artifacts.

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
selects the largest positive
`g_ik q_i abs(d_i dot u_k)^2`, with stable component-index ties. Unusable and
zero-evidence samples remain unassigned.

Assigned projective angular residuals `1-abs(d_i dot u_k)^2` are summarized by
a deterministic 256-bin weighted median/MAD histogram using `g_ik q_i` mass.
The cutoff is the larger of median plus three MADs and a five-degree angular
floor. Trimming is optional and capped at 20% of assigned mass; coherent data
therefore retains every sample. The complete cutoff bin is retained. The
limits are configurable with `--robust-max-trim`,
`--robust-mad-multiplier`, and `--robust-min-angle-deg`.

Each direction update is the principal eigenvector of retained
`g_ik q_i d_i d_i^T` and is installed directly, without angular interpolation
or angular line search. A non-unique retained tensor removes that component.
Supported close components are not merged before refinement; cross-cell NMS
remains responsible for genuine duplicates.

The retained aligned-evidence centroid is projected onto the plane through the
cell pivot normal to the updated direction and clamped to the local transverse
window. The previous and proposed positions are evaluated with the retained
spatial objective using deterministic position-only backtracking at fractions
`1, 1/2, ...` through the first displacement at or below the configured
peak-grid step. The first strict improvement is accepted; otherwise the
projected baseline is retained.
Every sampled lattice site contributes the same geometric denominator term
while only retained assigned evidence contributes positive signal. Because the
denominator does not depend on a site's presence, direction, assignment, or
trim state, rejected observations cannot create attractive or repulsive holes
in the normalization.

Each additional outer pass recomputes competitive assignments and robust
inliers from the preceding direction and position update. The default budget
is one pass.

Compact extraction records retained observation indices per component while
materializing each robust cutoff. The following centroid update traverses only
those indices in original logical order; expanded/public fitting retains its
defensive full-support scan. Worst-case index storage participates in worker
memory admission.

Compact extraction also evaluates configured direction/presence eligibility
once per unique sampled voxel. Proposal, centroid, owned-cell initialization,
and peak fitting reuse that cached result; direct/public observations retain
defensive validation.

The finite support bounds collected during that same refinement preparation
are carried into peak Voronoi ownership; peak setup does not rescan the support
to rediscover identical bounds.

For full-halo production cells, those bounds come directly from the fixed
support stencil translated by the cell sample origin. Clipped/general ranges
derive bounds from their actual compact indices.

The following direction-conditioned peak response evaluates its transverse and
axial Gaussian weights with `exp` after their exact support cutoffs, matching
the broad direction fit and final support calculations.

Peak preparation constructs denominator geometry for every in-support lattice
site, but direction validation and positive-evidence fields are evaluated only
after the site's final retained component assignment is known.

**Anchor quality knob:** `--maximum-iterations` trades extraction speed for
additional robust reassignment and refinement. Increase it above `1` when
nearby or crossing fibers need a second chance to separate after the first
geometry update. This can materially change anchor positions, directions, and
the retained anchor population; it is not merely a convergence or debugging
setting. Two passes cost about 18% more anchor wall time on the canonical
Paris4 replay and remain the first quality-oriented setting to try.

Refinement deliberately does not seek exact hard-assignment equality because
samples at histogram or component boundaries can flicker without a meaningful
geometry change. An earlier exit is allowed when axis and position updates are
already below their geometric tolerances. Empty, degenerate, and
below-threshold components are discarded independently. The output is zero,
one, or two anchors per cell before duplicate suppression.

The retired pre-refinement merge objective is not used. Its fields remain in
anchor artifacts for compatibility. New robust artifacts use schema version 2;
the loader retains strict version-1 support with the original parameter set.
`--nms-angle-deg` controls duplicate-axis compatibility in downstream NMS.

Finally, local-maximum NMS suppresses cross-cell copies of the same anchor.
The two supported components from one cell never suppress each other.
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
defaults to 10 degrees.
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
For the common case where all eight positive-weight interpolation corners lie
in one sparse `16^3` bitmap page, collection resolves the page once and sets
the eight local bits as one cell operation. Integer-coordinate, volume-edge,
and page-crossing cells retain the general per-corner path.

Local corridor membership is the union of the two continuous float32 segment
capsules incident to the node's curved-domain layer. Points strictly inside the
layer center's transverse-radius circle are admitted directly; boundary and
outer-square points test the incoming and outgoing segments. This keeps the
search tube local to each layer and prevents a distant bend of the same
candidate from admitting shortcut nodes.

`--batch` is a coordinate-call limit, 65536 by default. It partitions only
consecutive ranges of the global unique union. The path stage completes all
prediction calls, then all normal calls, materializes every prepared endpoint
score, and retains immutable prepared scoring voxels plus their page index while
running every DP candidate in parallel. Interior node scores are interpolated
at first search access and cached by candidate-local node index. Increasing
`--batch` changes
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
The layered graph is acyclic. States 0 through 8 encode the incoming transverse
step and state 9 is source-only. The predecessor packed key and incoming
geometry are derived from that state; reconstruction retains one predecessor-
state byte per node/state. Float32 cumulative costs roll through only the
current and next interior layers because alignment and curvature need no older
cost state. Source and sink transitions remain separate.

Each interior geometry node occupies 16 bytes: one checked row-major `uint32`
key and three `float32` prediction coordinates. On a lazy-cache miss, fiber and
normal axes pass through the same +Z compact `nx/ny` encoding as Lasagna and
presence uses the native byte scale. No per-node reason string or
interpolation-address object is retained. This re-quantization is intentional
for the experimental fiberlet objective; exact anchor endpoints use the same
float32 geometry representation. DP stores normalized solve-local scoring
records only for requested nodes. Each reached node resolves its at most nine
outgoing neighbors and edge geometry once, reusing them across incoming states;
no candidate-wide edge table is materialized.

Presence is trilinearly interpolated. Fiber directions are unoriented: the
native voxel axes are validated and normalized once into compact float32
symmetric outer products. Positive-weight corners accumulate those tensors,
then an analytic symmetric 3x3 resolver finds the unique principal axis. A
missing top-eigenvalue gap remains ambiguous and invalid. When the gap is clear
but closed-form eigenvector reconstruction fails its residual bound, the
existing iterative Jacobi resolver is used as a numerical fallback. This
preserves antipodal axes without sign cancellation; an invalid required corner
or ambiguous tensor invalidates the destination. Normals use the same
interpolation, but invalid normal data keeps the existing isotropic curvature
fallback rather than rejecting the path. Interpolation weights, tensor sums,
and principal-axis calculations stay float32 throughout fiberlet extraction;
only the external normal-sampler call uses its existing double-coordinate API
before immediately narrowing the returned sample.

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
The exported validating/prepared scoring functions and the prepared DP hot loop
share one source-private inline implementation of alignment, smoothness, and
metric composition. This preserves the public API and exact equations while
avoiding interposable shared-library calls for every interior DP transition.
The shared smoothness implementation evaluates isotropic curvature only when
an invalid normal or a degenerate projected tangent actually requires that
fallback. Normal-aware transitions with valid projected tangents therefore
avoid an otherwise unused inverse-cosine evaluation without changing any
returned cost. Each reached node also prepares the outgoing edge's destination
normal, projected candidate tangent, and candidate normal angle once, then
reuses those exact intermediates across every incoming DP state. The same edge
descriptor owns its sign-oriented candidate prediction axis and individual
candidate-only alignment factors. Each reached incoming state similarly
prepares its sign-oriented current prediction axis and previous/current factor
once. Only the four genuinely pair-dependent alignment dots remain in the
transition loop. Interior DP stores valid outgoing alignment inputs in a
compact fixed-capacity structure-of-arrays batch and evaluates those four dots
across the batch. Invalid outgoing slots are omitted, and valid lanes retain
ascending transition-slot order. Factor multiplication order is unchanged;
smoothness, accumulated component costs, strict-less comparisons, destination
updates, and backpointers remain scalar in their original order. Public and
non-DP metric callers prepare the same private descriptors on demand through
the shared scorer, while the standalone alignment API keeps its raw caller-
oriented semantics. There is no separate scoring equation.

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

`fiberlet-replay` exposes one canonical fiberlet graph in a tube around the
selected reference interval, starting at the first control point. By default,
anchor and fiberlet chunks are generated as the graph is traversed and reused
from local sparse cache roots; the complete corridor graph is not retained in
memory. It then runs two independent evaluators over exactly that interval: the
regular native 3D greedy tracer and the fiberlet graph tracer. The interval
reaches the reference end by default; `--length N` limits it to `N` base voxels
and clamps an oversized request at the reference end.

```bash
volume-cartographer/build/bin/vc_fiberlets fiberlet-replay \
  /path/to/fiber.lasagna.json \
  /path/to/reference-fiber.json \
  /tmp/fiberlet-replay \
  --normal-manifest /path/to/lasagna.lasagna.json \
  --beam 16 \
  --beam-step-distance 48 \
  --lookahead-distance 384 \
  --length 4096
```

`--beam-step-distance D`, `--lookahead-distance H`, and `--prune-distance P`
use base voxels. The defaults are `D=48`, `H=384`, and `P=48`.
`--search-width K` defaults to zero and runs the exact cost-bounded lookahead.
Full width refers to the untruncated lookahead search; replay still retains at
most the configured positive `--beam` width and enforces exact cycle rejection
and the per-decision state limit. Beam width defaults to 16 but has no fixed
upper policy limit. A positive `K` opts into approximate intermediate
pruning, must be at least the final beam width, and uses `--prune-distance P`.
The prune distance is ignored in exact mode.

`--cost-mode fiberlet` is the default. It uses the stored whole-fiberlet and
join costs directly from the segment seed through the common lookahead horizon.
Only the horizon-crossing fiberlet is prorated by length; an entering join is
charged in full. This mode does not load route cost profiles or walk the
integration grid. `--cost-mode stepped` explicitly enables the experimental
subsegment integration described below. The stepped-only `--cost-weight`,
`--cost-delay`, `--cost-step`, and `--cost-profile-weight` options are rejected
unless that mode is selected.

Cache-backed replay may apply the existing staged route reduction before graph
tracing by passing one or more ordered
`--stage SIDE,OFFSET_X,OFFSET_Y,OFFSET_Z` options. With no stage, replay is
unchanged. Replay stage offsets are global base-volume coordinates, normalized
modulo `SIDE`; they are not relative to the selected fiber interval. For
example:

```bash
volume-cartographer/build/bin/vc_fiberlets fiberlet-replay \
  /path/to/fiber.lasagna.json \
  /path/to/reference-fiber.json \
  /tmp/fiberlet-replay \
  --normal-manifest /path/to/lasagna.lasagna.json \
  --stage 512,0,0,0 \
  --stage 256,128,128,128
```

The final stage selects every complete global box intersecting the replay
corridor. Earlier stages are expanded backward by the dataset's declared
maximum Fiberlet endpoint reach, and source generation covers the resulting
support union. Filter boxes, generation cells, and persistent storage chunks
are separate globally anchored layouts and may use unrelated sizes. The
filtered graph is held in invocation-local overlay layers and deleted after
replay; only ordinary unfiltered anchor/Fiberlet source chunks remain in their
persistent caches. Graph traversal still uses the requested reference
corridor, so the expanded support cannot seed or traverse outside the original
replay domain. `--join-angle`, `--cost-profile`, and `--max-states` configure
the filter passes. Filtered replay requires the on-demand cache path and is
therefore incompatible with `--eager-graph`.

Every successful fiberlet stores one total-cost density per emitted route
segment next to its route lattice. Density is selected DP segment cost divided
by prediction-voxel segment length. It uses the fixed nonlinear `uint16`
mapping `round(65535 * sqrt(clamp(density / 256, 0, 1)))`; decoding squares
the unit-range code value and multiplies by 256. This codec mapping is the only
normalization involved. Decoded segment costs are used directly and are never
rescaled to make their sum equal the separately stored whole-edge cost.
Geometry and cost profiles have independent route-payload offsets. Prefixes
retain the five-component whole-edge cost used by committed-route diagnostics.

In stepped mode, `--cost-profile-weight A` linearly blends the density used by replay before
geometric weighting:

```text
effective_density = (1 - A) * fiberlet_average_density
                  + A * subsegment_density
```

`A` must be finite and in `[0,1]` and defaults to 1. The fiberlet average is
computed from the same decoded additive profile as
`sum(density * segment_length) / fiberlet_length`; it is not taken from, or
normalized to, the separately stored whole-edge cost. Consequently, every
complete unweighted fiberlet retains the same decoded total at all blend
values, apart from floating-point accumulation. Blend zero makes density
constant within each fiberlet while preserving geometric weighting across the
fiberlet. Blend one preserves the full stored subsegment profile and is the
stepped-mode default.
The option is replay-only and does not affect anchor or fiberlet cache identity.

Stepped lookahead ranking adds the authoritative unweighted route cost from the segment
seed through the current checkpoint to the decoded-profile cost from the
checkpoint through the common horizon. `--cost-weight W` is the geometric
weight per base voxel and
must be in `(0,1]`; it defaults to 1. `--cost-delay L` keeps weight equal to
one through the first `L` base voxels and then weights cost at checkpoint-local
base distance `s` by `W^(s-L)`; it must be finite and nonnegative and
defaults to zero, preserving immediate decay. A delay of 192 is half the
default 384-base-voxel lookahead and is the initial delayed-falloff experiment.
`--cost-step N` is the positive regular integration spacing in
base voxels and defaults to 16. The grid is anchored once at the checkpoint
and continues across fiberlet boundaries; a non-grid-aligned delay splits its
containing integration cell exactly. Partial interval cost comes from
linear interpolation of the piecewise-linear cumulative cost curve, and each
interval uses the weight at its midpoint. The same integration path is used for
all weights. At `W=1`, the decoded additive profile therefore reproduces its
own total apart from codec, interpolation, and floating-point rounding; no
special compatibility path or corrective scaling is applied. `--cost-step`
remains active at `W=1`, so small route sensitivity from those numerical effects
is possible, but material failure-count changes indicate a defect or a genuine
near tie. An entering join before the checkpoint belongs to the authoritative
prefix. A join at the checkpoint is included in the forward term with weight
one, and a join exactly at the horizon is excluded. Ranking's weighted scalar
is recorded separately from the unchanged unweighted component diagnostics.

On the Paris4 5,000-base-voxel validation interval, the pinned pre-profile
revision and repaired scorer all completed without a fiberlet failure. Five
serial hot-cache runs gave these timings; CPU is `user + sys` and p95 is the
nearest-rank sample:

| Mode | Wall mean/p50/p95 | CPU mean/p50/p95 | Peak RSS |
| --- | --- | --- | --- |
| Pinned `64e5341` | 0.558/0.550/0.640 s | 15.684/15.480/18.550 s | 96.4 MiB |
| Repaired `W=1`, step 16 | 0.666/0.650/0.800 s | 18.992/18.800/23.260 s | 97.6 MiB |
| `W=0.99`, delay 0, step 16 | 0.642/0.630/0.710 s | 18.584/18.410/20.800 s | 97.1 MiB |
| `W=0.99`, delay 192, step 16 | 0.616/0.610/0.650 s | 17.076/17.430/17.800 s | 97.4 MiB |

The repaired `W=1` median wall and CPU times are respectively 1.18x and 1.21x
the pinned baseline. `W=1` step 8 and step 32 selected the same canonical
55-edge route on this interval. Both `W=0.99` modes also had zero fiberlet
failures; delayed falloff remains an explicit experiment rather than a default
quality change.

On the full Paris4 radius-768 hot-cache corridor, `W=0.99`, delay 192, and
step 16 produced 5/5/3/2/2 fiberlet failures for profile weights
0/0.25/0.5/0.75/1 respectively. Wall time was 21.47-22.63 seconds and peak RSS
was 1.18-1.23 GiB. With `W=1`, profile weight zero reproduced the two-failure
aggregate baseline at exactly the same failure arcs in 21.68 seconds. Profile
weight one retained the current five-failure behavior in 21.44 seconds.

A matched-terminal-weight sweep then tested whether more lookahead helps the
0.75 profile blend. Each run used delay `H/2` and
`W = terminal_weight^(1 / (H - delay))`, so horizon changes did not also change
the requested endpoint decay:

| Lookahead H | Terminal 0.10 | Terminal 0.25 | Terminal 0.50 |
| ---: | ---: | ---: | ---: |
| 384 | 4 failures, 22.11s | 2 failures, 21.13s | 4 failures, 21.00s |
| 512 | 4 failures, 31.90s | 4 failures, 36.88s | 4 failures, 37.18s |
| 768 | 3 failures, 110.70s | 4 failures, 147.12s | route-state limit, 36.07s |

These full-reference radius-768 hot-cache runs used beam 16, checkpoint step
48, exact search, integration step 16, and the existing one-million-state cap.
No larger horizon improved on the two-failure H384 control; H768 also increased
peak RSS from about 1.2 GiB to 1.4-1.9 GiB.

The two H384/T0.25 failures were then compared with constrained-radius runs at
the same objective settings. Radius 64 was not a usable correctness reference:
it failed at 41,744 and 48,159 base-voxel reference arc. Radius 32 followed the
reference through both radius-768 failure regions and stopped only with graph
exhaustion at 50,753, near the reference endpoint and below the distance
threshold. That full run took 29.11 seconds wall time and 927,040 KiB peak RSS.
It is a supervised reference-following baseline, not proof that its exact
discrete route exists in the wider graph; radius-dependent cell selection and
anchor NMS can change graph topology.

Two focused common-start replays isolate the persistent failures without the
earlier route and restart history:

| Start arc | Length | Radius 32 | Radius 768 |
| ---: | ---: | --- | --- |
| 40,500 | 1,600 | no fiberlet failure | failure at 41,744 |
| 41,744.240 | 1,400 | no fiberlet failure | failure at 42,747 |

At reference arc 41,289, a radius-32-like continuation was present in the wide
frontier at rank 7. The selected and
reference-following losses were respectively 0.283476 and 0.284211 per
prediction voxel. The reference-following route had lower weighted edge loss
(12.2647 versus 12.3026) but higher weighted join loss (1.5219 versus 1.3826),
and matched the constrained geometry to 0.28 base voxels over the clipped
lookahead. The selected candidate was 7.08 base voxels from that continuation.
A close continuation remained available at the next two checkpoints, then the
closest top-16 candidate was already 32.2 base voxels away at checkpoint 108.
At the reported checkpoint the selected and rank-7 routes share their committed
prefix and differ only in future lookahead, so this does not identify a causal
wrong commit.

In the second focused window both radii selected the same geometry through
reference arc 42,404. At checkpoint 66 the selected wide route was 13.52 base
voxels from the constrained continuation, and no closer candidate survived in
the top 16. Increasing the beam to 128 recovered a marginally closer 13.11-
base-voxel alternative only at rank 99: its loss was 0.235793 versus 0.231951
for the selected route. The alternative's weighted edge loss was slightly
lower (7.4989 versus 7.5524), but its join loss was 0.7827 versus 0.2075. Beam
128 therefore delayed the failure to 42,781 but did not remove it. Radius 32
changes the extracted graph and is not a same-graph correctness oracle, so
these costs are descriptive and do not prove that join weighting caused either
failure. A causal comparison requires the best reference-admissible route in
the same radius-768 graph, with identical seed and checkpoint history.

The two replay events exceed the configured Lasagna-normal ellipsoid even
though their normal and tangential components are individually below 20 and 80
base voxels. Their component pairs and normalized ratios are respectively
`19.134/38.976, 1.0736` and `10.618/68.473, 1.0072`, where
`ratio = sqrt((normal/20)^2 + (tangential/80)^2)`. A future evaluation can also
test bounded excursions that rejoin the admissible region within a configured
reference arc, but must report excursion length and severity rather than
silently suppressing persistent switches.

The focused collections can support offline tuning of objective-only values
when they retain stable logical route IDs, raw decoded subsegment profiles,
edge and join components, route geometry, checkpoint history, and cache/profile
fingerprints. Fixed candidates can then be relabeled by distance to the
reference and rescored for profile blend, decay, delay, integration spacing,
and explicit component multipliers. They cannot faithfully tune radius,
fiberlet generation, beam/checkpoint/lookahead policy, state limits, or any
setting that changes the candidate frontier or later checkpoint history; those
settings require focused replay, although the hot-cache focused runs above take
only about 0.2 seconds at radius 32 and 1.3-1.5 seconds at radius 768. These two
windows are tuning diagnostics and must not also be used as validation data.

Exact mode keeps each winning route through the full common lookahead horizon,
not just through the next checkpoint. The following decision extends those
retained routes only from their existing endpoints; a whole terminal fiberlet
that already crosses the new horizon is rescored there without expansion. One
multi-source priority frontier feeds one global top-`beam` completion set. Routes
are ranked by checkpoint-relative weighted exact-horizon loss and canonical persistent logical
identity, and only identical complete logical routes are deduplicated. Several
retained routes may therefore share the same checkpoint prefix and diverge
later in the lookahead.

The global cutoff appears only after `beam` distinct complete logical routes
are known. Its value is the worst retained weighted total, shared by all input beams.
Increasing the beam therefore delays cutoff activation and can substantially
increase exact-search work; the generated-state limit remains the hard bound.
Lower bounds equal to the cutoff remain eligible. Expansion uses a fixed-size
batch independent of thread count; workers produce ordered successors and the
coordinator merges them canonically before updating the cutoff and the one-
million-state decision budget. Queue ranking and cutoff maintenance use scalar
cost/path state plus persistent route handles, never materialized route or
point vectors.

After selection, only the best route's prefix through the complete fiberlet
containing `C+D` is reference-matched and committed. Its retained future suffix
is not inspected for a reference failure until a later checkpoint commits it.

The optional bounded search maintains whole-fiberlet histories and one shared logical
checkpoint `C`. Its first exact front is the next checkpoint `C+D`; subsequent
fronts are no farther than `P` apart and end exactly at `C+H`. At each
intermediate front it keeps the best continuation for each stable `C+D` prefix
before filling the remaining `K` slots by global score. At the final front it
keeps one continuation per actual next-checkpoint prefix and retains the best
16. This can discard a route that would become better later in the horizon;
that is the explicit bounded-search approximation.

Expansion between fronts is a uniform-cost label search. Labels are keyed by
the logical incoming directed fiberlet and a 0.5-prediction-voxel front-offset
bin. If multiple histories reach that state, only the lowest accumulated-cost
history survives; ordered logical arc IDs break exact-cost ties. The survivor's
visited-node set is used for later cycle rejection. This deliberate state merge
closes reconvergent paths instead of enumerating every history. Exact crossing
scores integrate the segment profile through the terminal fiberlet, and equal-bound
labels are processed before a width cutoff. Diagnostics report dominated
labels separately from cost-bound pruning and exact completed candidates. Once
stable prefixes exist, a worker searches for one prefix representative plus
only enough extra candidates to fill globally unoccupied slots; it does not
try to prove a local top 128 for every prefix.

Front ranking retains the authoritative unweighted prefix through the decision
checkpoint and adds the decoded weighted profile through each exact front. The
entering join of a front-crossing fiberlet is charged at its checkpoint-relative
weight and its edge profile is integrated only through the front. The complete
crossing fiberlet remains in route geometry and visited state. Stable logical
IDs, canonical worker merging, and one decision-wide state budget make output
independent of expansion thread scheduling.

In bounded mode, after pruning the checkpoint advances by `D`. Each retained history is
committed through the complete fiberlet containing the new checkpoint. A beam
may therefore commit no new fiberlet, one fiberlet, or several fiberlets during
one iteration, and its stored endpoint may lie beyond the shared checkpoint.
Routes that cannot reach the common lookahead horizon are excluded, and
exceeding the explicit per-decision state bound fails rather than changing the
search.

Replay failures are printed immediately in both compact-progress and `--stats`
modes. In compact mode, the failure line interrupts the progress bar and the
bar is redrawn afterward; `--stats` only controls the additional stage and
per-tracer progress diagnostics.

Only final reference-end or failure materialization may clip the displayed
route inside a fiberlet; this does not alter search costs or beam state. A
reference failure, graph exhaustion, or the selected end closes the persistent
population.

Replay prefix state is incremental. Logical route keys are canonical immutable
parent/arc nodes with exact lexicographic ordering, while physical histories
remain distinct. Visited anchors live in an immutable exact-key Patricia trie,
so advancing a checkpoint does not copy the accumulated cycle set. Reference
matching and Lasagna-normal threshold evaluation resume at the nearest already
evaluated physical-history ancestor and process only the newly selected suffix.
Decision scoring initializes the authoritative prefix from cumulative scalar
cost at the history immediately before the checkpoint, then walks only the
checkpoint-to-horizon suffix. Expired logical-route interning entries are
reclaimed through a bounded persistent cursor rather than a whole-registry scan
at every checkpoint. These bounds keep per-checkpoint bookkeeping independent
of the distance already committed in the segment.
The public point, match, step, and consumed-node vectors are assembled once at
segment termination. Focused replay decision diagnostics are the deliberate
exception: they request complete route payloads for every recorded decision and
therefore may materialize those diagnostic histories.

The greedy and fiberlet evaluators run concurrently. `--threads N` is their
shared evaluator worker budget: replay deterministically divides it between the
two nested evaluators instead of creating up to `2N` workers. On the full
46,148-base-voxel Paris4 interval at radius 768, the repaired hot-cache run with
`--threads 32` crossed 64.7 percent at 11 seconds and completed in 22.34 seconds
wall time (41.27 seconds user, 4.97 seconds system, 1.19 GiB peak RSS). It
reported 14 greedy and 5 fiberlet failures; the complexity and scheduling
changes do not alter failure evaluation.

Beam-step, lookahead, prune distance, search width, geometric cost weight, and
cost integration spacing are replay state only.
They are absent from anchor/fiberlet cache fingerprints, extraction settings,
and serialized chunk payloads. A new horizon can request previously untouched
on-demand chunks, but a fully hot cache is reopened without rewriting it.

The default cache roots are `<output>/cache/anchors.zarr` and
`<output>/cache/fiberlets.zarr`. Override them independently with
`--anchor-cache PATH` and `--fiberlet-cache PATH`. `--cache-gib` is one shared
decoded-byte budget across both caches; active adjacency or route leases remain
pinned until the current graph query releases them. `--storage-chunk-side N`
selects the spatial chunk side in base voxels and must be an exact multiple of
the anchor cell side. The roots are local-only in this implementation.

The fingerprinted cache identity includes producer-generation contract version
3. This revision invalidates unpublished version-2 anchor and Fiberlet payloads
whose producers could emit different records under one namespace. Default
commands create and reuse new fingerprint directories and leave version-2
directories untouched; an explicitly supplied version-2 root is rejected as
incompatible rather than migrated, repaired, or overwritten. Compiler family,
compiler version, and build configuration are recorded as producer diagnostics
but excluded from cache identity and compatibility. When the neutral default
namespace is absent, default cache discovery can reuse a compatible version-3
directory created with the earlier toolchain-specific identity. Cached anchor
prediction and normal fields are not a consistency boundary. Fiberlet generation
uses its freshly sampled endpoint and interior evidence, while the replay anchor
view resamples prediction and Lasagna normal evidence at each effective anchor
position before transition scoring. Cached anchor geometry and stable IDs remain
authoritative. A scheduled failure reports its owner key, terminal cache status,
and original generator error.

### Cache portability follow-up

GCC Release and Clang Debug produced 2,526 versus 2,528 incident Fiberlets for
one focused workload before both staged reductions retained 2,275. The exact
divergent candidates and decision predicates have not yet been isolated. A
follow-up must diff canonical Fiberlet IDs, log every threshold margin for the
divergent candidates, define deterministic quantized decision and tie semantics,
and verify payload equivalence across GCC/Clang, QuickBuild/Release, amd64, and
arm64. Toolchain-neutral reuse is intentional in the meantime.

## Chunk-local optimal-route statistics

`chunk-route-stats` measures how much of the regular cached graph participates
in an optimal route through one geometric box. It uses the same on-demand
preprocessor, anchor/fiberlet cache participants, dependency halo, serializers,
and shared decoded-byte LRU as tracing. A cold run generates and persists
missing canonical chunks; compatible hot chunks are reused without rewriting:

```bash
volume-cartographer/build/bin/vc_fiberlets chunk-route-stats \
  /path/to/fiber.lasagna.json /path/to/output \
  --normal-manifest /path/to/normals.lasagna.json \
  --chunk 23040,17920,54784 --chunk-size 256 --threads 32
```

`--chunk` is the box minimum in base-volume XYZ. The selected box is half open
and its side defaults to 256 base voxels. There is no lookahead option or route
length guard: every candidate continues until its first exit, even if its route
is longer than the box side or normal replay horizon.

Cache preparation uses the normal replay progress display and reports
resolved/expected anchor and fiberlet chunks, elapsed time, and ETA. `--stats`
adds serialized per-chunk completion records; it is not required for progress.

For every directed fiberlet entering from an outside anchor, the command finds
the exact cheapest route until it first reaches an outside anchor again. A
route may curve back and leave through any box face, but it cannot revisit an
anchor or fiberlet. Every join uses the regular strict maximum-angle test,
prediction-validity check, and normal/tangent-aware join cost. Thus spatial
turn-back is not permission for cycles or arbitrary turns. Entry and exit edges
are included once in both loss and length.

The default `--cost-profile sqrt-u16` applies the production fixed square-root
`uint16` cost-density view with ceiling 256; `--cost-profile stored` uses the
stored float component totals. Joins remain float. Cache roots default to
fingerprinted namespaces below `<output>/cache`; `--anchor-cache` and
`--fiberlet-cache` select the same explicit roots accepted by replay. Cache
metadata is strict and incompatible datasets are rejected.

The population row counts inside anchors, incident physical fiberlets, directed
entries/exits, and admissible transitions. The optimum row reports the union of
anchors and physical fiberlets used by at least one exactly tied cheapest
entry-to-first-exit route. Unused counts are candidate pruning opportunities,
not a pruning operation or proof that a globally different analysis boundary
would retain the same graph. `--max-states` bounds generated exact-search states
per entry and defaults to 5,000,000; reaching it fails without returning partial
used/unused results.
The primary table reports before/after counts and reduction percentages for
anchors and all incident Fiberlets. A second table reports internal Fiberlets
separately, excluding mandatory boundary-crossing entry and exit edges.

Pass `--region-size N --mode staged` with one or more repeatable
`--stage SIDE,OFFSET_X,OFFSET_Y,OFFSET_Z` options to run an ordered reduction
chain. `--chunk` names the selected region minimum in base XYZ. Each stage
selects complete `SIDE`-voxel analysis boxes from the global base-volume
`OFFSET mod SIDE` lattice that lie inside that region. The same global
anchoring applies to replay filtering. Analysis boxes need not align to anchor
cells or storage chunks.

This reproduces the original 512/256 two-pass experiment with eight outer
boxes followed by one half-offset central box:

```bash
volume-cartographer/build/bin/vc_fiberlets chunk-route-stats \
  /path/to/fiber.lasagna.json /path/to/output \
  --normal-manifest /path/to/normals.lasagna.json \
  --chunk 23040,17920,54784 \
  --region-size 512 \
  --mode staged \
  --stage 256,0,0,0 \
  --stage 256,128,128,128 \
  --storage-chunk-side 128 \
  --anchor-cache /path/to/anchors.zarr \
  --fiberlet-cache /path/to/fiberlets.zarr \
  --threads 32
```

Appending `--stage 512,0,0,0` runs a third whole-region pass. Every stage gets
separate anchor and Fiberlet overlay datasets with exactly the initial cache's
storage grid and encoding. A missing upper chunk means unchanged data and
falls through to the preceding layer; an explicit empty chunk shadows lower
data. A partial Fiberlet prefix/route pair or corrupt upper payload is an
error, never a fallback. The initial caches remain persistent and unchanged;
derived stage directories are temporary for this experiment and are deleted
after reporting.

Temporary stage payloads are memory-first. One write-back LRU retains their
canonical serialized bytes under the existing `--cache-gib` allowance, so a
later box or stage reads the current overlay directly without an intermediate
file write and reread. Only memory pressure evicts entries. Evicted entries are
written atomically by a bounded background writer while route work continues;
queued buffers remain charged until the write releases them. Anchor chunks are
single entries, while each Fiberlet prefix/routes owner is one paired entry and
can never fall through or fail as a partial pair. The write-back allocation is
subtracted from the shared decoded-chunk budget rather than creating an
additional unbounded cache.

`--stats` reports `fiberlet_write_back_cache` residency, memory hits, spills,
spilled bytes, and peak charged bytes. Payload diagnostics hash the logical
memory-plus-disk layer without forcing clean memory entries to disk. At command
teardown, already queued writes drain, unspilled temporary entries are dropped,
the decoded-cache allowance is restored, and the invocation-local tree is
removed.

Boxes execute serially in deterministic Z/Y/X order. Later overlapping boxes
in one stage read earlier writes from that same stage. Only Fiberlets whose
canonical first endpoint lies in the current half-open box may be removed.
The writer rewrites every affected original-layout owner chunk, including a
canonical owner outside the geometrically intersected storage chunks. It may
only publish subsets of the effective lower records, and retained anchor,
prefix, and route fields must remain exactly unchanged. An inside anchor is
removed only when no surviving incident Fiberlet references it, including
outside-owned and lower-layer Fiberlets.

Each stage table is scoped only to that stage's complete analysis boxes. It
reports original/input/output counts for inside anchors, all incident
Fiberlets, and Fiberlets interior to at least one complete stage box, plus
reduction from the inherited input and from the same stage-local original
geometry. A Fiberlet crossing between adjacent stage boxes is `all`, not
`interior`. An offset stage therefore does not count untouched parts of the
selected region. The joint table separately compares original and final
canonical populations over the complete selected region.

Analysis and simplification share one immutable materialized graph per box.
Materialization loads each required anchor, prefix, and route owner once, then
constructs directed arcs and transitions from those immutable chunk payloads.
Exact entry searches, transition construction, serialization, and independent
overlay owner writes use deterministic index-addressed work on reusable thread
pools; boxes and stages retain canonical serial order. Fiberlet prefix/route
pairs publish together before anchor chunks, and all replacement payloads are
prepared before publication. `--stats` adds materialization, exact-search,
simplification, write, and population wall/CPU timings, effective-core
diagnostics, semantic ID hashes, payload hashes, and the existing per-box
simplification details.

Use the ordinary optimized build for performance measurements:

```bash
cmake -S volume-cartographer -B volume-cartographer/build \
  -DCMAKE_BUILD_TYPE=Release
cmake --build volume-cartographer/build --target vc_fiberlets -j 32
```

On the hot Paris4 example above, three Release `-O3 -DNDEBUG` runs measured
2.44/2.46/2.49 s wall (min/median/max), down from 7.97/8.09/9.32 s. The final
runs used 3.82-3.88 s user and 1.54-1.62 s system CPU time (217-221% process
CPU). All runs reproduced the pre-change retained-ID and complete
overlay-payload hashes exactly. A controlled one-versus-32-thread run reduced
stage-one exact route analysis from 0.852 s to 0.108 s. Search workers read one
immutable local graph through fixed strided index partitions and keep their
heap and ancestry scratch thread-local, so the per-entry trace loop performs no
scheduling or synchronization.

After adding the memory-first write-back layer, the established four-stage
Paris4 workload measured 2.89/2.90/2.90 s wall (min/median/max) versus the
immediately preceding 3.93 s median on the same checkout and workload. The
321 temporary logical entries occupied 5,556,963 bytes, stayed below the shared
budget, and caused zero spills. All four retained-ID hashes, all four logical
payload hashes, and the final 3,368-anchor/35,027-Fiberlet/4,469-interior
population remained exact.

For each processed box, the command also constructs an exact in-memory
simplified graph. Directed states outside the intersection of
entry-forward and exit-backward reachability are removed conservatively, then
every anchor not referenced by a surviving Fiberlet is removed. Reachability
ignores the path-specific no-revisit history, so it may retain an uncertain
state but cannot remove a valid simple entry-to-first-exit route. Outside
endpoints of surviving crossing Fiberlets remain explicit boundary portals.

The simplifier forms maximal physical macro-Fiberlets only across an interior
anchor with exactly two incident surviving Fiberlets and valid regular joins
in both directions. A macro references the complete ordered original directed
Fiberlet and anchor sequence, with the original edge losses, join losses, and
lengths. It is therefore evaluated with the same scalar order as expansion;
the printed aggregate is diagnostic only. Cycles, branch anchors, one-way
joins, and boundary portals stop physical contraction. The report separately
counts disjoint directed chains and precomputes maximal forced continuations
from every directed state with exactly one admissible successor. The latter
may overlap at convergences, but can skip choices during replay after the start
state is known. Every hidden anchor is checked atomically against route history
before such a continuation can be applied.

A physical Fiberlet ID is the canonical pair of exact endpoint anchor keys, so
two distinct same-endpoint physical Fiberlets cannot exist in the stored graph;
duplicate IDs are rejected during materialization. Different endpoint variants
are distinct routes and are not deleted merely because one currently costs
more: visited-anchor history can make either route relevant. Macro graphs are
not serialized as ordinary Fiberlets and are not yet consumed by regular
replay. The ordinary route lattice cannot encode concatenated geometry without
resampling, so persistent macro storage requires a dedicated format.

Sequential box-local pruning is deterministic and monotone, but local
entry-to-first-exit optimality does not prove that a globally optimal replay
route is preserved for every possible later analysis boundary.

## Whole-volume preprocessing

Use `preprocess-volume` to materialize sparse anchors and fiberlets for an
entire stored prediction volume without a reference fiber:

```bash
volume-cartographer/build/bin/vc_fiberlets preprocess-volume \
  /path/to/fiber.lasagna.json /path/to/fiberlets.zarr \
  --normal-manifest /path/to/normals.lasagna.json \
  --source-context /path/to/stable-source-context.json \
  --threads 32
```

`--source-context` supplies portable producer identity (sample/volume, manager
run UUIDs, model/level identities, and manifest SHA-256 values). It must not
contain runtime manifest, cache, or output paths. The combined dataset metadata
schema is version 2: it persists structured `sources` and the complete effective
processing/coordinate/layout/storage/codec contract. `openExisting()` builds a
reader from that metadata alone and rejects missing, unknown, or internally
inconsistent settings.

`algorithm_fingerprint` canonically hashes the complete scientific processing
contract. `dataset_fingerprint` additionally hashes stable source identities
and manifest content hashes. Consequently relocating identical manifests does
not change either identity, while changing source content does. Executable
paths/hashes and host details remain external audit provenance rather than
scientific dataset identity.

The command retains two outputs. The default intermediate anchor cache is
`<output-stem>.anchors.zarr`; override it with `--anchor-cache PATH`. The final
combined Zarr contains only its metadata and the sparse `anchors/`, `prefix/`,
and `routes/` payload arrays. It uses float positions, compact directions, and
fixed sqrt-density `uint16` costs with ceiling 256. It has no active-chunk index
or completion-marker files.

Sparse eligibility depends only on canonical stored presence chunks. A final
spatial chunk is active when it overlaps a present, decoded-nonzero presence
chunk. Direction channels do not affect this first-stage decision. Missing and
decoded-all-zero presence chunks do not activate output. The intermediate
anchor cache additionally materializes the exact neighboring dependency halo
needed to evaluate active fiberlet chunks; those halo-only chunks are not added
to the final sparse index.

Every invocation rescans canonical input presence to reconstruct the expected
final chunks, then checks the required intermediate anchors and final output
tuples directly. It does not generate the entire anchor cache before starting
fiberlets. Instead, one global `--threads` chunk-worker budget first runs every
ready fiberlet in the current Z slab and uses remaining slots for the earliest
missing anchor dependencies. Anchor generation can look ahead, but final-output
generation advances to a later Z slab only after the current slab completes;
work within a slab may finish out of order. Each chunk extraction is
single-threaded in this mode so nested worker teams cannot oversubscribe the
global budget. A final chunk is complete only when its anchor, prefix, and route
payloads all exist, decode against the expected dataset and chunk identity, and
form a matching prefix/route pair. Missing tuple members are regenerated while
matching members left by an interrupted run are reused. Corrupt or conflicting
members fail loudly.

The combined pipeline reports one live progress line about once per second and
preserves a newline at least once per minute and at completion. It includes the
current Z frontier and separate anchor/output completed counts, throughput,
ETA, and current/projected compressed payload size. Projection uses the mean
payload size of completed or resumed expected chunks and excludes Zarr
metadata.

Each payload file is independently published by temporary write, file sync,
atomic rename, and directory sync. The three final files are a logical
completeness unit rather than one filesystem transaction. The command holds
exclusive locks on both output roots, removes abandoned atomic-write temporary
files before scanning and after workers stop, and removes obsolete
`active_chunks.bin`, `dataset.complete`, and `complete/` artifacts from earlier
development versions. Concurrent writers to either root are rejected.

Stored graph readers for a combined output must perform the same input-presence
scan and configure the resulting expected set after every reopen. Chunks outside
that set resolve as canonical empty payloads; missing expected tuples keep the
dataset incomplete. Extra intermediate halo anchors remain reusable.

`--presence-floor` defaults to `0.05`. It is the inclusive owned-observation
eligibility floor: a cell with no usable owned observation at or above the
floor returns before seed generation or refinement. `--minimum-support` is a
separate post-fit acceptance threshold.

Pass `--storage-compression-chunks N` to select up to `N` spatial regions that
intersect the replay and run complete anchor and fiberlet extraction for every
cell in those regions. Extraction also includes the neighboring anchor-cell
halo needed to evaluate all fiberlets owned by a selected region. The
diagnostic remaps the extracted records to the default compact replay profile:
float endpoint positions, compact two-byte directions, and fixed sqrt-density
`uint16` costs with ceiling 256. It reports the field-wise Zstd payload size
alongside the existing float32-cache size for the same source-region records.
The diagnostic also reports the size produced by wrapping the complete payload
in an additional Zstd level-3 frame. It materializes the raw field blocks and
compares replacing field-wise compression with one whole-payload Zstd level-3
frame. Selection is deterministic; change
`--storage-compression-seed` from its default of `1` to sample a different
ordering. This diagnostic does not change or publish the authoritative float32
replay cache.

Use `--storage-compression-chunk-side` to set the extracted spatial-region and
compact ownership side in base voxels. It affects only the diagnostic payloads.

Cache-backed replay schedules exact tube intersections at storage-chunk
resolution. It does not first materialize the complete anchor-cell population
for the reference interval. When an anchor chunk is requested, its owned cells
are enumerated in canonical Z/Y/X order and admitted by the same exact
segment-to-cell distance test as eager extraction. The existing post-refinement
anchor-position test and fiberlet-interior point test still apply. Neighboring
anchor chunks remain dependencies of fiberlet chunks, preserving the context
needed by cross-chunk NMS. Cache identity contains the complete clipped
reference geometry, radius, source metadata, algorithm settings, and corridor
selector version rather than a serialized cell list.

The fiberlet root separates `prefix/` connectivity/cost blocks from `routes/`
interior geometry. Beam/frontier state contains stable anchor and endpoint-pair
IDs, not pointers or copied corridor-wide graph records. An incident query
batch-prefetches the complete declared endpoint-reach neighborhood and queries
the deterministic two-endpoint index built once in each decoded prefix chunk.
The separate anchor and fiberlet `ChunkCache` instances retain typed decoded
payloads directly and charge their vectors and indices to the shared LRU
budget; they do not retain another serialized copy or use a graph-private LRU.
Prefix records and exact endpoint steps remain sufficient for connectivity and
join scoring. Lookahead loads route blocks to obtain decoded segment densities
and the route-lattice segment lengths; only selected routes are retained after
the query. The current provisional best is additionally reconstructed for
reference-error evaluation and final output. Evicted chunks reload
transparently.
For the float cache, anchors own the exact interpolated prediction/presence and
normal samples used by eager graph construction, while prefixes own all five
float path-cost components, the authoritative float path length, and the exact
first/last base-space steps. Cached joins consume those records directly; they
do not resample source volumes or reconstruct scoring geometry. Reconstructed
committed routes apply the same adjacent-point epsilon suppression as DP
finalization. Therefore cold and warm float-cache replay are required to be
byte-identical to eager replay for the same graph. The compact profile remains
intentionally quantized and does not provide that identity guarantee.
The final `fiber_replay_cache` row reports disk materialization counts for
anchor, prefix, and route chunks; multiple committed edges in one route chunk
share one decode. `--eager-graph` runs the prior whole-tube graph construction
for diagnostics.

By default cached replay prints independent `cache/prep` and `trace` terminal
progress bars while those operations overlap. Cache progress covers the
deterministic scheduled prefetch keys: resolved anchor chunks have weight one
and resolved fiberlet prefix chunks have weight 16. Persisted and newly
generated chunks both count, while reloads count only once. Trace progress is
the minimum monotone reference-arc fraction of the greedy and fiberlet
evaluators. It is therefore actual progress through the selected reference
interval, not a weighted estimate of preprocessing work. Eager replay prints
only the trace bar. Cache progress covers the scheduled anchor and prefix
population only; data-dependent neighbor-prefix and committed-route reads are
not predicted by that denominator and occur as part of tracing.

The active compact line shows one overall replay elapsed time. While cache and trace
overlap, each retains its own ETA; once scheduled cache progress reaches 100%,
the cache field is removed and the remaining terminal line is cleared to avoid
stale text. Trace also reports `eta_current`, computed from its progress during
the latest ten-second window. A stalled window reports `n/a` instead of
reusing an old rate. A private 250-millisecond ticker repaints the line even
while a long chunk emits no callback. Non-finite, stale, and restart-local
tracer callbacks cannot move trace progress backward. After both evaluators
complete, requested overview, failure visualizations, and durable publication
use separately named output stages and cannot be mistaken for tracing.

Fiberlet decisions add `fiberlet_rollout_expansions`, the total number of search
states expanded across all fronts of the latest bounded lookahead decision. An
expanded state is one whose successors were enumerated, so this value tracks
the search work that grows during slow decisions. Bounded search also adds
`fiberlet_local_cutoff_loss_per_vx_min`. For each final-front input where the
existing strict cutoff actually stops the queue, this subtracts the input
route's loss at the front start from the cumulative cutoff and divides by the
front length in prediction voxels; the displayed value is the minimum of those
local densities. The search itself still compares unchanged cumulative raw
losses. Both diagnostics are absent in exact mode, and the cutoff is absent
when it never binds. The most recent values remain visible while the other
evaluator controls the combined trace fraction.

Pass `--stats` to replace the bars with the detailed machine-readable stage,
chunk, failure, evaluator, and cache rows. These retain the stable chunk schedule
indices, internal generation phases, worker and CPU timings, cache residency,
restart-local greedy step diagnostics, and the fiberlet rollout expansion and
local cutoff diagnostics used for profiling.

On-demand anchor fitting publishes completion while holding the same mutex
used by its ready-cell condition predicate. Cache waits are ordinary blocking
dependency waits: there is no polling, heartbeat, or timeout recovery. A
failed dependency reports its exact dataset stage, chunk key, status, and
underlying cache error.

Within one fiberlet chunk, canonical source anchors are enumerated in parallel
and their results are concatenated in source order. Preparation, sampling, and
DP keep their existing parallel paths. Finished prepared geometry is released
by the worker that solved it, avoiding a serial multi-gigabyte teardown after
the search phase. These scheduling changes do not alter candidate order,
floating-point evaluation order within a candidate, or serialized payloads.

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

The logical graph contains every successful fiberlet over the selected
reference tube; failure-local graphs are not used for evaluation. In the
default path, its chunks are generated and loaded on demand and unleased chunks
remain evictable. The eager diagnostic path still uses the complete globally
deduplicated coordinate union and `--batch` only limits coordinates per sampler
call, as described above. The final summary reports both evaluator failure
counts and confirms reference fraction one for both.

The root manifest contains one `threshold` descriptor with the normal radius,
fixed factor and tangential radius, strict comparison, and invalid-normal
policy. Greedy and fiberlet descriptors are generated from it. The earlier
unpublished ambiguous replay keys `error_base_voxels` and `error_ratio` are not
supported.

The hot replay-tube point filter snapshots the selected reference interval in
prediction coordinates. It stores float32 continuous segments in a packed
Boost.Geometry R-tree using radius-expanded segment bounds, then performs the
float32 point-to-segment test only for intersecting candidates. Queries are
immutable and safe to share across preparation workers. The ordinary
`FiberReplayTube` distance methods remain linear and double precision because
anchor diagnostics need an actual distance rather than the high-volume boolean
filter. Float32 classification may differ at the configured radius boundary;
performance comparisons must report resulting workload and artifact changes.

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

The same `--vis` run also writes indexed `fiber_replay.NNNNNN.jpg` files for
immediate inspection, even when neither evaluator fails. Every wrapped range
contains four rows: the reference fiber's VC3D top and side strips followed by
top and side strips built from the actual fiberlet replay geometry. The latter
make the fiberlet refinement visible in its own transported frame instead of
only projected into the reference frame. The reference centerline is yellow,
the regular greedy trace is red, and the fiberlet trace is cyan. Fiberlet reset
segments are rendered by separate default `buildLineViewSurfaces()` calls and
placed in source-segment order with an eight-column black gap. Their matched
reference geometry and covered greedy geometry are projected through stored
match arcs; no new nearest-point matching is performed.

All four strips use the same concrete CT group, default line-view surfaces, and
shared fine-to-coarse renderer as the per-failure strips, but request an 8x
render scale from that renderer for detailed inspection. This does not resize
or change the native-resolution per-failure OBJ/MTL/TIFF artifacts. Greedy
failures are marked by three-pixel vertical red bands at the pre-reset error
arc; fiberlet failures use cyan, and coincident bands are magenta. The later
reset seed is intentionally not marked.
An explicit `--length N` limits this JPEG
to that same selected `N`-base-voxel interval; without `--length`, it covers the
remaining reference fiber. Long 8x strips are split at equal progress fractions
into at most 32,000-column ranges. Each range is one labeled four-strip block.
Complete blocks stack in the same JPEG until another would exceed 65,000 rows;
then publication continues in the next indexed JPEG. Every JPEG dimension is
at most 65,000 pixels. Each of the four source rasters maps independently to
exact half-open ranges, so no raster is resampled and no column is lost.

The immutable copies are
`runs/<content-hash>/replay/full_strip.NNNNNN.jpg`; the root
`fiber_replay.json` records their hashes and stable aliases, selected arcs,
reference-point count, CT group transform, all four unwrapped dimensions, 8x
scale, marker semantics, exact page/block ranges, fiberlet component placement,
layout, and colors. The stable indexed JPEGs are direct-inspection output, not
napari replay manifests. The command prints every indexed path after
publication. Stale higher indices are removed when a later run has fewer pages.
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

Benchmark and replay extraction also emit a versioned
`fiberlet_extraction_profile version=26` row. Both commands use the same field
names and units. Replay writes the row to stderr after full tube extraction;
benchmark writes it to stdout after the existing summary. The row separates:

- anchor setup, tile planning, coordinate construction, prediction sampling,
  observation/gradient construction, fitting, selection, duplicate suppression,
  and finalization;
- fiberlet candidate generation, geometry preparation, node enumeration,
  interpolation-corner collection and merge, prediction/normal sampling,
  scoring preparation and index construction, interpolation materialization,
  node-index construction, and dynamic programming;
- deterministic workload counts including selected/context/work cells, tiles,
  sampler calls and submitted coordinates, observations and gradients, lattice
  nodes and corridor tests, corner insertion attempts and globally unique
  sampled voxels, and DP lookups/visits/relaxations.

Version 11 also reports a bounded one-in-4096-per-worker interpolation sample.
It separates page lookup, prediction/normal corner accumulation, and
prediction/normal principal-axis resolution without timing every scoring
point.

Version 12 adds complete prediction/normal closed-form resolution and iterative
fallback counts. Ambiguous tensors do not count as fallbacks because no unique
direction exists to recover.

Version 13 adds prepared-node counts, reached/generated/valid/reused edge
counts, solve-local prepared-node/direct-index/state byte maxima, and separate
node-preparation worker time. Lookup/visit counters omit dead outgoing work
from the final interior layer, which transitions directly to the sink. Rolling
state memory is the global predecessor bytes plus the largest adjacent pair of
cost layers; those layer populations are collected during node generation.
On the canonical 5,000-base-voxel replay at 32 threads, three runs measured
11.91 seconds median total wall time and 0.996 seconds median search wall time,
versus 12.42 and about 1.85 seconds before solve-local reuse. Median total CPU
fell from 310.98 to 283.73 seconds and search CPU from about 58 to 31.39
seconds. Selected geometry and replay failures were unchanged; float32
cumulative costs permit small serialized-cost differences.

Version 14 reports eager endpoint interpolations, lazy node requests, unique
node materializations, cache hits, maximum lazy node-map bytes, and immutable
shared scoring bytes retained through search. `interpolatedScoringPoints` is
now endpoint interpolations plus unique lazy node materializations. On the same
canonical replay, only 14.48M of 50.72M retained nodes were materialized.
Three runs measured 10.76 seconds median total wall time and 3.30 seconds
fiberlet wall time, versus 11.91 and 4.52 seconds for version 13. Median total
CPU fell from 283.73 to 248.84 seconds, fiberlet CPU from 108.20 to 69.38
seconds, and peak RSS from 2.46 to 2.11 GiB. Search itself increased from 1.00
to 1.29 seconds because lazy interpolation is included there; the former
all-node interpolation phase fell from 1.52 seconds to about 0.015 seconds.
Selected geometry, replay failures, and replay artifacts were unchanged.

Version 15 adds `anchor_support_stencil_cells` and
`anchor_clipped_support_cells`. Complete volume-interior cells with their full
sampling halo reuse one immutable ordered `(z, y, xBegin, xEnd)` support
stencil. The spans are translated through each tile's actual row and plane
strides, preserving canonical Z/Y/X observation order. Partial cells and cells
without a full volume halo retain the clipped scalar construction. On the
canonical replay all 13,027 work cells used the stencil; median observation-
construction worker time fell from 18.99 to 11.84 seconds, anchor CPU from
177.28 to 167.49 seconds, and total wall from 10.76 to 10.37 seconds. All
workload populations and replay artifacts remained identical.

Version 16 separates the larger support range used by robust refinement from
the exact owned-cell range used by initialization. Production traverses the
owned cube directly from validated dense-tile row and plane strides; it does
not rescan support coordinates or allocate owned indices. The public expanded-
observation API retains its stable input-order filter and historical count-only
coverage validation. `anchor_fit_owned_discovery_observation_visits`,
`anchor_fit_owned_initialization_observation_visits`, and
`anchor_fit_avoided_owned_support_observation_visits` distinguish those paths.
On the canonical replay, direct initialization visited 833,728 owned voxels and
avoided 858,114,544 support-range visits. Fit-setup worker time fell from a
10.996-second baseline median to 0.092 seconds in the final attribution-inclusive
validation; anchor CPU fell from 169.45 to 162.51 seconds and anchor wall
from 6.553 to 6.304 seconds, and total wall from 10.76 to 10.65 seconds. Replay
populations, failures, and artifacts remained identical.

Version 2 subdivides `anchor_fitting_work_seconds` into exclusive summed-worker
phases for weighted-observation setup, seed generation, seed-pair refinement,
initialized-component finalization, local direction/position refinement,
direction-conditioned peak search, and final evaluation. The corresponding
`anchor_fit_profiled_work_seconds` sum and
`anchor_fit_residual_work_seconds` expose instrumentation gaps.

Version 3 further partitions local refinement into axis-tensor proposal,
position-centroid proposal, and refined-state evaluation work. The evaluation
recomputes each observation's finite axial/Gaussian support, direction
agreement, component assignment, weighted objective numerator, and Gaussian
denominator for the initial state and every backtracking candidate.
`anchor_fit_local_control_work_seconds` is the enclosing local-refinement time
minus those three kernels, so it includes bounds/setup, interpolation,
acceptance, convergence, and profiling overhead. The four fields reconcile to
`anchor_fit_local_refinement_work_seconds`.

Production compact observations keep the robust direction-proposal scan in
float32, including component assignment, Gaussian/alignment arithmetic,
residual histograms, retained direction-tensor bins, robust cutoffs, and the
shared float principal-axis solver. The direct public-observation fitter uses
the same representation. On the canonical 5,000-base-voxel replay, median tensor-
proposal worker time fell from 25.63 to 23.74 seconds, anchor CPU from 143.16
to 140.75 seconds, and command wall from 9.65 to 9.58 seconds. Anchor/graph
populations and failures were unchanged; emitted route points differed by at
most 1.38e-6 base voxels.

Fixed-direction position objectives live in a separate private translation
unit so their production float specialization cannot perturb robust-proposal
code generation. The production path borrows tile observations and canonical
per-cell indices without materialization: assignment and membership arrays use
logical per-cell positions, while observation reads use the corresponding tile
index. Gaussian, alignment, ordinary numerator/denominator sums, and final
ratio are float32. Every finite-position site contributes to all active
denominators before evidence eligibility is checked. The direct public path
uses the same float objective equation and persistent component/accepted-position
state. On the canonical
replay, isolation reduced median local-state objective work from 22.59 to 13.86
worker-seconds, anchor CPU from 140.80 to 130.74 seconds, and command wall from
9.56 to 9.17 seconds, with byte-identical replay artifacts.

Final support evaluation is independently isolated from both the anchor fitter
and fixed-direction objective kernels. Compact production observations remain
float32 throughout Gaussian, direction, presence, and accumulation arithmetic.
The direct public fitter uses the same float32 reduction with scale-safe
direction normalization. Per-component aligned support/coherence, the combined
objective, and persistent output fields remain float. Invalid public evidence
cannot enter a numerator; a finite-position site still contributes to every
active denominator regardless of membership or direction usability. On the
canonical replay, median final-evaluation work fell
from 13.79 to 13.11 worker-seconds, anchor CPU from 132.40 to 130.55 seconds,
and command wall from 9.26 to 9.22 seconds, with byte-identical artifacts and
unchanged replay work/failures.

Version 4 reports robust components with no detected outliers, trimmed
components, candidate/actual trimmed and retained mass, components removed for
non-unique retained tensors, iteration-limit hits, and position candidates tested
and accepted by halving depth. Its local tensor phase includes competitive
assignment, histogram cutoff selection, and retained sampled-direction PCA;
the local state-evaluation phase contains fixed-direction position objectives.
Component and mass fields count every robust proposal across bounded passes;
they are work diagnostics, not a deduplicated final component population.

Version 6 distinguishes logical per-cell gradient uses from physical gradient
computations. Presence gradients are computed once over each dense anchor tile
and reused by overlapping cell halos. `anchor_gradient_attempts` and
`anchor_valid_gradients` retain their logical per-observation meaning, while
`anchor_gradient_computations`, `anchor_valid_gradient_computations`, and
`anchor_gradient_construction_work_seconds` report the physical tile work.
Robust residual histograms and retained direction tensors are accumulated
together, paired spatial objectives share one observation pass, and peak
responses use an equivalent two-dimensional transverse representation. These
grouped reductions can introduce small floating-point differences; exact
numeric identity with version 4 is not a requirement.

Peak-search observations, transverse response calculations, accepted anchor
positions, component state, aggregate numeric diagnostics, and serialized
output use float32. This removes widening at the peak-search boundary and
halves the repeatedly scanned working set relative to the earlier double
representation; small differences in peak ties and downstream path node counts
are expected.

The bounded peak hill climb stores grid geometry, feasibility, and computed
responses in contiguous row-major slots. Response values have independent
computed-state bytes, so every result, including a non-finite result, is
computed once and cached. The canonical
candidate order, response kernel, tie-breaking, float response coordinates,
and uncached subpixel checks are unchanged. On the canonical 5,000-base-voxel
replay, median peak-search worker time fell from roughly 43.9 to 42.84 seconds,
anchor CPU from 162.51 to 160.30 seconds, and command wall from 10.65 to 10.43
seconds while complete artifacts and deterministic counters remained exact.

Peak responses store transverse coordinates, axial weight, and signal in a
16-byte sequential record. A parallel 32-bit index addresses a 16-byte sparse
record containing alignment and projected-gradient data only for retained
usable evidence. Invalid-gradient evidence remains indexed because it still
contributes eligible aligned weight to gradient coverage. Version 17 profile
fields report prepared hot/evidence populations, their record sizes, maximum
temporary storage, all hot response visits, and evidence records actually read
inside the radial cutoff. On the canonical replay, 4.82% of prepared records
and 3.25% of response visits needed evidence; median peak-search worker time
fell from 42.84 to 39.94 seconds and command wall from 10.43 to 10.25 seconds.
The response equation remains unchanged, but exact accumulation order is not a
compatibility requirement; deterministic geometry and replay quality are.

Production extraction also constructs each sampled tile voxel once as a compact
float32 observation with a pre-normalized direction. Each overlapping cell
stores only canonical-order 32-bit indices into that tile plus its cell-local
gradient-validity byte. The public expanded-observation fitting API uses the
same templated fitter. Tile observation storage and maximum cell-reference
scratch are included in the existing concurrent sample-memory budget.
Complete cells whose configured sample halo fits inside the volume expand the
shared ordered support-span stencil directly into those indices. Crop and tile
boundaries do not affect eligibility. A volume boundary or partial final cell
uses the original clipped sample-cube scan. When gradients are enabled, the
extra halo voxel makes every retained stencil site gradient-eligible; sampled
tile-gradient validity remains authoritative.
Initialization does not rediscover owned observations from this support range.
It traverses the clipped cell's dense tile rows directly in canonical Z/Y/X
order after constant-time tile-shape, bounds-containment, and owned-cardinality
validation. Refinement continues to use the support indices and gradient bytes.
For compact production fitting, the same initial bounds pass also records the
ascending logical indices whose immutable validity, direction, and presence
meet robust-proposal eligibility. Robust axis and membership proposals traverse
that subset directly, but write assignments and retained state at the original
full-support indices. Expanded public fitting retains its general traversal.
The reusable eligibility vector is bounded as worker scratch and included in
the extraction memory budget.
Robust axis proposals retain their independent return-value storage, but
intermediate membership is not copied into fit state because the mandatory
post-update membership refresh replaces it. That final membership is moved
into the fit, and final support evaluation updates scalar summaries without
copying the full membership arrays. The live proposal arrays are included in
compact fitting worker admission.

Version 20 shares raw prediction samples across bounded exact-union partitions.
Canonical-order tiles are partitioned conservatively so workloads larger than
the sample-memory budget still stream. Within each partition, exact tile X
ranges are merged per structured `(z,y)` row and stored in one contiguous
float32 sample array. Deterministic bounded batches submit every partition-
union coordinate once, using one lower-level sampler thread per batch. Sampling
joins before any tile reads the shared array. Each tile then copies contiguous
row ranges, constructs its gradients and compact observations locally, and
publishes cells to the cooperative fitting queue. Its owner helps the same
queue and retains the immutable observations until every dependent cell is
complete.

Sampling workers are bounded by shared storage plus concurrent coordinate and
expanded-result scratch. Fitting workers are bounded separately by shared
storage, the pre-reserved ready-cell queue, timing storage, and complete
per-worker tile/gradient/observation/cell scratch. These allocations contribute
to `anchor_max_accounted_live_bytes`; partitions are split rather than requiring
the full extraction union to fit. `anchor_sampling_partitions` and partition
duration quantiles expose the phase boundary. The profile also reports shared
batch count/maximum size, shared-sampling wall/CPU, maximum shared bytes, and
tile-copy worker time. `anchor_submitted_prediction_voxels` counts partition-
union samples, `anchor_reused_prediction_voxels` is tile occurrences minus
submissions, and `anchor_unique_tile_prediction_voxels` remains the exact whole-
extraction union. Submitted and unique counts are equal for a one-partition
workload; bounded partitions may resample overlap at their boundaries.

On the canonical 5,000-base-voxel Paris4 replay, one partition reduced raw
prediction submissions from 26,741,712 to the exact 6,162,456-voxel union.
Against version 19, three warm QuickBuild runs reduced median anchor CPU from
126.98 to 111.61 seconds, anchor wall from 4.262 to 4.069 seconds, and command
wall from 6.97 to 6.82 seconds. Median peak RSS changed from 1,687,504 to
1,675,944 KiB. Anchor and graph populations, DP work, failures, and the complete
replay artifact remained exact.

Version 8 replaces the per-candidate packed-node-key hash map with a direct
`uint32_t` table over the already validated packed-key range. The table uses an
invalid sentinel for absent corridor nodes; node generation, transition order,
DP state, and path-cost evaluation are unchanged. Peak search memory accounting
uses the direct table's actual payload. `fiberlet_dp_node_index_entries` counts
stored nodes and `fiberlet_dp_node_index_slots` counts allocated direct-table
slots, exposing occupancy and sparse-lattice overhead.

Version 9 stores sampled scoring-voxel indices in sparse `16^3` pages. Each
interpolation stencil caches up to its eight touched pages, then resolves each
corner by a dense page-local offset. This changes only lookup: corner order,
weights, tensor accumulation, principal-axis resolution, and compact node
quantization are unchanged. `fiberlet_scoring_page_count` reports occupied
pages, `fiberlet_scoring_page_slots` reports their dense index capacity, and
`fiberlet_scoring_page_directory_probes` reports actual sparse-directory
lookups after per-stencil reuse.

On the canonical 5,000-base-voxel Paris4 replay, three alternating QuickBuild
baseline/candidate pairs reduced median DP search wall from 1.165 to 1.051
seconds and DP worker time from 36.88 to 33.23 seconds. Median fiberlet wall
fell from 2.129 to 2.008 seconds and total command wall from 7.87 to 7.75
seconds. Replay artifacts, populations, DP counters, routes, and failures were
exact.

Anchor-fit counters distinguish fitter invocations from nonempty cells and
report seeds, seed pairs, seed-pair iterations, local-refinement attempts and
accepted steps, and backtracking evaluations. Observation visits are counted
separately for seed assignment/tensors/objectives, local tensors/centroids,
refined-state evaluations, peak preparation/responses, and final evaluation.
They are logical work counts: an exact broad phase can skip detailed kernel
calculations without reducing the corresponding visit count.
Profile version 21 additionally splits robust axis-producing and membership-
only calls. Its logical counts describe full support, eligible counts describe
the immutable contributing subset, and indexed/cutoff counts describe physical
compact-path traversal.
Profile version 22 reports proposal-buffer initializations and bytes plus any
membership bytes copied into evaluation state; production copy-elimination
keeps the latter at zero.
Profile version 23 stores direction-conditioned peak response geometry in a
12-byte hot record and signal with sparse 20-byte evidence. Denominator scans
still visit every response record; numerator and gradient work follows the
parallel evidence index. Nonzero signal requires evidence, while zero-presence
evidence remains valid and carries zero signal.
Profile version 29 removes that dense evidence-index stream. Denominator work
traverses the 12-byte response records, while numerator and gradient work
traverses self-contained 32-byte sparse evidence records in original
observation order. The profile no longer emits an evidence-index record size.
Profile version 24 reports extraction-wide compact-observation construction
separately from per-tile shared-index-map construction. Shared observations and
presence gradients are built once per exact-union prediction voxel instead of
once per overlapping tile occurrence. Gradient validity at tile boundaries is
still evaluated per tile, preserving the fitting support seen by each cell.
Profile version 25 prepares one contiguous 32-byte record for each robust-
proposal-eligible compact observation in a cell. Records preserve logical
order and carry their original logical destination, allowing both axis passes
and final membership to reuse position, normalized direction, and presence
without repeatedly dereferencing shared observation maps. Full-support fitting
phases remain unchanged. The profile reports prepared record count/size and
summed preparation work; these records replace the old eligible-index scratch
in bounded worker-memory admission.
Prepared proposal records store the finite absolute position, normalized
direction, presence, and logical destination for each eligible observation.
The compact proposal kernel reuses these records while retaining the original
position-to-pivot and position-to-component arithmetic. Invalid arbitrary input
is still handled by the checked expanded/detail fitting paths.
Peak grid-response requests include cache hits;
`anchor_fit_peak_computed_grid_responses` counts cache misses, while
`anchor_fit_peak_acceptance_responses` counts uncached subpixel checks. The
legacy `anchor_fit_iterations` field remains for compatibility and counts a
cell's accepted local-refinement steps once per output component; use
`anchor_fit_local_refinement_accepted_steps` for the unambiguous cell-level
count.

Fields ending in `_seconds` are enclosing wall phases unless their name contains
`_work_`; work fields sum per-candidate or per-worker elapsed time and may exceed
wall time. CPU fields are process CPU time. Corner insertion attempts count all
positive-weight insertion attempts, while `fiberlet_unique_sampled_voxels` is
the deterministic global union. `*_profiled_seconds` is the sum of disjoint
wall phases and `*_residual_seconds` is unassigned elapsed wall time. Existing
progress callbacks remain inside their enclosing wall phases. Profiling is
diagnostic only and does not alter extraction decisions, ordering, or artifacts.

## Storage quantization benchmark

`quantization-benchmark` compares two cache-backed production replays. The
baseline uses the float cache profile. Every run shares one canonical float
anchor dataset under `OUTPUT/cache` for the exact source, extraction settings,
corridor, grid, and chunk layout. The selected scenario derives rounded
positions or compact fitted directions from those anchors, then performs fresh
candidate generation, Hermite geometry, dense sampling, and DP in a separate
fiberlet namespace. Quantization therefore never reruns anchor fitting or
changes persisted anchor chunks. These are evaluation caches, not final compact
interchange data.

```bash
vc_fiberlets quantization-benchmark FIBER_MANIFEST FIBER_JSON OUTPUT --normal-manifest NORMAL_MANIFEST --radius 768 --threads 32
```

The default scenario is `compact_axis_cost_sqrt_u16_max256`: exact float
endpoint positions, compact two-byte fitted directions, and the fixed
sqrt-density `uint16` cost view with ceiling 256. `--scenario NAME` selects another standard scenario;
`--scenario all` runs one baseline followed by all 18 non-baseline scenarios in
their fixed matrix order. An unknown name is an error. `--length N` limits the
reference interval in base voxels for a shorter comparison.

`position_q1_8_compact_axis_cost_sqrt_u16_max256` replaces the unpublished
`combined_q1_axis_cost_u8` matrix row. It globally rounds endpoint coordinates
to the nearest 0.125 base voxel, uses compact fitted directions, and applies the
fixed sqrt-density `uint16` cost view with ceiling 256. Positive half steps
round upward. At prediction-to-base scale 8, the position quantum is 0.015625
prediction voxel; it is not one eighth of a prediction voxel. Endpoint
prediction direction, presence, validity, and Lasagna normal are resampled at
the rounded coordinates before fresh Hermite construction, dense sampling,
DP, and graph replay. Join costs remain float and are added once.

`compact_axis_cost_u8` and `compact_axis_cost_u16` keep float endpoint
positions, use compact two-byte fitted directions, and decode the stored float
component totals through per-owner-chunk `uint8` or `uint16` cost views. Along
with `compact_axis`, all three reopen the same compact-axis prefix/route cache;
cost precision is graph-view state and never rewrites stored geometry or float
component costs. The cache is populated on demand. A cost view can therefore
complete missing compact-axis chunks in that same cache while establishing the
stable minimum/maximum for a first-endpoint storage chunk; it does not create a
cost-specific geometry cache.

`compact_axis_cost_sqrt_u16_max256` reopens that same compact-axis cache and
uses a fixed two-byte-equivalent evaluation view. It encodes
`round(65535 * sqrt(clamp((total_cost / path_length) / 256, 0, 1)))` and
decodes the edge total as `256 * (code / 65535)^2 * path_length`. The fixed
ceiling is global: no chunk statistics, dataset scan, or observed min/max
contributes to the mapping. The authoritative positive float32 path length
comes from the existing prefix. Join costs stay in their original float
representation and are added once by graph search.

The fixed-sqrt scenario does not persist a two-byte cost field. It derives an
ephemeral decoded cost view from existing float component costs and lengths.
Cost domain and ceiling are absent from geometry identities and cache paths, so
a completed compact-axis cache is reused without range scans, anchor
extraction, fiberlet DP, new payloads, or payload rewrites. Result rows and
replay JSON identify `sqrt_per_prediction_voxel` and the fixed ceiling.

Normal cache-backed `fiberlet-replay` uses this profile by default. It shares
the canonical float anchor cache, generates or reopens the compact-direction
fiberlet namespace, and applies the fixed sqrt-density `uint16` view while
ranking graph edges. Its replay bundle records the position quantum, direction
encoding, cost width/domain/ceiling, storage chunk side, and persistent payload
profile under `fiberlet_evaluation_profile`. `--eager-graph` remains an explicit
exact-float diagnostic and records `exact_float_oracle` instead.

The exact-float profile remains the correctness oracle used as the baseline of
every quantization comparison. It is selected through an explicit named
profile, not inherited from the production replay default. This default change
does not alter the unpublished `CompactQuantized` serializer: on-demand cache
payloads still use `Float32Cache`, while compact in-memory/persistent payloads
will use this accepted profile when that representation is materialized.

On the full Paris4 46,148-base-voxel reference interval at radius 768, beam 16,
checkpoint 48, lookahead 384, and exact search, both the exact oracle and the
new default produced two failures. Mean/median reference distance changed from
5.712/3.625 base voxels to 5.611/3.549. The default differed from the oracle by
1.172 base voxels mean and 0.172 median; its 71.778 maximum occurred around a
shifted restart.

Fiberlet caches are grouped by endpoint position quantum and fitted-direction
encoding. Float, raw `uint8`, raw `uint16`, and fixed-sqrt `uint16` cost views
over the same geometry reopen the same fiberlet prefixes, routes, endpoint
steps, path lengths, and float component costs; cost decoding happens only in
the replay graph. All scenarios
read the same canonical anchor cache, and cost-only scenarios also reuse the
exact baseline fiberlets. Geometry-
quantized scenarios retain the pre-existing internal `cost_bits=8` cache tag so
completed experimental caches remain reusable, but that tag is opaque and is
never passed to anchor extraction or DP. The selected cost precision remains
the `cost_bits` field in each output row. `geometry_cache_cost_tag_bits` reports
the internal compatibility tag. Storage chunk side remains part of both the
physical cache layout and its identity.

Evaluation position quanta are finite positive base-voxel spacings; zero means
exact float positions. Fractional values are represented in cache identities
and reports without integer truncation. They affect only derived evaluation
geometry: canonical float anchor payloads and the unrelated compact physical
storage header remain unchanged. Thus the 0.125 scenario reuses canonical
anchors, creates one distinct position-plus-direction fiberlet cache on its
first run, and reopens that cache without DP or rewrites on later runs.

Each comparison writes complete baseline and scenario graph-replay JSON files
under `OUTPUT/quantization-replays/<interval-hash>/`. It also prints one
`fiberlet_quantization_failure_window` row per failure. That row gives the
exact segment reset arc, the length through the complete failure-containing
fiberlet, and the original graph seed key. A single failure can then be rerun
without tracing the full reference fiber:

```bash
vc_fiberlets quantization-benchmark FIBER_MANIFEST FIBER_JSON OUTPUT --normal-manifest NORMAL_MANIFEST --radius 768 --threads 32 --scenario position_q1_compact_axis --arc ARC --length LENGTH --seed-key Z,Y,X,V
```

The same directory contains `route-cost-statistics-<scenario>.json`. Its
`baseline_all` and `scenario_all` objects summarize only fiberlets actually
committed by replay, not overlapping beam-lookahead candidates. Each objective
term and the edge/transition/combined losses are divided by the committed
edge's prediction-voxel path length before reporting count, sum, minimum, mean,
median, and maximum. Raw per-fiberlet totals are deliberately omitted because
route ranking normalizes unequal-length candidates. The artifact retains only
aggregate total route loss, prediction-voxel path length, and their
length-weighted whole-route density. `baseline_away_from_failures` and
`scenario_away_from_failures`
exclude any committed fiberlet whose covered reference-arc interval is within
128 base voxels of a baseline failure. The same baseline windows filter both
runs so their regions are comparable. The interval test is inclusive. Repeated
commitments are counted as separate route occurrences, and each entry owns its
edge cost plus only the incoming join from the preceding edge in that replay
segment. Change the distance with `--route-stats-failure-margin N`.

`--arc` and `--length` are base-volume arc lengths. `--seed-key` is the
diagnostic anchor identity printed by the full run; it makes a later segment
exact even though the full replay excludes graph nodes consumed by earlier
segments. Focused diagnostics reopen the completed full-corridor cache identity
and schedule the chunks intersecting the focused reference tube. Missing chunks
are generated with the full-corridor containment predicate, so their persisted
contents remain compatible with an unfocused replay.

Focused replay JSON includes each decision's retained beam frontier, current
and next logical checkpoints, whole-fiberlet scoring horizon, expansion thread
count, and evaluated/retained candidate counts. Every route records its
committed whole-fiberlet prefix IDs separately from its best lookahead
continuation, plus checkpoint-to-lookahead route points in base coordinates,
the path-length origin of those points, edge and transition costs,
actual path length including final-edge overshoot, total loss, and loss per
prediction voxel. A sibling
`decision-comparison-<scenario>.json` reports the first
selected-route difference, cross-ranks each run's choice in the other run when
that route exists, and reports the maximum symmetric distance between paired
selected route geometries. This distinguishes a disproportionate scoring bug
from a small cost change that flips an already near-tied discrete decision.

For full-history replay diagnostics, `--stats --decision-window BEGIN,END`
retains those decision records only when the selected route's matched reference
arc lies in the inclusive base-voxel interval. The option is repeatable. Search,
matching, restart history, and cache identity are unchanged; only diagnostic
materialization is filtered. This is preferable to unrestricted `--stats` for
long fibers because each retained route owns its lookahead IDs and geometry.
Decision-frontier recording is disabled by default; `--decision-window` is
rejected unless `--stats` is also present.

`fiberlet-replay --arc BEGIN --length LENGTH` starts both evaluators at an
absolute base-voxel reference arc. This is a diagnostic interval, not a new
fiber control point. The greedy evaluator uses the sampled reference point and
tangent, and graph replay selects its seed there. Focused runs retain the full
reference corridor's cache identity and containment rules while scheduling
only the requested interval, so compatible completed cache chunks are reused
and any missing chunk is generated with the same contents as a full replay.

The `fiberlet_cached_quantization` row is machine-readable. It reports the
explicit position/direction/cost settings, failure counts, completed fractions,
logical-key collision diagnostics, cache residency, timing, and symmetric
Euclidean, Lasagna-normal, and Lasagna-tangential line-distance distributions.
Restart segments remain disconnected; both lines are sampled at no more than
one base voxel spacing and projected onto the other line's actual segments.
The row is flushed as soon as its comparison finishes. Before a batch context
is released, speculative fiberlet jobs are cancelled and drained, followed by
their anchor dependencies and issued persistent writes. This prevents late
cache generation from racing process-wide worker-pool teardown while leaving
unrelated cache users running.
Exact anchor, candidate, point-count, and point-index agreement are not quality
criteria. Separate
`baseline_reference_*` and `scenario_reference_*` summaries measure each replay
toward the annotated reference only. They report count, minimum, mean, median,
and maximum Euclidean, normal, and tangential distance in base voxels. The
Lasagna normal is sampled at the matched reference point; invalid normal samples
remain in the Euclidean summary and are counted separately. Quantized anchor
geometry uses rounded positions, but stable logical IDs remain the source cell
coordinate plus its zero-or-one component. This preserves the two-anchor limit
per cell without merging anchors from adjacent cells that round to the same
position. Unrepresentable endpoint deltas fail the scenario instead of silently
changing identity.

Derived endpoint views are chunk scoped. Compact direction changes the fitted
axis, and prediction/presence/normal fields are always resampled at the effective
anchor position. A position quantum first rounds that position. Single-flight
construction prevents overlapping fiberlet chunks from repeating that sampling,
and a bounded LRU releases derived chunks after use. Replay seed selection,
endpoint lookup, route reconstruction, transitions, and compact-cost ownership
all use the same view as fiberlet DP.
