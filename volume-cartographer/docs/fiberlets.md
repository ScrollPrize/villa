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
  --lookahead 3 \
  --length 4096
```

The default cache roots are `<output>/cache/anchors.zarr` and
`<output>/cache/fiberlets.zarr`. Override them independently with
`--anchor-cache PATH` and `--fiberlet-cache PATH`. `--cache-gib` is one shared
decoded-byte budget across both caches; active adjacency or route leases remain
pinned until the current graph query releases them. `--storage-chunk-side N`
selects the spatial chunk side in base voxels and must be an exact multiple of
the anchor cell side. The roots are local-only in this implementation.

The fiberlet root separates `prefix/` connectivity/cost blocks from `routes/`
interior geometry. Beam/frontier state contains stable anchor and endpoint-pair
IDs, not pointers or copied corridor-wide graph records. An incident query
batch-prefetches the complete declared endpoint-reach neighborhood and queries
the deterministic two-endpoint index built once in each decoded prefix chunk.
The separate anchor and fiberlet `ChunkCache` instances retain typed decoded
payloads directly and charge their vectors and indices to the shared LRU
budget; they do not retain another serialized copy or use a graph-private LRU.
Prefix records and exact endpoint steps are sufficient for beam ranking and
join scoring. A route block is loaded and its full polyline reconstructed only
after an edge is selected for commitment. Evicted chunks reload transparently.
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

By default replay prints one terminal progress bar with elapsed time and ETA.
Its tracing fraction is the minimum monotone reference-arc fraction completed
by the greedy and fiberlet evaluators, so a fast greedy completion cannot hide
unfinished graph work. Non-finite, stale, and restart-local callback values
cannot move it backward. Completion is reserved until requested visualization
and bundle publication finish.

Pass `--stats` to replace the bar with the detailed machine-readable stage,
chunk, failure, evaluator, and cache rows. These retain the stable chunk schedule
indices, internal generation phases, worker and CPU timings, cache residency,
and restart-local greedy step diagnostics used for profiling.

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

`quantization-benchmark` extracts anchors once with the production
implementation. It quantizes anchor endpoints and fitted directions before
running the regular candidate generation, dense sampling, and fiberlet DP for
each distinct geometry. Geometry results are cached and reused by their
`uint8`/`uint16` cost variants. It writes no persistent quantized format.

```bash
vc_fiberlets quantization-benchmark FIBER_MANIFEST FIBER_JSON --normal-manifest NORMAL_MANIFEST --length 5000 --threads 32 --radius 64
```

Use `--scenario combined_q4_axis_cost_u8` to run only the baseline and the
4-base-voxel endpoint-position, compact two-byte fitted-direction, 8-bit-cost
case. The selector names are the exact `scenario=` values printed by the full
matrix; an unknown name is an error.

The fixed 16-row matrix contains the float32 baseline; isolated position quanta
`1`, `2`, and `4` base voxels; isolated compact fitted axes; all three
position-plus-compact-axis float-cost geometries; isolated `uint8` and `uint16`
per-chunk total costs; and all six combined position/axis/cost cases.
`--storage-chunk-side` defaults to 512 base voxels.

Every `fiberlet_quantization` row is machine-readable. A scenario is
`valid=false` if any anchor key, variant, scalar, or endpoint cannot be
represented by the proposed format. Ordinary candidate or no-path rejection
after valid quantization is instead recorded as a changed DP result. Valid rows
report candidate additions/removals, graph and transition populations, global
and within-chunk cost ordering changes, top-100 agreement, comparisons against
both the float baseline and matching float-cost geometry, baseline/scenario
tracing failure counts, and symmetric maximum Euclidean, Lasagna-normal, and
Lasagna-tangential line distance. Restart segments remain disconnected; both
lines are sampled at no more than one base voxel spacing and projected onto the
other line's actual segments. Exact anchor, candidate, point-count, and
point-index agreement are not quality criteria. Separate
`baseline_reference_*` and `scenario_reference_*` summaries measure each replay
toward the annotated reference only. They report count, minimum, mean, median,
and maximum Euclidean, normal, and tangential distance in base voxels. The
Lasagna normal is sampled at the matched reference point; invalid normal samples
remain in the Euclidean summary and are counted separately. Extraction, DP,
baseline-to-scenario distance, and replay-to-reference distance stages emit
machine-readable progress and ETA updates.
