# Fiberlet crop tracing

`vc_fiber_trace_chunk` fills a base-coordinate crop with traces from a
preprocessed combined Fiberlet dataset. It does not read or regenerate the
original dense Fiber prediction. The authoritative output is a sparse Fiberlet
Zarr trace dataset; OBJ files are derived visualization artifacts.

```bash
volume-cartographer/build/bin/vc_fiber_trace_chunk \
  trace \
  /path/to/fiberlets.zarr \
  --normal-manifest /path/to/normals.lasagna.json \
  --bbox X0 Y0 Z0 X1 Y1 Z1 \
  --output crop_traces.zarr \
  --obj crop_fibers.obj
```

The crop is half-open and ordered base-volume XYZ. Stored anchor variants in
intersecting cells are processed from greatest prediction presence to least,
with the storage key as the deterministic tie break. Each uncovered anchor is
traced in both directions of its fitted axial direction. Fiberlets must pass
the stored join-angle constraint, revisiting an anchor is forbidden, and a
side stops at its first crop-boundary crossing or when the graph has no usable
continuation. The boundary segment is clipped to the crop.

Route selection is not limited to that output boundary. The tool expands the
internal graph and speculative-lookahead box on every face by `--lookahead`
base voxels. Seeds, coverage, stored paths, and visualization remain limited to
`--bbox`, but a seed near a crop face ranks candidates using the same lookahead
context as an interior seed. No extra maximum-Fiberlet-length padding is
needed: graph materialization retains each complete Fiberlet incident to an
anchor in the search box, including the final edge that crosses its boundary.
The graph-preparation cost therefore follows the expanded box rather than only
the requested crop.

Coverage suppression uses the same anisotropic measurement as Fiber replay.
The default radius is 20 base voxels along the local Lasagna normal and 80 base
voxels in its tangent plane. An anchor is suppressed only when its unoriented
fitted axis agrees with the covering trace tangent within 25 degrees, so a
crossing direction remains available as another seed. This first version does
not compare, split, merge, or deduplicate already accepted output lines.

The combined Fiberlet Zarr is authoritative. Present tuples contain graph data;
wholly absent sparse chunks are empty. A present anchor/prefix/route tuple must
be complete and valid. The original Fiber manifest and an expected-chunk index
are not inputs.

Crop materialization reads prefix and route owner chunks from the bounded
dependency halo around the lookahead-expanded search box, filters those records
to Fiberlets incident to an actual search-box anchor, and only then loads their
endpoint anchor chunks. It separately returns only requested-crop anchors as
seed candidates. An incomplete tuple required by a retained Fiberlet is an
error; incomplete tuples referenced only by unrelated owner-halo Fiberlets are
outside the search graph and are not read.

The normal manifest need not be the same file used during Fiberlet generation.
Its path and exact JSON bytes are not compared. It must describe the same base
coordinate domain: `base_shape_zyx` must ceil-downsample to the stored Fiberlet
prediction grid at the recorded prediction-to-base scale, and its uint8 3D
`nx`, `ny`, and `grad_mag` arrays must cover that base shape at their declared
scales. `nx` and `ny` must have equal shapes and effective base spacing, though
their storage chunk shapes may differ. Ordinary Lasagna array padding of up to
one chunk is accepted.

## Stored trace dataset

`crop_traces.zarr` uses the existing Fiberlet Zarr v2 envelope with dataset
kind `traces`, encoding profile `float64_traces`, and one opaque `traces`
array. Sparse chunks are aligned to the source Fiberlet spatial chunk side and
owned by the trace seed position. A missing chunk is empty; the root metadata
inventories every populated chunk and the total record count, so a missing or
unexpected file is rejected rather than silently treated as complete.

Each stored trace contains its deterministic result ordinal, float64 base-XYZ
seed position, float32 seed presence, float64 total metric cost, float64 traced
length in prediction voxels, and complete float64 base-XYZ polyline. These are
complete crop traces, not the short endpoint/lattice Fiberlets used by the
preprocessed graph. Trace chunks retain the shared field-wise Zstd and checksum
format but do not quantize trace geometry.

Total metric cost is the sum of selected edge and join costs. An edge clipped
at the crop boundary contributes the same retained fraction used for its
stored traced length. A bidirectional trace includes the central join once when
the graph defines that transition. Speculative lookahead cost is not stored.
The comparable visualization quality is
`total_metric_cost / path_length_prediction_voxels`; lower is better.

Every command that consumes a stored trace artifact accepts
`--quality-fraction F`. It keeps the best `ceil(F*N)` traces by this exact cost
density (stored ordinal breaks ties), restores their original order, and only
then performs visualization, direction fitting, splitting, constraint
extraction, consensus, or BP. For example, add `--quality-fraction 0.25` to
process the best quarter. The command prints the original and retained counts,
effective fraction, and worst retained density. Equal-density ties may be split
by ordinal, so the printed cutoff is diagnostic rather than a complete filter
predicate. The complete artifact is still read and validated; this reduces
downstream computation, not Zarr input I/O. Crop bounds and provenance remain
unchanged, and diagnostics retain the original stored trace IDs.

## H/V constraint diagnostics

The `constraints` mode derives candidate H/V and winding links directly from a
stored crop-trace dataset. It does not need the source Fiberlet graph and does
not write a constraint data artifact yet:

```bash
volume-cartographer/build/bin/vc_fiber_trace_chunk \
  constraints \
  crop_traces.zarr \
  --normal-manifest /path/to/normals.lasagna.json \
  --output crop_constraints.obj
```

`--output` is an OBJ basename; its final extension is removed. The command
writes `crop_constraints_perpendicular_same_winding.obj`,
`crop_constraints_perpendicular_separate_winding.obj`,
`crop_constraints_parallel_same_winding.obj`, and
`crop_constraints_parallel_separate_winding.obj`. It also solves the piece
labels and writes `crop_constraints_h_even.obj`,
`crop_constraints_h_odd.obj`, `crop_constraints_v_even.obj`,
`crop_constraints_v_odd.obj`, and `crop_constraints_broken.obj`. If omitted for
`crop_traces.zarr`, the basename defaults to `crop_traces_constraints` beside
the trace dataset. Each OBJ line joins a measured constraint's two closest
sampled points and is named `constraint_piece_A_B` from stable ascending global
piece IDs. Hard same-trace continuity links are excluded.

Pass `--constraints-per-fiber K` to apply opt-in strength pruning before the
constraint OBJ files and either labeling path. A cross-fiber link with parallel
score `p`, closest distance `d`, and extraction radius `D` is ranked by
`abs(2*p-1) * max(0, 1-d/D)`. This rewards decisive parallel/perpendicular
evidence and nearby fibers. Zero-strength links are discarded. Each endpoint
fiber nominates its strongest `K` individual piece-pair links, and only mutual
nominations form the initial sparse graph. Multiple piece links to the same
neighboring source fiber consume multiple slots. The reducer then adds the
minimum number of strongest discarded positive-strength bridges needed to
restore the input graph's connected components. It first prefers bridges that
keep both endpoint degrees within `K`, then uses a deterministic least-overflow
fallback. Consequently `K` is a target, not an unconditional final cap. Hard
same-trace continuity links remain unchanged and consume no slots.

The command prints extraction and pruning separately. The pruning table reports
input, mutual, and recovered cross-link degree and connectivity over
nondegenerate source fibers; isolated fibers count as components. It also
reports exact expected/accepted bridge counts, cap-respecting bridges, fallback
overflow bridges, and fibers ending above `K`. Fallback overflow describes this
greedy result and does not prove that a different degree-bounded forest is
impossible. `--exclude-parallel-separate-winding` is still a later
labeling-only filter, and the labeling report gives the final HiGHS edge count
after both stages. Omitting `--constraints-per-fiber` preserves the current
graph exactly.

The perpendicular view requires normalized perpendicular score `> 0.5` and
aligned winding distance `> 0.3`. Both parallel views require normalized
parallel score `> 0.5`; winding `< 0.5` is classified as same-winding and
winding `>= 0.5` as separate-winding. Threshold comparisons are exact, so a
score of `0.5` is absent and winding `0.5` belongs to separate-winding.
H/V-only diagnostics and winding BP discard measured links with aligned
winding distance greater than or equal to `4.0` by default. Legacy parity
labeling retains its representable default cutoff of `1.5`. Both are exclusive
cutoffs; same-trace continuity remains at winding zero.
Use `--winding-cutoff N` to select another positive exclusive cutoff.
Pass `--no-winding-cutoff` with `--hv-only` to retain every finite measured
winding distance for diagnostics. Invalid and non-finite samples remain
rejected. Joint parity labeling retains the finite `<1.5` invariant.

The five labels are H/V crossed with even/odd plus broken. For a retained link
with parallel score `p` and winding distance `d`, two active pieces pay
`1-p` when their H/V labels agree and `p` when they differ. Equal parity costs
`d`; different parity costs `abs(1-d)`. The two terms have equal weight.
Breaking a piece disables every incident link term and costs `0.5` times the
piece's retained incident-link count by default. Set another finite,
nonnegative coefficient with `--broken-cost-per-link`. HiGHS uses relative MIP
gap `1e-4` and absolute gap `1e-6` by default; `--mip-gap 0` requests an exact
relative-gap proof, and the achieved gap is reported. Within each active connected component, the equivalent
global H/V and parity flips are canonicalized so its lowest piece ID is
H/even. Isolated pieces with no evidence are broken.

Pass `--lp-relaxation` to replace the three binary piece variables by
continuous `[0,1]` variables while retaining the same linear objective and
constraint envelopes. The command does not threshold or canonicalize that
solution and does not write the five discrete label OBJs. Instead it writes
`<output-stem>_values.csv` with stable piece, trace, and within-trace piece
IDs followed by the raw `active`, `vertical`, and `odd` values. Console output
reports deciles for each variable. This mode is a diagnostic of relaxation
strength, not a discrete labeling.

For direct inspection, LP mode also thresholds the raw values into five OBJ
layers. `vertical >= 0.5` selects V, `odd >= 0.5` selects odd, and
`active >= mean(active)` selects an active label; lower activity is broken.
The literal suffixes are `_h_even.obj`, `_h_odd.obj`, `_v_even.obj`,
`_v_odd.obj`, and `_broken.obj`. The report prints the
actual mean activity threshold and every class count. These layers are only a
threshold visualization and are not presented as an optimized integer result.

LP edges use a stable gated XOR value for each label family: it is zero when
either endpoint is broken, otherwise zero means equal labels and one means
different labels. The lowest piece in each input connected component has H/V
and parity fixed to zero, but remains free to become broken. Every graph
triangle receives the four cut-polytope inequalities for H/V and again for
parity. These prohibit three mutually different binary relations around a
triangle while allowing the remaining active edge to differ if the third
piece is broken. Reports include gauge-root, triangle, and triangle-row counts.
All triangles are materialized deterministically; this diagnostic can consume
substantial time and memory on dense crops and does not silently omit cuts.

Pass `--exclude-parallel-separate-winding` for a solver-only ablation. It omits
non-hard measured links with `parallel_score > 0.5` and
`winding_distance >= 0.5` from degree penalties, adjacency, gauges, triangle
cuts, and objective terms. Exact `parallel_score == 0.5` remains included, and
hard continuity links are never removed. The four constraint OBJ files still
represent the complete extracted constraint set. Labeling output reports both
retained and excluded link counts.

Pass `--perpendicular-only` to retain only measured links whose normalized
perpendicular score is strictly greater than `0.5`. Exact `0.5` evidence is
ambiguous and excluded. Hard same-trace continuity links remain as the same
strong objective evidence used by the default model; this option does not turn
them into equality constraints. The filter applies identically to MILP and LP
before degrees, components, cuts, and costs are built. Constraint OBJ files
still show the complete extracted set. The option conflicts with the redundant
`--exclude-parallel-separate-winding` filter.

Pass `--hv-only` to omit parity from the labeling problem completely. This
keeps active/broken penalties, retained-link filtering, H/V costs, H/V gauges,
and H/V triangle cuts unchanged, but creates no parity piece columns, parity
XOR columns, winding objective terms, parity gauges, or parity triangle cuts.
For `N` pieces, `E` retained links, and `T` LP triangles, the reduced model has
`2N + 2E` columns and `N + 8E + 4T` rows; an integer solve has `2N` integer
columns. Reported winding cost is exactly zero. For output compatibility, raw
`odd` values are exact zero and every active discrete or thresholded piece is
classified as even. The existing CSV and five OBJ paths are therefore
unchanged, with valid but empty H/odd and V/odd OBJ files. This is opt-in; the
default joint H/V-plus-parity model is unchanged.

Pass `--exact-perpendicular-milp` together with `--hv-only` for a mixed
diagnostic with binary active/broken decisions and continuous `[0,1]` H/V
values. It conflicts with `--lp-relaxation` and LP-backend controls. For an
active retained link with parallel score `p`, its orientation loss is
`(1-p) + (2*p-1)*abs(h_a-h_b)`. Thus a fully parallel link pays the actual H/V
difference and a fully perpendicular link pays one minus that difference. A
perpendicular link incident to values `0.9` and `0.0` retains loss `0.1`; the
model cannot erase it with an independent edge-XOR value.

The exact mode keeps pair activity and difference columns continuous. Piece
activity is binary. Links with `p <= 0.5` receive a binary endpoint-order
column and two pair-gated big-M rows, while links with `p > 0.5` obtain exact
absolute differences from their positive objective coefficient. Consequently
all active-edge differences derive from shared piece values and no graph
triangles are enumerated or cut. For `N` pieces, `E` retained links, and `P`
links with `p <= 0.5`, the model has `2N + 2E + P` columns, `N + P` integer
columns, `N + 8E + 2P` rows, and zero triangle rows. It writes the continuous
values CSV and threshold OBJ layers used by LP diagnostics; activity values
are exact zero or one, parity remains zero, and H/V values remain continuous.

## Iterative H/V consensus

The separate `consensus` command reuses the same stored-trace loading, Lasagna
normal validation, piece extraction, spatial search, orientation scoring, and
exclusive winding cutoff as `constraints`, but does not construct a HiGHS
model. It operates on original stored traces: every retained cross-trace
piece-pair constraint contributes one item of evidence, while same-trace hard
continuity links are ignored. Multiple piece-pair constraints between the same
two source traces remain multiple evidence items.

The first trace is assigned H. The crop's nominal side is its smallest stored
XYZ extent, and the primary seed must have arc length strictly greater than
half that side. Eligible traces are ranked by endpoint-chord/arc-length
straightness, then by the smallest 3D Euclidean distance from the crop center
to the complete polyline, then by arc length and trace index. Comparisons use
the computed values directly without a tolerance. Later disconnected-component
seeds use the same ranking without the primary length cutoff, so short fibers
remain labelable. Subsequent candidates are ranked by
`constraint_count / mean_closest_distance_base_voxels` over links to already
assigned H/V traces. Zero mean distance has infinite priority. Remaining ties
prefer greater evidence count, smaller mean distance, and lower trace index.
Parallel confidence and winding do not affect this growth priority.

For the chosen trace, an H/V match across a link costs `1-parallel_score` and
an H/V mismatch costs `parallel_score`. These costs are summed over currently
active evidence. Broken costs `broken_cost_per_link * evidence_count` and wins
only on a strict improvement; exact ties prefer H, then V, then broken. This is
an irreversible greedy objective, not a rescore of the final graph. Once a
trace is broken, its links are disabled for later decisions and never charged
retroactively. A valid trace connected only to broken assignments starts a new
active-evidence component as H. Degenerate stored lines are broken immediately
and do not count as growth steps.

Final complete-trace layers are `<base>_h.obj`, `<base>_v.obj`, and
`<base>_broken.obj`. Snapshot triplets `<base>_step_N_h.obj`,
`<base>_step_N_v.obj`, and `<base>_step_N_broken.obj` are written after 10, 20,
..., 100 assignments and then every 100 assignments. A snapshot contains the
labels immediately after assignment `N`, including a broken choice made at
that step. `N` includes component seeds and broken decisions but not degenerate
input lines. Degenerate non-assignments occur in none of the three layers.
All three files are created even when a class is empty, and final files remain
separate when the final count is also a snapshot milestone. A user-supplied
`--output` basename owns and overwrites all three final and milestone layers.

```bash
volume-cartographer/build/bin/vc_fiber_trace_chunk consensus \
  /tmp/crop_traces_central_384.zarr \
  --normal-manifest /path/to/normals.lasagna.json \
  --output ./384 \
  --piece-length 1000000000 \
  --max-distance 256 \
  --broken-cost-per-link 0.25
```

Constraint extraction options and `--broken-cost-per-link` are accepted.
HiGHS-only options such as `--lp-relaxation`, `--hv-only`, and
`--exact-perpendicular-milp` are rejected. Console output includes each seed's
straightness, crop-center distance, and arc length, plus a detailed table for
the first 100 assignment choices with connectivity evidence and all three
candidate costs. The complete assignment, label-count, and objective summary
is printed after that table as the final output block.

HiGHS' LP backend can be selected explicitly for this diagnostic. Add
`--lp-parallel` to request parallel execution, and use `--lp-solver choose`,
`simplex`, `hipo`, or `ipm` to select the LP algorithm. These flags require
`--lp-relaxation`; they do not apply to the MILP. The default remains HiGHS
automatic solver and parallel selection (`choose`), which may choose serial
simplex even when `--threads` is greater than one. The relaxation report prints
the requested solver, requested parallel mode, and thread count; CPU-versus-wall
measurements are needed to determine whether the selected algorithm ran in
parallel.
Backend availability depends on how HiGHS was built. In particular, a HiGHS
CLI may advertise `hipo` even when its linked build lacks the linear-algebra
backends required to select it; that condition is reported as an error rather
than silently falling back to another solver.

Only the three piece columns are integer. Pair-active and gated difference
columns are continuous in `[0,1]`; their exact AND/XOR linear envelopes force
binary values for binary endpoints, avoiding three unnecessary integer columns
per link without changing the feasible labels or objective.

All distances are in base voxels. By default, traces are resampled every 32
voxels and divided into evenly sized overlapping pieces with a maximum target
length of 512 and overlap 128. Distinct traces are searched within 128 voxels.
The point R-tree is only a broad phase: reported neighbors pass an exact
Euclidean-distance test, and only the closest sampled pair for each unordered
piece pair is scored. Pieces from the same trace are excluded from that search;
each consecutive pair instead receives a hard parallel-continuity link with
parallel score 1 and winding distance 0.

For a measured pair, centered 32-voxel secants determine the initial tangent
orientation. The closest sampled pair remains the seed and first winding
connector. By default, both pieces are walked in both directions at the sample
spacing and a bounded anti-correlated phase shift of one twentieth spacing is
retained when it reduces connector distance. Pass
`--parallel-correspondence perpendicular-grid` to instead walk incrementally
from the last accepted pair and independently vary both advances in
one-twentieth-spacing increments by up to one quarter of the spacing. The grid
minimizes the sum of the two squared target-step-normalized advance residuals
and the squared alignment of the unit connector with each centered tangent;
connector length is not scored.
The experimental grid can be swept without recompilation using
`--parallel-grid-step`, `--parallel-grid-limit`, `--parallel-step-weight`, and
`--parallel-perp-weight`. Optional `--parallel-direction-weight` and
`--parallel-length-weight` terms penalize changes from the previous connector;
both default to zero. Fractions are relative to `--sample-step`; objective
weights are nonnegative and cannot all be zero. These options do not affect the
default `distance` correspondence.
The winding-factor CSV includes correspondence sample count, normalized
advance residual, connector/tangent absolute dot, normalized connector-length
change, connector-direction change, and grid-limit hit fraction. These columns
are populated when `--parallel-diagnostics` is passed. They do not affect
candidate selection, winding integration, or BP; collection is off by default
to avoid adding work to the normal distance path.

On the fixed 1024-crop, eight-reference diagnostic, the distance mode retained
the best measured result (`1784/2839`, 62.839%, with 8/8 exact reference
windings). Tested perpendicular grids ranged from 62.359% to 62.698%. The best
grid improved mean connector perpendicularity from 0.266 to 0.254, but changed
261 quantized parallel-winding bins; 246 of those baseline measurements were
within 0.10 winding of a half-integer boundary. This threshold sensitivity is
why the distance mode remains the default.
The mean consistently oriented tangent dot is clamped to `[0,1]` as raw
parallel evidence. Raw perpendicular evidence is `1 - abs(initial tangent
dot)`; the two values are divided by their sum, so the reported normalized
scores add to one.

Winding distance uses the ordinary Lasagna straight-connector integral, but
each endpoint density sample is multiplied by the absolute alignment between
the connector and the decoded local Lasagna normal before trapezoidal
integration. This suppresses winding evidence where the connector lies in the
local tangent plane. Missing required density or normal samples reject that
candidate.

Constraint orientation scores are computed in deterministic parallel slots.
Accepted connectors are then sampled through the shared grouped Lasagna corner
path in bounded float-coordinate batches: `grad_mag`, `nx`, and `ny` chunk
dependencies are laid out once per batch, values are materialized in parallel,
and winding integrals are reduced independently. When density and normals use
different compatible grids, density is one batch and the paired normal channels
are another. There are no per-point channel-cache lookups. Console timing
separates orientation and winding work.

Use `--sample-step`, `--piece-length`, `--piece-overlap`, `--max-distance`,
`--tangent-window`, and `--winding-step` to change the defaults. `--threads`
defaults to the host CPU count. The console report contains input and rejection
counts, phase timings, and `q0` through `q100` deciles for measured closest
distance, normalized parallel/perpendicular evidence, and aligned winding.
Parsed normal-manifest equality with trace provenance is reported only as a
diagnostic. Compatibility requires valid normal channels in base coordinates
whose declared base shape covers the stored trace crop.

HiGHS is a required C++ build dependency for `vc_fiber_tracer`: Ubuntu uses
`libhighs-dev`, macOS uses the Homebrew `highs` formula, and Windows uses the
vcpkg `highs` port. CMake consumes the package target `highs::highs`.

Publication is all-or-nothing: the command writes and fully reopens a unique
sibling temporary dataset, validates its inventory, ownership, ordinals, and
record count, then renames it to the requested path. The output path must not
already exist. Trace mode generates line artifacts only from that reopened
dataset, never from its in-memory tracing result.

To regenerate line visualization later without source Fiberlets, normals, or a
CT volume, run:

```bash
volume-cartographer/build/bin/vc_fiber_trace_chunk \
  visualize \
  crop_traces.zarr \
  --output crop_fibers.obj
```

If `--obj` is omitted in trace mode, the line OBJ defaults beside the trace
dataset with the `.zarr` suffix replaced by `.obj`.

## CT box visualization

Pass one concrete uint8 CT OME-Zarr array/group with `--volume`:

```bash
volume-cartographer/build/bin/vc_fiber_trace_chunk \
  trace \
  /path/to/fiberlets.zarr \
  --normal-manifest /path/to/normals.lasagna.json \
  --bbox X0 Y0 Z0 X1 Y1 Z1 \
  --output crop_traces.zarr \
  --obj crop_fibers.obj \
  --volume /path/to/ct.ome.zarr/2
```

The command uses the existing VC3D fine-to-coarse coordinate sampler and the
OME-Zarr transform advertised for that exact group. It writes six independent
`crop_fibers_volume_slices_<face>.obj` meshes, six TIFF textures, and one shared
MTL beside the line OBJ. `--texture-max` limits either texture dimension.

Useful controls are `--lookahead`, `--beam`, `--coverage`,
`--coverage-angle`, `--cache-gib`, `--max-attempts`, and `--max-fibers`.
`--threads` controls bulk graph preparation and independent seed tracing, and
defaults to the host CPU count. Before tracing, the tool loads the search
box's incident Fiberlets, route geometry, endpoint anchors, and joins once into
an immutable in-memory graph. Trace workers do not query the chunk cache. The
graph-preparation line reports the requested crop, expanded search bounds, and
padding; timing output reports graph preparation and tracing separately. Both
limits use zero for
unlimited:
`--max-attempts` counts uncovered seed attempts, including failed/no-edge
attempts, while `--max-fibers` counts accepted lines. Seeds are attempted from
highest prediction presence to lowest, with storage key as the deterministic
tie break.

Pass `--profile-memory` to print a profiling sample approximately once per
second throughout graph preparation and tracing. Each sample reports current
and peak process RSS on Linux, the live/capacity bytes and pending work for the
independent anchor, path, and normal caches, chunk-loading progress, and the
current anchor, Fiberlet, route-point, profile-segment, successor, and replay-
transition counts. On platforms without `/proc/self/status`, RSS fields are
reported as `NA`; the portable structure and cache counters remain available.

The materialized crop graph stores admissible joins as CSR adjacency keyed by
dense directed-arc indices. Each entry contains the outgoing 32-bit arc index
and the original `FiberletPathCost`; it does not quantize or recompute costs.
This avoids retaining duplicate successor and full storage-ID transition
records and lets tracing search only the contiguous transitions for its current
incoming arc.

Seed graph traversal over the materialized graph is read-only and concurrent.
The canonical seed set is unchanged; additional endpoint anchors needed to
close crop traversal do not become new starts. Results are integrated
serially in the same strongest-first order, including all counters and
anisotropic coverage updates. Workers are fed continuously through dense seed
tickets; a bounded queue holds completed work until every preceding ticket is
ready. If an earlier integrated line covers a speculative seed, that result is
discarded and does not consume an attempt. Attempt/fiber limits and failures
are also applied at the ordered frontier, so work beyond the equivalent serial
stop point cannot affect output.

The materialized graph uses sorted contiguous records and flat directed
adjacency. Lookahead reads adjacency and forward/reverse route geometry through
stable views and allocates clipped geometry only for a selected continuation.
Immutable views borrow graph storage without reference counting. The common
view interface also supports one shared owner for a complete cache-derived
result; ownership is per query, never per point, edge, or search state.

Each lookahead keeps the already committed visited anchors as one read-only
set. Speculative branches add compact parent-linked route nodes instead of
copying that set and their complete arc prefix. Local cycle checks walk the
short rollout ancestry. Full arc lists are reconstructed only when the
intermediate beam cap requires lexicographic ranking; terminal candidates are
compared directly through their parent links. The cost accumulation, density
ordering, lexicographic tie break, generated-state limit, and chosen route are
unchanged.

The final timing line reports `lookahead_route_nodes_max` and
`lookahead_route_bytes_max`, the largest retained parent arena and its allocated
capacity observed in any computed seed candidate. These are diagnostic
high-water marks, not cumulative memory totals across workers.

## Principal direction groups

After tracing, the tool analyzes every nonzero consecutive step of every
accepted polyline. Each normalized step is an unoriented axial observation
weighted by its base-voxel length. A deterministic multi-seed two-line fit
maximizes
`sum(length * max((step dot direction1)^2, (step dot direction2)^2))`.
The two fitted directions are independent and therefore need not be
orthogonal; the global axial PCA tensor is used only to seed this fit.

The line grouping retains gradual angular information instead of converting
each step to a binary vote. Let `q=(direction1 dot direction2)^2`. A unit step
axis `u` has calibrated support
`clamp(((u dot direction)^2-q)/(1-q),0,1)` for each direction. Support is
multiplied by step length and accumulated over the fiber. An exact fitted-axis
step therefore has support one for that direction and zero for the other even
when the fitted axes are non-orthogonal. Off-axis and bend segments contribute
less. The two supports are independent affinities and can sum to more than one;
they are never renormalized against one another.

A complete fiber is direction-1- or direction-2-dominant when its larger
accumulated support divided by its actual valid arc length reaches the selected
fraction. The default is 0.75; pass `--direction-dominance F`, with finite `F`
in `(0.5, 1]`, to `trace` or `visualize` for a stricter or looser split. The
comparison is inclusive and an exact support tie prefers direction 1. Other
fibers, including a degenerate fiber with no nonzero step, are mixed. Nearly
identical fitted axes use direction 1's ordinary squared alignment and zero
direction-2 support rather than dividing by an unstable axis separation.

Direction identities remain deterministic: the fit's hard nearest-axis
assignments are used only to order the axes by total assigned length, with
canonical axis order breaking an exact tie. They do not classify fibers. The
command reports both fitted axes, analyzed step count and length, selected
fraction, and all three counts.

For example, reclassify an existing stored crop without retracing:

```bash
volume-cartographer/build/bin/vc_fiber_trace_chunk visualize \
  crop_traces.zarr \
  --output crop_fibers.obj \
  --direction-dominance 0.9
```

The requested `crop_fibers.obj` contains the complete line set. The same
directory also receives independently displayable subsets and actual
seed-anchor point objects:

| Contents | Lines | Seed anchors |
| --- | --- | --- |
| All accepted fibers | `crop_fibers.obj` | `crop_fibers_anchors.obj` |
| Direction 1 dominant | `crop_fibers_dir1.obj` | `crop_fibers_dir1_anchors.obj` |
| Direction 2 dominant | `crop_fibers_dir2.obj` | `crop_fibers_dir2_anchors.obj` |
| Mixed | `crop_fibers_mixed.obj` | `crop_fibers_mixed_anchors.obj` |

The anchor artifacts use OBJ point (`p`) elements at the stored trace seed,
not a polyline endpoint. Empty groups still produce valid empty OBJ files.
This classification is output-only and cannot change tracing, coverage, the
stored trace artifact, fitted axes, per-direction supports, or quality ranks.
`visualize` still rewrites the complete all/direction/anchor/quality artifact
family beneath the selected output base.

## Direction-label MILP diagnostic

Use `direction-diagnostic` to treat the gradual direction split as a diagnostic
reference for the constraint-based H/V labeling:

```bash
volume-cartographer/build/bin/vc_fiber_trace_chunk direction-diagnostic \
  crop_traces.zarr \
  --normal-manifest normals.lasagna.json \
  --output crop_direction_diagnostic \
  --direction-dominance 0.9
```

The command first writes the complete, unfiltered direction visualization as
`crop_direction_diagnostic_initial.obj` and its `_dir1`, `_dir2`, `_mixed`, and
anchor siblings. It then removes mixed fibers before piece splitting,
constraint extraction, optional `--constraints-per-fiber` pruning, and the
ordinary discrete H/V-plus-broken MILP. Parity is disabled because the initial
direction split supplies only H/V reference labels. The existing constraint
connector and H/V/broken piece OBJ files use `crop_direction_diagnostic` as
their basename.

Raw H/V labels are not directly comparable across disconnected graph
components because each component has an arbitrary binary orientation gauge.
For reporting, the command independently flips each active component only when
that reduces disagreement with the initial direction split. It prints the raw
solver counts, a gauge-aligned confusion table, separate orientation and broken
errors, piece and represented-fiber error rates, and one row for each erroneous
piece with its original fiber ID and arc interval. An all-mixed input remains a
successful empty diagnostic and still writes every expected output family.

To measure how uncertain fibers affect the labeling graph, run the cumulative
mixed-fiber ablation:

```bash
volume-cartographer/build/bin/vc_fiber_trace_chunk direction-ablation \
  crop_traces.zarr \
  --normal-manifest normals.lasagna.json \
  --output crop_direction_ablation \
  --direction-dominance 0.9
```

Mixed fibers are admitted five at a time by default from highest to lowest
directional support fraction; use `--ablation-step N` to change the stride, and
use `--ablation-limit N` to stop at a fixed ranked prefix while tuning solver
costs. Without a limit the complete mixed cohort is admitted, and the final
remainder is always included. Confidence ranking controls membership
only; every Mixed
fiber remains a defect reference and is expected to optimize to Broken rather
than H or V. Every retained checkpoint remains in original stored-fiber order.
Constraint extraction, optional pruning, the discrete H/V-only MILP, and its
LP relaxation are recomputed at every checkpoint. LP activity and H/V are
thresholded at 0.5. Each checkpoint prints both solve times, solver status,
gap, objective, graph size, raw labels, and separate H/V-fiber and mixed-defect
errors for MILP and thresholded LP. Component gauge alignment is chosen exclusively from
trusted active pieces, preventing uncertain additions from changing the frame
used to judge the original split.

The initial direction OBJ family is written once beneath
`crop_direction_ablation_initial`. Only the final admitted checkpoint
writes constraint and H/V/broken OBJ layers; intermediate checkpoints are
statistics-only.

For a no-split checkpoint-40 comparison using only perpendicular measured
links, set an effectively infinite piece length explicitly:

```bash
volume-cartographer/build/bin/vc_fiber_trace_chunk direction-ablation crop_traces.zarr --normal-manifest normals.lasagna.json --output crop_direction_ablation_perpendicular_40 --direction-dominance 0.9 --broken-cost-per-link 0.2035 --ablation-step 40 --ablation-limit 40 --piece-length 1000000000 --perpendicular-only
```

Remove only `--perpendicular-only` and change the output basename to obtain the
directly comparable all-constraint no-split baseline.

Add `--post-iterations N --post-influence I` to a perpendicular-only,
no-split direction ablation to derive continuous post-solve H values. H starts
at `1`, V at `0`, and Broken at `0.5`. A perpendicular neighbor contributes
`1-v`, weighted by:

```text
clamp((abs(v - 0.5) - 0.5 * (1 - I)) / (0.5 * I), 0, 1)
```

Updates are synchronous. With `I=1`, confidence rises linearly from zero at
`0.5` to one at either extreme. With `I=0.5`, values in `[0.25,0.75]` have no
influence. A fiber with no positively weighted neighbor keeps its prior value.
This diagnostic requires exactly one solved piece per represented fiber and
does not add non-admitted fibers or alter the MILP result.

The output files are `<base>_p0.obj` through `<base>_p9.obj`, covering fixed
H-value bands `[0,0.1)` through `[0.9,1]`. Exact boundaries use the higher
band, so equal values remain together. Each file contains complete source
fibers and is written even when empty.

The band table also reports the original H, V, and Mixed populations and the
existing gauge-aligned labeling errors in each band. H and V errors are
trusted-direction mismatches; Mixed errors are admitted diagnostic defects
that the labeling left active. The table only stratifies the existing
comparison and does not introduce a post-filter error threshold.

### BP-only constraint-consistency diagnostic

Use `--bp-only --perpendicular-only` with `direction-ablation` to build only
the final admitted cohort, extract and optionally prune its constraints, and
run natural binary min-sum BP without constructing either HiGHS model:

```bash
volume-cartographer/build/bin/vc_fiber_trace_chunk direction-ablation crop_traces.zarr --normal-manifest normals.lasagna.json --output crop_bp --direction-dominance 0.9 --piece-length 512 --perpendicular-only --bp-only
```

Every extracted piece is one BP node. Consecutive overlapping pieces from one
source fiber have exact edge-local continuity by default: two active endpoints
must share H/V, while either endpoint may be Defect and thereby split the
source into independent active runs. Pass `--split-continuity finite` to
restore their canonical parallel-score-1 finite same-label evidence. The
established central-straight rule still selects a
source fiber, then the exact clipped piece of that source closest to the crop
center is fixed to H. Optional balance uses piece arc lengths, including
overlap. The Mixed coefficient is multiplied by each piece's retained incident
measurement count and applied once as a node-local unary.

`--ablation-limit N` selects a ranked prefix; omission admits every Mixed
fiber. Natural BP fixes the established central straight seed to H and emits
`<base>_orientation_p0.obj` through `_p9.obj`. A nonconverged status remains in
the report and in every CSV row; it is not silently treated as converged.

`<base>_consistency.csv` records global piece ID, original stored source
trace ID, source-local piece ID, and begin/end base-voxel arcs alongside the
initial `dir1`, `dir2`, or `mixed` source diagnostic reference. Band, state,
confusion, consistency, and AUROC outputs are piece-weighted. Their OBJs contain
the exact dense source polyline clipped to the piece interval; overlapping
geometry therefore appears twice intentionally. A full-range one-piece input
retains the original source polyline exactly. The diagnostic uses
V values at or below `0.25`, H values at or above `0.75`, and treats values
between them as unresolved. Duplicate measurements between the same two
pieces form one factor: degree counts that factor once, `incident_measurements`
retains its measurement count, and strength is the absolute difference of its
summed same-label and different-label costs.

Hard mismatch rates include only links whose endpoints are both resolved.
Undefined ratios are serialized as `NA` and excluded from group summaries.
`soft_mismatch_proxy` is the strength-weighted independent-endpoint
probability of violating each factor's preferred same/different relation; it
is not a calibrated BP edge marginal.
`neighbor_support_balance` measures how evenly neighbors support H and V and
can be high when neighbors themselves are uncertain, so
`neighbor_certainty` is reported alongside it. Console summaries give
equal-per-piece min/mean/median/p90/max values by initial reference group and
tie-aware AUROC values for Mixed versus trusted fibers.

Add `--bp-inference sum-product` to run binary sum-product BP over the same
merged orientation factor graph and the same hard-H central straight seed.
Omit `--perpendicular-only` to retain both parallel- and
perpendicular-preferring measurements:

```bash
volume-cartographer/build/bin/vc_fiber_trace_chunk direction-ablation crop_traces.zarr --normal-manifest normals.lasagna.json --output crop_bp --direction-dominance 0.9 --piece-length 512 --bp-only --bp-inference sum-product
```

For same-label and different-label factor costs `E_same` and `E_diff`, this
mode uses potentials `exp(-E_same/T)` and `exp(-E_diff/T)`, where `T` is
`--bp-temperature`. Messages are normalized H-vs-V log ratios and the reported
horizontalness is `P(H)`, obtained directly from the node log odds. These are
exact marginals on trees and loopy-BP approximations after convergence on
cyclic graphs. A `message_limit` status exposes the final finite iterate but
does not claim convergence.

For every raw measurement or continuity link with parallel score `p`,
same-label cost receives
`1-p` and different-label cost receives `p`. Measurements for the same fiber
pair are summed, then the smaller merged cost is subtracted from both. The
remaining nonzero gap `abs(E_same-E_diff)` is the constraint's decisiveness:
scores near `0.5` are weak, while scores near `0` or `1` are strong. Exactly
canceled merged factors are omitted from inference, degree, components, and
mismatch statistics and are reported separately as neutral factors and
measurements. `--perpendicular-only` remains available to reproduce the
restricted graph.

Temperature has different semantics in the two modes. Sum-product applies it
to factor energies during inference; min-sum applies it only when mapping a
min-marginal advantage to a display value. Sum-product accepts the shared
message-iteration, damping, and residual controls but rejects `--bp-balance`
and the min-sum-only target, balance-strength, balance-iteration, and balance-
tolerance controls.

Sum-product writes `<base>_orientation_p0.obj` through `_p9.obj` and
`<base>_consistency.csv`. A later run intentionally replaces these current-result
artifacts instead of encoding the selected solver in every filename. The CSV
records its inference name, temperature, and convergence status. The
soft mismatch diagnostic remains an endpoint-independence proxy; it is not a
sum-product pairwise marginal.

Add `--bp-inference sum-product-mixed` to test an explicit categorical
V/Mixed/H variable. Mixed means an orientation defect, not a third physical
direction. After the same merged-factor normalization, its pairwise energies
are:

```text
E(V,V) = E(H,H) = normalized_E_same
E(V,H) = E(H,V) = normalized_E_diff
E(Mixed,*) = E(*,Mixed) = 0
```

At least one normalized oriented energy is therefore zero. An indecisive or
exactly canceled factor has both energies zero and is omitted, so it cannot
spuriously favor Mixed merely because the unnormalized oriented costs shared a
positive constant offset.

Every non-seed piece assigned Mixed instead pays one node-local unary energy
`U(Mixed)=mixed_cost_per_constraint * incident_measurements`, with
`U(V)=U(H)=0`. Set the coefficient with `--bp-mixed-cost F`; its default is
`1`. Incidence sums raw measurements in retained non-neutral merged factors;
omitted neutral measurements count zero. Mixed therefore conditionally disables incident orientation
terms: a source known to be Mixed sends a uniform factor message, rather than
encouraging its neighbor to become Mixed. A soft piece with residual V/H
probability can still transmit that residual orientation evidence.

The hard seed remains exactly H and replaces the unary at that node. Every
directed message contains three normalized log-values. For a non-seed outgoing
cavity, the solver adds the node's scaled unary exactly once; it adds the same
unary exactly once to the final node marginal. The reported `p_v`,
`p_mixed`, and `p_h` are normalized node marginals. An isolated unseeded fiber
has uniform probabilities because its incident count is zero. An
unseeded connected component retains exact V/H gauge symmetry, although its
Mixed marginal generally differs from one third.

The scalar `orientation_projection = p_h + 0.5*p_mixed` exists only for the
legacy H/V band visualization and explicitly labeled heuristic consistency
output. It is not `P(H)` and is not a calibrated binary marginal. Direct Mixed
diagnostics report a tie-aware `p_mixed` AUROC, state-marginal summaries, and
an argmax V/Mixed/H confusion table with exact ties in a separate column.
Mixed-state soft mismatch and neighbor-support diagnostics use the explicit
V/Mixed/H probabilities instead: Mixed endpoint mass contributes no orientation
violation or neighbor support.

This mode writes `<base>_orientation_p0.obj` through `_p9.obj` for the
orientation projection, `<base>_error_probability_p0.obj` through `_p9.obj`
for `P(Mixed)`, and `<base>_consistency.csv` with all three probabilities. It
also writes the direct argmax partition as `<base>_v.obj`, `<base>_err.obj`,
`<base>_h.obj`, and `<base>_tie.obj`. Like binary sum-product, it rejects
population-balance controls.

### Standalone Lasagna normal sign alignment

`vc_lasagna_normal_align` resolves local sign ambiguity in normals sampled
from a regular Lasagna `grad_mag`/`nx`/`ny` manifest. It uses the same shared
binary sum-product message engine as fiber BP, but it does not run H/V
classification and does not use the legacy `NormalGridVolume` format.

```bash
volume-cartographer/build/bin/vc_lasagna_normal_align \
  /path/to/normals.lasagna.json \
  --bbox X0 Y0 Z0 X1 Y1 Z1 \
  --output normal_alignment
```

The bbox is half-open base-voxel XYZ. The default spacing is the physical
`nx`/`ny` Lasagna level spacing. Samples are globally anchored at integer
multiples of that spacing, so overlapping bboxes contain identical sample
coordinates. Remote manifests additionally require `--remote-cache-dir`.

After stable removal of invalid (`grad_mag <= 0`, missing, or degenerate)
samples, the default graph connects each retained pair in a one-cell
Chebyshev neighborhood exactly once. For signed normalized dot product `d`,
the pair costs are:

```text
E_same      = (1-d)/2
E_different = (1+d)/2
```

Thus the preferred relation follows the sign and its strength follows
`abs(d)`. Exact neutral links are omitted. Each connected component fixes its
lowest retained node to its original sign because pairwise ambiguous axes do
not define an absolute outward/inward orientation. Posterior flip probability
strictly above `0.5` flips a sample; ties do not. A finite nonconverged final
iterate is reported and still visualized.

The command writes `<base>_unaligned.obj` and `<base>_aligned.obj` using the
exact same centers. Each sample has two short crossed strokes in its normal
plane under `g normal_bases` and one longer center-to-normal stroke under
`g normal_directions`. Glyph dimensions can be overridden in base voxels.
The source Lasagna Zarr is never rewritten.

`--threads` controls both Lasagna batch sampling and binary BP. Large BP
graphs use deterministic OpenMP parallelism in GCC builds; Clang and MSVC use
the project's serial OpenMP shim. The summary prints the effective worker
count plus BP setup, node-total, message-update, solve, and total times. Node
totals retain original factor order and message iterations remain synchronous,
so worker count does not change inference results. `--damping 1` can converge
faster for well-behaved graphs, but the more conservative default remains
`0.5` because undamped loopy BP is less robust in general.

The regular spatial lattice is also used to orient winding evidence during
crop BP. The crop path samples the stored crop plus a one-normal-spacing halo,
clipped to `base_shape_zyx`, through the already open normal sampler and cache.
It does not reopen or rewrite the manifest.

### Signed winding inference

`direction-ablation` BP aligns the crop normals once before extracting its
first constraint cohort. For an ordered measured constraint `A -> B`, the
existing `winding_distance` remains a nonnegative normal-modulated magnitude.
An additional BP-only target is signed by
`dot(point_B-point_A, aligned_normal_at_midpoint)`, so it denotes
`winding(B)-winding(A)`. The nearest globally anchored normal samples at A,
the midpoint, and B must all exist and belong to the same normal-alignment
component. Otherwise the H/V constraint remains valid but contributes no
perpendicular winding evidence. A connected winding graph that would combine
signed evidence from independently gauged normal components is rejected.

The H/V-aware winding BP solvers split each measured factor into two winding
components. The parallel component uses the unsigned measured
`winding_distance`, rounded to the nearest nonnegative integer:

```text
[0,0.5) -> 0
[0.5,1.5) -> 1
[1.5,2.5) -> 2
```

Its loss is `parallel_weight * abs(abs(latent_delta) - integer_distance)`, so
the same-winding zero bin needs no signed normal observation. Gain/scale never
calibrates this exact integer ladder. The perpendicular component uses the
aligned signed observation and maps every nonzero magnitude to its
half-integer interval center while preserving sign:

```text
(0,1] -> 0.5
(1,2] -> 1.5
(2,3] -> 2.5
(3,4) -> 3.5 with the default exclusive cutoff
```

Exact signed zero and hard same-trace continuity remain zero. The extraction
cutoff is not re-applied to either quantized value. Alternating and joint-grid
H/V winding solvers, including continuous initialization and decoded energy,
share these conversions.
The independent integer-only winding diagnostic remains raw because it has no
half-offset H/V lattice.

The dominant hypothesis is multiplied by
`2^-floor(abs(effective_target))`. Parallel integer distances `0`, `1`, `2`,
and `3` therefore use multipliers `1`, `0.5`, `0.25`, and `0.125`, while
perpendicular half-integer targets `0.5`, `1.5`, `2.5`, and `3.5` use the same
sequence starting at `1`. It does not scale the separate H/V relation energy,
hard same-trace continuity, or the strict sign requirement on a nonzero signed
perpendicular observation. The formula continues for larger bins admitted by
an explicit extraction-cutoff override. Parallel and perpendicular hypotheses
are mutually exclusive for winding: only the larger score contributes winding
energy. Ties select perpendicular. The H/V relation potential still uses the
complementary scores as alternative same/different costs. Factor CSV output
retains the raw scores and targets while its effective weights identify the
selected winding term.

`--parallel-winding-cutoff N` applies an additional exclusive filter only to
parallel-dominant integer winding evidence. For example, `0.5` retains only
same-winding distance `0`. A filtered parallel factor still contributes its
ordinary H/V orientation score. Perpendicular-dominant factors, extraction,
and stored constraints are unchanged; reference diagnostics expose the
suppression through `used_w`. The default is no additional filter.
Factor/message connectivity remains intact after filtering. Each message
component retains one H/V class gauge, while disconnected effective-winding
subgraphs receive separate integer-zero gauges. An extra integer gauge does
not fix H or V, so orientation-only evidence continues to propagate across a
filtered winding component.

Fixed phase `0.5` and measurement scale `0.822` are the joint-grid CLI
defaults. The scale comes from the combined continuous reference/reference
constraint fit; the perpendicular model predicts measured winding delta as
`latent / 0.822`. Pass `--winding-adaptive-calibration` to infer phase and
scale instead. Adaptive calibration can move latent phase or measurement scale
even though the observations are quantized.

For `sum-product-mixed`, `--winding-solver joint-grid` is the default. It runs
one inference over `(H,k)`, `(V,k)`, one winding-free Defect state, the global calibration grid, and one local
ladder-order sign per connected component. It does not run the earlier
orientation BP first. The alternative
`--winding-solver alternating` retains the established orientation pre-pass,
multi-start calibration, and numerical behavior for direct comparison.

Either solver can instead freeze orientation before winding:

```bash
volume-cartographer/build/bin/vc_fiber_trace_chunk direction-ablation crop_traces.zarr --normal-manifest normals.lasagna.json --output crop_bp_fixed_orientation --bp-only --bp-inference sum-product-mixed --winding-fixed-orientation
volume-cartographer/build/bin/vc_fiber_trace_chunk direction-ablation crop_traces.zarr --normal-manifest normals.lasagna.json --output crop_bp_fixed_orientation_alternating --bp-only --bp-inference sum-product-mixed --winding-solver alternating --winding-fixed-orientation
```

`--winding-fixed-orientation` runs the ordinary H/V/Mixed sum-product BP first.
Set its Mixed coefficient with `--bp-mixed-cost`. Set the later winding-stage
Defect coefficient independently with `--winding-defect-cost`. The orientation
Mixed coefficient defaults to `1`; the winding-stage Defect coefficient defaults
to `100`. Neither inherits an explicitly changed value from the other.
Each piece's unique MAP direction is then fixed; an exact class tie becomes
Defect. During winding, a fixed H piece can be H or Defect and a fixed V piece
can be V or Defect, but neither can switch to the opposite direction. A
pre-pass Defect remains Defect. A directional piece has one fixed-direction
state per integer winding plus one Defect state; Defect is not duplicated over
the winding support. The chosen backend still controls phase/scale calibration,
component sign, integer-support expansion, and winding factor evaluation.

A late Defect pays `--winding-defect-cost` times its retained incident winding
measurement count at `--bp-temperature` and is
selected only from winding evidence; fixed mode does not repeat the H/V
same/different orientation factor. A Defect assignment makes every incident
pair factor neutral, disabling orientation, winding, calibration, and component
sign evidence. Constraints attached to a pre-pass Defect are also removed from
the continuous initializer and component graph.
Split-piece continuity is exact and edge-local by default. Two active endpoints
of a same-source continuation edge must have the same H/V class and integer
winding. A Defect endpoint neutralizes that edge, so an active run may resume
with a different state after a Defect gap. The orientation prepass and final
winding decode enforce this pairwise invariant so independently decoded
loopy-BP marginals cannot publish an active-active mismatch. Pass
`--split-continuity finite` to recover the previous pairwise behavior. In that
compatibility mode, `--piece-break-cost F` discourages a late Defect boundary:
the nonnegative cost is charged once when exactly one endpoint is active and is
divided by `--bp-temperature`. It is not charged for active/active,
Defect/Defect, or measured cross-trace pairs. Its default is `0`.
The consistency CSV preserves the soft pre-pass `p_v`, `p_mixed`, and `p_h`
columns and records `winding_prepass_class`, `winding_final_class`, and final
H/Defect/V probabilities. It writes `winding_valid=0` and `NA` for numeric
winding fields of final Defect pieces. Defect pieces are excluded from `w_N`
OBJ layers and remain visible in the ordinary Defect class OBJ.
The exact fixed assignment passed to winding is written separately as
`<base>_prepass_h.obj`, `<base>_prepass_v.obj`,
`<base>_prepass_err.obj`, and `<base>_prepass_tie.obj`; the tie layer is empty
because prepass ties become Defect. Final `<base>_{h,v,err,tie}.obj` layers
continue to describe the post-winding result.

```bash
volume-cartographer/build/bin/vc_fiber_trace_chunk direction-ablation crop_traces.zarr --normal-manifest normals.lasagna.json --output crop_bp --bp-only --bp-inference sum-product-mixed
volume-cartographer/build/bin/vc_fiber_trace_chunk direction-ablation crop_traces.zarr --normal-manifest normals.lasagna.json --output crop_bp_alternating --bp-only --bp-inference sum-product-mixed --winding-solver alternating
```

In joint-grid mode, a non-Mixed pair factor charges orientation and winding
evidence once. Orientation and the Mixed unary retain `--bp-temperature`;
winding retains its established temperature `0.25`. A Mixed endpoint disables
the complete pair factor. Consequently, Mixed-cost tuning from the alternating
model is not expected to produce identical posteriors in the joint model. Every
split piece remains a distinct variable. Default hard continuity permits a
Defect boundary, but forbids an H/V or winding change across a continuation
edge whose endpoints are both active. `--split-continuity finite` restores the former parallel-score-1,
zero-difference factor that other evidence can override.

Without fixed orientation, joint-grid uses `--winding-defect-cost` as its only
Defect unary. Non-fixed alternating uses the orientation BP posterior as its
orientation prior and does not charge that unary again. The winding summary and
`winding_defect_cost_per_constraint` CSV column reports the configured winding value
separately from `bp_mixed_cost_per_constraint`. It also reports the configured
piece-boundary value and stores it as `winding_piece_break_cost`.

The two oriented classes occupy interleaved integer lattices. Local class A is
at `k`; local class B is at `k + sign*phase`, where `k` is integer, phase is in
`[0,0.5]`, and each disconnected component has a deterministic sign/class
gauge. Physical H/V names and absolute integer offsets between disconnected
components are not observable. A Mixed endpoint has no winding coordinate and
does not retain winding connectivity.

For latent difference `delta`, a joint-grid measurement contributes:

```text
parallel * abs(delta)
    + perpendicular * abs(gain * delta - signed_target)
```

The five canonical winding terms can be scaled independently with
`--winding-weights P05,PFAR,P0,P1,P2`. The tuple order is perpendicular next
half-step (`0.5`), perpendicular farther half-steps (`1.5+`), parallel same
winding (`0`), parallel one winding (`1`), and parallel farther windings
(`2+`). Standard runs default to `0,2,2,2,1`; pass `1,1,1,1,1` explicitly for
neutral class weighting. Each value must be finite and nonnegative. A zero
value removes that class's finite winding magnitude loss. If its dominant
signed observation also has an enabled hard-sign rule, the sign remains an
active winding constraint; otherwise the observation remains orientation-only
and does not join winding components or contribute winding Defect incidence.
The class multiplier is applied after the existing `2^-floor(abs(target))`
distance decay and only to finite winding energy. It does not scale H/V
orientation evidence, hard perpendicular order, hard continuity, the Defect
unary, or piece-break cost.
When a signed parallel target is available, its quantized magnitude selects
the parallel class; the unsigned median distance is only the fallback.
Reference group `raw_w` includes distance decay and this class multiplier
before optional parallel-cutoff admission; `used_w` applies the cutoff.
The class multipliers apply to the H/V-aware joint-grid and interleaved winding
solvers. The standalone raw-integer winding solve does not quantize H/V ladder
targets and therefore retains its existing unscaled measurement behavior.

Use `--winding-hard-signs none|perpendicular|parallel|both` to select which
dominant signed observations impose nonzero ordering. The default is `both`.
A constraint contributes
at most one sign: its dominant perpendicular or parallel hypothesis. Unsigned
and zero-target observations never impose a sign. Parallel signs also respect
`--parallel-winding-cutoff`. Passing `--winding-weights 0,0,0,0,0
--winding-hard-signs both` therefore runs sign-only winding inference; passing
the same weights with `none` removes all measured cross-trace winding terms.
Same-trace hard continuity is unchanged in both cases.

Finite winding evidence can additionally be confidence-weighted with
`--winding-decision-confidence legacy|linear|cosine` and
`--winding-normal-confidence none|linear|cosine`. Let `s` be the selected
dominant normalized parallel/perpendicular score in `[0.5,1]` and
`x=2s-1`. `legacy` uses `s`, `linear` uses `x`, and `cosine` uses
`(1-cos(pi*x))/2`. Normal `none` contributes one. Normal `linear` uses
`1-2*acos(abs_dot)/pi`, while normal `cosine` uses `abs_dot`. Perpendicular
alignment comes from the closest connector. Parallel alignment is the median
over all component-compatible signed connector samples. Missing alignment is
neutral for `none` and contributes zero confidence for weighted modes.

Pass `--winding-sign-cost F` to assign each enabled sign a finite
penalty. A nonzero signed target whose predicted delta has the wrong sign or is
exactly zero adds `F * decision_confidence * normal_confidence`. This term is
independent of magnitude class weights and distance decay; the ordinary BP
temperature still scales its log potential. The default is `44`; pass
`--winding-sign-cost hard` for strict rejection, or `F=0` to remove the enabled
sign at weak normal alignment.

`--winding-hard-sign-angle DEG` promotes an admitted, enabled, nonzero dominant
sign to exact rejection when its connector lies within `DEG` of the aligned
Lasagna normal. The comparison is inclusive and uses the raw absolute dot
product, before decision- or normal-confidence transforms. Perpendicular uses
the closest connector alignment; parallel uses the median alignment over its
accepted signed connector samples. The default is 30 degrees. Missing alignment
is not promoted, and `--winding-hard-sign-angle off` leaves all signs under the
configured finite cost. Global `--winding-sign-cost hard` remains stricter and
makes every admitted enabled sign exact. Dominant-hypothesis admission, nonzero
target, sign enablement, and the parallel cutoff are evaluated before this
alignment gate.

These confidence controls only
change the dominant winding factor; they do not change H/V orientation evidence,
same-trace hard continuity, or the discrete Defect cost.
The winding-factor CSV records both raw alignments, the transformed decision
and normal multipliers, effective finite sign penalties, and whether each sign
was promoted by alignment.

After every winding solve, `fiber winding constraint agreement` reports
continuity, perpendicular `0.5`/`1.5+`, and parallel `0`/`1`/`2+` rows plus a
sum. `prepared` is the final solver factor population after dominant selection,
cutoffs, fixed-orientation removal, and component filtering. `active` means both
endpoints have an active winding and is the denominator for `infringed_%`.
`neutralized` counts factors disabled by at least one Defect endpoint, including
hard continuation edges split by a Defect.
An active factor is infringed once when any required H/V relation, enabled sign,
or canonical winding bin is wrong. Neutralized factors are shown explicitly and
are not silently counted as correct or included in the percentage denominator.

The hard-continuation 1024-crop sweep selected linear normal confidence,
cosine decision confidence, and class weights `0,2,2,2,1`. With both hard sign
classes at the 30-degree gate, finite sign cost 44, Defect cost 100, and
temperature 1.25, it retained 8/8 exact reference windings and matched
1,369/2,100 evaluated reference constraints. These measured settings are the
standard defaults. Explicit confidence, weight, sign, Defect, and temperature
options reproduce prior modes.

With reference fibers loaded, `--winding-weight-search V0,V1,...` exhaustively
tests the Cartesian product of the listed values in all five tuple positions.
It conflicts with `--winding-weights`. Constraint extraction, topology,
effective-winding component selection, fixed orientation, and reference cross
constraints are reused; only winding inference is repeated. Each scenario
prints elapsed time, ETA, exact/wrong/missing calibrated reference estimates,
constraint right/total accuracy, convergence, and the tuple. The final ranking
uses the fixed reference-source denominator first, then constraint matches, so
turning difficult endpoints into Defect cannot improve the primary score by
abstaining. Failed scenarios are reported and skipped. The selected report and
tuple drive the ordinary CSV, OBJ, and reference diagnostics.

For iterative tuning, pass one starting tuple with `--winding-weights` and add
`--winding-weight-search-local`. Each iteration evaluates the ten tuples made
by multiplying exactly one coordinate by `0.5` or `2`, then moves to the best
strict benchmark improvement. Exact power-of-two exponent tuples cache both
successful and failed scenarios, so a previously visited neighbor is not
solved again. Tied benchmark metrics do not cause a move; tuple ordering and
message residual only make reporting deterministic. The search stops when no
single-coordinate neighbor improves the benchmark. Exponents are bounded to
`[-16,16]` relative to the supplied tuple; reaching the iteration guard before
an optimum is an error rather than a successful termination. The selected
tuple is solved once more to produce the ordinary output artifacts.

where `gain=1/measurement_scale`. In alternating mode the equivalent stored
parameterization is:

```text
parallel * abs(delta)
    + perpendicular * abs(delta / measurement_scale - signed_target)
```

An admitted enabled nonzero dominant sign promoted by the alignment gate, or by
global hard-sign mode, requires:

```text
signed_target * predicted_delta > 0
```

Zero and reversed deltas have exactly zero active-active factor probability;
they are not approximated by a large finite penalty. Exact signed zero, missing
signed targets, disabled sign classes, cutoff-suppressed parallel targets, and
zero-difference continuity do not activate this rule. A Defect endpoint remains
neutral, so contradictory order evidence is resolved by disabling at least one
involved piece. Admitted non-promoted signs use the finite penalty described
above.

The log-space message implementation tracks finite sums and counts of
negative-infinite terms separately. This permits exact exclusions with damped
messages and gauge-adjacent factors without `-inf - -inf` cavity arithmetic.
Alternating component-sign selection and phase/scale backtracking apply the
same hard compatibility rule as the pair factors.

Loopy BP reports retain their soft node marginals. Because independently
selected node-marginal MAP states need not form a feasible edge assignment,
publication performs deterministic feasibility decoding. Reversed hard-sign
edges disable the lower-confidence endpoint while preserving a gauge endpoint
when possible. Each mismatched active-active hard-continuity edge similarly
disables one endpoint; existing Defect endpoints split the chain. Sign and
continuity projection repeat once so the published assignment satisfies both
invariants. The CLI treats
`winding_valid=0` as authoritative for H/V/Mixed OBJ output and reports the
number of `hard_sign_projected_defects` in the winding calibration table.

The perpendicular residual stays in the raw Lasagna-integral units. This is
important: multiplying the observed target by scale would also scale its noise
and spuriously favor the minimum scale. Alternating calibration fits inverse
scale `g=1/scale` and `h=g*phase` from oriented pair beliefs, solves the exact
bounded `(g,h)` wedge, and accepts updates only when expected L1 energy does
not increase. Rank-deficient evidence retains the previous parameters.
Deterministic phase/scale starts are ranked by the complete decoded assignment
score.

Joint-grid progress reports one coherent warm-started lifecycle: message
iteration/residual, calibration-posterior residual, candidate-state count,
gain/phase grid size, shifts, MAP and posterior-mean phase/scale, gain-boundary
mass, absolute gain bounds, and elapsed time. Support-change and terminal events
are always printed; ordinary messages are throttled to roughly once per second.
There is no initialization or calibration-pass counter because calibration
cells are simultaneous states of one model. Alternating mode retains the older
orientation, initialization, calibration, and empirical ETA progress stream.
Library callbacks are synchronous and observational in both modes.

A continuous weighted least-squares solve only centers conservative integer
support. Joint-grid fixes the crop-central state of every connected component
to `(A,0)`, shares one crop-global calibration posterior across components, and
retains messages when settled boundary pressure expands integer support or
moves the absolute log-gain window. Resource and shift guards fail explicitly;
there is no automatic fallback to alternating.

The joint calibration controls are:

```text
--winding-gain-cells N
--winding-phase-cells N
--winding-log-gain-step F
--winding-boundary F
--winding-max-gain-cells N
--winding-max-shifts N
```

Defaults are 5 gain cells, 6 phase cells, `log(1.1)` log-gain spacing, 0.25
boundary pressure, 17 maximum gain cells, and 32 one-cell shifts. Phase always
spans the canonical `[0,0.5]` interval.

The fixed defaults can be overridden explicitly:

```text
--winding-fixed-phase 0.5 --winding-fixed-scale 1.0
```

Both flags are required together. Phase must be finite and in `[0,0.5]`; scale
must be finite and positive. Fixed flags cannot be combined with
`--winding-adaptive-calibration`, explicitly supplied adaptive-grid controls,
or `--winding-solver alternating`. Fixed mode
uses the same joint H/V/Mixed, integer-winding, and component-sign model, but
the supplied values are scalar factor parameters: there is no latent
calibration variable, calibration message, posterior, or gain-window update.
Adaptive integer support remains active. Progress, console summaries, and the
consistency CSV identify calibration as `fixed`; the exact supplied scale is
retained for reporting rather than reconstructed from reciprocal gain.

The ordinary BP consistency CSV includes continuous winding, integer MAP,
posterior mean, MAP probability, entropy, candidate bounds, component, and
incident signed/skipped counts. `<base>_winding_factors.csv` records the raw
original-order target, raw canonical-order target, effective quantized
canonical target, and selected scale-calibrated latent target. The consistency
CSV additionally records posterior latent coordinate,
phase, scale, component phase sign, solver and calibration modes, calibration
posterior means,
and the component-sign posterior when available. The solver's arbitrary
zero-centered
relative MAP labels are shifted by the global minimum for publication, so the
OBJ groups are consecutively numbered `<base>_w_0.obj`, `_w_1.obj`, and so on.
The consistency CSV records both `winding_relative_map` and `winding_output`.
Every published winding group is additionally partitioned by orientation state
as `<base>_w_<number>_h.obj`, `_v.obj`, `_err.obj`, and `_tie.obj`.

### Napari winding layers

The state-partitioned winding OBJs can be inspected directly without loading a
volume:

```bash
PYTHONPATH="$SRC/vesuvius/src" python -m vesuvius.scripts.view_fiber_windings "$VES/data/workdir3/fiber-crop-1024/fibers"
```

The positional argument is the output base passed to `direction-ablation`; a
trailing `.obj` is also accepted. Matching `_h`, `_v`, `_err`, and `_tie` files
are discovered independently. Missing state files and whole intermediate
winding labels are allowed; every present file remains strictly validated.
Aggregate `_w_N.obj` files and CSV diagnostics are ignored.

Every state slot from the lowest through the highest observed integer winding
becomes one managed 3D path layer. Missing or empty artifacts become empty
placeholder layers, so their visibility bits still exist. H and V layers at
the same winding share one bright winding-specific color; Broken and Tie remain
visually distinct. `H`,
`V`, and `Broken` show that category across all winding labels; Broken includes
both `_err` and exact `_tie` layers. `All` and `None` set all winding-layer
visibility at once. Full-size Previous and Next buttons circularly rotate the
entire managed H/V/Broken/Tie visibility mask by one winding in the contiguous
observed range while preserving state. Visible and hidden bits both move, so
any arbitrary pattern, including missing and empty slots, wraps intact.
The reference and unrelated layers are untouched. The label lists all currently
visible managed windings and follows manual layer-panel visibility changes. The
first nonempty H/V winding is the initial view. `--width` changes the displayed
path width.

The shown winding number is the nonnegative publication offset, not an absolute
physical winding. Likewise, physical H/V identity and absolute winding are not
comparable across separately gauged components.

For comparison against an existing VC3D annotation stack, add both reference
options to the `direction-ablation` command:

```bash
--reference-fiber-dir /path/to/vc3d-fibers --reference-fiber-tag hendrik_crop1
```

The tag match is exact and case-sensitive. The command scans only regular JSON
files directly in that directory, validates selected files with the normal
VC3D fiber parser, and clips their dense base-XYZ lines to the stored trace
artifact's crop before writing `<base>_reference.obj`. A fiber that leaves and
re-enters the half-open crop becomes multiple ordered OBJ paths; boundary
crossings are retained, while geometry merely lying on the exclusive maximum
face is not. The reference fibers are diagnostic only and do not affect BP.
The same cropped runs are also passed once through the normal constraint
extractor. Selected source-fiber filename order defines diagnostic virtual
winding values `0.0, 0.5, 1.0, ...`; all crop runs from a source inherit its
value, and a selected source without crop geometry still occupies its slot.
Generated within-run continuity links and links between separate crop runs of
the same source fiber are excluded. Every remaining link is presented exactly
once under its lower-winding source and only points to a higher winding.

For each source and matching BP benchmark, the command prints a perpendicular
table followed by a parallel table. Dominant score determines the table, with
exact ties assigned to perpendicular. The raw signed step is first oriented
from the lower virtual winding to the higher one. The benchmark's fitted global
sign then produces `calibrated_step`; the per-gauge additive offset cancels from
a difference and is not applied. A constraint lacking a signed observation
retains its nonnegative magnitude rather than receiving a fabricated sign.
The calibrated step is quantized with the same signed half-integer or integer
rule used by the winding solver, and both `calibrated_minus_gt` and the
canonical summary use that value. Thus these rows, `est_w`, and the reference
accuracy table share one sign convention. These diagnostics use the active constraint extraction geometry
and distance settings, but not downstream constraint pruning,
perpendicular-only selection, or labeling filters.

Before those detailed tables, `reference constraint measurement-scale
calibration` fits the filename-ordered reference differences against both the
globally signed continuous raw steps and the canonical targets. Both use the
matching winding configuration's dominant-factor admission and effective
class, distance-decay, decision-confidence, and normal-confidence weights. Its magnitude-only objective is
`sum(w*abs(gt/scale-target))`; finite and hard sign penalties are omitted
because they do not vary with positive scale. In reciprocal scale the exact L1
fit is a weighted median of `target/gt` with weights `w*abs(gt)`. The reported
`[0.5,2.0]` range is a fixed diagnostic convention, not the joint-grid
adaptive gain support. `used` counts positive effective weights, `fit_n`
counts rows with positive `w*abs(gt)`, `sum_w` sums solver magnitude weights,
and `fit_w` sums the reciprocal-scale median weights. `loss_s1`, `fit_scale`,
`fit_loss`, `reduce_%`, and `bound` compare scale one with the bounded fit.

The `raw` rows estimate continuous integration bias before half/integer target
snapping. The `canonical` rows describe the exact targets consumed by current
inference; because snapping can place more than half the weighted observations
at an exact target/GT ratio of one, their L1 optimum can be exactly one without
implying a statistically exact physical calibration.
Canonical `perpendicular_all` is directly compatible with the current
measurement-scale parameter. The report also includes `parallel_all` and
`all_constraints` aggregates plus the five class rows for perpendicular `0.5`,
perpendicular `1.5+`, parallel `0`, parallel `1`, and parallel `2+`. All
parallel-containing aggregates and class rows are explicitly counterfactual:
current winding inference does not apply measurement scale to its integer
parallel residual. With the displayed
objective, fitted scale below one means the selected raw or canonical targets
are larger than the known latent reference separation; scale above one means they are smaller.
The scale table is diagnostic only and does not alter inference or artifacts.

The preceding `reference raw signed step distributions` table is deliberately
independent of solver weights. It uses every signed source-oriented reference
measurement and groups it by dominant perpendicular/parallel classification,
the H/V transition implied by the selected phase gauge, and virtual reference
separation. Opposite-parity transitions use `0.5`, `1.5`, and `2.5+` bands;
same-parity transitions use `1`, `2`, and `3+`. Each nonempty row reports count,
minimum, arithmetic mean, median, and maximum of the raw signed winding step.
This exposes asymmetric H-to-V and V-to-H measurements instead of hiding them
inside one fitted phase.

`reference constraint phase calibration` then fits all signed dominant
perpendicular opposite-parity rows with unit weight. It enumerates increasing
and decreasing winding direction and both even-reference H/V assignments, and
minimizes `sum(abs(predicted/scale-raw_signed))` over phase `[0,0.5]` at the
run's fixed positive measurement scale. Production class, confidence, and sign
weights do not enter this physical reference diagnostic. The fit is diagnostic
only and does not alter solver phase or output artifacts.

After the last source table, `reference constraint canonical summary` reports
`correct`, `false`, and `total`. Every displayed measured row contributes once,
including repeated piece-pair measurements between the same source fibers. A
row is correct when its parallel integer or perpendicular half-integer
canonical step equals the virtual GT step; otherwise it is false. Consequently
`correct + false` always equals both `total` and the number of displayed rows.
The Napari command above automatically loads this sibling as the independent
`Reference fibers` layer when present. A later successful
`direction-ablation` run without the reference options removes an older
reference sibling owned by that output base.

For `--bp-only --bp-inference sum-product-mixed`, the same reference data also
benchmarks the solved BP winding result. Reference runs reuse their canonical
constraint pieces, while the represented BP lines remain exactly one input
piece each. A filtered cross extraction measures only reference-to-BP pairs
after that solver run has retained its main winding component; reference
geometry never enters the solve.

Before the final BP-only winding solve, the command retains only the largest
component of the exact effective-winding graph. Fixed-orientation runs repeat
their orientation prepass and component filtering until one component remains,
then reuse that stable prepass. Source traces, split pieces, constraints, and
output provenance are remapped together. If filtering removes pieces between
two retained runs of one source trace, each contiguous retained run becomes a
separate represented trace; the removed arc gap is never treated as hard
continuity. The `fiber BP main winding component`
line reports cumulative trace, piece, and constraint counts before and after
filtering. Equal sizes prefer the crop-central piece and then the lowest piece
index.

The benchmark uses each BP piece's authoritative post-projection MAP latent
coordinate. Perpendicular constraints infer the reference coordinate from the
signed canonical half-step and solved measurement scale. Parallel constraints
use the scale-independent canonical integer distance and test both signs;
distance zero and nonzero are reported separately. Constraints ending at a
final Defect/Mixed or otherwise winding-invalid BP piece are excluded from both
offset calibration and right/wrong totals. Unsigned perpendicular observations
also have no candidate and are excluded.

Immediately before each reference benchmark, `BP final states by source-piece
cohort` reports H, V, active H/V, Defect, and Defect percentage for the original
`source_piece == 1` cohort, all other pieces, and their total. The cohort bit is
captured before main-component filtering and carried through remapping, so
filtering cannot move pieces between rows. This is the stable cohort used by
the existing central-piece tuning diagnostic; it is not a general geometric
center definition for arbitrary piece counts. States come from the final
post-projection winding result. Mixed and winding-invalid pieces count as
Defect, and an empty cohort prints `NA` percentage. The table is also emitted for
BP runs without reference fibers.

`BP admitted winding evidence by source-piece cohort` follows the state table.
It reports the winding terms that survived preparation for the final solve:
hard continuity, perpendicular, parallel-same, and parallel-other. A measured
link appears in exactly one perpendicular or parallel row according to its
dominant hypothesis. The `measurement` row counts each admitted link once and
can be smaller than the degree basis used by the Defect unary when the parallel
cutoff suppresses a parallel-dominant winding term.
`inc` counts endpoint incidences. `act_i/p` and `def_i/p` first separate those
incidences by final state and then normalize by the corresponding piece count;
`act_c/p` and `def_c/p` do the same for finite coefficient.
`hard` counts endpoint incidences with a nonzero signed perpendicular target;
`def_hard_%` is the percentage of those incidences landing on final Defect
pieces. The finite coefficient on a Defect endpoint is evidence admitted into
the prepared problem that Defect can neutralize, not realized final pair
energy. These statistics identify associations for tuning; they do not alone
establish which constraint caused a Defect decision.

The filename-ordered reference stack is calibrated only after the dominant-factor
scorer has inferred one raw winding for every `(reference, gauge)` pair with
admitted evidence. Raw inference does not use the filename-order label, a gauge
mapping, or the reporting tolerance, and every pair contributes one calibration
vote regardless of its constraint degree. A single global sign and one
half-integer offset per gauge maximize exact raw-estimate matches; residual
error breaks match-count ties. Remaining global-sign ties prefer `+1`, while
offset ties prefer the value closest to zero and then the lower value. Gauges
without admitted raw estimates are omitted. Output first lists each gauge's
offset, exact matches, and estimate votes, then prints a weighted
constraint-group diagnostic. The later constraint-accuracy rows retain their
inclusive `0.5` reporting tolerance, which does not affect calibration. For each reference winding
it splits dominant perpendicular evidence into canonical step `0.5` and
`1.5+`, and dominant parallel evidence into canonical step `0`, `1`, and
`2+`. Every member contributes only its dominant BP winding term. `raw_w` sums
the winning scores after fixed power-of-two winding-distance decay. `used_w` additionally reflects
parallel-distance cutoff suppression. `true_h` and `infer_h` count violated hard
signed-order constraints. Candidate selection first minimizes that count, then
the admitted dominant-hypothesis BP winding energy; exact ties prefer the lower
signed half-integer winding. `true_L1`/`true_avg` give total
and admitted-coefficient-mean energy at the globally calibrated true winding;
`infer_w` and `infer_L1`/`infer_avg` give the half-integer winding and energy
preferred by that group alone. Perpendicular coordinate residuals are divided
by the solved measurement scale. Gauge-local candidates are transformed by the global sign
and their gauge offset before aggregation, including when one reference source
reaches multiple gauges. `NA` denotes a group with no positive raw coefficient.

An additional `all` row evaluates every constraint for the reference with the
same scorer. Its preferred winding is exactly the `est_w` printed in the next
table. This removes the previous difference between weighted-L1 group inference
and support-count/squared-residual reference inference. In BP, any hard-sign
violation makes an active pair state impossible and Defect provides an escape.
The forced-active reference diagnostic instead minimizes the number of hard
violations before finite energy when hard evidence is contradictory.

The command then prints one compact error row for every selected reference JSON in source order,
identified only by its filename-ordered virtual winding. Each row
contains `est_w`, `parity_ok`, and right/wrong/right-fraction columns for perpendicular, parallel-same,
parallel-other, and sum; a zero-total fraction is `NA`. Multiple cropped runs
and pieces accumulate into the same row. These rows reuse the one global gauge
calibration and each class and sum count agrees with its aggregate. The command
then reports:

```text
class  right  wrong  total  right_percent
perpendicular  ...
parallel_same  ...
parallel_other  ...
sum  ...
```

`parity_ok` compares the normalized parity of `round(2*est_w)` with the
reference source index. It is `NA` when no winding was estimated. A whole
winding insertion or deletion preserves parity, while a half-step H/V switch
does not. The already calibrated winding coordinate is used directly; the H/V
gauge described below does not enter this field.

Before the winding-error table, `reference H/V component calibration` fits the
otherwise arbitrary mapping between even/odd reference parity and published
H/V names independently for every BP component. It prints how many active
reference-to-BP endpoints agree under even-to-H and even-to-V, choosing the
larger count and choosing even-to-H on an exact tie. The following
`reference H/V endpoint consistency` table reports perpendicular, parallel,
and combined correct/wrong/fraction counts per reference plus a sum row.
Parallel dominance expects the same H/V class; perpendicular dominance,
including an exact score tie, expects the opposite class. Repeated cross
measurements count separately. Final Mixed/Defect or otherwise inactive BP
endpoints are excluded, but an active endpoint remains in this orientation
diagnostic even if its winding target is unsigned, suppressed by a cutoff, or
has zero magnitude weight. Consequently a mismatch demonstrates inconsistent
solved orientation and dominant geometric relation, but cannot alone identify
which of those two inputs is wrong.

Exact orientation-score ties are perpendicular. Every accepted measured cross
piece pair contributes once, including repeated constraints involving the same
source fibers, provided its final BP endpoint has a valid active winding
candidate. A zero-total class reports `NA` rather than a percentage. The full
reference/reference section and every reference-to-BP table are buffered and
printed after all ordinary command output.

## Quality groups

Visualization stably sorts traces by ascending cost density and then stored
ordinal. It partitions sorted rank `r` among `N` traces with
`min(9, floor(10*r/N))`, producing ten independently displayable files:

```text
crop_fibers_quality_00_10.obj
crop_fibers_quality_10_20.obj
...
crop_fibers_quality_90_100.obj
```

Every trace occurs exactly once. For fewer than ten traces some rank bins are
empty; their OBJ files are still valid and present. The sibling
`crop_fibers_quality_histogram.csv` and console table report each bin's count
and min/mean/max total cost and cost density. Blank numeric CSV fields denote
an empty bin.

The BP cohort diagnostic separates perpendicular winding evidence into
`perp_value` and `perp_sign`. `perp_value` is the finite winding-distance
coefficient. `perp_sign` is the independent hard ordering incidence and has
`NA` coefficient columns. This distinction matters because a piece can have a
small finite value residual while no integer winding satisfies all incident
hard signs.
