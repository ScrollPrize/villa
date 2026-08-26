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
dependency halo, filters those records to fiberlets incident to an actual
in-crop anchor, and only then loads their endpoint anchor chunks. An incomplete
tuple required by a retained fiberlet is an error; incomplete tuples referenced
only by unrelated halo fiberlets are outside the crop graph and are not read.

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

The perpendicular view requires normalized perpendicular score `> 0.5` and
aligned winding distance `> 0.3`. Both parallel views require normalized
parallel score `> 0.5`; winding `< 0.5` is classified as same-winding and
winding `>= 0.5` as separate-winding. Threshold comparisons are exact, so a
score of `0.5` is absent and winding `0.5` belongs to separate-winding.
Measured links with aligned winding distance greater than or equal to `1.5`
are discarded before diagnostics and labeling. This is an exclusive cutoff;
same-trace continuity remains at winding zero.
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
orientation. Both pieces are walked in both directions at the sample spacing.
At each step a bounded counter-shift of one twentieth of the spacing may be
retained when it decreases point distance, allowing a combined relative phase
adjustment of one tenth of the spacing. The mean consistently oriented tangent
dot is clamped to `[0,1]` as raw parallel evidence. Raw perpendicular evidence
is `1 - abs(initial tangent dot)`; the two values are divided by their sum, so
the reported normalized scores add to one.

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
defaults to the host CPU count. Before tracing, the tool loads the crop's
incident Fiberlets, route geometry, endpoint anchors, and joins once into an
immutable in-memory graph. Trace workers do not query the chunk cache. The tool
reports graph-preparation and trace times separately. Both limits use zero for
unlimited:
`--max-attempts` counts uncovered seed attempts, including failed/no-edge
attempts, while `--max-fibers` counts accepted lines. Seeds are attempted from
highest prediction presence to lowest, with storage key as the deterministic
tie break.

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

Each step is assigned to the fitted direction with the larger absolute dot
product. A complete fiber is direction-1- or direction-2-dominant when at
least 75% of its valid arc length is assigned to that direction. Other fibers,
including a degenerate fiber with no nonzero step, are mixed. Direction labels
are deterministic: greater total assigned length is direction 1, with
canonical axis order breaking an exact tie. The command reports both fitted
axes, analyzed step count and length, and all three fiber counts.

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
This classification is output-only and cannot change tracing or coverage.

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
