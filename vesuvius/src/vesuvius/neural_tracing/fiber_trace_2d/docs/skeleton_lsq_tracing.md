# Skeleton-topology least-squares fiber tracing (prototype)

Status: **implemented prototype** (2026-09-18). First measured result:
[case31 skeleton report](benchmarks/contrast_case31_skeleton_20260918.md).

This document holds the design, every clarification made while planning, the
implementation layout, and the implementation notes and results in section 10.
The task-level plan is `planning/task_plan.md`.

## 1. Motivation and baseline

Current short200 results (same 200 directed pieces, 662.089848 mm, full-res
best EMA999 cache, see
[fiberlet vs beam](benchmarks/short200_fullres_fiberlet_vs_beam_20260918.md)):

| tracer | restart episodes | failure-free pieces |
|---|---:|---:|
| native beam | 56 | 145/200 |
| bounded fiberlet, width 64 | 60 | 147/200 |

Both tracers decide locally (beam lookahead 384 base voxels; fiberlet graph
beam over DP paths). The new method makes the topology explicit first: the
thresholded presence is skeletonized, unambiguous line stretches are refined
continuously with fixed endpoints, and the discrete connection choices inside
ambiguous regions are enumerated and scored as a joint explanation of the
region.

Development benchmark: the explicit contrast-choice case31
(`/ephemeral/hendrik/fiber3d_train2/evaluations/explicit_contrast_choice_tests_20260918_v1`,
piece `053_5de02b9d2fd3c9919f58`, PHerc0139, reverse direction, 3200 base
voxels, 7.6768 mm). Beam fails once at reference arc 3093.6 of 3200 (normal
error 73.8, tangential 57.7, threshold error 79.2 against 40). Fiberlet
scoped r32 (v2) and bounded w64 completed with 0 failures; the unrestricted
fiberlet control is incomplete (`incomplete.json`). The rendered evidence shows
the reference climbing to a neighbouring sheet near the end while the beam
trace continues straight through a thick, bright merged presence region.

Second development case (to be packaged later): merge/split case000
(PHerc0139 piece `000_08e44ead4019b9400317`, 1485.5 base voxels) with the
user-supplied companion fiber. Its Python skeleton diagnostic (v15, raw
threshold 212/255, 879 component voxels at prediction scale 8) reduced to a
graph with 5 terminals, 3 junctions, 2 junction bridges and cycle rank 0.

## 2. Existing code that is reused or serves as the reference

| Purpose | Location | Use in the prototype |
|---|---|---|
| Threshold search and Lee-thinning skeleton graph (Python diagnostic) | `fiber_trace_3d/fiber_pair_connectivity.py` (`search_connectivity`, `skeleton_topology`) | Reference semantics for the native port (26-neighbour graph, junction merge 64 base, degree-2 suppression). Not used at runtime. |
| Shared local metric (alignment + normal-aware smoothness, length scaled) | `volume-cartographer/core/src/fiber_tracer/FiberLocalScoringInternal.hpp`, public `FiberLocalScoring.hpp` (`fiberLocalMetricCost`) | Called directly for the discrete initial loss on skeleton paths; defines the terms and weights the Ceres residuals mirror. |
| Beam tracer | `core/src/fiber_tracer/FiberTrace.cpp` | Baseline; it uses the unscaled per-step form plus a cumulative tangent smoothness (4 steps, weight 2) that is not mirrored. |
| Shared reference replay core (seed on reference, forward matching, threshold, excursion, failure records, deterministic reset) | `FiberTrace.cpp` anonymous-namespace `traceReferenceReplayCore` with `ReplaySegmentRunner`/`ReplayPointObserver`; used by beam and Lasagna replay | Exported to a header and reused unchanged by the skeleton driver (see Stage 6). |
| Fiberlet DP / graph replay | `FiberPaths.cpp`, `FiberGraph.cpp` | Baseline only. **Not** used as optimizer or initializer (user decision). Its private seed-scan/reset is not reused either. |
| Ceres (2.0) and Eigen already linked into `vc_fiber_tracer` | `core/CMakeLists.txt` (`vc_fiber_tracer`), `FiberTraceWindingLeastSquares.cpp` | No new dependency; in-repo example of residual/config validation style. |
| Stored prediction grid access | `FiberPredictionField::storedGridInfo`, `sampleStoredPresenceBatch` (index based) and the coordinate corner samplers `sampleCornerBatchWithNormals` / `visitCornerBatchWithNormals` (`LasagnaCornerBatch`: ordered 8 corners, fractions, validity, with normals) | One dense uint8 presence read of the tube for thresholding/thinning; the corner batch feeds trilinear interpolation of presence, direction and normal inside the Ceres functors, refreshed every solver iteration. |
| Reference matching, threshold ellipse, excursion grace | `PolylineGeometry.hpp::matchForwardPolylinePoint`, `FiberReplayMetric.hpp::measureFiberReplayThreshold`, `FiberReplayExcursion` | Used through the exported replay core, not re-implemented. |
| Result/artifact types | `FiberReplayTraceResult`, `FiberReplayTraceSegment`, `FiberReplayFailure`, `writeFiberReplayTraceArtifacts`, `fiberReplayTraceJson`/`fiberReplayTraceFromJson` (`FiberReplay.cpp`) | Output contract; `FiberReplay.cpp` gains a generic `geometry_backend` label (see Stage 6). |
| Benchmark packaging and runner | `fiber_trace_3d/contrast_choice_benchmark.py`, `short_benchmark_evaluate.py`, `scripts/run_classic_fiber_evaluation.py` | Backend dispatch (`beam`, `fiberlet`, `anchors`) to be extended with `skeleton`. |
| Native build/test entry points | `vesuvius/scripts/run_fiber3d_native_build`, `run_fiber3d_native_replay_tests` | Unchanged entry points; the build target list gains the new test. |

There is no existing 3D thinning in the native code (`core/util/Thinning.hpp`
is a 2D distance-transform tracer for slices), so the Lee (1994) 3D thinning
used by scikit-image is ported natively.

## 3. Clarifications agreed during planning

1. **Threshold selection.** The Python diagnostic chose the threshold from
   endpoint connectivity, which a tracer cannot do. The prototype uses a fixed
   uint8 threshold parameter on the full-resolution presence that
   is swept through benchmark configs. Selection by ground truth is not
   allowed at runtime.
2. **Ambiguity is defined by skeleton degree first.** Ambiguous region =
   junction cluster (graph degree >= 3). The pair case is known to consist of
   two separate start and endpoints with one long shared stretch (observed at
   low resolution); width-based or gap-based ambiguity classes are deferred
   until case31/case000 show they are needed.
3. **Presence channel.** All current fiber models have one presence channel
   (the earlier planning question about per-option presence was wrong). The
   stored uint8 presence grid is thresholded directly.
4. **Cost model.** Same in spirit as the DP/beam local metric: same terms,
   similar weights and order. The prototype mirrors the **length-scaled DP
   form** (`fiberLocalMetricCost`), which is the per-unit-length consistent
   one; the beam's unscaled per-step form and its cumulative tangent term are
   not mirrored (see section 5). Exact numerical equality with DP/beam costs
   is not required.
5. **Line parametrization.** Polyline with a fixed point count per segment
   and a smoothed spacing: one free per-segment spacing scalar keeps points
   evenly spread while the total length may grow or shrink.
6. **Joint explanation.** Ambiguous regions are scored jointly: a region with
   ports can only be connected in a finite number of ways; the initial test
   enumerates all matchings by brute force and adds a penalty for support
   left unexplained.
7. **Initialization of a connection.** The path through the region's own
   skeleton subgraph that minimizes the initial (discrete) loss, then refined.
   Neither fiberlet DP nor graph beam search is used.
8. **Entry offset.** 32 base voxels from the ambiguous region along the line.
   If the line component is shorter than that, the neighbouring non-line
   components are connected directly with one shared port at the midpoint.
9. **Processing unit.** For now a small tube around the benchmark fibers
   (reference plus explicit choice lines). Later: a tube around the test
   fiber, or a full bounding box.
10. **Language and scope.** Native C++ prototype, case31 only for the first
    run, existing 40/80/160 evaluation protocol, planning first.

Note on `planning/conventions.md`: its "no brute-force search" rule sits under
"never do inversions of some forward transform". Enumerating discrete
connection choices is an explicitly requested search over a finite
combinatorial set, not an inversion; this interpretation is logged in
`planning/task_log.md`.

## 4. Method

All lengths are base voxels unless noted. The full-resolution best EMA999
cache (`fullres_refinement_213753_best_ema999_20260917_v2_chunk64`) stores
presence at one prediction voxel per base voxel (`source_to_base` 1,
`scaledown` 1, uint8, 64^3 chunks). The prediction scale is always read from
`FiberPredictionGridInfo::predictionToBaseScale` at runtime; the DP
longitudinal step of 2 prediction voxels therefore equals 2 base voxels for
case31 and 16 base voxels for the older scale-8 caches.

### Stage 0. Crop and fields

- Tube of radius `tube_radius_base` (default 64) around the reference polyline
  and every `contrast_choices` polyline of the case; axis-aligned bounding
  box padded by `crop_margin_base` (default 16). Only tube voxels are
  enumerated (about 4e7 voxels for case31), never the whole box.
- Presence (uint8) is read once for the tube voxels through
  `sampleStoredPresenceBatch` in chunk-ordered batches into a dense uint8
  crop (box-shaped, zero outside the tube; about 40 MB for case31). This crop
  is used only for thresholding and thinning. Direction and normal values are
  not materialized: the initial-loss evaluation on skeleton voxels and the
  Ceres refinement use the existing coordinate corner-batch samplers (8
  ordered corners with fractions and validity, normals included) through the
  chunk cache.
- Voxels outside the tube are invalid for both fields. The tube is a
  diagnostic scoping like the fiberlet contrast corridor and is reported in
  the artifacts; this makes the prototype a diagnostic, not a production
  tracer.

### Stage 1. Thinning and graph (full resolution)

1. No downsampling: thinning runs on the full-resolution presence crop so
   neighbouring fibers keep their separation (user decision). The tube for
   case31 is about 4e7 voxels; native Lee thinning is expected to take seconds
   to tens of seconds and its time is reported per stage.
2. Threshold at `threshold_u8` (one config parameter, default 128, swept
   through benchmark configs like any other parameter; no built-in ladder).
   Keep components with at least `min_component_voxels` voxels. No dilation
   or closing (the Python `_transforms` options are dropped).
3. Lee 3D thinning (native port of the ITK/scikit-image algorithm, 26
   adjacency, Euler-invariant simple-point test). The Euler lookup table is
   embedded as plain data; no platform intrinsics.
4. Voxel graph: degree via the 26-neighbour count; clusters of degree != 2
   voxels become nodes; degree-2 chains become edges with measured length
   (same as the Python `skeleton_topology`).
5. Reductions, in order: merge junction clusters joined by bridges shorter
   than `junction_merge_max_base` (64); prune terminal spurs shorter than
   `spur_prune_max_base` (32, new, needed at higher resolution); suppress
   degree-2 nodes. Output: nodes (terminal/junction) and edges as base-space
   polylines, plus the statistics the Python report produced (voxels,
   components, cycle rank, terminals, junctions, arm lengths).

### Stage 1b. What the prediction constrains

Thinning reports the axis of whatever blob it found. That is the fibre only
where the blob is one fibre. The section of the thresholded presence, measured
in the plane normal to the reference on case31, is a ribbon rather than a tube:
one sigma is 1.3 base voxels across the sheet and 3.6 along it, and the
reference sits within 0.3 of the ribbon's centre across the sheet but up to
17.6 from it along the sheet. Presence places the fibre in one direction only,
and it is saturated across the ribbon, so where two fibres merge nothing local
says which side is the fibre being traced. That choice is made in routing
(stages 2 and 5), and the skeleton is used for topology alone.

### Stage 2. Regions, ports and line segments

- Ambiguous region = merged junction cluster. Each incident edge gets a port
  at arc distance `entry_offset_base` (32) from the region boundary.
- An edge joining two regions that is shorter than twice the offset gets one
  shared port at its midpoint (both regions use it as a fixed point). An
  edge shorter than the offset that ends at a terminal keeps the terminal as
  a free end.
- Each port carries sub-ports. The section of the thresholded prediction in
  the plane normal to the edge, at the thinning voxel and restricted to the
  piece of the mask it lies in, gives the centre and the two extremes along
  the section's long axis; an extreme closer than `candidate_min_offset_base`
  (6) to the centre is the same fibre and is dropped. Each sub-port is centred
  by a few steps of a presence centroid of radius
  `candidate_refine_radius_base` (4) in that plane, which walks an extreme in
  from the mask's edge onto the fibre it belongs to. A shared midpoint port
  keeps the centre only. The centre of a wide section is the seam between its
  fibres; it is a routing target only with `route_through_centre`.
- Line segment = edge part between two ports, or between a port and a
  terminal. Ports carry candidates and the outward tangent (skeleton edge
  direction).

### Stage 3. Initial geometry by path search

Every line variant (one per pair of sub-ports at its ends) and every
connection variant starts from the cheapest path through the thresholded
prediction between its two sub-ports, inside a corridor of
`search_radius_base` (16) around the skeleton piece, over 26-neighbours under
the tracing metric's presence and direction-alignment terms, length-scaled,
with the smoothness terms off since a voxel search carries no incoming
direction. It is never the skeleton's own voxels: those run between merged
fibres and would put a kink at each end. The skeleton stretch with its ends
blended onto the sub-ports is the fallback when the mask does not connect
them inside the corridor; on case31 it was never needed.

The port optimizer of the first prototype (a Gaussian centroid fit around the
skeleton position) is removed: inside a ribbon it pulled toward the merged
axis, which is the wrong direction.

### Stage 4. Fixed-endpoint line refinement (Ceres)

Every line is refined once per pair of end sub-ports, from the stage 3
initial geometry, ends held fixed. All variants are kept, and each sub-port
records the cheapest variant into the line from its side.

- Initialization: the skeleton chain (mapped back to base coordinates)
  resampled at `step_base` (default 8; sweep {2, 4, 8, 16}; the residuals are
  length scaled, so the step mainly sets the resolution) into N+1 vertices.
  First and last vertex are the fixed ports (a terminal end stays free).
- Parameters: interior vertices (3 doubles each) and one spacing scalar s per
  segment (initialized to the mean initial spacing).
- Residuals: see section 5. Presence, direction and Lasagna normal are
  trilinear interpolations of the 8 voxel corners around each vertex. A
  `ceres::EvaluationCallback` batch-loads the corners for all current vertex
  positions before every solver iteration (one corner-batch call per field,
  through the chunk cache); the interpolation cell is fixed for that
  evaluation, and the fractions are Jets, so automatic differentiation flows
  through the interpolated fields. The prediction sign alignment is chosen
  from the scalar part. Invalid corners (outside the tube or invalid stored
  values) contribute presence 0 and a zero direction, which saturates the
  alignment residual; an invalid normal selects the isotropic fallback.
- Normals are therefore sampled at every solver step at the current vertex
  positions (no frozen normals, no rounds).
- Solver: Levenberg-Marquardt, `SPARSE_NORMAL_CHOLESKY` with the Eigen sparse
  backend, `num_threads` 1, residual blocks inserted in vertex order,
  `max_iterations` 50, function tolerance 1e-6. Reports per-term cost before
  and after, iterations and time.

### Stage 5. Ambiguous-region choice

For a region with ports q_1..q_k:

1. Pair cost `c_ab` for every unordered port pair (k(k-1)/2 refinements,
   cached): initial path = minimum initial-loss path from q_a to q_b through
   the region's skeleton voxels (Dijkstra; edge cost = `fiberLocalMetricCost`
   evaluated on consecutive skeleton voxels with their step lengths and the
   sampled predictions, i.e. the shared discrete metric, not a re-derivation).
   If the skeleton does not connect the two ports, a cubic Hermite curve
   between the ports is the fallback (logged as such). Refine with Stage 4
   holding q_a and q_b fixed.
2. Enumerate every matching M of the ports (partial matchings allowed; an
   unmatched port means the fiber terminates there). Cost
   `J(M) = sum_{(a,b) in M} c_ab + c_end * |unmatched| + w_u * U(M)` where
   `U(M)` is the presence sum over the region's own thresholded voxels
   (junction-cluster voxels plus the skeleton voxels up to the ports; regions
   are disjoint, a shared midpoint port belongs to both regions but its
   voxels to neither) that are farther than `explain_radius_base` (8) from
   every chosen cached refined path. Defaults `c_end` 8, `w_u` 0.05 per base
   voxel volume, both to be swept.
3. Brute force over all matchings for k <= 8 (764 partial matchings of 8
   ports). Larger k is an explicit prototype limit and raises an error that
   is recorded in the artifact and the case summary. Ties are broken by the
   lexicographic order of the matching's port index pairs.
4. The traced fiber entering through port a leaves through M(a), or
   terminates if a is unmatched.

Sub-ports in the region (2026-09-19): a port with k sub-ports is k items,
and every pair of items on different ports gets its own connection, refined
from the stage 3 initial. Nothing is chosen ahead of tracing: each item
records its cheapest available connection, so a fibre arriving on any sub-port
leaves along that sub-port's own route, and two fibres sharing a merged edge
each take their own sub-ports. With three entry and three exit sub-ports a
segment carries six routes. The brute-force matching, the end cost and the
unexplained-support term of the first prototype are gone; an interim layer
that committed one matching per region and per line before tracing was the
source of every dead end and was removed. `choice.max_ports` counts items
and is 14 in the committed configs; one region on case31 still exceeds it.

### Stage 6. Tracer driver and evaluation

- `traceReferenceReplayCore` (currently private to `FiberTrace.cpp`) is
  exported to a header (`FiberReferenceReplayCore.hpp`) as a mechanical move
  with no behaviour change; the beam and Lasagna replays keep calling it. The
  skeleton driver is one more `ReplaySegmentRunner`. This reuses, unchanged:
  seeding every segment exactly on the reference (`tracePointsBase[0]` is the
  reference point), forward matching, the 40/80 ellipse with the tangential
  factor taken from the evaluation config (default 4, the current benchmarks
  use 2), the 160 excursion grace, failure records, the rule that a pending
  excursion at termination counts as a failure, and the deterministic reset
  `resetArc = min(end, max(previousArc, segmentStart + nominalStep))`
  (`nominal_step_base`, default 48 as for beam).
- Segment runner: from the reference start point, find the nearest point on
  any line segment or port; if none lies inside the 40/80 ellipsoid the
  runner returns termination reason `no_skeleton_seed` with no points (the
  core records the failure and resets; the summary counts these separately so
  a missing seed is never credited as tracing). Otherwise walk: line segment
  -> far port -> region choice -> matched port -> next line segment, until a
  terminal, an unmatched port (`skeleton_terminal`, `unmatched_port`) or the
  tube boundary (`tube_boundary`), emitting refined vertices through the
  observer with the cumulative refined cost as the per-point loss (so the
  ordinary `cumulativeLosses == tracePointsBase` artifact validation holds).
- Evaluation and restart episodes (80 reference voxels) are therefore the
  unchanged protocol; credited coverage (`traced_length_mm`,
  `length_weighted_success_percent`) is reported as for fiberlets.
- Output: `FiberReplayTraceResult` with a new optional generic
  `geometry_backend` label (`FiberReplay.cpp` serializes and restores it;
  `graphGeometry` stays fiberlet-only; the Python export author becomes
  `evaltrace-skeleton`), `skeleton.json` (`format`
  `vc_skeleton_trace`, `version` 1, config, per-stage statistics and
  timings, regions, ports, every matching with its cost,
  refinement cost breakdowns, seed/limit events), `skeleton_graph.obj` /
  `skeleton_reduced.obj` in base XYZ, and the regular `trace.json`, so
  rendering, `report-metrics` and review indexing work unchanged. `render`
  is allowed for this backend.

## 5. Cost model mapping

The shared discrete metric per step of length `L` (`fiberLocalMetricCost`,
used by the fiberlet DP) is

- alignment: `L * (1 - p * prod(six clamped dots))`, where the dots are
  prev-step x cand-step, prev-step x current prediction, prev-step x candidate
  prediction, current prediction x cand-step, current prediction x candidate
  prediction, cand-step x candidate prediction, predictions sign-aligned to
  the previous step first;
- smoothness: `(0.1 * theta_n^2 + 10 * theta_t^2) / max(1, mean step length)`
  with `theta_n = |asin(cand . n) - asin(prev . n)|` and `theta_t` the angle
  between the two step directions projected into the sheet plane; isotropic
  fallback `2 * theta^2` (full angle) when the normal is invalid or a
  projected tangent is degenerate; free angle 0;
- invalid prediction: `4 * L`.

The beam uses the unscaled per-step form plus a cumulative tangent term; it is
not mirrored. The Ceres objective is a sum of squared residuals, so each
residual is the square root of the corresponding discrete term, which keeps
the per-unit-length weighting and order of the DP metric:

| DP term | Ceres residual (`L_i = |P_{i+1} - P_i|`, `Lbar_i` = mean of adjacent step lengths) |
|---|---|
| alignment | `sqrt(max(L_i * (1 - p(P_{i+1}) * prod(six clamped dots)), eps))`, `eps = 1e-3`; the six factors are built from the two step directions and the trilinear predictions at `P_i`, `P_{i+1}`, sign-aligned to the previous step |
| normal smoothness | `sqrt(0.1 / max(1, Lbar_i)) * |asin(clamp(c . n)) - asin(clamp(a . n))|` |
| tangent smoothness | `sqrt(10 / max(1, Lbar_i)) * atan2(|a_t x c_t|, a_t . c_t)` on the sheet-plane projections |
| isotropic fallback | `sqrt(2 / max(1, Lbar_i)) * atan2(|a x c|, a . c)` when the normal is invalid or a projection is degenerate |
| invalid prediction | not mirrored as `4 * L`: invalid corners give presence 0, so the alignment residual saturates at `sqrt(L_i)` (a factor 4 below the DP penalty); stated deviation |
| spacing (new) | `w_s * (L_i - s) / s`, `w_s` 1, `s` free per segment |

`a`, `c` are the previous and candidate step directions, `n` the Lasagna
normal interpolated at the vertex. The clamps to [0, 1] and the `asin` clamps to
`[-1 + eps, 1 - eps]` are piecewise smooth and handled by Ceres Jets. The
prev-step x cand-step factor carries an implicit smoothness exactly as in the
DP. Because the alignment residual is length scaled and the endpoints are
fixed, the free spacing `s` cannot shrink a segment below its chord; the
synthetic-tube test checks that total length stays within tolerance.

## 6. Configuration (planned `skeleton` block)

```json
{
  "backend": "skeleton",
  "skeleton": {
    "tube_radius_base": 64,
    "crop_margin_base": 16,
    "threshold_u8": 128,
    "min_component_voxels": 8,
    "junction_merge_max_base": 64,
    "spur_prune_max_base": 32,
    "entry_offset_base": 32,
    "section_radius_base": 24,
    "candidate_min_offset_base": 6,
    "candidate_refine_radius_base": 4,
    "search_radius_base": 16,
    "route_through_centre": true,
    "dump_crop_presence": false,
    "step_base": 8,
    "nominal_step_base": 48,
    "weights": {"normal": 0.1, "tangent": 10.0, "isotropic": 2.0, "spacing": 1.0},
    "choice": {"max_ports": 8},
    "solver": {"max_iterations": 2000, "function_tolerance": 1e-6},
    "workers": 1
  }
}
```

Those are the code defaults. The committed case31 configs override
`threshold_u8` to 224, which the sweep in the
[case31 report](benchmarks/contrast_case31_skeleton_20260918.md) selected,
`choice.max_ports` to 14, and `route_through_centre` to false, measured in the
[geometry report](benchmarks/contrast_case31_skeleton_geometry_20260919.md).

`section_radius_base` is how far across the section to look for the piece of
the mask the port's thinning voxel lies in. `candidate_min_offset_base` is how
far an extreme must be from the section centre to count as another fibre.
`candidate_refine_radius_base` is the presence-centroid window that walks a
sub-port onto its fibre. `search_radius_base` is the corridor around a
skeleton piece for the path search. `route_through_centre` makes the seam
centre of a wide section a routing target. `dump_crop_presence` writes the
thresholding input beside the other artifacts as raw bytes in the crop's Z, Y,
X order, for measuring geometry offline (about 200 MB per case), and
`render_line_variants` writes every variant of the listed lines as named
polylines.

The block lives under `evaluation` in a runner workload (like `fiberlet`).
Unknown keys are rejected (same allowed-set validation as the `fiberlet`
block). Ordinary `trace_args` do not apply. The skeleton backend needs no
fiberlet cache or `fiberlet` block; the tube radius replaces the fiberlet
contrast corridor radius.

## 7. Benchmarks and commands

- First run: case31 with the explicit contrast-choice manifest v1 and a fresh
  output root; control: the same case with `manifest_unrestricted.json`
  (tube around the reference only).
- Second: package merge/split case000 plus its companion fiber as an
  additional explicit-choice case in a v2 manifest (multi-case packaging,
  companion role `companion-fiber-hypothesis` with provenance fields
  `source_manifest`, `source_case`, `companion_path`, `companion_sha256`;
  `manifest_unrestricted.json` drops choices from every case), evaluated with
  the same full-res model (missing predictions are filled by the existing
  on-demand inference).
- Commands are the approved runner form; the exact configs are listed in
  `benchmark_runbook.md` once they exist, labelled as diagnostic scope.

## 8. Testing

Native doctest `test_fiber_skeleton_trace` on synthetic uint8 volumes,
registered with `vc_add_test`, built by `run_fiber3d_native_build` and
executed by `run_fiber3d_native_replay_tests`:

- Lee thinning: a solid straight bar thins to a single 26-connected chain of
  the expected length; a solid ring keeps cycle rank 1; a solid X keeps one
  junction cluster and four terminals.
- Graph reductions: short bridges merge, short spurs prune, degree-2 nodes are
  suppressed; statistics match the hand-computed values.
- Ports: entry offset placement, the midpoint rule for short bridges, and
  port optimization moving a deliberately offset port back to the tube axis.
- Refinement: a curved tube with a laterally displaced initialization
  converges to within 0.5 voxel of the axis with endpoints unchanged; spacing
  stays uniform; total length within tolerance; per-term costs decrease.
- Choice: an X crossing with straight arms selects the straight-through
  matching; a Y merge with one weak arm compares end cost against the
  unexplained-support penalty deterministically.
- Replay: a synthetic reference through the tube yields a
  `FiberReplayTraceResult` that round-trips through `fiberReplayTraceJson` /
  `fiberReplayTraceFromJson` with the `skeleton` label and passes the artifact
  structure validation; the `no_skeleton_seed` path produces a failure and a
  reset.
- Determinism: two runs give byte-identical `skeleton.json` and `trace.json`;
  `workers` 1 and 4 give identical results.

Python: `test_short_benchmark_evaluate.py` gains the `skeleton` backend
binding, multi-case packaging and `report-metrics` on a synthetic skeleton
`trace.json`. Then the existing native build and replay-test runners and the
case31 run.

## 9. Open assumptions (non-blocking)

- `step_base` 8 and `threshold_u8` 128 are starting values, not measured.
- Terminal-gap regions (bridging nearby skeleton ends) are deferred.
- The pair case is re-evaluated with the full-res model to share one manifest
  and cache with case31.
- `c_end` and `w_u` are unknown until the first case31 report.
- The DP's `4 * L` invalid penalty is not mirrored.

## 10. Implementation notes and results

### Files

Native library, all in `volume-cartographer/core`, registered in
`core/CMakeLists.txt` under `vc_fiber_tracer`:

| File | Contents |
|---|---|
| `FiberReferenceReplayCore.{hpp,cpp}` | The shared replay driver, moved out of `FiberTrace.cpp` unchanged so beam, Lasagna and skeleton replay all use one protocol. |
| `FiberSkeletonThinning.cpp` | (26,6) simple-point test, Lee thinning, 26-connected labeling, thresholding. |
| `FiberSkeletonGraph.cpp` | Voxel graph, junction merge, spur prune, degree-two suppression, polyline helpers, OBJ output. |
| `FiberSkeletonFields.cpp` | Trilinear interpolation of a corner block and batch scalar sampling. |
| `FiberSkeletonRefine.cpp` | Ceres residuals, evaluation callback and the fixed-endpoint solver. |
| `FiberSkeletonRegions.cpp` | Regions, ports, entry offsets, port optimization, line refinement. |
| `FiberSkeletonChoice.cpp` | Region Dijkstra, connection refinement, matching enumeration and joint scoring. |
| `FiberSkeletonReplay.cpp` | Model build, segment runner, artifacts. |
| `FiberSkeletonSource.cpp` | Tube crop from the stored prediction volume and the corner/voxel provider. |
| `core/test/test_fiber_skeleton_trace.cpp` | Twelve doctest cases; run by `run_fiber3d_native_replay_tests`. |

`FiberReplay.cpp` gained an optional generic `geometry_backend` label;
`apps/src/FiberTraceEvaluationCli.cpp` gained the `skeleton` backend, its
allowed-key config block and the per-case artifact write. On the Python side,
`short_benchmark_evaluate.bind_benchmark` accepts the backend and
`contrast_choice_benchmark.package_contrast_choice` accepts several cases.

### Deviations from the plan, and why

1. **Terminal line ends are fixed, not free.** With a free end the objective is
   flat along the line once the geometry is aligned, so the endpoint slides at
   no cost. Both ends of every line are anchored; a terminal is anchored at its
   skeleton endpoint.
2. **The invalid-prediction penalty is not mirrored.** The discrete metric
   charges four per voxel for an invalid prediction; the continuous alignment
   term saturates at one per voxel instead.
3. **No free angle.** The discrete smoothness charges the excess over a free
   angle; the residuals charge the full angle. The two agree at the default
   free angle of zero, which every current config uses.
4. **The alignment residual has a floor of 1e-3** inside its square root so the
   derivative stays bounded. Every step therefore contributes a constant 1e-3
   to the reported alignment cost.
5. **The spacing residual carries `sqrt(weight)`** like the other terms, so all
   four configured weights are comparable. The plan's literal form would have
   squared that weight.
6. **`tube_radius_base` pads a bounding box and a scope lattice**, not an exact
   tube: presence is fetched only for voxels the scope lattice accepts, but the
   crop array itself is the padded bounding box.
7. **The skeleton backend requires a published prediction cache.** Stored-grid
   sampling is the offline path, so the backend binds the ungated prediction
   field. A crop with no published coverage raises rather than producing an
   empty topology, and every crop reports its scoped and missing voxel counts.
8. **A contrast choice may omit its coordinate-binding keys.** The packaging
   records which keys were absent and both the Python runner and the native CLI
   accept exactly those; a stated key must still agree. This was needed for the
   user-supplied case000 companion, which lost its export provenance.
9. **The port limit is 12, not 8,** in the committed configs, because regions
   with nine to twelve ports occur on real crops. A region above the limit is
   reported through `port_limit_exceeded` and leaves its ports unrouted.
10. **`max_ports` was not swept.** Only the threshold was, and later
    `route_through_centre`.

### Cost

Case31 builds its model in 8.7 s and runs in 11.7 s end to end. Graph
construction dominates and scales poorly: case000, with a crop six times
larger, spends 146.8 s of its 252.7 s model build in the graph stage.

### Results

See the [case31 skeleton report](benchmarks/contrast_case31_skeleton_20260918.md)
for the first run and the
[geometry report](benchmarks/contrast_case31_skeleton_geometry_20260919.md) for
what the prediction constrains and how the lateral choice is made.

In short: on case31 the backend traces the full eligible length as a single
segment without a restart, and its distance from the refined reference is a
median 2.02, p95 5.77 and maximum 8.69 base voxels, down from 2.52, 13.75 and
21.17 at the start of the geometry task. The unrestricted control reproduces
it. The two-case run is at two failures, both in its second case, which is a
separate routing failure.

### Not done

- No short200 or full-cohort run for this backend.
- Terminal-gap ambiguity (bridging nearby skeleton ends) remains deferred.
- One region on case31 exceeds the 14-item limit and is left unresolved.
- Whether the seam centre should be a routing target by default; it is off
  in the committed configs.
- Belief propagation over regions is not implemented and, with per-sub-port
  routing, no longer needed for the trace.
