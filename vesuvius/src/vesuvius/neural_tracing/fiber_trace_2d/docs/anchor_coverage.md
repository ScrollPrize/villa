# Anchor placement coverage benchmark

Use the existing approved runner:

```bash
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py benchmark-full --config /home/hendrik/villa2/vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/configs/eval_short200_anchor_coverage_u22000_20260916_v2.json
```

The initial baseline uses all200 locked shortened GT pieces and best EMA999u22000,
without embedding. Existing eligible CP/non-Lasagna spans, source bindings,
canonical extraction parameters, prediction and anchor caches are unchanged.
Placement is not directional: ordinary bidirectional annotations are evaluated once.
This benchmark is a failure-targeted sample, not an unbiased whole-scroll measure.

## Native work

Backend `anchors` reuses replay-set source/normal binding and the shared
`FiberletOnDemandPreprocessor`. It schedules only anchor owner chunks intersecting
the GT tube: maximum acceptance radius plus the existing conservative anchor-cell
diagonal padding. The padding helper is also used by regular graph anchor lookup.
Full owner chunks are extracted so cache identities/results never depend on GT.
No path-connection halo, fiberlet path construction or graph search is requested.

The coordinator retains ordinary demand inference with GPU-manager acquisition
only on ready missing-inference work and release after30idle seconds. Cache-only
operation needs no GPU grant. Physical scrolls run sequentially, pieces in parallel.

## Metric

Each canonical anchor is matched to its nearest Euclidean point on the eligible
GT polyline, using a native spatial segment index and shared projection arithmetic.
At that projection, the existing sheet-normal acceptance measurement applies:
normal radius40, tangential radius80base voxels. Invalid normals use the existing
isotropic fallback; its incidence is reported. Equality is accepted.

Flat endpoint caps exclude anchors beyond the first/last reference tangent plane.
This prevents extrapolated locations outside the eligible span from covering its
ends. It is a nearest-projection test, not membership in the union of differently
oriented ellipses around every point on a bent line.

Accepted stable anchor IDs are retained individually. Identical projected arcs
count once for spacing. Successive arclength gaps are reported in base voxels in
per-piece artifacts and normalized by actual cell side for pooled histograms.
Currently cells are4prediction voxels ×8base/prediction =32base voxels.
These units do NOT mean the number of crossed 3D lattice cells.

- Interior gaps exclude endpoints. Zero/one-position pieces have no interior gap.
- Leading/trailing gaps have separate distributions.
- No-anchor pieces record their entire length as uncovered, without inventing anchors.
- Arc0-phased bins of one cell side record empty-bin fraction and consecutive
  empty-bin runs. The final partial bin counts; an anchor exactly at the endpoint
  occupies that bin. Bin-count and length-weighted empty fractions are both saved.
- Aggregate quantiles pool raw individual gaps, not per-piece quantiles.
- Reversing a piece reverses its interior gaps; arc-bin phase can change occupancy.

## Artifacts and validation

Output root: `/ephemeral/hendrik/fiber3d_train2/evaluations/short200_anchor_coverage_u22000_20260916_v2`.
`report.md` and `metrics.json` contain overall/per-scroll histograms and summaries.
`piece_ranking.json` sorts by largest gap including endpoints (empty pieces use
full length). Each row links to its native `anchors.json`: accepted positions,
stable IDs, GT projections, original-fiber arcs, interior/end gaps and source fiber.
Case IDs are local to each physical-scroll output directory.

Immutable workload, model/manifest hashes and native binary hash are recorded.
Use a fresh output_root for a new run; do not overwrite previous measurements.
Build/tests use `run_fiber3d_native_build` and `run_fiber3d_native_replay_tests`.
Synthetic tests cover anisotropic boundaries, endpoint caps, empty/singleton
cases, invalid normals, bend projection, reversal, aggregation, and cold/warm
native anchor extraction with no fiberlet construction.

First attempt v1 retained77PHerc0139 measurements but failed during coordinator
postprocessing because restart correction was applied to anchor summaries.
The backend-aware summary finalizer fixes this and has a regression test;
v1 remains an incomplete immutable artifact, not the reported full result.
