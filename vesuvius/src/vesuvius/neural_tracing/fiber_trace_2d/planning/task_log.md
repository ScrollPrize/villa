# Task log: integer-DP fiberlet paths

## Planning

- The existing Python Trace2CP DP is structurally relevant because it advances
  monotonically while retaining two lateral coordinates and the previous move,
  but it operates in a prepared strip. The new C++ solve operates directly on
  integer voxels of the stored 3D prediction grid.
- General A* was not selected for the initial implementation. The local
  objective has no useful positive lower bound for an informative admissible
  heuristic, while a strict positive chord-projection constraint makes the
  integer corridor graph acyclic and directly solvable by deterministic DP.
- Fitted sub-voxel anchors remain authoritative virtual endpoints. Only the
  searchable dense-volume states are integer voxels.
- Fiber direction channels and regular Lasagna normals are separate inputs.
  The path stage must never interpret fiber `nx/ny` as surface normals.
- Cumulative tangent-history smoothness is deliberately omitted because its
  continuous running heading would add another quantized state dimension.
  Direct one-step curvature uses the shared native equations.

## Review

- Independent review identified underspecified virtual endpoint attachment,
  invalid-prediction costs, mixed-edge curvature scaling, Hermite rasterization,
  normal/isotropic breakdown, locator matching, and scorer regression coverage.
- The plan now connects exact anchors to every eligible nearby integer voxel and
  scores/constrains the actual virtual edges; defines invalid prediction cost as
  `4 * edge_length`; defines a dense sampled-Hermite polyline corridor and
  radius; normalizes turn cost by effective adjacent edge length; makes
  normal-aware and isotropic fallback components mutually exclusive; permits
  content-identical manifest relocation; and adds direct smoothness golden
  tests.
- Initial testing found that a floor/ceil cube can contain no attachment within
  45 degrees of a valid endpoint axis when all endpoint coordinates are
  fractional. Attachments were broadened to the isotropic integer neighborhood
  within `sqrt(3)` prediction voxels. This still bounds attachment length to one
  26-neighbour step while providing the stencil's intended angular coverage.

## Implementation

- Added `FiberLocalScoring` as a shared implementation of the native tracer's
  local normal/tangent smoothness equations. The native tracer now delegates to
  it, and the fiberlet DP uses the same equations without the native cumulative
  heading term.
- Added strict anchor-artifact loading and verification of the fiber manifest
  content hash and prediction grid before path search.
- Added deterministic radius-four cell-shell pairing, 45-degree endpoint-axis
  gating, a sampled-Hermite corridor, positive-chord 26-neighbour moves, and a
  directed integer-voxel DP whose state includes the incoming move.
- Added exact virtual endpoint edges, batched fiber-prediction and Lasagna-normal
  sampling, decomposed cost accounting, and JSON plus base-coordinate OBJ
  serialization.
- Added `vc_fiberlets paths`; it accepts local or remote fiber and normal
  manifests through the existing cache layer and requires a separate regular
  Lasagna manifest for surface normals.
- Added focused scorer, shell, path, failure, strict-loader, scale, and
  deterministic-output tests.

## Validation

- Configured the existing `volume-cartographer/build` tree and built with
  `cmake --build build --parallel 32 --target vc_cli_all test_fiberlet_paths
  test_fiber_trace3d test_fiber_anchors test_lasagna_manifest
  test_lasagna_normal_sampler`.
- Passed `test_fiberlet_paths` (8), `test_fiber_trace3d` (46),
  `test_fiber_anchors` (13), `test_lasagna_manifest` (14), and
  `test_lasagna_normal_sampler` (11).
- Ran the path command twice on the S1 scale-four anchor crop using
  `fiber_s1_002.lasagna.json` and the separate `las_tmp.lasagna.json` normal
  manifest. Each run considered 23,493 pairs, searched 3,087 after axis
  rejection, and found 3,087 paths. Runtime was 51.94 s and 51.63 s with 32
  threads and a 0.5 GiB cache.
- The two 24 MiB JSON reports and 1.6 MiB/53,543-line OBJ files were
  byte-identical. Successful path costs ranged from 1.99265 to 24.5183 with a
  mean of 9.93674.
- The output artifact is available at
  `/tmp/vc_fiberlet_paths_s4_run1/fiberlets.obj`; interactive visual inspection
  in a viewer was not performed in this pass.

## Base-coordinate CLI adaptation

- The initial implementation exposed a prediction-coordinate crop flag and
  prediction-coordinate JSON positions. The user clarified that spatial
  coordinates in CLI interfaces must always use base-volume coordinates.
- Cell indices and counts, prediction-grid shape/scale metadata, cell size,
  Gaussian sigma, and per-prediction-voxel objective parameters remain lattice
  quantities. They are not point or extent coordinates.
- No compatibility aliases or old artifact fields will be retained. The short
  CLI flags are `--crop` and `--corridor-radius`, with base voxels as their only
  coordinate convention.
- Independent review rejected the initial floor/ceil mapping because stored
  prediction indices are point centres, not owned spatial bins. The implemented
  half-open mapping uses ceil on both base boundaries, with scale-aware snapping
  at exact lattice boundaries. It rejects overflow, empty mapped intervals, and
  out-of-volume selections.
- Anchor JSON now declares `base_volume`, stores fitted
  `position_base_xyz`, and records the effective prediction interval using
  base-coordinate boundaries plus exact discrete cell bounds. The strict loader
  divides fitted positions without snapping, then applies its existing cell
  ownership tolerance. Fiberlet JSON stores base-coordinate endpoints and
  polylines, and stores corridor radius in base voxels.
- The artifact and CLI contain no compatibility reader, alias, or duplicate old
  coordinate field. Prediction shape/scale and explicit lattice parameters
  remain available for interpreting cell IDs and objective values.
- Added aligned, non-aligned, and decimal-scale crop conversion tests; updated
  serialization tests to reject the former coordinate contract. Passed 14
  anchor tests, 8 fiberlet-path tests, and the existing native tracer, manifest,
  and normal-sampler suites after building `vc_cli_all` with 32 jobs.
- Re-ran the 192-base-voxel S1 crop with
  `--crop 13600,20256,18144,192,192,192`: 216 anchors, 5,710 candidate pairs,
  792 searched/successful paths, and 13.35 s. Its OBJ is byte-identical to the
  earlier prediction-coordinate invocation. An explicit
  `--corridor-radius 32` run took 13.57 s and produced byte-identical JSON/OBJ
  to the one-cell-width default.
- Made the demonstrated values actual CLI defaults: four prediction voxels per
  anchor cell, radius four, one-cell-width corridor, hardware worker count, and
  a 0.5 GiB decoded cache. Examples now specify only the base crop and required
  normal manifest.

## Path statistics

- The current output has no score threshold: every DP path that reaches a sink
  is accepted and written to OBJ. Axis-mismatch and no-path candidates have no
  comparable final DP score and must be counted as unscored rather than treated
  as zero-cost paths.
- Added independent `score_valid` and acceptance state. JSON omits the cost for
  every unscored feasibility failure instead of serializing a misleading zero;
  non-finite final scores fail loudly.
- Independent review required explicit score presence separate from search and
  acceptance, a searched-but-unscored count, finite-score enforcement, and
  `n/a` rather than numeric sentinels for empty score populations.
- The real-crop OBJ contains 792 groups and 12,046 vertices, but only 792
  multi-index `l` records with 11--24 indices each. Although valid OBJ, that
  encoding is not rendered reliably by MeshLab. The output will instead contain
  one explicit two-index line record per adjacent path edge.
- Added `paths --stats` and reusable summary coverage, including distinct
  scored/accepted populations and empty ranges. The real crop reports 216
  anchors, 5,710 total pairs, 4,918 pre-DP rejections, 792 searches, zero
  searched-but-unscored failures, and 792 accepted fiberlets. Both current
  score populations have min 3.18031, mean 10.737, and max 20.8916 because no
  quality threshold exists yet.
- Regenerated the OBJ with 792 groups, 12,046 vertices, and 11,254 explicit
  two-index line segments; every line record has exactly two indices. Built
  `vc_cli_all` with 32 jobs and passed all five focused suites, including nine
  fiberlet-path cases.
