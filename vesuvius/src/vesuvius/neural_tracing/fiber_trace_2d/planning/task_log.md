# Task Log: fiberlet graph replay

## Discovery

- Fiberlet pair generation currently enumerates only the half-open radius-four
  shell. The requested shorter pairs can use the same deterministic offset
  enumeration with a zero lower bound.
- Fiberlet DP already preloads the dense fiber direction at every integer
  search voxel for alignment scoring. The hard direction-feasibility test can
  therefore run without additional sampling. Lasagna normals remain separate
  inputs used by the existing curvature split.
- Accepted fiberlet paths retain exact anchor identities, dense integer paths,
  endpoint positions, and component loss. This is sufficient to build graph
  topology without reparsing OBJ output.
- Existing dense replay owns the forward exact-reference matching rule and
  extracts a bounded failure tube. The new graph replay will reuse that tube as
  a controlled comparison region and publish its route beside the greedy trace.

## Interpretations

- "Deviation from the sampled normal" is the unoriented angular deviation from
  the dense fiber prediction axis at that voxel, not deviation from or relative
  to the separate Lasagna surface normal.
- The 25-degree and 45-degree limits are strict, matching "less than" in the
  request. Invalid fiber predictions reject nonzero DP steps. Invalid Lasagna
  normals retain the existing isotropic curvature fallback.

## Independent Review

- Review required an exact filled-neighborhood definition, systematic removal
  of shell terminology, and strict boundary tests.
- Review initially proposed applying the sample convention to virtual endpoint
  attachments too. Focused tests showed that this makes half-voxel fitted
  anchors unreachable because of integer-lattice quantization. The hard gate
  therefore applies only to actual lattice moves, using the destination fiber
  prediction; the existing endpoint-axis constraint owns virtual attachments.
- Review required two directed arcs per undirected fiberlet, endpoint tangents
  from dense path geometry, strict directed 45-degree joins, canonical graph
  ordering, and node-cycle prevention across committed and lookahead routes.
- Review made beam pruning and seeding deterministic and required a shared
  variable-step forward matcher instead of copying the greedy observer.
- Review required partial-edge failure truncation, separate greedy/graph
  statuses, content-addressed graph/route artifacts, and a stable independent
  napari route layer/reload contract.

## Implementation Finding

- An initial implementation interpreted the 25-degree rule as a Lasagna
  tangent-plane constraint. A real Paris4 replay produced zero accepted paths,
  exposing that this was the wrong field and geometry. The implementation and
  contract now use the sampled dense fiber axis; no compatibility alias or old
  artifact field is retained because these formats are experimental.

## Implementation

- Replaced shell-only pair enumeration with the canonical filled neighborhood
  `0 < norm(offset) < radius+margin`. The default radius-four/0.5 neighborhood
  has 388 symmetric offsets and retains canonical pair deduplication.
- Added the strict destination-prediction lattice gate, renamed the experimental
  CLI/artifact fields directly, and kept Lasagna normal sampling unchanged for
  the shared normal/tangent curvature terms.
- Added `FiberGraph` as a reusable core module. Every accepted fiberlet supplies
  exact forward/reverse arcs; dense endpoint tangents create canonical directed
  transitions only below 45 degrees.
- Extracted the monotone variable-step reference matcher into
  `PolylineGeometry` and made both greedy and graph replay use it.
- Added deterministic receding-horizon beam routing with global per-depth beam
  pruning, node-cycle prevention, configurable beam/lookahead, partial terminal
  edge truncation, and distinct failure/end/exhaustion/start statuses.
- Added `fiberlet-replay`, content-addressed graph and route JSON/OBJ artifacts,
  strict bundle bindings, and a separate reloadable napari route layer that is
  independent of fiberlet display-radius filtering.

## Validation

- Built with 32 jobs:
  `cmake --build volume-cartographer/build -j32 --target vc_fiberlets test_fiberlet_paths test_fiber_replay`.
- The complete configured Volume Cartographer build, including VC3D, also
  passed with `cmake --build volume-cartographer/build -j32`.
- Focused C++ tests passed:
  `ctest --test-dir volume-cartographer/build --output-on-failure -R '^(test_fiberlet_paths|test_fiber_replay)$'` (2/2).
- The broader `ctest --test-dir volume-cartographer/build --output-on-failure -R 'fiber'`
  selection also passed (9/9).
- Viewer tests passed with unrelated installed pytest plugin autoload disabled:
  `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src python -m pytest -q vesuvius/tests/test_view_fiber_presence.py`
  (54/54). Plain pytest currently fails before collection because the host's
  installed plugin entry point imports absent `zarr.testing`.
- Ruff lint and formatting checks passed for the changed Python files.
- A small Paris4 replay with `--threads 32 --along 128 --radius 64` produced
  166 anchors, 561 accepted fiberlets, 152 graph nodes, 2,788 transitions, and
  a five-edge/27-point route with `reference_end`. The Python strict loader
  accepted and hash-verified the published bundle and its route JSON/OBJ,
  including status-specific error indices and aggregate loss consistency.
- Repeating the same replay produced identical root bytes:
  SHA-256 `5f5ce7a59d379899ac58e88b62b7a41edc414f27fdb46636b35c9de115661fb6`.

## Follow-up Default

- Changed the shared `fiber-replay`/`fiberlet-replay` failure-tube defaults from
  `--along 512 --radius 128` to the verified `--along 128 --radius 64`. Explicit
  overrides remain available and all values remain in base voxels.

## Follow-up Progress Reporting

- Extended the final `fiberlet-replay` line with the exact graph stop reason and
  comparable greedy/fiberlet reference progress in the extracted interval.
  Documented that graph `reference_end` currently denotes that interval's end,
  which explains the apparently arbitrary stop at the `--along` boundary.

## Follow-up Unified Comparison Extent

- Removed the independent experimental `--after` CLI setting. `--along` now
  selects reference arclength before/after failure, supplies the graph replay
  interval, and derives greedy postroll as
  `ceil(along / effective_trace_step_base)`.
- Independent review identified that independently clipping each side could
  still produce unequal comparisons near reference/trace boundaries. Added a
  shared core comparison-window helper that reduces the requested value to one
  effective symmetric half-extent available on all four sides, then uses it
  for the reference tube, greedy display, and graph interval.
- Kept the full greedy trace, losses, matches, and failure index as diagnostic
  bundle data. Added strict `comparison_trace_points_base_xyz`; the trace OBJ
  and napari greedy layer use its exact arclength crop around failure. The
  bundle persists requested/effective extents plus reference/trace arc bounds;
  C++ publication and Python loading validate and reconstruct the slice.
- A default Paris4 smoke run resolved an 8-base-voxel greedy step and therefore
  requested/completed 16 postroll steps for `--along 128`. The published
  effective symmetric half-extent was 127.999273 base voxels because the 16
  realized greedy steps were slightly shorter than their nominal length. Both
  reference and displayed greedy intervals therefore use the same 255.998547
  base-voxel scope. The graph route reached that same interval end; its matched
  reference progress was 252.102 base voxels.
- Rebuilt `vc_fiberlets`, `test_fiber_replay`, and `test_fiberlet_paths` with
  `-j32`. All nine `ctest -R fiber` tests passed. The strict viewer module
  passed 55/55 tests with plugin autoload disabled, Ruff passed, and the strict
  loader accepted and hash-verified the real replay bundle.

## Follow-up Fiberlet Graph Postroll

- Replaced the stop-at-first-crossing graph contract. The first strict
  reference-distance failure is retained with route point, candidate, directed
  arc, local edge point, reference match, and error, but later threshold
  crossings neither stop routing nor replace it.
- Graph replay now completes the failure-containing fiberlet and every later
  selected fiberlet. Routed geometric base-voxel arclength after the failure
  drives postroll; stopping occurs only at the first target anchor at or beyond
  the effective `--along` request. Complete edges may overshoot; graph or
  comparison-interval exhaustion reports truncation and shortfall.
- Experimental route JSON now stores `failure_with_postroll` or
  `failure_truncated`, full candidate/directed-arc arrays, the immutable first
  failure, final stop node, requested/completed postroll, complete flag, and
  overshoot/shortfall. Full-edge loss and length metrics include the failure
  edge. The strict Python loader verifies phase/status fields and recomputes
  completed postroll from route geometry.
- Repeating the Paris4 workload with `--along 512 --threads 32` produced 83
  route points over 16 complete fiberlets. Its graph component ended at an
  anchor after 449.926/512 base voxels of postroll, so it correctly reported
  `failure_truncated`, `postroll_graph_exhausted`, and a 62.074-voxel shortfall
  instead of the prior 11-point partial-edge result. The strict viewer loader
  accepted and hash-verified the expanded route artifact.
