# Plan: integer-DP fiberlet paths

## 1. Artifact and coordinate contracts

1. Add a strict loader for the existing `vc_fiberlet_anchors` version-1 JSON.
   Validate format/version, source manifest locator/hash, prediction ZYX shape,
   positive prediction-to-base scale, cell size, cell/component array shape,
   finite retained positions/support/axes, unit nonzero axes, component indices,
   cell ownership, deterministic cell ordering, and grid bounds. Do not repair
   malformed artifacts. Keep anchor identity as `(cell_zyx, component_index)`;
   do not introduce opaque encoded identifiers.
2. Add a `vc_fiberlets paths` command taking a fiber manifest, `anchors.json`,
   output directory, and required `--normal-manifest`. Open local, sidecar-
   remote, HTTP, and S3 manifests through the existing cache-first Lasagna
   APIs. Require `--remote-cache-dir` for direct remote manifests. Verify the
   materialized fiber-manifest content hash and stored prediction grid against
   the anchor artifact before sampling. The stored locator is informational and
   is not required to match, so an identical manifest may be relocated without
   invalidating its anchors. Configure the normal dataset so path
   prediction coordinates map to base coordinates with the anchor artifact's
   authoritative prediction-to-base scale. Never use fiber `nx/ny` as Lasagna
   surface normals.

## 2. Fixed-radius candidate pairs

3. Flatten retained anchors in canonical cell/component order. Precompute
   integer cell offsets in the half-open shell
   `radius-0.5 <= norm(offset) < radius+0.5`, initially radius four. Exact
   Euclidean equality to four is not used because it would strongly bias the
   shell toward grid axes. For every occupied source/target cell pair on the
   shell, enumerate all retained component pairs once in canonical anchor-ID
   order.
4. Reject zero-length pairs and pairs for which either unoriented endpoint axis
   differs from the anchor chord by more than a configurable angle, initially
   45 degrees. Orient both accepted endpoint axes from source toward target for
   the directed solve. Record generated, axis-rejected, out-of-grid, searched,
   successful, and no-path counts deterministically. Do not enforce anchor
   degree, mutual-best selection, overlap rejection, or path deduplication.

## 3. Integer directed search graph

5. Preserve each fitted endpoint as an exact virtual source/sink. Connect it to
   every distinct in-bounds integer voxel within `sqrt(3)` prediction voxels
   whose nonzero
   virtual edge has positive chord progress and satisfies the endpoint-angle
   bound; an exact integer endpoint uses a zero-length attachment carrying the
   oriented endpoint axis. Score nonzero source attachments at their integer
   destination and include their direct curvature; score sink-attachment
   curvature but do not rescore its already visited integer source. This avoids
   rounding away endpoint precision or leaving an unvalidated endpoint kink.
   Build a cubic-Hermite reference from exact endpoints and their chord-oriented
   axes, with each derivative magnitude equal to the endpoint distance. Sample
   it uniformly in parameter with `max(8, ceil(4*endpoint_distance))` segments
   and define corridor membership by exact point-to-polyline-segment distance,
   not distance to samples. The default corridor radius is one anchor cell side
   in prediction voxels. Always insert valid virtual-attachment voxels even at
   the corridor boundary so sampling cannot disconnect an endpoint.
6. Use the 26 nonzero integer offsets in `{-1,0,1}^3`. For a pair, retain only
   oriented moves having strictly positive projection onto its chord. This
   induces an acyclic graph ordered by chord progress. The DP state is
   `(integer_voxel, incoming_move)`, plus one source-attachment state at each
   eligible source voxel whose incoming direction is that exact virtual edge.
   Finalize over eligible sink attachments using the actual final virtual edge.
   Constrain both real endpoint segments by the endpoint angle and reconstruct
   deterministic predecessor states between them. Use stable voxel/move and
   attachment ordering and exact tie rules.
7. Batch-sample every corridor node once from the canonical stored fiber
   prediction grid and the regular Lasagna normal sampler. Integer prediction
   nodes are the only searchable/sampled fiber positions; interpolation inside
   the independent normal grid remains required by the established compact-
   normal sampler. An invalid fiber sample pays only
   `invalid_prediction_cost_per_prediction_voxel * edge_length`; its presence and
   direction components are zero, while independently valid Lasagna normals
   may still govern curvature. The default invalid cost is 4.0. Thus a short
   missing-prediction gap is bridgeable and, because this stage has no quality
   cutoff, even a mostly invalid geometrically feasible route may succeed and
   report its invalid-cost total. Volume bounds, endpoint attachment failure,
   and unreachable graph states are hard `no_path` failures.

## 4. Additive local objective

8. For every valid sampled unoriented prediction axis `d`, compute its local
   stencil floor
   `theta_q(d)=min_m acos(abs(dot(d,m)))` over the pair's forward move stencil.
   A candidate move pays
   `direction_weight * max(0, theta(d,m)-theta_q(d))^2 * edge_length`.
   Consequently the best direction representable by that discrete stencil has
   zero direction cost. Add
   `presence_weight * (1-clamp(presence,0,1)) * edge_length`; multiplying by
   Euclidean edge length prevents diagonal moves from becoming cheaper merely
   because they use fewer graph nodes.
9. Extract the native tracer's local isotropic/Lasagna-normal split smoothness
   calculation into a reusable helper. Preserve the tracer's existing float
   equations and behavior exactly. The fiberlet DP applies that helper between
   the incoming and candidate move using a configurable free angle, initially
   45 degrees to tolerate ordinary 26-neighbour lattice switching, and the
   native default isotropic/normal/tangent weights `2/0.1/10`. Convert its
   per-turn cost to a mixed-edge lattice bending cost by dividing by
   `max(1, (previous_edge_length+candidate_edge_length)/2)`; axial unit moves
   therefore exactly retain the native value, while a turn distributed over
   longer diagonal edges represents gentler curvature. The shared helper
   returns a breakdown and mode: valid normals produce mutually exclusive
   tangent and normal components, while invalid or degenerate normals produce
   only the isotropic fallback, exactly preserving native tracer summation.
   Store invalid, direction, presence, isotropic, tangent, and normal cost
   contributions separately.
   Do not add cumulative history smoothness or continuous refinement.

## 5. Results and CLI output

10. Define reusable path/report types containing endpoint anchor IDs, exact
    anchor endpoints, integer interior points, success/rejection reason, total
    and component costs, and aggregate diagnostics. Successful paths contain
    exact fitted endpoints plus the integer DP path without duplicate adjacent
    points. Candidate ordering and output must not depend on worker count.
11. Write `fiberlets.json` with format/version, credential-free source
    identities and hashes, coordinate/scale contract, anchor artifact identity,
    fixed-shell/search/objective parameters, diagnostics, and every considered
    pair's success or rejection. Write `fiberlets.obj` with a named line group
    per successful path in base coordinates. Both use atomic replacement.
    Rejected pairs stay in JSON for tuning but never produce OBJ geometry.

## 6. Tests and manual validation

12. Add focused unit tests for radius-four shell isotropy and half-open bounds;
    duplicate-free deterministic pair generation; unoriented endpoint-axis
    gates; integer-only graph paths; isotropic `sqrt(3)` virtual attachments; actual first
    and final segment angle enforcement; exact virtual endpoints; monotone progress;
    quantization-floor zero cost for the best representable move; worse-move
    direction cost; axial/diagonal edge-length weighting; presence preference;
    direct curvature preference; mixed axial/diagonal curvature normalization;
    separate tangent/normal curvature behavior and isotropic fallback; finite
    invalid-prediction accounting and gap bridging; geometric no-path reporting;
    deterministic ties; and
    JSON/OBJ topology and scale conversion.
13. Add loader/integration tests for malformed/wrong-version anchor JSON,
    source hash/grid/scale mismatch, separate normal-manifest use, and a tiny
    local end-to-end fixture. Add direct golden cases for the extracted native
    smoothness equations with valid normals, invalid normals, degenerate tangent
    projection, and nonzero free angle, and confirm the existing native tracer
    wrapper and focused tests remain numerically unchanged.
14. Build `vc_fiberlets`, the new focused tests, and the complete CLI target
    set with 32 threads. Run the focused anchor, fiberlet-path, native tracer,
    Lasagna manifest, and normal-sampler tests. Run `paths` twice on a small
    representative crop, require byte-identical JSON/OBJ, and record command,
    configuration, candidate/path counts, timing, and manual OBJ inspection
    status. If suitable local data for the regular normal manifest is not
    discoverable, report the real-data run as explicitly pending rather than
    substituting fiber directions for normals.

## Spec update

- Extend `planning/specs.md` with the strict anchor-artifact input contract,
  radius-four half-open cell shell, endpoint-axis gating, integer 26-neighbour
  forward DAG, Hermite corridor, virtual endpoints, quantization-relative
  direction loss, edge-length weighting, shared direct Lasagna-aware
  smoothness, invalid-prediction handling, and fiberlet JSON/OBJ formats.
- State explicitly that cumulative smoothness, continuous/sub-voxel search,
  global graph selection, deduplication, extension, H/V/winding optimization,
  CUDA, and production radius calibration remain out of scope.

## Docs updates

- Extend `volume-cartographer/docs/fiberlets.md` with the `paths` command,
  candidate shell, integer-DP state and costs, separate normal-manifest rule,
  output formats, limitations, and small-crop inspection workflow.
- Update the fiber-tracing code-structure documentation to name the reusable
  path module and shared local smoothness helper.

## Changelog and workflow

- Add a changelog entry after implementation for the first integer-DP
  anchor-connection/fiberlet output stage without claiming graph construction
  or final over-segmentation is complete.
- Replace `planning/task_log.md` with this task's review, implementation
  decisions, deviations, commands, and validation results. Update
  `planning/status.md` incrementally.

## Base-coordinate CLI adaptation

15. Replace `--crop-prediction-xyzwhd` with the short flag `--crop`. Interpret the
    supplied integer XYZ origin and WHD extent as a half-open base-volume box,
    convert it to stored-prediction point centers using ceil on both half-open
    boundaries, with scale-aware snapping at exact lattice boundaries, and then
    retain the existing whole-cell selection semantics. Do not retain an alias
    for the unshipped prediction-coordinate flag.
16. Keep the short `--corridor-radius` flag but redefine its unshipped contract
    exclusively in base voxels. Convert the explicit radius to prediction voxels only after the fiber
    manifest establishes `prediction_to_base_scale`. The default remains one
    anchor-cell width, represented internally in prediction voxels.
17. Change version-1 anchor JSON to declare `base_volume` position space and
    store the effective `prediction_interval_origin_base_xyz`,
    `prediction_interval_size_base_xyz`, and
    `position_base_xyz`. Its strict loader converts those values back into the
    internal prediction grid and validates cell ownership there. Change
    fiberlet JSON analogously to store base-coordinate endpoints and path
    points plus `corridor_radius_base_voxels`. OBJ remains base-coordinate.
    Cell ZYX indices, prediction shape/scale metadata, cell size, Gaussian
    sigma, and lattice costs remain in prediction-grid units.
18. Update serializer/strict-loader tests to require only the new field names,
    add non-unit-scale conversion coverage including a non-aligned base crop,
    rebuild with 32 jobs, rerun the five focused suites, and rerun the small
    local crop using the base-coordinate CLI.

### Spec update

- Make base-volume coordinates the only external spatial-coordinate contract
  for fiberlet CLI flags and JSON/OBJ artifacts. Keep prediction-grid units only
  for explicit discrete-lattice indices and algorithm parameters.

### Docs update

- Replace the prediction-coordinate crop examples, state that CLI spatial
  values are base-volume coordinates, and explain the
  conservative base-box conversion at prediction-grid boundaries.

## Path statistics

19. Add a flag-only `--stats` option to `vc_fiberlets paths`. After tracing,
    report anchor count, total candidate pairs, pre-DP rejections, DP searches,
    scored paths, accepted fiberlets, and unscored candidates.
20. Compute finite min/mean/max total objective values for all scored paths and
    independently for accepted paths. A DP path has a score only after a sink
    path is found; axis-rejected and `no_path` candidates are unscored. Current
    endpoint/path feasibility is the only acceptance rule, so both populations
    intentionally match until quality filtering is added.
21. Put the summary calculation in the reusable fiberlet path module, add unit
    coverage for distinct scored/accepted subsets and empty populations, expose
    it through the CLI, document the semantics, and rerun focused tests plus the
    real crop with `--stats`.
22. Represent score presence independently from acceptance. Reject a non-finite
    final objective loudly. Count pre-DP unscored candidates separately from
    searched-but-unscored feasibility failures, and print `n/a` for empty score
    ranges.
23. Serialize each successful path as one OBJ group containing an explicit
    two-index `l a b` record for every adjacent vertex pair. Add a regression
    assertion that line-record count equals the sum of `path_points-1`, then
    regenerate the real-crop OBJ for MeshLab inspection.
