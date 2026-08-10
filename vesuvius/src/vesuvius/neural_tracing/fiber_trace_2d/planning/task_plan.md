# Plan: cell anchor extraction

## 1. Contracts and coordinates

1. Add reusable C++ anchor types under `vc::fiber_tracer`, separate from the
   CLI. An accepted anchor records its integer cell ZYX index, sub-voxel XYZ
   position in canonical prediction-grid coordinates, unit unoriented XYZ axis,
   aligned presence support and directional coherence. Persist the authoritative
   prediction-to-base scale once with the artifact and derive base coordinates;
   do not store two position triples that can disagree. A cell result records
   the two fitted components and why each was retained or discarded.
2. Define a cubic `cell_size_prediction_voxels` on the actual stored
   `presence/nx/ny` grid. Obtain the authoritative prediction-to-base scale and
   prediction shape from the manifest-backed bindings. Require the production
   fiber format's single canonical `presence/nx/ny` triplet and reject a missing
   or incomplete triplet. Other manifest groups are not observations and do not
   change the fit. The three channels must share shape and spacing, while
   different physical chunk layouts remain valid. Cell lattice origins are
   anchored at prediction-grid origin zero so crops,
   batching, and thread counts cannot move anchors.
   Convert output positions to base voxels explicitly. Report both cell side
   length and cube diagonal in base voxels when evaluating whether two parallel
   fibers can occupy a cell. Report micrometres only when an optional positive
   base-voxel size is supplied; it never affects the solve.
3. Extend the existing prediction-field API with a stored-grid boundary that
   exposes prediction-grid shape/spacing metadata and samples integer
   prediction-grid indices as one unoriented direction plus presence. Internally
   it must reuse the same compact `nx/ny` tensor decoder, local/remote Zarr
   access, and transparent cache as tracing. Do not call the tracing
   reference-direction API with a dummy direction, and do not introduce a
   second manifest reader, Zarr reader, compact-axis decoder, or remote cache
   path. With working scale one, define
   `base_xyz = prediction_xyz * prediction_to_base`, where
   `prediction_to_base = source_to_base * group.scaleFactor()`.

## 2. Deterministic cell observations

4. Treat integer XYZ prediction indices as voxel centres. Cell `(cz,cy,cx)`
   owns the ZYX half-open index ranges
   `[c*S, min((c+1)*S, shape))`, including clipped cells only at the global
   volume boundary. Its nominal Gaussian centre on each axis is
   `c*S + (S-1)/2`, including a clipped final cell. Define
   `--crop-prediction-xyzwhd` in half-open prediction
   index coordinates. It selects every globally anchored cell it intersects
   and samples those cells in full; it must not redefine or truncate an
   interior cell, so a crop run and full-volume run produce identical results
   for the same cell. Enumerate cells and voxels in Z/Y/X order and sample the
   prediction field in chunk-friendly batches.
   An observation consists of prediction-grid position, decoded unit axis,
   presence, and validity. Invalid/non-finite axes and presence
   below the configured inclusive observation floor contribute nothing.
5. Give each observation the fixed weight
   `gaussian(position - cell_center, sigma) * presence`. The Gaussian is
   centered on the nominal fixed global cell centre and uses a configurable
   sigma expressed in prediction voxels. It is not recentered during fitting,
   which keeps adjacent cells independent and the solve deterministic. All
   directional terms use squared or absolute dots, so
   arbitrary `d` versus `-d` decoding cannot affect a result.

## 3. Non-orthogonal two-direction fit

6. Always attempt a two-component fit over the valid observations. For unit
   unoriented directions `u0,u1`, maximize the exact normalized support
   objective
   `Q = sum(w_i * max_k((d_i dot u_k)^2)) / sum(cell_gaussian_i)`, where
   `w_i = cell_gaussian_i * presence_i`. Assign each observation exclusively to
   the component with the greater squared dot, using a deterministic tie break.
   The denominator includes the Gaussian weight of every voxel centre owned by
   the cell, including zero-presence, below-floor, invalid, and missing
   predictions, so unsupported coverage reduces support. For clipped edge
   cells it includes every actually owned voxel and no out-of-volume samples.
   This is a mixture of two directional PCA components, not the first two
   eigenvectors of one covariance matrix; `u0` and `u1` are not constrained to
   be orthogonal.
7. With assignments fixed, update each component independently using the
   principal eigenvector of its weighted dyadic tensor
   `M_k = sum(assigned_i * w_i * d_i * d_i^T)`. Alternate assignment and PCA
   updates to convergence or a fixed iteration limit. Construct a deterministic
   seed list from the global weighted principal direction followed by up to
   seven valid observed directions selected greedily by greatest
   `w_i * min_j(1-(d_i dot seed_j)^2)`, with canonical voxel-order ties. Enumerate
   every unordered pair from that list; if it contains only one direction, use
   that direction for both initial components and assign exact ties to component
   zero. Retain the converged pair with greatest `Q`, choosing the earliest
   seed pair on an exact tie; do not seed the second line from the necessarily
   orthogonal second eigenvector of a global PCA. This handles
   two non-orthogonal direction modes and selects the best-supported two when a
   rare cell contains more than two modes. Canonicalize component ordering from
   position and sign-invariant axis keys. For serialization only, choose the
   equivalent axis sign whose largest-absolute component is positive, breaking
   equal-magnitude ties in X/Y/Z order. This is not a directed fiber
   orientation and requires no epsilon-dependent sign decision.
8. For each fitted component, define its aligned presence support using the same
   `p=2` term:
   `support_k = sum(assigned_i * gaussian_i * presence_i *
   (d_i dot u_k)^2) / sum(cell_gaussian_i)`. Define coherence diagnostically as
   the aligned numerator divided by `sum(assigned_i * gaussian_i * presence_i)`.
   Compute the anchor position from the aligned-support centroid
   `sum(assigned_i * gaussian_i * presence_i * (d_i dot u_k)^2 * position_i) /
   aligned_numerator`. Independently discard each empty, degenerate, or
   below-minimum-support component; do not apply a one-versus-two gain test or
   require the two retained directions to be orthogonal. The cell consequently
   emits zero, one, or two anchors. Discarding a component does not reassign its
   observations or refit the survivor: retained anchors describe the selected
   two-component optimum. Define the support threshold boundary inclusively and
   restrict ordinary cell size to integer `[2,8]`.

9. Make floating reductions reproducible: preserve the canonical observation
   order before every reduction, use fixed-order compensated sums for dyadic
   tensors, masses, and centroids, define deterministic eigenvalue-degeneracy
   rejection/hints, enumerate starts in fixed order, and use deterministic
   component ordering. Parallelism may distribute complete cells only; it must
   not change reductions inside a cell. Stop an iteration when assignments are
   unchanged and the maximum projective update
   `1-abs(dot(u_previous,u_updated))` is at most `1e-12`, with a default hard
   limit of 64 iterations. Detect an assignment two-cycle, retain its higher-Q
   state with the earlier state winning an exact tie, and stop.

## 4. C++ tool and artifacts

10. Add an anchor-stage command to a C++ `vc_fiberlets` CLI. It accepts a local
   path or remote URL to a fiber `.lasagna.json`, output directory, optional
   prediction-grid crop, cell size, Gaussian sigma, observation-presence floor,
   minimum aligned-support threshold, PCA start/iteration controls,
   decoded-chunk cache GiB, persistent remote-cache directory, and worker count.
   Full manifest extent is the default, while the crop makes tuning runs
   practical. Require
   `--remote-cache-dir` for a direct remote manifest and use cache-first
   manifest/Zarr reads; continue to support a local manifest accompanied by the
   existing `lasagna-remote.json` sidecar without requiring a second format.
   Validate all parameters before reading prediction chunks and print resolved
   prediction/base scales, grid/crop/cell counts, cache settings,
   rejection counts, anchor counts, and stage timings.
11. Write a versioned `anchors.json` with a credential-free normalized source
    locator and materialized manifest content hash, never a cache path or signed
    URL. Include explicit XYZ/ZYX field naming, prediction shape and
    prediction-to-base scale, selected crop/cells, parameters,
    aggregate zero/one/two/rejection diagnostics, and accepted non-empty cells
    in deterministic cell order. Do not serialize one verbose record for every
    empty cell in a whole volume. Write `anchors.obj`
    with one named line glyph centred at each anchor and aligned with its axis;
    use a configurable glyph length in base voxels. The JSON is the later
    connection stage's input. The OBJ is diagnostic only. Use atomic replacement
    for completed artifacts and do not leave a
    valid-looking partial JSON after interruption. Serialize finite floats with
    a stable maximum-precision format so repeated runs are byte-identical.

## 5. Tests and calibration

12. Add unit tests with synthetic prediction sources for: empty/invalid cells;
    one straight direction; arbitrary per-sample sign flips; two directions at
    15, 30, 45, 60, and 90 degrees; three-mode cells selecting the
    greatest-support pair; weak second directions; curved or noisy directions;
    equal-eigenvalue degeneracy; edge cells; aligned-support centroid
    localization; empty components; support-threshold boundaries; and
    deterministic equality across arbitrary decoded-axis sign, batch size,
    crop-versus-full selection, and worker count. Assert finite normalized axes,
    positions inside their cells, stable component ordering/sign
    canonicalization, and exact prediction-to-base coordinate conversion.
    Add a direct fixed-assignment test proving that the weighted PCA update does
    not decrease `sum(w_i * (d_i dot u)^2)` and reaches the principal-eigenvector
    objective.
13. Add manifest integration tests using tiny local Zarr fixtures for exactly
    one canonical `presence/nx/ny` triplet; missing/incomplete or
    malformed canonical channels; unrelated additional manifest groups that do
    not change results; missing or invalid `source_to_base`;
    mixed-shape/spacing rejection; different valid chunk layouts; half-open
    crop bounds; crop/full-cell equivalence; JSON
    schema and credential/cache-path exclusion; and OBJ topology. Exercise both
    a direct remote manifest's required-cache/second-run-hit behavior and a
    local manifest with the existing remote sidecar through the shared test
    transport rather than creating a second fake network stack.
14. Build the tool and focused tests with the established CMake build, then run
    a real representative crop at cell sizes 2, 4, and 8 prediction voxels.
    Record command, manifest/crop, Release build, CPU/thread count, wall time,
    peak memory, cells/s, zero/one/two-anchor counts, rejection histogram, and
    visually inspect the OBJ over the source volume. Compare each cell's base-
    voxel side and diagonal against the intended minimum sustained fiber/sheet
    separation and choose the largest empirically safe size. The stage is not
    accepted until repeated runs produce byte-identical JSON/OBJ. A 4-voxel
    micrometre spacing claim may be reported only when a positive base-voxel
    size is supplied explicitly; manifest metadata alone determines base-voxel
    scale, not micrometres.

## Deferred connection-stage questions

15. Use the measured anchor density, localization error, and cell-size result to
    set the neighbor radius and prediction-gap budget. Only then specify the
    straightened directed search lattice, Lasagna normal/tangent constraints,
    CPU versus CUDA implementation, path-quality metric, overlap
    deduplication, and extension rules. None of those decisions belong in the
    anchor implementation.

## Spec update

- Add a `Fiberlet anchor extraction` section to `planning/specs.md` defining
  stored-prediction-grid coordinates, fixed-origin cells, unoriented weighted
  observations, deterministic non-orthogonal two-component assignment/PCA,
  squared-alignment-modulated presence support, acceptance diagnostics,
  structured output, and the explicit current-stage
  exclusions.
- Do not add connection/path-search behavior to the normative spec until its
  design is based on measured anchor output.

## Docs update

- Add `volume-cartographer/docs/fiberlets.md` describing the overall staged
  pipeline, current anchor algorithm, CLI parameters, JSON/OBJ formats,
  coordinate scales, local/remote manifest behavior, cache use, and real-crop
  calibration workflow.
- Link the reusable anchor module and CLI from the fiber tracing code-structure
  documentation.

## Changelog and workflow

- After implementation and validation, add one changelog entry for the C++
  manifest-to-anchor stage; do not claim fiberlet connection or extraction is
  complete.
- Replace `planning/task_log.md` with current-task findings, deviations,
  commands, test results, and calibration measurements as work proceeds, and
  update `planning/status.md` incrementally.
