# Kollesis candidate diagnostics (not anatomical classification)

Use saved 20-sweep CCW and CW solutions (same state as inspected surfaces),
without changing or waiting for the continuing 100-sweep solves. No endpoint
density. CPU-only via approved crop-benchmark runner; no inference or optimizer.

## Metrics and localization
- Parallel same-family pair crowding from existing distinct-source constraints:
  average absolute connector projection onto both endpoint normals; transverse
  distance from both projections; normal agreement >=0.9, transverse <=32vx.
  Report fractions in normal-distance bands 0-8,8-16,16-32,32-64,64-128vx,
  normalized by eligible encounters within128vx. This is encounter crowding,
  not an anatomical extra-sheet count or an exhaustive neighbor search.
- Lasagna discrepancy: exact posterior mean absolute discrepancy between
  absolute model layer difference and measured winding distance, including the
  model's phase correction and H/V half-layer offset. Evaluate all hypotheses
  using sorted weighted cumulative sums (not a 100x100 tensor per event).
- Rejection probability, length weighted; localize rejection mass using the
  conditional winding distribution, not a hard chosen winding.
- Conditional winding entropy / log(100), length weighted by retention mass.
  Label uncertainty separately from rejection. Include rejected-piece conditional
  localization as uncertain inferred positions, not measured geometry.
- Constraint maps split half their weight to each endpoint; weight by both
  retention probabilities and endpoint winding probability. No interpolation
  between remote endpoint charts. Keep separate parallel/perpendicular residuals.
- All winding hypotheses retained. Coordinates reuse shared UV coordinate code.
  Fiber length quadrature at <=250vx spacing; U/V buckets1000vx each. Full-height
  U profiles are ratios of summed numerators/denominators, not unweighted pixel
  means. Heatmaps use identical bins and show missing support as missing.

## Outputs and limitations
Per convention: SVG and compressed TIFF profiles/heatmaps; numeric NPZ and CSV;
support maps, supported-V fraction and elevated-V fraction. Elevation is an
exploratory support-weighted global75th-percentile threshold, not a detector.
Heatmap physical axes use equal base-voxel scale; long U is divided into panels
for legibility. No endpoint metric, combined score, overlap inference or rejection
of data. Sampling/extraction bias, same-layer fibers, compression and duplicate
traces can cause peaks; winding gauge uncertainty can smear candidate bands.

## Implementation, tests, review
Read-only database scan into reusable geometry arrays keyed by exact event IDs;
validate database and stage1 hashes against frozen tensor signatures. Bounded
vectorized event batches, progress/timings. Reuse coordinate and inward-normal
conventions, no copied optimizer geometry. Compare cumulative-sum expectation
to brute-force pairs; test mass conservation, entropy, normalization, missing
support, symmetric endpoint deposition and source identity validation. Independently
review plan and implementation, run full plots and report observations/caveats.

## Spec update / docs updates
Derived diagnostics only, no solve/constraint changes. Update task/status/log,
README/changelog and standalone results with definitions, run commands and paths.
