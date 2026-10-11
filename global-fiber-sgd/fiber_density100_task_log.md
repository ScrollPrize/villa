# Density maps from saved100 solutions

- Independently reviewed fiber_density100_plan.md. Use enabled-only, all saved
  MAP, all with disabled-only conditional placement against enabled MAP neighbors.
- Primary source arc-length per million inferred UV vx²; H/V separate. Counts
  secondary, split-sensitive. No implication of measured3D area or ground truth.
- Shared exact deposition/canonical geometry/costs. Diagnostic placement omits
  unary priors intentionally; no rejected peers, no changed solver state. Original
  N/E cost normalization and degree-calibrated sign charge; continuation included.
- Refinement: common rectangle snapped to full heat cells (not tiny edge slivers)
  so fixed-coordinate fibers on exact grid maxima retain positive area. Profile
  edge areas clip to this same rectangle; zero-area outside bins remain NaN.
- Exact cost ties keep saved placement. Unsupported disabled pieces explicitly
  retain saved placement; cost-softmax confidence is descriptive, not calibrated.
- Completed approved runner both100 states: CCW58.30s /CW58.14s including plots;
  placement8.98s/8.41s, all calculations19.48s/18.36s. No GPU.
- Independent review no blockers; added shared-helper/source-config hashes,
  continuation-only reversed-endpoint oracle. First oracle test used float64
  logits unsupported by old categorical unary float32 accumulation; corrected
  test to actual solver float32 logits. Production refit remains float64 costs.
- 21tests pass; git diff --check clean. Inspected both profiles and heatmaps.
- Conditional replacement moves57.13%/57.14%disabled buckets; only7/2 lack enabled
  support. Coarse profile redistribution ~2.4–2.6% of total length; no convincing
  common full-height kollesis bands in inspection. Diagnostic residual sign
  conflicts remain large; these are disabled fibers, not re-enabled solutions.
- Added-only maps and count density plots included. Numeric grids and provenance
  saved. Results/links: fiber_density100_results.md; semantics:metrics.md.
  No task items deferred. Earlier-stage rejected fibers absent by input scope.
