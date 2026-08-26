# Task Log: Direction-label MILP diagnostic

- The direction reference is per stored trace, while the existing solver labels
  extracted pieces. Every retained piece will inherit its source trace's
  direction reference; reports will include piece and unique-trace errors.
- Disconnected MILP components have independent H/V gauge symmetry. Directly
  comparing canonical H/V to global direction 1/2 would count arbitrary flips,
  so comparison will minimize disagreement independently per active component.
- Mixed traces must be removed before extraction rather than merely ignored in
  the comparison, so they cannot influence neighbors, scores, degree penalties,
  broken decisions, or the optimized graph.
- Independent review fixed the CLI matrix and report denominators. The new mode
  will force ordinary discrete H/V-only labeling, reject explicit `--hv-only`
  and incompatible solver ablations, but allow MIP-gap and no-winding-cutoff.
  It will translate every filtered-local piece index back to the original trace
  in user-facing errors and emit valid empty outputs for an all-mixed input.
- Implemented the reusable comparison beside the HiGHS labeling API. It builds
  components from the exact retained active-piece graph, resolves each binary
  gauge deterministically, and reports orientation and Broken errors in stable
  piece order.
- Added `direction-diagnostic`, which writes the unfiltered `_initial` direction
  artifact family, removes mixed traces before extraction, reuses canonical
  constraint/pruning/OBJ/MILP paths, and prints the filtered-to-original trace
  mapping for every error.
- Built `vc_fiber_trace_chunk` and `test_fiberlet_crop_trace` with `-j32`; all 41
  focused test cases passed.
- Centered-384 validation at dominance 0.90 classified 179 source fibers as 50
  direction 1, 45 direction 2, and 84 mixed. The retained 95 fibers produced 95
  pieces and 755 constraints. The discrete H/V-only MILP was optimal with no
  Broken pieces. Its one active component required a gauge flip and then
  matched all initial labels: zero orientation errors and zero erroneous source
  fibers.
