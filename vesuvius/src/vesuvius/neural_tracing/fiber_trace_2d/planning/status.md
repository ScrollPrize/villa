# Status: integer-DP fiberlet paths

- [x] Capture the fixed-radius integer-DP fiberlet task.
- [x] Inspect the anchor artifact, native 3D scorer, old 2D DP, normal sampler,
      CLI, and build/test boundaries.
- [x] Draft artifact, pairing, graph, objective, output, and validation plan.
- [x] Complete independent plan review against task, plan, and specs.
- [x] Incorporate review findings.
- [x] Add strict anchor-artifact loading and source/grid verification.
- [x] Extract and reuse native local smoothness scoring.
- [x] Implement radius-four candidate generation and integer directed DP.
- [x] Add fiberlet JSON/OBJ serialization and `vc_fiberlets paths`.
- [x] Add focused unit tests and local manifest/CLI integration validation.
- [x] Build and run focused plus compile-coverage validation.
- [x] Run deterministic small-crop output validation.
- [x] Update specs, docs, changelog, task log, and final status.
- [x] Capture the base-coordinate-only CLI and artifact adaptation.
- [x] Independently review the coordinate-contract plan.
- [x] Convert CLI crop and corridor-radius inputs from base voxels.
- [x] Convert anchor and path JSON spatial coordinates to base voxels.
- [x] Update strict tests and documentation for the new contract.
- [x] Rebuild and rerun focused plus small-crop validation.
- [x] Capture the `paths --stats` reporting contract.
- [x] Independently review the statistics plan.
- [x] Add reusable path statistics and CLI output.
- [x] Add tests and documentation for score-population semantics.
- [x] Rebuild and validate `--stats` on the small crop.
- [x] Emit MeshLab-compatible explicit OBJ path edges and validate counts.

Cumulative smoothing, continuous refinement, quality filtering beyond path
feasibility, deduplication, extension, final graph construction, H/V and
winding optimization, CUDA, and production radius selection remain deferred.
