# Status: anchor-seeded fiberlet over-segmentation

- [x] Capture the full staged fiberlet task and current anchor-only scope.
- [x] Inspect stored prediction products and reusable C++ sampling APIs.
- [x] Draft coordinate, fitting, artifact, testing, and documentation plan.
- [x] Complete independent plan review against task, plan, and specs.
- [x] Incorporate review findings.
- [x] Correct the plan to the single-direction prediction field and `p=2`
      non-orthogonal two-component PCA objective.
- [x] Obtain user approval for implementation.
- [x] Add reusable cell-anchor types and prediction-grid metadata access.
- [x] Implement deterministic zero/one/two-anchor extraction.
- [x] Add the `vc_fiberlets` anchor-stage CLI and JSON/OBJ writers.
- [x] Add unit and manifest integration tests.
- [x] Build and run focused tests plus full CLI compile coverage.
- [x] Run cell sizes 2, 4, and 8 on a representative real crop and record
      counts, timing, memory, determinism, and diagnostic projection.
- [x] Update specs, docs, changelog, task log, and final status.

Connection search, path optimization, path filtering, deduplication, and
extension remain deferred to later task stages.

A production cell size is deliberately not selected yet. It requires a 3D
volume overlay and an agreed minimum sustained fiber/sheet separation; the
current task records the calibration artifacts and measurements needed for
that decision.
