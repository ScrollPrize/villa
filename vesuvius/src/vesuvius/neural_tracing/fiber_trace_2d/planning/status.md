# Status: Truly Rolling Shared 3D Tiled Inference

- [x] Read local workflow, specs, overarching plan, and code-structure docs.
- [x] Diagnose full-Z mmap reservation and full-XY flush allocations.
- [x] Inventory independent Lasagna and Fiber inference paths.
- [x] Discard the rejected RAM-budget/macroblock/re-inference design.
- [x] Draft fixed-depth circular mmap and single-runner plan.
- [x] Preserve the detailed Lasagna/Fiber divergence audit.
- [x] Include spec, docs, tests, changelog, and task-log work.
- [x] Independent review of the replacement plan.
- [x] Incorporate review findings on ring sizing, weight/resume semantics, crop
  storage, chunked clearing, portable lifecycle, and remaining divergences.
- [x] Correct weight design to one geometric weight ring per scale with
  cross-product liveness tracking.
- [x] User approval of the replacement plan.
- [x] Add behavior-characterization and circular-layout regression tests.
- [x] Implement circular mmap planner/store and chunked flush.
- [x] Consolidate Lasagna and Fiber onto one runner.
- [x] Remove legacy runner, fake rolling mmap, and caller-owned flush loops.
- [ ] Validate byte compatibility, resume/crop behavior, scratch sizing, and RSS.
  Unit-level ring and one-pass multi-scale tests pass; Zarr-backed tests are
  blocked in this environment by `zarr.open` hanging before inference, and a
  representative GPU volume run remains outstanding.
- [x] Update specs, code-structure docs, changelog, status, and task log.
