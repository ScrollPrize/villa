# Status: bounded intermediate fiberlet lookahead

- [x] Capture the three search options and select intermediate pruning first.
- [x] Write the implementation, testing, spec, docs, and benchmark plan.
- [x] Complete independent plan review and incorporate findings.
- [x] Add replay configuration, CLI validation, and diagnostics.
- [x] Implement deterministic intermediate expansion and pruning.
- [x] Replace exhaustive between-front expansion with bounded distance-label
  search.
- [x] Add focused correctness, scaling, and determinism tests.
- [x] Update specs, documentation, and changelog.
- [x] Build and run the focused replay test target with `-j32`.
- [ ] Benchmark widths 64, 128, and 256 on focused hot-cache intervals.
- [x] Run the selected width on the full radius-768 fiber; result is rejected
  (about 19m13s and seven fiberlet failures).
