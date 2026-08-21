# Status: exact cost-bounded fiberlet lookahead

- [x] Capture the corrected task and alternative search candidates.
- [x] Write the implementation, spec, docs, and validation plan.
- [x] Complete independent plan review and incorporate findings.
- [x] Correct terminal-edge scoring to the exact horizon.
- [x] Implement deterministic parallel fixed-prefix exact lookahead, admissible
  relaxed A* bound, and strict per-prefix cutoff.
- [ ] Add focused scoring, optimality, pruning, and determinism tests.
- [ ] Update specs, documentation, and changelog.
- [ ] Build and run relevant tests with `-j32`.
- [x] Run the focused radius-768 hot-cache validation.
- [ ] Complete the full-fiber validation; exact search remains impractical at
  the dense 95.8% decision.
