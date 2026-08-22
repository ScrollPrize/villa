# Status: restore baseline fiberlet search around weighted lookahead

- [x] Capture the regression-repair task.
- [x] Write the implementation and validation plan.
- [x] Review the plan independently against task and specifications.
- [x] Implement prefix-plus-weighted-forward score decomposition.
- [x] Implement direct decoded-profile linear subsegment integration.
- [x] Implement incremental exact and bounded search scoring.
- [x] Restore effective admissible relaxed pruning.
- [x] Add focused correctness and determinism coverage.
- [x] Build and run focused tests with `-j32`.
- [x] Benchmark the actual pre-change executable and repaired 5k modes.
- [x] Validate the longer regression corridor.
- [x] Finish specification, documentation, changelog, and task log.
- [x] Remove score-initialization walks over the committed prefix.
- [x] Replace full logical-route registry sweeps with bounded cleanup.
- [x] Add long-prefix regression coverage and remeasure hot-cache replay.
