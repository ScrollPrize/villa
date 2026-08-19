# Task log: re-merge fiber-lets2

## Findings

- Benchmark/config documentation was committed separately as `22d1f4098` and
  records the exact pre-documentation code revision
  `9095ba351b6fea1c37253190cedddab5c97373f6`.
- The merge produced conflicts in the four task-local planning files and one
  implementation location in `FiberGraph.cpp`.
- The implementation conflict was between the current anisotropic replay seed
  broad-phase and the source branch's older isotropic distance check. The
  source branch float extraction changes do not require reverting the newer
  anisotropic behavior.

## Deviations

- Stale task-local records from both completed tasks were replaced with this
  concise merge record, as explicitly permitted by the user.

## Validation

- Independent review confirmed that the `FiberGraph.cpp` resolution retains
  the complete float graph/extraction changes while restoring anisotropic
  replay measurement, the 4x tangential seed broad-phase, threshold-ratio seed
  ordering, and threshold diagnostics. Existing replay coverage exercises a
  tangential seed outside the isotropic threshold.
- The first build attempt and first test run failed on temporary-file writes in
  `/tmp`; no source assertion failed. Re-running with
  `TMPDIR=$PWD/volume-cartographer/build/tmp` avoided the constrained tmpfs.
- Built `test_fiber_anchors`, `test_fiberlet_paths`, `test_fiber_replay`, and
  `vc_fiberlets` using 32 jobs.
- Focused CTest run passed all three suites: anchor, path, and replay.
- Two merged canonical Paris4 runs used 32 threads, a 5,000-base-voxel
  interval, and one anchor-refinement iteration. Both retained 2,604 anchors,
  51,780 searched / 26,494 accepted fiberlets, 170,792 sampled voxels, 2,563
  graph nodes, 26,494 graph edges, and 63,008,017 DP relaxations. Both tracers
  reached the reference end with 1 greedy failure and 0 fiberlet failures.
- Both merged replay manifests were byte-identical with SHA-256
  `72282f7fbd3a4dcdc6397b93bf66bed6ad5e518eb1d38da7bb08647007eb2106`.
  The failure counts intentionally differ from the source branch's Euclidean
  replay because this merge retains the newer anisotropic threshold evaluator.
