# Task Log: compact float-position fiberlet default

## Findings

- The accepted benchmark scenario is
  `compact_axis_cost_sqrt_u16_max256`: exact float endpoint positions, compact
  fitted directions, fixed sqrt-density `uint16` costs with ceiling 256.
- The normal cache-backed replay still selected exact-float geometry and float
  costs implicitly. Changing only the benchmark scenario would not change that
  runtime behavior.
- Evaluation caches persist float payloads; this task selects the compact
  logical/runtime profile and future storage baseline. It does not silently
  revise the separate unpublished `CompactQuantized` payload schema.
- Eager replay has no cache-backed compact projection and remains an explicit
  exact-float diagnostic path.

## Measurements

- Paris4, full 46,148-base-voxel reference interval, radius 768, 32 threads,
  beam 16, checkpoint 48, lookahead 384, exact search:
  - exact-float oracle: 2 failures, mean reference distance 5.712 base vx,
    median 3.625 base vx;
  - accepted default: 2 failures, mean reference distance 5.611 base vx,
    median 3.549 base vx;
  - accepted default versus oracle: mean 1.172 base vx, median 0.172 base vx,
    maximum 71.778 base vx. The maximum lies around a shifted restart.
- The q1/8 combined profile completed with 3 failures versus 2 for the oracle;
  mean/median line distance were 1.499/0.211 base vx.

## Validation

- Independent plan review required explicit named default/oracle profiles, one
  profile flowing through geometry and graph ranking, canonical float anchors,
  distinct compact-direction fiberlets, artifact provenance, and an explicit
  eager exact-float contract. All were incorporated.
- Built `vc_fiberlets`, `test_fiberlet_storage`, `test_fiberlet_paths`, and
  `test_fiber_replay` with `cmake --build volume-cartographer/build ... -j32`.
- `test_fiberlet_storage`: 16 test cases passed.
- `test_fiber_replay`: 12 test cases passed.
- `test_fiberlet_paths` retains 298 known pre-existing bitwise/fixture failures
  at lines 379 and 991-993, identical to the pre-task baseline. No new failure
  remains.
- Real-data smoke test: Paris4, first 5,000 base voxels, radius 64. Cold
  cache-backed replay completed in about 6 seconds; the warm repeat completed
  in about 1 second. Both completed the reference interval with zero fiberlet
  failures. The complete cache hash remained
  `24b3484c339cdf9552fe0f80c1cc873486d3fad4485da457262efd283c6aceb5`.
- The published `fiberlet_evaluation_profile` reports exact float positions,
  compact directions, 16-bit sqrt-density costs, ceiling 256, and the explicit
  `float32_cache` persistent payload limitation.

## Intentional limitation

- This task chooses and applies the compact logical/runtime default but does not
  replace the cache payload with a compact resident SoA. Current decoded cache
  payloads remain float and therefore do not yet realize the eventual resident
  memory reduction. The accepted profile is now the required baseline for that
  subsequent representation change.
