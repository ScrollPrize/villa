# Task log: normal-alignment progress output

## Findings

- The CLI prints its normal-alignment summary only after the complete
  `sampleAndAlignLasagnaNormalLattice` call returns.
- That call performs an opaque batch normal-volume read, deterministic lattice
  factor construction, connected-component discovery, and up to 500 binary-BP
  message iterations.
- Binary BP currently exposes only its final report even though each iteration
  already has a serial completion boundary with iteration count and residual.
- Independent review required safe OpenMP callback exception propagation, a
  real parallel-path regression, phase-local rather than global ETA semantics,
  exact progress units, bounded callback frequency, exclusion of callback time
  from BP timing fields, and success-only terminal completion.

## Deviations

- None.

## Validation

- Built the production CLI and focused tests:

  ```text
  cmake --build volume-cartographer/build --target vc_fiber_trace_chunk test_lasagna_normal_alignment test_fiber_trace_winding_bp test_fiberlet_crop_trace -j 16
  ```

- Focused results:

  ```text
  test_lasagna_normal_alignment: 8 test cases passed
  test_fiber_trace_winding_bp: 41 test cases passed
  test_fiberlet_crop_trace: 80 test cases passed
  ```

- The normal-alignment test exercises more than 32,768 factors so OpenMP uses
  the production parallel path. It verifies ordered serialized callbacks,
  success terminal events for early convergence and message-limit completion,
  callback exception propagation outside OpenMP, and exact probabilities,
  log odds, residual, convergence, component gauges, and aligned normals with
  and without reporting.
- Ran a CLI smoke against the established Paris4 `crop_traces.zarr` with two
  quality-selected fibers and one configured BP iteration. The automatic
  output reported all six phases in order with exact work totals, message
  residual, `eta_to_limit`, and terminal completion before the existing normal
  alignment summary. The remainder of the direction-ablation command also
  completed successfully.
- `git diff --check` passed.
