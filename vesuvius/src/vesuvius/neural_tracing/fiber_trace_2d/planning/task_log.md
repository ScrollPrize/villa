# Task Log: BP-aligned Lasagna normals

## Findings

- The regular Lasagna normal source is the manifest-backed `grad_mag/nx/ny`
  dataset sampled by `LasagnaNormalSampler`; it is unrelated to the legacy
  `NormalGridVolume` and `vc_ngrids --align-normals` path.
- Current connector winding scoring is deliberately sign invariant through
  `abs(connector_direction dot normal)`, so the persisted normal axes remain
  unaligned.
- Current binary fiber sum-product BP has the needed pairwise normalized-log
  message update, but it is private to `FiberTraceBeliefPropagation.cpp`.
- Absolute normal sign is unobservable from pairwise axis evidence. Every
  disconnected graph component therefore needs an explicit deterministic
  gauge; this does not claim physical outward/inward orientation.

## Deviations

- The planned additional synthetic regular-manifest integration fixture was
  not duplicated here. Existing `test_lasagna_normal_sampler` coverage already
  exercises regular-manifest opening, coordinate scaling, channel validation,
  batch sampling, missing chunks, and zero `grad_mag`. The new focused suite
  covers the alignment lattice, stable holes, exact factors, BP gauges, and OBJ
  structure, and the standalone path was instead exercised end-to-end against
  the real `las_008.lasagna.json` manifest.
- The requested 0.5--1.0 second default BP target was not fully reached without
  changing inference tuning. Exact default-damping BP reached a 1.465-second
  median. Existing explicit `--damping 1` reached 1.019 seconds with identical
  decisions on the benchmark graph, but it remains opt-in because undamped
  loopy BP is less robust generally.

## Plan Review

- Independent reviews required exact factor equations, unchanged fiber-only
  gauge semantics, regular-manifest validation, explicit base-coordinate bbox
  and lattice rules, stable invalid-sample compaction, deterministic ties and
  nonconvergence, and stronger integration tests. The plan now includes them.
- The reviews identified cross-sheet spatial coupling and absent `grad_mag`
  confidence as limitations of a standalone regular lattice graph. They are
  documented rather than silently inherited by the future H/V integration,
  which will supply its own graph to the shared API.
- Parallel-BP review confirmed deterministic per-node CSR accumulation is
  sound and required explicit phase barriers, an appended/default-serial
  worker API, effective-worker reporting, GCC coverage above the serial
  threshold, Clang fallback coverage, and repeated split-phase measurements.
  The follow-up plan now includes each requirement.

## Validation

- Release build:

  ```bash
  cmake --build volume-cartographer/build \
    --target vc_lasagna_normal_align vc_fiber_trace_chunk -j 8
  ```

- Focused Clang tests:

  ```bash
  volume-cartographer/build/dev-quickbuild-clang/bin/test_lasagna_normal_alignment
  volume-cartographer/build/dev-quickbuild-clang/bin/test_fiberlet_crop_trace
  ```

  Results: 6/6 normal-alignment cases and 72/72 existing fiber BP cases passed.

- Real regular-Lasagna smoke:

  ```bash
  volume-cartographer/build/bin/vc_lasagna_normal_align \
    /home/hendrik/business/aiconsulting/vesuviuschallenge/data/lasagna3d_inf/las008_s1_full/las_008.lasagna.json \
    --bbox 23456 17840 54928 23584 17968 55056 \
    --output volume-cartographer/build/normal_alignment_smoke --threads 8
  ```

  The command sampled 512/512 valid regular Lasagna normals, built 5,068
  factors in one component, flipped 325 samples, reduced negative neighbor
  links from 1,446 to zero, and increased mean signed neighbor dot from
  0.429402 to 0.998221. BP converged in 38 iterations at residual
  `5.89696e-09`; both OBJ files contain 512 complete glyphs.

### Parallel BP validation

- Baseline/parallel command (only `--threads` differs):

  ```bash
  volume-cartographer/build/bin/vc_lasagna_normal_align \
    /home/hendrik/business/aiconsulting/vesuviuschallenge/data/lasagna3d_inf/las008_s1_full/las_008.lasagna.json \
    --bbox 23456 17840 54928 24480 18864 55952 \
    --output volume-cartographer/build/normal_alignment_benchmark \
    --threads 1
  ```

  Environment: GCC 15.3.0 Release, Ryzen 9 5950X, 262,144 valid samples,
  3,298,428 factors, 98 iterations. Five measured runs per configuration:

  | BP workers | min BP ms | median BP ms | max BP ms | speedup |
  |-----------:|----------:|-------------:|----------:|--------:|
  | 1          | 12012.2   | 12074.2      | 12188.8   | 1.00x   |
  | 16         | 1454.68   | 1465.40      | 1485.29   | 8.24x   |

  All runs retained 233,511 flips, zero negative final links, residual
  `7.21667e-09`, and identical posterior decisions. At 16 workers, median
  phase cost was approximately 53 ms setup, 115 ms node totals, and 1,296 ms
  message updates. An explicit `--damping 1` check converged in 67 iterations
  at 1,019.49 ms total BP with the same final flip/link diagnostics.

- Exact serial/parallel report parity passed in both configurations:

  ```bash
  volume-cartographer/build/dev-quickbuild-clang/bin/test_lasagna_normal_alignment
  volume-cartographer/build/dev-release-gcc-tests/bin/test_lasagna_normal_alignment
  ```

  Clang reported the expected one-worker shim path; GCC exercised more than
  one OpenMP worker above the parallel threshold. Existing fiber BP remained
  green through the 72-case `test_fiberlet_crop_trace` suite.

- Discarded experiments:
  - per-factor plus per-node hybrid message storage increased update traffic
    and regressed BP from 1.46 to 1.59 seconds;
  - exact cavity/raw memoization reused 332.6 million updates but added about
    100 MiB and regressed BP to 1.93 seconds.
