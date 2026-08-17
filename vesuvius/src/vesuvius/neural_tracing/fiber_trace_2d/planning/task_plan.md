# Plan: accelerate anchor local refinement

## Baseline And Invariants

1. Use the version-2 three-run canonical baseline: wall time
   24.83/24.92/25.15 seconds and CPU time 712.79/717.37/724.51 seconds
   min/median/max. Representative local-refinement worker time is 427.26
   seconds. Confirm close variants against a same-session control.
2. Preserve seed axes, assignment tie-breaking, Gaussian and axial support,
   principal-axis and centroid accumulation order, backtracking, convergence,
   precision, thresholds, iteration limits, and serialization.
3. Reject any variant that changes the complete output inventory hash
   `48eac30b92ce088aace367b19a03e8bbf82d6de4ac343ecb074d1efba4aebfb8`
   or does not produce a repeatable speedup.

## Optimization

1. Measure the proposed component-scan fusion and normalized-direction cache
   separately. Record and remove either one if dynamic dispatch, working-set
   growth, or cache traffic outweighs reduced arithmetic.
2. Add a conservative broad-phase test before repeated refined-state kernel
   evaluation. Derive it from each actual component axis `u` and position `p`:
   a contributing observation is at most `axial_half_width / |u|` from the
   pivot along the normalized axis and at most
   `Gaussian_cutoff + |p - pivot|` transversely. Take the maximum resulting
   pivot-centered squared radius over active components. This directly covers
   initial, clamped/backtracked, and post-peak positions, including small axis
   normalization error and the peak-domain position tolerance.
3. Expand each sphere by a scale-relative numerical margin and one outward ULP.
   For observations outside it, retain the original zero additions to each
   compensated denominator while skipping component-specific axial,
   transverse, and exponential work. Evaluate every observation inside the
   conservative sphere through the original code path.
4. Keep profile version 2 and logical visit counters unchanged so measurements
   remain comparable. Do not add clocks or atomics inside hot loops.

## Tests And Measurement

1. Cover invalid, zero, NaN/Inf, non-unit, below-floor, near-normalization-
   threshold, and far halo observations. Compare exact fitted values and
   iteration/profile counters with an equivalent baseline fixture.
2. Build `vc_fiberlets`, `test_fiber_anchors`, `test_fiberlet_paths`, and
   `test_fiber_replay` in the existing RelWithDebInfo tree; run focused CTest.
   Repeat focused compilation/tests in the Clang quick-build tree and run
   `git diff --check`.
3. Run three canonical replays with fresh output directories, identical input,
   warmed caches, build type, and thread count. Report wall and CPU
   min/median/max plus local-refinement worker time.
4. Hash every relative output path and file payload for each run. Any mismatch
   from the baseline hash blocks acceptance.

Canonical command, with a fresh output directory per run:

```bash
/usr/bin/time -f 'PERF_TIME wall_s=%e user_s=%U sys_s=%S max_rss_kib=%M' \
  volume-cartographer/build/fiberlet-perf/bin/vc_fiberlets \
  fiberlet-replay \
  /home/hendrik/business/aiconsulting/vesuviuschallenge/data/s1/PHercParis4.volpkg/volumes/fiber_s1_002.lasagna.json \
  /home/hendrik/business/aiconsulting/vesuviuschallenge/data/fibers/david/Paris4_fibers/dj_20260805T025256484_000003.json \
  /tmp/fiberlet-replay-anchor-broadphase-N \
  --normal-manifest /home/hendrik/business/aiconsulting/vesuviuschallenge/data/lasagna3d_inf/las008_s1_full/las_008.lasagna.json \
  --length 5000
```

## Spec Update

- Specify the actual-component local-refinement support sphere, its numerical
  margin, and preservation of compensated denominator zero additions.
- Clarify that version-2 visit counters are logical operation counts and remain
  stable when an exact broad phase avoids detailed kernel evaluation.

## Documentation Updates

- Document the local-refinement broad phase and exactness argument in
  `volume-cartographer/docs/fiberlets.md`.
- Record retained/rejected measurements in `task_log.md`, update `status.md`,
  and add a concise changelog entry.
