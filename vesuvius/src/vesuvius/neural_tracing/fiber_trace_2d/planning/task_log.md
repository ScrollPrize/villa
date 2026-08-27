# Task Log: Binary sum-product fiber belief propagation

## 2026-08-27

- Started from committed binary min-sum and consistency diagnostics at
  `fd7aa3c2e`.
- Chose a separate sum-product entry point over the same merged factor graph so
  existing min-sum numerics and the comparison cohort remain unchanged.
- The first experiment has no population-balance field. Sum-product will fail
  explicitly if combined with the current min-sum-only balance draft.
- Independent plan review fixed the directed log-ratio sign convention,
  required post-damping residuals and explicit final-iterate `message_limit`
  semantics, and separated factor temperature from min-sum's post-hoc display
  temperature. It also required exact-tree, sign, damping, low-temperature,
  nonconvergence, seed-only, and deterministic-loop tests.
- Extracted shared graph preparation and report initialization so min-sum and
  sum-product consume exactly the same merged perpendicular factor graph and
  central straight seed without duplicating either implementation.
- Implemented synchronous log-space sum-product with stable two-term
  log-sum-exp updates, common factor-potential offsets, exact hard-H seed
  messages, and distinct output artifacts. Sum-product balance fields are not
  reported as meaningful values.
- Focused build and test after implementation:
  `cmake --build volume-cartographer/build --target vc_fiber_trace_chunk test_fiberlet_crop_trace -j32`
  and `volume-cartographer/build/bin/test_fiberlet_crop_trace` passed all 61
  test cases.
- Centered-384 full-Mixed cohort used the existing 179-fiber, 1324-factor
  no-split perpendicular graph. At `T=0.25`, sum-product converged in 45
  iterations and 0.003079 s. Soft same-label AUROC was `0.646966`, effectively
  unchanged from committed min-sum `0.646835`; hard mismatch AUROC remained
  `0.633989`.
- Deterministic temperature sweep soft same-label AUROC:
  `T=0.1: 0.637838`, `0.5: 0.665353`, `1.0: 0.685317`,
  `1.5: 0.690833`, `2.0: 0.693853`, `2.5: 0.694641`,
  `3.0: 0.691883`, `4.0: 0.679669`, `8.0: 0.598503`,
  `16.0: 0.575650`, and `32.0: 0.569609`. All runs converged in 22-95
  iterations and 0.0025-0.0079 s solver time. This is a single-crop diagnostic,
  so the existing default `T=0.25` was not silently retuned.
- Sum-product at high temperature leaves most fibers unresolved under the
  fixed 0.25/0.75 hard thresholds; those hard-mismatch AUROCs therefore use a
  changing cohort and are not directly comparable across the entire sweep.
- At the best tested `T=2.5`, all 94 constrained trusted fibers lie on the
  correct side of the `P(H)=0.5` majority boundary: all 50 Direction1/V fibers
  are below it and 44 of 45 Direction2/H fibers are above it. The remaining
  Direction2 fiber is exactly `0.5` because it has degree zero; there are no
  majority H/V reversals. Under the stricter confidence bands, 45/50 V fibers
  are below `0.25`, 38/45 H fibers are above `0.75`, 11 supported fibers are
  weakly but correctly classified, and the one unsupported fiber remains
  uncertain. Median `P(H)` is `0.018161` for Direction1/V and `0.980703` for
  Direction2/H. The 84 Mixed references remain distributed across both
  orientations and the uncertainty interval rather than being forced into a
  separate state.
