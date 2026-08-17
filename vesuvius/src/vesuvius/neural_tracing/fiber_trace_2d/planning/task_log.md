# Task Log: accelerate anchor local refinement

## Baseline

- The version-2 profile attributes 427.26 worker-seconds, or 86.3% of anchor
  fitting time, to `refineLocalComponents()` in the representative canonical
  replay.
- Logical local-refinement work is 9.64 billion tensor visits, 9.64 billion
  centroid visits, and 13.20 billion refined-evaluation visits.
- Three canonical baseline runs took 24.83/24.92/25.15 seconds wall and
  712.79/717.37/724.51 seconds CPU, min/median/max.
- Complete baseline output inventory hash:
  `48eac30b92ce088aace367b19a03e8bbf82d6de4ac343ecb074d1efba4aebfb8`.

## Decisions

- Preserve double precision and the existing `normalized()` implementation.
- Retain a conservative broad phase derived from actual component axes and
  positions; preserve every compensated denominator addition and its order.
- Keep version-2 profile counters as logical-work metrics so before/after
  values remain directly comparable.
- Independent review required the direction cache to live in the fitter so the
  final evaluation can reuse it, and required rejected observations to pass
  state-independent eligibility checks before normalization. The plan was
  corrected accordingly.
- The independent review confirmed that observation-outer fusion preserves
  each component's compensated-addition subsequence provided both principal
  axes are finalized before the fused centroid pass.

## Deviations

- The first implementation fused tensor and centroid scans across components.
  Two exact-output canonical runs regressed wall time to 25.66 and 25.87
  seconds and local-refinement worker time to 444.60 and 460.80 seconds.
  Dynamic component dispatch and larger live accumulator sets outweighed the
  skipped assignment checks. The fusion was removed before testing normalized
  direction reuse independently.
- Normalized-direction caching alone regressed wall time to 26.46 seconds and
  local-refinement worker time to 472.04 seconds. Its second large memory stream
  cost more than repeated normalization saved, so it was also removed.
- A third variant reused one normalized direction only within each
  refined-state observation evaluation to avoid persistent cache traffic.
- Same-session control measurement was 24.70 seconds wall and 422.64 seconds
  local-refinement worker time. A lazy register-only normalization variant then
  regressed to 26.15 and 460.05 seconds, so direction reuse was removed.
- The next measured variant adds only a conservative pivot-distance broad
  phase. It preserves zero additions to compensated denominators but avoids
  component geometry for observations that cannot enter any feasible line
  kernel.
- Independent review found that a config-radius proof omitted the post-peak
  domain tolerance. The retained bound is instead derived from every actual
  component's axis norm and position offset, so it covers initial,
  backtracked, and post-peak evaluation directly.
- Non-finite squared pivot distances caused by overflow of otherwise finite
  coordinates fall back to the original kernel. Focused coverage includes this
  case and a contributor at the combined axial/transverse support boundary.

## Validation

- GCC RelWithDebInfo focused build and tests passed:

  ```bash
  cmake --build volume-cartographer/build/fiberlet-perf \
    --target vc_fiberlets test_fiber_anchors test_fiberlet_paths \
    test_fiber_replay --parallel 32
  ctest --test-dir volume-cartographer/build/fiberlet-perf \
    --output-on-failure \
    -R '^(test_fiber_anchors|test_fiberlet_paths|test_fiber_replay)$' \
    --parallel 3
  ```

- Clang quick-build focused compilation and the same three tests passed. Its
  existing doctest-compatibility `CAPTURE` comma warning remains unchanged.
- The regular `volume-cartographer/build/bin/vc_fiberlets` target was rebuilt.
- Final canonical replay wall times were 22.05/22.09/22.36 seconds
  min/median/max, versus 24.83/24.92/25.15 seconds at baseline. Median wall
  time improved 11.4%.
- Final CPU times (user plus system) were 628.82/633.96/640.07 seconds
  min/median/max, versus 712.79/717.37/724.51 seconds at baseline. Median CPU
  time improved 11.6%.
- Local-refinement worker times were 337.50/339.01/343.42 seconds
  min/median/max, versus the representative 427.26-second baseline, a 20.6%
  median reduction.
- Every final replay produced complete inventory hash
  `48eac30b92ce088aace367b19a03e8bbf82d6de4ac343ecb074d1efba4aebfb8`,
  exactly matching the baseline.
- `git diff --check` passed.
