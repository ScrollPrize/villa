# Task Log: Full orientation constraints in fiber BP

## Findings

- The solver already accumulates `sameCost += 1-p` and
  `differentCost += p`; therefore decisiveness is the energy gap
  `abs(1-2p)` per unmerged measurement and requires no new weight.
- The current graph validator and CLI explicitly require perpendicular-only
  evidence.
- Current hard mismatch and neighbor-support diagnostics assume every factor
  prefers different labels, so merely relaxing validation would produce wrong
  reports for parallel factors.
- Independent review found that raw common factor costs are irrelevant to
  binary BP but currently bias ternary BP toward zero-energy Mixed. The solver
  must normalize merged oriented costs to their minimum so only decisiveness
  remains. Exact canceled factors then carry no information and must not dilute
  diagnostics or connect components.
- Ternary soft diagnostics must use explicit V/Mixed/H marginals rather than
  treating Mixed as half V and half H through the orientation projection.

## Plan Review

- Independent review conditionally approved full-relation BP after requiring
  explicit ternary factor normalization, neutral-factor handling, and
  ternary-native consistency diagnostics. The plan now includes these fixes.

## Deviations

- None.

## Validation

- Built `vc_fiber_trace_chunk` and `test_fiberlet_crop_trace` from the existing
  `volume-cartographer/build` configuration with `-j32`.
- `volume-cartographer/build/bin/test_fiberlet_crop_trace`: 68 test cases
  passed.
- The 1024 full-constraint run used 7,174 effective factors, converged in
  0.345 seconds, and produced no neutral factors on this crop. At temperature
  2.5 and Mixed unary cost 1.0, the direction-confusion counts were:
  `dir1 -> V/Mixed/H/tie = 1/54/153/1`,
  `dir2 -> V/Mixed/H/tie = 115/42/0/0`, and
  `mixed -> V/Mixed/H/tie = 24/95/15/0`.
- A matching `--perpendicular-only` smoke run still converged and selected the
  expected smaller 4,941-factor graph.
- Rewrote the established 1024 visualization basename at
  `$VES/data/workdir3/fiber-crop-1024/fibers` with that full-constraint result.
- `git diff --check` passed.
