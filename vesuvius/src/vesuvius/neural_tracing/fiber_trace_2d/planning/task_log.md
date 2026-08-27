# Task Log: Non-transitive Mixed-state fiber BP

## Findings

- The current model charges Mixed per raw incident measurement inside each
  factor. It therefore makes the Mixed prior depend on graph degree and sends a
  non-neutral preference through factors touching Mixed.
- The intended model needs a node unary because uncertainty belongs to one
  fiber, while pairwise constraints only compare two oriented states.
- The current final-marginal accumulation already includes each incoming
  directed message once; the new unary must be added without changing that
  accounting.

## Deviations

- None.

## Plan Review

- Independent review approved the node-local unary formulation after requiring
  explicit conditional-neutrality, isolate, seed, degree-independence, and
  empirical comparison coverage. Those requirements are incorporated into the
  plan.

## Validation

- Built with:
  `cmake --build volume-cartographer/build --target vc_fiber_trace_chunk test_fiberlet_crop_trace -j32`.
- `volume-cartographer/build/bin/test_fiberlet_crop_trace` passes all 66 test
  cases.
- CLI validation rejects `--bp-mixed-cost` outside `sum-product-mixed` and
  rejects nonfinite input.
- `git diff --check` passes.
- Centered-384 evaluation used 179 fibers (50 Direction1, 45 Direction2, 84
  Mixed), 1324 perpendicular factors, `T=2.5`, and the full admitted cohort.
  The node-unary sweep produced:

  | Mixed unary cost | P(Mixed) AUROC | Trusted H/V reversals | Mixed argmax |
  | ---: | ---: | ---: | ---: |
  | 0.125 | 0.399060 | 0 | 81/84 |
  | 0.325 | 0.399937 | 0 | 78/84 |
  | 0.5 | 0.400564 | 0 | 76/84 |
  | 1 | 0.402068 | 0 | 68/84 |
  | 2 | 0.402068 | 0 | 51/84 |
  | 4 | 0.434524 | 0 | 17/84 |
  | 8 | 0.479511 | 0 | 3/84 |
  | 16 | 0.486404 | 0 | 0/84 |

- The earlier measurement-scaled ternary model reached 0.754073 AUROC at cost
  0.325, and the recorded binary consistency proxy reached 0.694641. The new
  formulation therefore fixes the requested semantics but does not improve
  Mixed ranking on this diagnostic. Low unary costs classify most fibers as
  Mixed; high costs recover the trusted H/V partition but suppress Mixed. The
  default remains 0.5 because no tested node-unary value is justified as a
  better general default from this crop.
