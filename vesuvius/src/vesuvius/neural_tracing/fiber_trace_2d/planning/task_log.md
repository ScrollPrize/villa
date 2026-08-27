# Task Log: Interleaved-lattice winding inference

## Findings

- The checkpoint implementation places every piece on one integer winding
  lattice. It tolerates a direct `0.8` target by preferring integer difference
  one, but it cannot represent complementary fractional A-to-B and B-to-A
  offsets whose sum is one.
- Physical H/V naming is a global symmetry. The existing crop-central H seed is
  suitable as an arbitrary class-A gauge.
- Phase and measurement scale are only two global continuous parameters, so
  deterministic multi-start alternating inference is practical.
- Independent review required explicit per-component class gauges, normalized
  Mixed potentials, rank-deficient calibration fallback, an L1 acceptance rule,
  decoded-energy naming, nonconvergence behavior, and unambiguous output
  semantics. The plan now defines each contract.
- The first real-crop implementation compared latent residuals as
  `delta-scale*d`. On the 384 crop it selected the minimum scale `0.5` and a
  near-zero phase because shrinking scale also shrank the represented noise.
  The authoritative residual is now `delta/scale-d`; calibration fits
  `g=1/scale` and `h=g*phase` over the exact bounded wedge.
- Letting winding and orientation factors jointly re-solve from uniform node
  unaries classified 185/192 real-crop pieces as Mixed. The practical
  two-stage formulation now uses the established Mixed-state orientation
  posterior as a soft node prior and applies only winding factors in stage two.
  This pays orientation evidence and the Mixed unary once while still allowing
  winding evidence to refine uncertain classes.
- The interleaved solve has no honest fixed work denominator: each of four
  starts can terminate calibration early, every calibration can expand integer
  support and cold-restart BP, and iteration cost changes with state count.
  Progress therefore reports exact nested counters and elapsed time rather than
  a fabricated percentage or pre-initialization ETA.
- Follow-up user direction requested an ETA before the first initialization
  completes. After calibration one, the CLI now extrapolates mean calibration
  duration over all maximum remaining slots and labels it
  `eta_basis=calibration_max`. Once initialization one completes, it switches
  to mean initialization duration and `eta_basis=initialization`. Both subtract
  time spent in the active unit, can move with later support expansion, and are
  not described as conservative bounds.
- Quality filtering must preserve stored ordinals independently of the compact
  retained vector. The implementation composes this retained-ordinal map into
  direction/BP diagnostics; the 10% real-crop smoke run reported seed original
  trace 22 despite only 19 retained inputs, confirming it does not silently
  relabel the filtered vector.

## Deviations

- The reviewed plan initially described repeating normalized same/different
  orientation energy inside the joint winding factors. Real-crop validation
  showed that this duplicates the already solved orientation model and lets
  winding inconsistency destroy useful H/V/Mixed marginals. The implementation,
  specs, and docs now explicitly use the proposed practical two-stage variant.
- The test plan's exhaustive internal-potential matrix was reduced to public
  solver regressions for calibrated interleaving, no-signed-evidence rank
  fallback, malformed soft priors, serial/parallel identity, and the existing
  independent solver behavior. Exact internal Mixed-potential symmetry and
  forced resource/nonconvergence cases are not separately exposed as public
  tests; the real-crop run exercises Mixed states, adaptive support, multiple
  starts, component gauge, and finite convergence together.

## Validation

- Release build:

  ```bash
  cmake --build volume-cartographer/build \
    --target test_fiber_trace_winding_bp test_fiberlet_crop_trace \
             vc_fiber_trace_chunk -j32
  ```

- Focused regression suites:

  ```bash
  volume-cartographer/build/bin/test_fiber_trace_winding_bp
  volume-cartographer/build/bin/test_fiberlet_crop_trace
  ```

  Results: 9/9 winding cases and 74/74 crop-trace/BP cases passed. The winding
  suite recovers `A_0 -> B_0 -> A_1` from raw offsets `0.32 + 0.48`, with phase
  approximately `0.4` and measurement scale approximately `1.25`; it also
  covers rank-deficient fallback, malformed priors, and exact serial/parallel
  marginal identity.

- Real 384-base crop:

  ```bash
  volume-cartographer/build/bin/vc_fiber_trace_chunk direction-ablation \
    /home/hendrik/vesuvius/crop_traces_central_384.zarr \
    --normal-manifest /home/hendrik/business/aiconsulting/vesuviuschallenge/data/lasagna3d_inf/las008_s1_full/las_008.lasagna.json \
    --output /tmp/interleaved-winding-384-two-stage/fibers \
    --direction-dominance 0.9 --piece-length 512 --bp-only \
    --bp-inference sum-product-mixed --bp-temperature 2.5 --bp-mixed-cost 1
  ```

  Results: 192 pieces, 3,259 factors, one component, 1,857 final candidate
  states, 1,251 accumulated message iterations, convergence residual below
  printed precision, and 9.653 s joint discrete/calibration time. The selected
  fit was phase `0.5`, measurement scale `0.517399`, and five consecutive
  winding layers (`0..4`). Refined class argmax counts were H=76, V=86,
  Mixed=30. These values are diagnostics, not ground-truth accuracy claims.

- `git diff --check` passed.

## Progress follow-up validation

- Added an optional synchronous callback covering preparation, message
  passing, calibration updates, initialization completion, and terminal
  completion. CLI message rows are throttled to approximately one per second;
  all phase transitions are forced and flushed immediately.
- The callback regression observes exactly four initialization-complete events
  and one terminal event, validates nested counters, and proves that callback
  enablement leaves discrete labels, marginals, phase, and scale bit-identical.
- Rebuilt `vc_fiber_trace_chunk`, `test_fiber_trace_winding_bp`, and
  `test_fiberlet_crop_trace` with `-j32`. Results remain 9/9 and 74/74 cases.
- The real 384-base crop completed with 4,491 total message iterations across
  all starts. The selected result is unchanged: phase `0.5`, scale `0.517399`,
  decoded energy `1142.920133`, and H/V/Mixed counts `76/86/30`.

## Quality-filter follow-up validation

- Extracted the canonical cost-density ranking shared by quality deciles and
  `selectFiberletCropQuality`. Tests cover `ceil` selection, full-fraction
  identity, equal-density ordinal ties, restored order, empty input, and
  invalid fractions.
- Rebuilt with `-j32`; `test_fiberlet_crop_trace` remains 74/74 and
  `test_fiber_trace_winding_bp` remains 9/9.
- A real 384-base smoke run with `--quality-fraction 0.1` retained 19/182
  traces (`0.104396` effective fraction), used cutoff density `0.189188`, and
  completed direction classification, constraint extraction, and BP output.
