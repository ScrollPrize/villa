# Task Log: Lasagna-oriented replay failure threshold

## Findings

- Greedy replay and fiberlet route evaluation both use the same forward
  Euclidean reference projection, then independently test `match.distance > T`.
- Fiberlet reseeding separately rejects projected nodes at Euclidean distance
  above `T`; it must use the new decision or route failures and restarts would
  use inconsistent acceptance regions.
- Greedy already receives the trace-scale Lasagna normal sampler. Fiberlet graph
  replay currently receives no sampler, although its CLI caller owns a canonical
  sampler configured at the prediction-grid working scale.
- The error vector and threshold are in base voxels. Only the matched reference
  point is divided by the sampler working-to-base scale; uniform scale means the
  decoded normal direction itself needs no coordinate-vector rescaling.

## Plan review

- Independent review accepted the ellipsoid, unchanged Euclidean reference
  matching, reference-point normal sampling, and conservative fallback.
- The plan was tightened to replace ambiguous error keys, make both sampler
  scale contracts explicit, preserve graph-exhaustion last-match diagnostics,
  validate all numeric relationships strictly, define equality and `T=0`, use
  the shared decision for initial seeds, and publish one authoritative threshold
  descriptor.

## Deviations

- None.

## Implementation

- Added `FiberReplayMetric`, the single evaluator and serializer for the fixed
  4x Lasagna tangent-plane ellipsoid, conservative isotropic fallback, strict
  boundary decision, finite zero-threshold ratio, and component validation.
- Greedy replay now requires an explicit normal sampler and sampler scale, and
  evaluates every matched dense point with the shared metric while leaving its
  forward Euclidean reference matching and native trace objective unchanged.
- Fiberlet replay now receives the canonical normal sampler and scale. Route
  samples and seeds use the same metric; seed search retains only an inclusive
  Euclidean `4T` broad phase before exact normal-aware evaluation.
- Replaced the unpublished ambiguous match/failure error members and JSON keys
  with one typed measurement and explicit Euclidean, normal, tangential,
  threshold, ratio, and validity fields. Graph-exhaustion retains last-match
  diagnostics; unevaluated failures serialize nulls.
- The strict bundle writer verifies both engine thresholds, geometric Euclidean
  distance, component reconstruction, ellipsoid formula, ratios, fallback
  state, and distance-failure identity before publishing. Root, greedy, and
  fiberlet threshold descriptors share one serializer.

## Validation

- Isotropic pre-change baseline:
  `vc_fiberlets fiberlet-replay fiber_s1_002.lasagna.json
  dj_20260805T025256484_000003.json /tmp/vc_replay_isotropic.X638YJ
  --normal-manifest las_008.lasagna.json --length 2048`.
  Result: greedy failures `1`, fiberlet failures `0`; the greedy failure was at
  reference arc `6297.9910697605383` with Euclidean error
  `21.711638236047378` base voxels.
- Built with all requested jobs:
  `cmake --build volume-cartographer/build --target test_fiber_trace3d
  test_fiberlet_paths test_fiber_replay vc_fiberlets -j32`.
- `test_fiber_trace3d`: 54 test cases passed.
- `test_fiberlet_paths`: 41 test cases passed.
- `test_fiber_replay`: 9 test cases passed.
- Repeated the exact 2048-base-voxel Paris4 command with the new binary and
  output `/tmp/vc_replay_anisotropic.5O5TN0`. Result remained greedy failures
  `1`, fiberlet failures `0`. The same greedy event reports Euclidean
  `21.711638236047378`, normal `20.74104335917513`, tangential
  `6.4190618680467466`, threshold-equivalent `20.803032274918387`, and ratio
  `1.0401516137459192`, with a valid local normal.
- Inspected the strict JSON with `jq`: root/greedy/fiberlet descriptors agree;
  the distance failure exactly matches the terminal greedy match; old ambiguous
  keys are absent.
