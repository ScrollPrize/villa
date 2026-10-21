# Width-aware native fiber tracing

Width scoring is opt-in on `FiberTraceConfig`. Ordinary tracing defaults are
unchanged. For cached inference, construct `FiberPredictionField` with
`FiberPredictionFieldOptions::enableWidth = true`; custom prediction sources
can implement `sampleWidths`. Width interpolation uses the existing shared
channel sampler, including its zero-presence corner validity handling.

## Scoring

- `widthRelativeTolerance = 0` disables width reads, gates and accumulation.
- A positive tolerance gates candidate presence by relative width agreement.
- `widthCosineWeight = false` uses `max(0, 1 - d)`.
- `widthCosineWeight = true` uses `cos(pi*d/2)` for `d < 1`, otherwise zero.

Here `d = abs(candidateWidth / referenceWidth - 1) / tolerance`.
For example, tolerance 0.4 gives zero evidence at 40% deviation; at 20%
deviation the linear gate is 0.5 and the cosine gate is approximately 0.7071.
The gate multiplies presence **before** the existing direction agreement and
smoothness costs are combined. It is not an extra additive loss. A zero gate
does not itself reject a candidate or add a stopping threshold. Embedding
costs, if enabled, are still added separately.

Width values retain the units supplied by the prediction source. Fixed and
initial references must use those same units; relative agreement is unitless.
Nonpositive/nonfinite candidate widths give zero evidence. Invalid committed
widths or a malformed batch fail explicitly rather than corrupting the mean.

## Committed reference

By default (`widthReferenceMean = 0`) the reference starts at the sampled seed
width. A positive `initialWidthMean` substitutes one initial observation for
the seed sample. Thereafter only irrevocable points shared by **all surviving
beams** update the arithmetic mean. Speculative paths may have their widths
sampled for scoring, but never contribute observations. A reference therefore
stays unchanged while surviving paths still diverge.

On return, the selected path's remaining points are counted once, in order,
including intermediate lookahead points. This also applies to target-plane
completion and exhausted-frontier fallback. `widthMean` and `widthSamples` in
`FiberTraceOneWayResult` describe these committed native path nodes, before
optional final crossing/length clipping of the returned polyline.

A positive `widthReferenceMean` instead freezes the reference for the entire
one-way trace. The result reports that mean and zero accumulated samples.
This supports callers that want to reuse a forward reference for a reverse
trace; automatic reverse-consistency behavior is not added here.

## Validation

From the checkout root, using an already configured native build:

```sh
cmake --build volume-cartographer/build/pr-native --target test_fiber_width_scoring test_fiber_prediction_channels test_fiber_embedding_scoring test_fiber_trace3d -j 8
ctest --test-dir volume-cartographer/build/pr-native --output-on-failure -R '^(test_fiber_width_scoring|test_fiber_prediction_channels|test_fiber_embedding_scoring|test_fiber_trace3d)$'
```

Tests cover gate formulas, invalid inputs, disabled behavior, fixed/initial
references, committed-only accumulation, lookahead depths 1–3, early exits,
fork selection, eager/lazy equivalence and serial/parallel scoring. The shared
prediction-channel fixture exercises width and embedding scoring together on
the fused normal-corner path.

An optional cache-only real-data test needs a width-enabled manifest and a
fiber whose first annotated control point is covered by that cache:

```sh
export VC_TEST_WIDTH_MANIFEST=/path/to/inference.lasagna.json
export VC_TEST_WIDTH_FIBER=/path/to/reviewed-fiber.json
volume-cartographer/build/pr-native/bin/test_fiber_width_scoring
```

Verified on GCC 13 Release with existing Paris4 predictions from checkpoint
`a56e9dc2116af3db535d0c6283b009833bd788eb77d1056ab175c11edc46db10`,
reviewed piece `0039_anon_20260815T025206849_000041_run0.json`, starting at
approximately `(19537.9, 18908.4, 36800.4)` base voxels. For a 64-base-voxel
trace with 16-voxel steps, both baseline and width-gated traces returned five
points. With cosine tolerance 0.4, the committed mean was 22.0988 in stored
width units. Independent resampling reproduced the mean, and serial/four-thread
paths and means agreed exactly. This is integration evidence, not an accuracy
or speedup claim. No GPU inference was run; macOS/arm64 were not tested locally.
