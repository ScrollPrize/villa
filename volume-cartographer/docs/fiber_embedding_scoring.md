# Embedding-guided native fiber tracing

The native tracer can add an optional cosine-distance cost to its existing
presence, direction and smoothness costs. This helps distinguish nearby fibers
with similar predicted directions. It does not replace those costs or guarantee
that a trace stays on the same anatomical fiber.

## API

```cpp
vc::fiber_tracer::FiberPredictionFieldOptions options;
options.enableEmbeddings = true;
vc::fiber_tracer::FiberPredictionField field(dataset, options);

vc::fiber_tracer::FiberTraceConfig config;
config.embeddingWeight = 0.3; // example, not a universally tuned value
config.embeddingHistorySpacingBaseVoxels = 32;
config.embeddingHistoryWindowBaseVoxels = 256;
config.embeddingHistoryCount = 8;
config.embeddingHistoryFalloff = "uniform";
config.embeddingReferenceMode = "mean";
// Pass config and field to traceFiberOneWay / traceFiberExtrapolation / segment tracing.
// Default normal-aware smoothness still requires a normal sampler.
```

Keep `dataset` alive for the field's lifetime. See
[prediction channels](fiber_prediction_channels.md) for the float32 manifest
layout. Embedding channels are interpolated before unit normalization, using the
shared channel sampler and cache. Custom sources must return finite unit vectors
of a consistent dimension, or an empty vector for an invalid sample.

`embeddingWeight` defaults to zero: no embedding channels are read and ordinary
tracing remains unchanged. There is no new CLI or Python binding in this layer.

## Cost and history

At each candidate, the tracer adds `embeddingWeight * distance` to the existing
per-step loss. Distance is `1 - dot(candidate, reference)`, clamped through the
dot product to [0,2]; unlike geometric fiber direction, embedding sign matters.

The seed initializes the history. Each surviving beam owns its own bounded
history. A candidate is scored against its parent's history before admission;
speculative or rejected candidates cannot contaminate sibling histories.
References are admitted only after the configured arclength spacing, expired
after the window, and capped at `embeddingHistoryCount` (1–64). Distances are
base voxels, using accumulated path length times `traceToBaseScale`, not steps.

`mean` averages distance over retained references, with weights:

- `uniform`: 1;
- `linear`: max(0, 1 - age/window);
- `gaussian`: exp(-2 * (age/window)^2).

`best` takes the smallest distance and requires uniform falloff. If no reference
has positive weight inside the window, the embedding cost is zero. These
formulas and the inclusive window boundary preserve the existing tracer's
bounded-history implementation; the extraction does not retune its parameters.

Missing candidate embeddings reject that candidate. A missing seed, malformed
batch size or inconsistent dimensions fails explicitly. Disabled scoring does
not require embedding availability. Scalar, parallel and fused-corner scoring
all apply the same cost before frontier selection. Lazy lookahead retains its
valid lower bound because this additional cost is nonnegative.

## Validation

With an already configured native build:

```bash
cmake --build build --target test_fiber_embedding_scoring test_fiber_prediction_channels test_fiber_trace3d -j 8
ctest --test-dir build --output-on-failure -R '^(test_fiber_embedding_scoring|test_fiber_prediction_channels|test_fiber_trace3d)$'
```

The tests cover signed cosine, all falloffs, history expiry/count/spacing,
sibling isolation, missing data, changed path selection, unchanged paths for
neutral embeddings, eager/lazy agreement, serial/parallel agreement and the
fused normal-corner path.

For a real, already inferred local cache (no GPU inference is launched):

```bash
export VC_TEST_EMBEDDING_MANIFEST=/path/to/inference.lasagna.json
export VC_TEST_EMBEDDING_FIBER=/path/to/annotated-fiber.json
export VC_TEST_EMBEDDING_NORMAL_MANIFEST=/path/to/normals.lasagna.json
build/bin/test_fiber_embedding_scoring
```

The optional test starts at the first annotated control point, not an
extrapolated tail. It traces 64 base voxels, compares serial/parallel paths and
independently recomputes the cosine costs. Without these variables only the
fixture cases run. This is a numerical/integration check, not a new accuracy
benchmark or a speedup claim. It requires cached coverage of the candidate cone.

Validated with GCC 13 Release on the cached paired-embedding EMA999 prediction
(`7ee3f7c19d12188a297b87c7ab02dde0de01792dcea97e4c691384240cc17a15`)
for Paris4, fiber `anon_20260815T020414697_000008.json`, starting at its first
control point. The trace returned five points over 64.0013 base voxels with
summed unweighted cosine cost 0.155241. Scalar/parallel coordinates were exact,
both with and without normal-aware smoothness; the cost matched an independent
dot-product calculation. An equal-grid fixture separately exercises fused
normal-corner scoring. No model inference, cache writes or training were run.

Width-based committed-path scoring, reverse-consistency/replay integration,
EMA/anchor policies and experiment runners are outside this extraction.
