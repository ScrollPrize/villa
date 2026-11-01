# Optional fiber prediction channels

`FiberPredictionField` can expose predicted widths and embeddings alongside
its existing presence/direction samples. Both are opt-in; existing constructors
and ordinary tracing behavior remain unchanged. This API supplies samples only:
it does not enable width or embedding scoring in the tracer.

```cpp
auto dataset = vc::lasagna::LasagnaDataset::open(manifestPath);
vc::fiber_tracer::FiberPredictionFieldOptions options;
options.enableWidth = true;
options.enableEmbeddings = true;
vc::fiber_tracer::FiberPredictionField field(dataset, options);

std::vector<cv::Vec3f> points{{100.f, 200.f, 300.f}};
std::vector<float> widths;
std::vector<std::vector<float>> embeddings;
field.sampleWidths(points, widths);
field.sampleEmbeddings(points, embeddings);
```

Keep the dataset alive while using its field, as for existing channel bindings.
Coordinates use the dataset's configured working coordinate system. Widths
retain the stored channel's units; they are not automatically converted to
base voxels. Both operations use the existing shared chunk cache and native
channel interpolation, with no separate loading implementation or cache.

- Width requires a float32 `width` channel on the first bound prediction
  option's spatial lattice. Corners whose presence is zero are excluded and
  remaining interpolation weights are renormalized. Missing/outside/invalid
  samples return zero. Consumers must treat nonpositive widths as invalid.
- Embeddings require 1–256 contiguous float32 `embedding_000`,
  `embedding_001`, … channels on that lattice. Signed components are
  interpolated **before** L2 normalization. Missing/outside/invalid or zero
  vectors return an empty vector. Presence does not mask embeddings.
- Auxiliary channels are shared rather than per-option. Different chunk layouts
  are permitted; spatial shape and spacing must agree.
- Calling a disabled/unavailable auxiliary operation throws instead of silently
  supplying a default. Existing custom prediction sources inherit this behavior.
- Immutable fields support concurrent sampling with separate output vectors.
  No live inference/readiness coordination is introduced here; use completed
  prediction datasets.

## Tests

With a configured native build:

```sh
cmake --build build --target test_fiber_prediction_channels -j 8
ctest --test-dir build -R '^test_fiber_prediction_channels$' --output-on-failure
```

Tests cover opt-in behavior, invalid channels, interpolation order,
presence-masked width interpolation, out-of-volume handling and concurrency.
To compare existing and options-based constructors on real anchor positions:

```sh
export VC_TEST_CHANNEL_PREDICTION_MANIFEST=/path/to/fibers.lasagna.json
export VC_TEST_METRIC_ANCHORS=/path/to/anchors.json
build/bin/test_fiber_prediction_channels
```

The real check reads existing data only. It verifies ordinary predictions stay
identical; auxiliary interpolation correctness uses controlled float32 fixtures.
