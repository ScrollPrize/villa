# Prediction publication readiness

`FiberPredictionField::setReadinessGate` connects the native sampling API to a
caller-managed live inference producer. Without a gate, offline sampling is
unchanged. This API does not launch inference, reserve GPUs or implement a
second cache.

Configure the gate before any reads from an incomplete prediction namespace:

```cpp
field.setReadinessGate([&](const std::vector<FiberPredictionChunkDependency>& chunks) {
    producer.ensurePublished(chunks); // caller's synchronous publication interface
});
```

Each dependency identifies an array path and a ZYX storage chunk. Before any
sampling reads/cache locks, the field plans interpolation dependencies for the
complete batch, sorts/deduplicates them and calls the gate once. The gate must
return only after coherent publication of every requested product, or throw
on cancellation/failure. Presence/nx/ny and all enabled width/embedding channels
participate, even when auxiliary channels use different chunk grids. Independent
normal samplers remain the caller's responsibility.

The gate is synchronous and may be invoked concurrently by separate callers;
it must be thread-safe and must not recursively sample the same field. Do not
change the gate while sampling is active. Publication must be immutable once
acknowledged. Do not first read missing chunks offline and then attach a live
producer: readiness does not invalidate existing cached fill values.

Planning uses the existing channel interpolation planner, with both double and
float-level coordinates. This covers optimized float corner sampling at chunk
boundaries where rounding can select a different corner. Exact grid equality
is required to share a dependency plan between channels. Duplicate sample points
do not result in duplicate dependencies.

Supported entrypoints include scalar sampling, double/float batches, prefetch,
width/embedding batches, fused normal-corner sampling/visitors, and indexed
stored-grid sampling. Whole-volume sparse scans reject a configured gate:
missing unpublished chunks cannot safely be interpreted as empty volume.

Exceptions propagate; the field does not remember successful or failed
acknowledgements. Repeated batches may call the gate again. Publication caching,
cancellation and scheduling belong to the producer. Empty/out-of-volume batches
with no dependencies do not invoke it. Optional `FiberTraceProfile` fields record
planning seconds, point count and dependency count, excluding the gate's wait.

## Validation

Using an existing configured native build:

```sh
cmake --build build --target test_fiber_prediction_channels -j 8
ctest --test-dir build --output-on-failure -R '^test_fiber_prediction_channels$'
```

The suite tests all entrypoints, float boundary rounding, independently chunked
auxiliary data, deduplication, failed-publication retry, publication before the
first read, concurrency and unchanged fused tracing results. Set
`VC_TEST_WIDTH_MANIFEST` and `VC_TEST_WIDTH_FIBER` to an existing float-width
prediction cache and a covered reviewed fiber JSON to enable its real-data case.
It performs no inference or downloads. Without those inputs that case is skipped.

On cached Paris4 predictions (GCC13 Release), 32 line points required eight
dependencies in one gate call. Offline serial and gated four-thread samples
were identical. This validates extraction/correctness, not producer publication
or inference performance. macOS and arm64 have not been tested locally.
