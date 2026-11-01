# Pinned prediction regions

`FiberPredictionField::pinRegion(lo, hi)` prepares an inclusive box in the
field's working coordinates for repeated chunk-local sampling. It uses the
existing `PinnedChunkRegion` ownership view, not a separate cache.

The field plans every prediction and enabled auxiliary channel, waits for its
publication gate once, then retains the decoded payloads. Interpolation corners
and the shared planner's float-rounding guard are included. Add any independent
normal channels before sealing the region:

```cpp
auto region = field.pinRegion(lo, hi);
normals.pinRegion(*region, lo, hi);
region->seal();

field.sampleWidths(points, widths, *region);
field.sampleEmbeddings(points, embeddings, *region);
field.visitCornerBatchWithNormals(normals, points, threads, context, visitor,
                                 lookaheadDepth, profile, region.get());
```

Only call auxiliary methods for channels enabled on the field. Keep the field,
normal sampler and region alive until all synchronous users finish. Seal once
after populating all channels, then share read-only across workers. Do not change
the publication gate while using the region. A region prepared by another field
is rejected, even if its manifest is identical.

Explicit pinned sampling skips repeated readiness planning and uses the shared
pin-aware channel readers. There is no thread-local state or cache fallback:
an unsealed region, an unregistered normal channel or a missing retained chunk
fails. Sampling can use any retained chunk, including the interpolation guard;
the box is a planning bound, not a geometric clipping boundary. Out-of-volume
points retain the ordinary sampler's invalid-value semantics.

Ordinary sampling overloads still use the cache and readiness gate. This change
does not automatically pin entire traces or alter their numerical scoring.
Chunk-tracing callers must bound their working set and pass the region explicitly.
Release it when jobs finish; retained payloads remain alive despite eviction of
the cache's own references, so callers must budget pinned memory themselves.

## Validation

With an existing native build:

```sh
cmake --build build --target test_fiber_prediction_channels -j 8
ctest --test-dir build --output-on-failure -R '^test_fiber_prediction_channels$'
```

Tests compare pinned and ordinary fused corner values, check concurrent width
and embedding reads, publication failure, sealing/ownership errors, invalid
bounds and out-of-region failure without a second publication request.

Set `VC_TEST_WIDTH_MANIFEST` and `VC_TEST_WIDTH_FIBER` for the cache-only real
case. On 32 line points of a reviewed Paris4 fiber, pinned widths exactly matched
ordinary width sampling. This uses existing cached predictions and performs no
inference. No performance improvement is claimed; macOS/arm64 have not been
tested locally.
