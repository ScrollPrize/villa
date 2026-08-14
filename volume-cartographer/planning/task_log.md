# Task log

## Findings

- The status estimator and adaptive epoch both divide aggregate completed bytes
  by `latest completion - earliest request start`.
- Unsaturated samples are skipped without necessarily ending the old window,
  so later activity can include a long idle gap in that denominator.
- The common curl write callback already receives exact response-body byte
  increments and is shared by ordinary and range GETs.
- `ZarrChunkFetcher` can scope an observer around one encoded read, avoiding
  metadata traffic and preserving the shared Zarr implementation.

## Implementation

- Added a thread-local `HttpClient::ScopedDownloadObserver`; the common curl
  body callback reports response bytes and restores nested observers safely.
- Added progress-aware encoded chunk fetching. HTTP Zarr reads report bytes;
  existing local and custom fetchers remain unchanged.
- Added transfer handles to the shared scheduler. Concurrent body bytes are
  measured on an active-time axis with a five-second rolling window. The first
  callback is timestamped immediately and later callbacks are batched to at
  most one scheduler update per 256 KiB or 100 ms.
- Adaptive epochs use the same aggregate bytes, require five active seconds and
  one completion per admitted worker, and reset on failure or underfilled work.
- Non-streaming fetchers use mean individual request rate multiplied by the
  represented admission; they no longer divide by a multi-request wall span.
- Routed both `ChunkCache` and `vc_zarr_download_bench` through the shared
  streamed transfer lifecycle.

## Deviations

- None.

## Validation

- `cmake --build volume-cartographer/build --target test_chunk_cache test_http_fetch_errors vc_zarr_download_bench -j2` - passed.
- `volume-cartographer/build/bin/test_chunk_cache` - 74 test cases passed.
- `volume-cartographer/build/bin/test_http_fetch_errors` - 6 test cases passed.
- `cmake --build volume-cartographer/build --target VC3D -j2` - passed.
- `cmake --build volume-cartographer/build -j2` - passed.
- `ctest --test-dir volume-cartographer/build --output-on-failure -L vc-core -j2` - 132/132 tests passed.
