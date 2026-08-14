# Task plan

## Problem

The current status and adaptive estimates sum recently completed chunk bytes
and divide by the span from the earliest request start to the latest completion.
That span includes idle gaps and can report a tiny bandwidth while many chunks
are completing. Adaptive admission uses the same flawed measurement.

## Implementation

1. Add a thread-local scoped HTTP response-body observer around Zarr chunk
   fetches. The common curl write callback reports received body bytes; metadata
   opens and unrelated HTTP work remain unobserved.
2. Extend `IChunkFetcher::fetchEncoded()` with an optional progress-aware path.
   Existing fetchers remain source-compatible and simply provide no streaming
   observations. `ZarrChunkFetcher` installs the HTTP observer only for the
   duration of one encoded chunk read, including range reads for sharded data.
3. Add measured-transfer lifecycle and byte-report APIs to
   `ChunkRequestScheduler`. Batch callback traffic before entering the
   scheduler to avoid a lock for every small curl write.
4. Maintain a five-second rolling aggregate payload-byte window. Starting the
   first streaming transfer after idle creates a fresh window; overlapping
   transfers share it; idle time never enters its denominator.
5. Use streamed aggregate bytes for adaptive epochs. Reset an epoch after idle,
   admission changes, failures, or underfilled demand. Complete an epoch only
   after both five active seconds and at least one successful completion per
   target admission slot.
6. Preserve a non-streaming fallback: average each successful request's
   `bytes / duration`, then multiply by the common admission represented by the
   sample window. Do not use earliest-to-latest wall span.
7. Route the standalone Zarr download benchmark through the same streaming
   observer and scheduler lifecycle as VC3D.

## Tests

- Verify the HTTP write callback reports incremental body bytes.
- Verify overlapping streamed transfers produce aggregate bandwidth.
- Verify idle time between bursts is excluded.
- Verify the fallback computes mean individual rate times admission.
- Verify adaptive epochs require both five active seconds and the admission
  sample count, and that underfilled work resets the epoch.
- Run `vc_test_core`, focused chunk-cache/Zarr/HTTP tests, the complete
  `vc-core` shard, the download benchmark unit/smoke coverage, and build VC3D.

## Spec update

Update `planning/spec.md` to replace completion-span bandwidth with streamed
HTTP payload measurement, document the five-second/admission epoch gate, and
retain the non-streaming fallback.

## Documentation updates

Update `docs/remote_file_cache.md` and scheduler API comments with measurement
scope, active-window behavior, and payload-byte semantics.

## Changelog

Record that VC3D status and adaptive admission now measure aggregate streamed
HTTP payload bandwidth without idle-gap dilution.

## Independent review

- The shared curl callback is the lowest reusable transport boundary and avoids
  duplicating Zarr reads.
- A scoped observer prevents metadata and unrelated HTTP requests from
  contaminating the chunk metric.
- Streaming bytes naturally account for actual concurrency; no parallelism
  multiplier is applied to that path.
- The multiplier exists only for fetchers that cannot provide byte progress.
- Render values, chunk contents, queue priority, and cache policy are unchanged.
- Byte callbacks must be batched because serializing every curl write on the
  scheduler mutex would create avoidable contention at high concurrency.
