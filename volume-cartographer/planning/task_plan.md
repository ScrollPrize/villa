# Task plan

## Discovery

- Use `ChunkCache` as the source of truth for unresolved requests and existing
  recent remote-download throughput.
- Extend the existing shared cache status emitted by `CChunkedVolumeViewer` and
  already rendered by the application status bar.

## Implementation

1. Extend `ChunkCache::Stats` with unresolved request counts indexed by pyramid
   level. Maintain the counts under the existing cache mutex across queue,
   resolve, invalidation, and stale-view cancellation paths.
2. Add a small shared VC3D formatter for the compact `qK ...` item.
   Trim only leading/trailing zero levels and retain interior zeros.
3. Normalize the existing status fields to fixed GiB RAM/disk units and compact
   `net N@XMiB/s` formatting.
4. Append queue counts after network speed only for active remote fetches; show
   `net idle` for an idle remote volume and no network field for local volumes.

## Testing

- Extend `test_chunk_cache` with a blocked multi-level fetch fixture that checks
  per-level unresolved counts before and after resolution and cancellation.
- Add a focused Qt unit test for empty queues, trimming, interior zero
  preservation, and starting-level selection.
- Build the touched VC3D and test targets.
- Run the focused chunk-cache and label-format tests.

## Spec update

- Add the status-bar syntax and queue-count semantics.
- Record the remote-only and active-download-only display conditions.

## Docs updates

- Document the compact status-bar fields in `docs/remote_file_cache.md`.

## Changelog update

- Add one entry for per-scale queue diagnostics after tests pass.
