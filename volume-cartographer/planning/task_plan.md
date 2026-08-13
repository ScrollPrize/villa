# Task plan

## Scope and invariants

- Adapt only normal interactive remote source fetching.
- Preserve probe and decode worker counts and all queue ordering.
- Keep explicit `maxConcurrentReads` configurations fixed.
- Measure encoded bytes and successful source transfers only.

## Implementation

1. Add a dynamic admission limit to `ChunkRequestScheduler`; worker threads
   remain allocated while only the allowed number can select work.
2. Add successful-transfer samples containing encoded bytes, start time, and
   completion time. Retain at most `64 * 4` samples.
3. Compute aggregate bandwidth from the latest `current_limit * 4` successful
   samples, spanning their earliest start through latest completion. Compute
   average encoded chunk size over the same samples.
4. Recalculate the admission limit with the configured 0.25-second pipeline
   window and `[2,64]` clamp. No incomplete window changes the limit.
5. Configure default remote `Volume::sharedChunkCache()` reads as adaptive;
   keep local/default and explicitly configured caches fixed.
6. Report the controller bandwidth estimate through `ChunkCache::Stats`, so
   the existing status formatter uses the same value.

## Testing

- Add deterministic scheduler tests for:
  - initial adaptive limit of two;
  - 2 MiB chunks at 100 MiB/s selecting 13 downloads;
  - low bandwidth retaining the minimum of two;
  - fixed schedulers ignoring adaptive samples;
  - active admission never exceeding the current limit.
- Build VC3D and focused cache/status tests.
- Run focused CTest cases and `git diff --check`.

## Spec update

- Document the successful-chunk bandwidth window, formula, bounds, and fixed
  caller behavior.
- State that adaptation affects admission only, not priority.

## Docs updates

- The render/fetch specification is the durable implementation document; no
  separate user guide is required.

## Changelog update

- Add a dated adaptive-download entry.

## Independent plan review

- With no exploration, a low starting limit can underestimate latent capacity;
  this is an explicitly accepted limitation.
- Failed and missing requests do not contaminate bandwidth or chunk-size
  estimates.
- A reduced limit cannot cancel running requests; it delays new admission until
  active work falls below the limit.
- No requirement is deferred.
