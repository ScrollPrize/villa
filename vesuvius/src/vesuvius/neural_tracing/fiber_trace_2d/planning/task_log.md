# Task log: show live cached replay preprocessing progress

## Findings

- The concise bar is rendered only from tracer/post-trace callbacks. During a
  long cache dependency, no callback reaches `ReplayOverallProgress`, so its
  elapsed text freezes along with percentage and ETA.
- The existing fiberlet-generation callback does not cover persisted chunk
  reads. Counting it would leave warm-cache preprocessing invisible.
- The deterministic replay schedule already provides the complete expected
  prefix-key set and anchor dependency-key set. Cache fetch resolution is the
  correct common event for generated and persisted chunks.

## Independent review

- Added explicit callback exception isolation and shared-state lifetime,
  synchronized expected-key installation/deduplication, an exact estimator
  formula, timer shutdown ordering, and live-ETA semantics.
- Scheduled prefetch keys define the preprocessing estimate. Data-dependent
  reach-neighborhood prefix and committed-route reads remain in the reserved
  tracer term because they cannot be known exactly before traversal and are
  small relative to measured extraction work.
- Focused automated coverage will exercise generated and persisted resolution
  plus observer isolation. Concurrent display shutdown and timer behavior will
  be validated through the real CLI because the progress renderer is private
  to the executable; exhaustive synthetic interleaving tests are intentionally
  not introduced for this narrow UI fix.

## Implementation

- Added an optional terminal-resolution observer to generated fiberlet caches.
  It runs for both generated and persisted chunks after final status is fixed;
  observer exceptions are contained and cannot change the fetch result.
- Forwarded resolution through on-demand preprocessing into a shared,
  mutex-protected expected/resolved-key state. Worker callbacks capture only
  that state, deduplicate reloads, ignore failures and unscheduled keys, and can
  be disabled safely during progress teardown.
- Added the documented 1:16 anchor/prefix weighted cache estimate and 95:5
  preprocessing/tracer combination. A 250-millisecond ticker snapshots it and
  repaints independently of worker callbacks. Shutdown signals and joins the
  ticker before closing the line.

## Verification

- Built `vc_fiberlets`, `test_fiberlet_storage`, and `test_fiber_replay` using
  `cmake --build volume-cartographer/build -j32` (Release build).
- `test_fiberlet_storage`: 11 cases passed. Its sparse-dataset test now proves
  exactly one resolution for generation, exactly one for persisted reuse, no
  second generator invocation, and successful data despite a throwing
  resolution observer.
- `test_fiber_replay`: 12 cases passed.
- A fresh-cache full-fiber radius-768 run bounded at nine seconds advanced from
  `0.02%` at three seconds to `0.05%` at nine seconds, updated elapsed time every
  second, and displayed a finite live ETA. A separate three-second `--stats`
  run contained no concise progress line.
- A complete 5,000-base-voxel radius-64 replay finished in 5.77 seconds, wrote
  one final newline, and retained SHA-256
  `9781e00ae129b5fef098246c163ba1f737eca3b8a3fcceba6c90e45087b10a91`.

## Limitation

- External `SIGINT` terminates the process without C++ stack unwinding, so it
  cannot guarantee a final newline from the progress destructor. Normal
  completion and handled C++ errors use the idempotent ticker shutdown path.
