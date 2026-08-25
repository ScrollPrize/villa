# Task log: write-back memory cache for temporary Fiberlet reduction layers

## Baseline

- Committed predecessor: `9230f63d7`.
- User four-stage Release run: 4.369 s wall, 7.766 s user, 2.646 s system.
- Local four-stage Release run: 4.18 s wall, 8.16 s user, 2.73 s system,
  540,220 KiB peak RSS.
- Local detailed phases totaled approximately 0.34 s materialization, 0.26 s
  exact analysis, 0.48 s simplification, 1.28 s overlay writes, and 0.73 s
  population scans. Temporary dataset setup, metadata, reporting, and cleanup
  account for the remaining time.
- `/tmp` is tmpfs on this host while the benchmark output is compressed Btrfs.
  The implementation will nevertheless use a bounded memory-first abstraction,
  not rely on a host-specific temporary filesystem optimization.

## Invariants

- No numerical, acceptance, canonical-order, retained-ID, or serialized-byte
  change.
- Prefix and route chunks are one atomic logical owner entry.
- Authoritative input caches remain durable and unchanged.

## Independent plan review

- The store must retain serialized buffers only; decoded payloads remain owned
  and accounted by `ChunkCache`.
- Pending writer buffers remain charged until release. The store dynamically
  reduces/restores the existing shared decoded budget and uses backpressure
  when dirty plus pending memory exhausts it.
- Owner generations, pair-level states, deterministic error selection, an
  explicit `finish()`, and logical memory-plus-disk hashing are required.
- Mutable stage replacement remains separate from immutable authoritative
  publication. Runtime prefix/route visibility is atomic, while the unchanged
  two-file layout is explicitly not crash-transactional.
- Tests must cover forced spill, failures, rewrites, pair visibility, budget
  restoration, no-spill teardown, and one-versus-many-thread equivalence.

## Implementation

- Added one shared `FiberletChunkWriteBackCache` for all invocation-local
  reduction stages. It retains only canonical serialized buffers; decoded
  payloads remain owned by the ordinary `ChunkCache` LRU.
- Anchor owners are individual LRU entries. Fiberlet prefix/routes owners are
  one paired entry for replacement, pending visibility, eviction, write, and
  failure handling.
- Dirty and queued entries reduce the existing shared decoded-cache ceiling.
  Deterministic LRU pressure queues a batch to a 75% resident low-water mark,
  then applies backpressure only until actual completed writes return total
  charged memory below the hard limit.
- Reads resolve latest resident/pending bytes before spilled disk bytes. Stage
  hashes overlay those same logical bytes on metadata and spilled files without
  flushing the cache.
- `finish()` drains queued writes, restores the decoded budget, discards cleanly
  unspilled invocation-local data, and precedes temporary-tree cleanup.

## Validation

- Release build:
  `TMPDIR=volume-cartographer/build/tmp cmake --build volume-cartographer/build --target vc_fiberlets test_fiberlet_storage -j 32`
- Focused Release suite:
  `TMPDIR=volume-cartographer/build/tmp volume-cartographer/build/bin/test_fiberlet_storage`
  passed all 31 test cases.
- The Clang Debug `vc_fiberlets` and `test_fiberlet_storage` targets built, and
  the same focused suite passed all 31 cases there as well.
- New tests cover no-spill memory reads and teardown, forced spill with a
  sub-entry budget, decoded-budget restoration, prefix/route pair visibility
  and injected write failure cleanup, deterministic LRU order, and reading a
  pending entry while its asynchronous writer is deliberately blocked.
- The detailed 32-thread Paris4 run reproduced exact stage ID/payload hashes:
  `7f6182d7e61b00da/93f875a8fd522366`,
  `fa6a9290546392be/4eed9c714ad148ec`,
  `7149465030b4c810/ffce8c90f177a00b`, and
  `3179a829e68eee48/b4a201a335778b17`.
- A detailed one-thread run reproduced those same four ID/payload pairs and all
  stage/joint counts exactly.
- Final counts remained 3,368 anchors, 35,027 all Fiberlets, and 4,469 interior
  Fiberlets. The cache reported 321 resident entries, 5,556,963 peak/live
  bytes, 636 memory hits, and zero spills.
- Three warm Release runs measured 2.90, 2.89, and 2.90 s wall; 7.85, 7.94,
  and 7.82 s user; 1.81, 1.65, and 1.70 s system; and 537,360, 538,844, and
  537,928 KiB peak RSS. Wall min/median/max is 2.89/2.90/2.90 s versus the
  3.93 s preceding median, a 26.2% reduction.

## Environment note

- `/tmp` reached its per-user tmpfs quota during validation even though `df`
  reported nominal free capacity. Builds and tests therefore used the existing
  build tree as `TMPDIR`; this does not affect the implementation or results.
- Focused tests consolidate related cases rather than injecting every possible
  anchor/prefix failure separately. The pair route-write failure exercises the
  shared pair cleanup/error path; the default atomic writer remains covered by
  both ordinary dataset tests and forced successful spill.
