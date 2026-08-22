# Task Log: whole-volume fiberlet preprocessing

## 2026-08-22

- Confirmed sparse eligibility is based only on canonical presence chunks.
- Confirmed Z/Y/X priority scheduling is sufficient; parallel completion need
  not be strictly ordered.
- Confirmed the intermediate anchor cache is a durable output and remains by
  default. The final output independently contains anchors and fiberlets.
- Existing `--presence-floor` defaults to `0.05`; cells with no usable owned
  observation at or above it return before seed generation and refinement.
- Existing `--minimum-support` is a separate post-fit acceptance threshold.
- Existing on-demand preprocessing already owns anchor dependency expansion,
  extraction, typed generated caches, and paired prefix/route publication; the
  new command will reuse those paths.
- Independent review identified that sparse absent chunks and a combined-root
  graph reader needed explicit contracts. Added a persisted active-chunk index,
  completion binding, inactive-as-empty reads only after full completion, and
  read-only anchor/path cache facets for one combined root.
- Verified the current fitting implementation already gates seed generation on
  usable owned-cell observations at `--presence-floor`; halo evidence cannot
  start an otherwise empty cell.
- Focused `test_fiberlet_storage` and `test_fiber_anchors` runs pass with
  combined-root graph, sparse mapping, and missing/empty/nonempty presence scan
  coverage.
- Follow-up requirement supersedes the persisted activity/completion design:
  remove `active_chunks.bin`, `dataset.complete`, and per-chunk completion
  markers. Every invocation must reconstruct expected chunks from input
  presence and inspect anchor/final payloads directly.
- Individual payload files remain fsync-and-rename atomic. A final chunk is
  complete only when anchors, prefixes, and routes all exist and validate as a
  compatible tuple. Partial tuples are resumable, not complete.
- Atomic-write helpers remove temporary files on normal failures, but a hard
  process exit can leave `.tmp.<process-tag>.<counter>` files. Resume and final
  shutdown cleanup for that exact suffix pattern is part of this follow-up.
- Independent review required the fresh presence-derived expected set to be
  configured for every stored combined reader, not only during preprocessing;
  no standalone reader may infer activity from absent output files.
- The review also required marker removal from ordinary prefix/route datasets,
  migration cleanup for legacy marker/index artifacts, validation of every
  anchor dependency, and safe shared temp cleanup under exclusive root locks.
- Unexpected final payloads are physically retained but hidden by the current
  expected set. This avoids unrelated destructive cleanup while preventing
  stale data from entering the graph. Extra intermediate anchors remain valid
  reusable cache entries.
- The focused Clang build exposed an existing aggregate-narrowing error in
  `FiberletQuantization.cpp`. Added the explicit `int`-to-`double` cast required
  by Clang; this is a compile-only portability correction with unchanged value.
- Removed all activity/completion persistence and marker-gated reads. Ordinary
  fiberlet chunks now require a valid prefix/route pair; combined chunks require
  a valid anchor/prefix/route tuple. Partial tuples remain invisible and resume
  through matching immutable payload reuse.
- `preprocess-volume` now rescans input presence every invocation, configures
  the expected set in memory, visits every required anchor dependency, and
  validates each existing or newly published final tuple exactly once.
- Added shared exact-suffix atomic-temp cleanup and exclusive directory locks.
  Both roots are cleaned before opening, after workers stop, and via a
  non-throwing scope guard during exception unwinding.
- Focused validation:
  `ctest --test-dir volume-cartographer/build/dev-quickbuild-clang
  --output-on-failure -R 'test_(fiberlet_storage|fiber_anchors)'` passed 2/2
  tests (21 storage cases and 86 anchor cases). Both the Clang quick build and
  regular `volume-cartographer/build/bin/vc_fiberlets` build succeeded.
- A bounded production `preprocess-volume` smoke run was not performed because
  the command currently has no bounded-region mode; focused synthetic storage
  and sparse-selection tests cover the changed resume semantics.
- Follow-up requirement: slow whole-volume stages need a live one-second
  progress/ETA/size line plus a persistent newline every minute. Size estimates
  use the mean payload bytes of all visited expected chunks and therefore become
  more stable as each stage advances.
- Implemented one independent progress ticker for each long-running stage so a
  blocked chunk does not suppress updates. Both generated and resumed chunks
  contribute their actual compressed payload-file sizes; projected size excludes
  Zarr metadata.
- Rebuilt `vc_fiberlets`, `test_fiberlet_storage`, and `test_fiber_anchors` in
  the quick Clang tree and reran both focused tests successfully. Rebuilt the
  regular `build/bin/vc_fiberlets` target successfully.
- The initial implementation incorrectly treated anchor and final Z/Y/X order
  as two independent phases. Replaced that global barrier with a tested dynamic
  scheduler: ready fiberlets in the current Z slab have priority, remaining
  slots generate earliest-needed anchors, and anchor lookahead cannot cause
  later-Z final output to overtake the active slab.
- Whole-volume extraction now uses single-threaded chunk kernels behind one
  `--threads`-sized pool, so ready fiberlets and anchor dependencies share a
  single extraction-worker budget. Resume-only anchor-cache repairs rank after
  dependencies that unblock incomplete final outputs.
- The quick Clang `vc_fiberlets`, `test_fiberlet_storage`, and
  `test_fiber_anchors` targets rebuilt successfully; both focused tests passed.
  The regular `build/bin/vc_fiberlets` target also rebuilt successfully.
