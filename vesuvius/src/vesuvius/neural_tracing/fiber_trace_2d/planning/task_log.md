# Task log: incremental fiberlet replay prefixes

## Finding

- `materializeSelected()` currently walks from the segment seed after every
  checkpoint, reloads all historical route geometry, reconstructs all points,
  and repeats reference matching plus normal-threshold evaluation. With fixed
  checkpoint spacing this makes a failure-free segment approximately quadratic
  in length and can repeatedly churn route chunks through the bounded LRU.
- Search also repeatedly materializes full logical-arc vectors for candidate
  keys and copies every retained beam's complete visited-node set. Those costs
  are absent from `fiberlet_rollout_expansions` and contain substantial serial
  work.

## Constraints

- Search decisions and numerical results must remain unchanged.
- Logical identity and cycle rejection must remain exact; hashes may accelerate
  lookup but may not define equality.
- Normal replay must perform only incremental prefix work. A single final
  linear assembly is required by the full-route output contract.

## Baseline

- Build: current `volume-cartographer/build` executable at `f93a8fa96`.
- Input: Paris4 `fiber_s1_002`, David fiber `...025256484_000003`, Lasagna
  normals `las_008`, 5,000 base voxels from the default start, radius 64, 32
  threads, hot float cache.
- Result: 7.57 s wall, 222.22 s user, 6.51 s system, 202,872 KiB maximum RSS;
  one greedy failure and zero fiberlet failures.
- The baseline replay JSON is retained at
  `/tmp/fiberlet-incremental-baseline.json` for exact post-change comparison.

## Independent review

- Use an exact fixed-key-depth Patricia trie for visited anchors; a balanced
  tree would still grow logarithmically with prefix length.
- Canonical logical identity must not merge physical states or use allocation
  order. Exact ancestor/first-divergence ordering must match vector ordering.
- Incremental materialization must retain all matcher continuation fields,
  authoritative cumulative costs, partial-edge semantics, and terminal state.
- Evaluation state follows live beam ancestry rather than an unbounded strong
  side cache. Full diagnostic payloads remain explicitly diagnostic-only.
- Validation must cover both search modes, diagnostics modes, thread counts,
  cached/eager float graphs, terminal edge cases, and exact replay artifacts.

## Implementation

- Canonical seed-local logical-route nodes replace root-to-tip logical vectors
  in ranking, equality, diversity selection, and reconvergent-label tie breaks.
  Binary lifting preserves exact vector lexicographic order; physical histories
  are never merged by logical identity.
- Exact cycle state is an immutable Patricia trie over all 200 bits of the
  signed Z/Y/X plus variant key. Checkpoint advancement shares the trie root and
  no longer copies a complete `std::set`.
- Selected-route evaluation stores matcher continuation, appended points and
  matches, per-edge committed steps, cumulative component costs, terminal
  state, and physical history identity. A segment-local history table resumes a
  later branch at its nearest evaluated ancestor. Unselected route payloads are
  not read.
- Normal checkpoints read scalar evaluator state only. Complete output vectors
  are assembled once at failure/reference end. Stable diagnostic IDs are
  assigned incrementally from newly selected history suffixes, preserving the
  former serialized ordering without rebuilding the route.

## Validation

- Built `vc_fiberlets`, `test_fiberlet_paths`, `test_fiber_replay`, and
  `test_fiberlet_storage` from `volume-cartographer/build` with `-j32`.
- `test_fiber_replay`: 12 test cases passed.
- `test_fiberlet_storage`: 14 test cases passed.
- `test_fiberlet_paths` retains exactly the known 298 pre-existing checks at
  the float bitwise fixture and Q4 fixture; the new long-route incremental
  matching regression passed and introduced no additional failure.
- Three final hot-cache runs on the baseline workload measured 3.84/3.87/3.89
  seconds wall (min/median/max), 65.61/68.03/69.35 seconds total CPU, and
  102,424/104,440/104,900 KiB peak RSS (min/median/max). Baseline was 7.57
  seconds wall, 228.73 seconds total CPU, and 202,872 KiB RSS.
- Every final run produced SHA-256
  `a15fdd46fdcc38085adaa262ec3b0038bb3e5ead677b918bc646b12dbb6e5318`,
  byte-identical to `/tmp/fiberlet-incremental-baseline.json`; failure counts
  remained one greedy and zero fiberlet.
- A sampling-profiler capture could not be produced because `perf` is not
  installed in this checkout. The removed serial hotspots were established by
  direct call-site accounting: per checkpoint, `materializeSelected()` walked
  every prior history, reloaded every route payload, rematched every point, and
  resampled every normal; logical-vector construction and visited-set
  compaction performed two additional full-prefix traversals/copies. The new
  long-chain regression bounds scalar normal sampling linearly, and the exact
  command-level CPU reduction validates removal of that repeated work.

## Deviations

- No user-visible or numerical behavior was intentionally changed. The
  implementation keeps selected evaluator contributions for the segment rather
  than eagerly attaching evaluation state to every retained beam; this avoids
  reading unselected route payloads while still resuming branch switches from
  shared evaluated ancestry.
