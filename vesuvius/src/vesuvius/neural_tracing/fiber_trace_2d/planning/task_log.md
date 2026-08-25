# Task log: compact Fiberlet crop lookahead state

## Baseline

- Build: GCC Release `volume-cartographer/build`, confirmed `-O3 -DNDEBUG`.
- Dataset: Paris4 combined `fiberlets.zarr`, crop base XYZ
  `[10240,22016,6144)` to `[11264,23040,7168)`, 500 attempts.
- Current result: 500 accepted, 27,715 covered anchors, 724 computed
  candidates, 224 discarded candidates.
- Current timing: 79.66 s wall, 1,381.65 s user, 33.44 s system; graph
  preparation 11.64 s wall / 62.83 s CPU and tracing 67.39 s wall /
  1,351.61 s CPU.

## Findings

- Every accepted lookahead branch currently copies both the complete committed
  `std::set<FiberletStorageKey>` and the complete rollout arc vector.
- The committed set grows with trace length, making branch construction depend
  on the full prefix despite cycle checks needing only a read-only prefix query
  plus the short current rollout ancestry.
- Full route sequences are semantically required only for existing
  lexicographic ranking and completion records.
- Independent review identified exact parity coverage needed for committed and
  rollout-local cycles, the generated-state cutoff, dead-end/terminal
  completion, and the `beamWidth * 64` intermediate pruning boundary.
- To bound retained history better, generated route ancestry will store only a
  compact parent link, endpoint anchor, and directed arc ID. Full incoming arcs
  remain only in active frontier states, not the historical arena.
- Performance validation must include repeated Release timing, a practical
  hotspot profile, and measured peak arena population rather than a single
  canonical run.

## Initial result and profile

- Three GCC Release runs after the route-node arena measured 69.84, 69.87, and
  71.80 seconds wall; trace time was 57.38, 57.63, and 59.51 seconds. The
  previous single canonical run was 79.66 seconds wall / 67.39 seconds trace.
- All eight OBJ artifacts from the first run are byte-identical to the previous
  implementation. Candidate, coverage, acceptance, and direction counters are
  also identical.
- A `gprofng` 100-attempt profile still attributes 45.25% exclusive CPU to
  `selectLookaheadFirstArc`, including 7.29% in reconstructing completed arc
  vectors and 3.22% in sorting completions. Since only the minimum completion
  is observed, the plan now includes index-backed completions and exact linear
  minimum selection.

## Implementation

- Replaced each branch's copied committed `std::set` and arc vector with a
  compact route-node arena. Nodes contain only the endpoint anchor, parent
  index, and directed arc ID; active frontier states retain accumulated values
  and the incoming arc.
- Cycle checks query the immutable committed-prefix set and then walk the
  current rollout ancestry, including the current node.
- Intermediate `beamWidth * 64` pruning reconstructs one route per ranked
  state and keeps the exact previous density/lexicographic comparator.
- Terminal and dead-end completions retain arena indices. A linear minimum
  scan uses the same density order and compares parent-linked routes
  lexicographically only on exact density ties. Sorting, truncating, and
  returning the first completion had no other observable effect.
- Added CLI high-water diagnostics for route-arena nodes and allocated bytes.

## Final performance

The final-code Release measurements used three iterations of the canonical
500-attempt command.

| Metric | Previous | Final min | Final median | Final max |
| --- | ---: | ---: | ---: | ---: |
| Wall | 79.66 s | 46.81 s | 46.97 s | 47.00 s |
| Graph preparation | 11.64 s | 11.83 s | 11.92 s | 11.92 s |
| Tracing | 67.39 s | 34.26 s | 34.40 s | 34.52 s |
| User CPU | 1,381.65 s | 742.69 s | 754.73 s | 767.10 s |
| System CPU | 33.44 s | 42.99 s | 48.40 s | 48.55 s |

- Median wall time improves by 41.0%; median tracing time improves by 49.0%.
- Every run retained 500 accepted lines, 27,715 covered anchors, 724 computed
  candidates, and 224 discarded candidates.
- The largest observed per-candidate route arena contained 224,096 nodes and
  had 29,360,128 bytes of allocated capacity. Process peak RSS was
  11,389,868-11,455,404 KiB, still dominated by the materialized graph.
- Every generated line, direction group, and anchor OBJ is byte-identical to
  the pre-change `/tmp/fiber-crop-parallel-smallq` artifacts.

## Profile

- `gprofng` sampled the same crop with 100 attempts in GCC Release.
- After the optimization, inlined lookahead/`traceSide` accounts for 47.1%
  exclusive CPU, immutable transition lookup 18.2%, route-view lookup 6.7%,
  and the remaining intermediate-prune sorting about 2.5%.
- Completion route reconstruction and completion sorting no longer appear as
  separate hotspots. The next meaningful exact optimization target is graph
  transition/route lookup rather than search-prefix ownership.

## Validation

- GCC Release `test_fiberlet_crop_trace`: 12 cases pass, repeated 20 times.
- GCC Release `test_fiberlet_storage`: 37 cases pass.
- Clang Debug `test_fiberlet_crop_trace`: 12 cases pass.
- The new 65-way equal-density test crosses the intermediate pruning boundary,
  verifies the lexicographic winner, exercises committed/rollout back edges,
  and repeats under a one-generated-state cap.
- `git diff --check` passes.

## Deviations

- None. The arithmetic, traversal order, cutoff semantics, counters, geometry,
  and output ordering are unchanged.
