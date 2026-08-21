# Task log: bounded intermediate fiberlet lookahead

## Baseline

- Exact fixed-prefix A* is committed at `d5b9582d2`.
- The focused 2500-base-voxel radius-768 tail interval completed with zero
  failures in 4.61 seconds after sharing the relaxed cost-to-go table.
- The uninterrupted full replay retained zero failures through 95.8%, but the
  dense late decision remained combinatorial and was interrupted after 1m22s.
- Exact A* therefore remains useful as a focused oracle but is not a practical
  full-fiber default.

## Search options

1. Active: wider intermediate pruning at equal-distance fronts, initially
   `K=128`, `P=48`, `H=192`, and final beam width 16.
2. Deferred: deterministic adaptive horizons 192/144/96/48 selected from
   generated-state pressure, not timing.
3. Active: uniform-cost distance labels with reconvergent-state dominance.
   The existing relaxed cost-to-go was tested and removed from bounded mode
   after it provided no measurable dense-tail improvement while serializing its
   recursive memo table.

## Deviations

- The first intermediate front is the next checkpoint `C+D`; later fronts add
  `P` and the last front is the exact horizon. Existing focused boundary tests
  with `D >= H` remain supported by clamping the first front to the horizon;
  their complete crossing fiberlet must still cover the checkpoint.
- Local workers retain both their top `K` global candidates and top `K` stable-
  prefix representatives. Retaining only the raw local top `K`, as the initial
  plan stated, could erase a locally expensive prefix before the diversity
  pass. The combined bounded set is necessary to implement the specified
  prefix protection.
- Bounded fronts use a 0.5-prediction-voxel state bin and deliberately discard
  alternate visited histories that reach the same logical incoming fiberlet
  and front-offset bin. The retained winner supplies subsequent cycle state.
  This is the requested reconvergence approximation; exact mode remains the
  oracle for focused comparisons.

## Independent review

- Fronts must be constructed as `C+D`, then repeated
  `min(previous+P, C+H)`, with `D <= H` and an exact final horizon.
- Prefix identity is the segment seed plus ordered stable logical arc IDs
  through `C+D`; history-pointer equality is not a contract.
- Scoring is cumulative from the seed to each exact front, with exact-anchor
  edge/join ownership and complete terminal-edge state.
- Parallel jobs must stream-retain local top `K`, merge canonically, and expose
  deterministic per-front diagnostics. Initial root successors must be
  independently schedulable.
- Focused performance runs require at least three repetitions and prefix/cost
  comparisons in addition to failure and geometry metrics.
- One generated-state budget spans the entire rolling decision. Parallel work
  uses canonical indexed outputs and aborts the complete decision on overflow,
  so worker scheduling cannot select a partial population.
- Front scheduling clamps `C+D` and repeated `+P` advances to the exact final
  horizon after converting all distances through prediction-to-base scale.
- Diagnostics include per-front and cumulative counts, stable selected prefix,
  and terminal output continues to rematerialize and clip from the segment seed.

## Validation

- `cmake --build volume-cartographer/build --target vc_fiberlets
  test_fiberlet_paths -j32` succeeds.
- `volume-cartographer/build/bin/test_fiberlet_paths` has no new failures. It
  still reports the pre-existing 298 checks at the float bitwise fixture
  (`:406`) and the Q4 extraction fixture (`:1188-1190`).
- Added a non-unit-scale, nondivisible-front fixture. Bounded search matches
  exact output and produces byte-identical diagnostics with one and four
  expansion threads.
- Full hot-cache radius-768 `K=128, P=48, H=192` reached 96.0% in 103.26
  seconds before manual interruption, with 1,033.07 user seconds, 94.72 system
  seconds, 1,767,076 KiB peak RSS, and about 10.9 effective CPU cores. It
  crossed the previous exact-search 95.8% wall but remained too slow.
- Full hot-cache radius-768 `K=128, P=24, H=192` reached 97.3% in 610.45
  seconds before manual interruption, with 1,594.81 user seconds, 98.93 system
  seconds, 1,819,300 KiB peak RSS, and about 2.77 effective CPU cores. It
  traversed well beyond the former failure region without a reported tracing
  failure, but reducing the between-front exponential depth alone is not
  practical.
- The first label-search full hot-cache radius-768 run used the shared relaxed
  cost-to-go lower bound. It crossed the former exact-search wall, reaching
  96.0% in 149 seconds before manual interruption. Replacing that heuristic
  with uniform-cost ordering reached the same dense region at effectively the
  same rate, showing that recursive cost-to-go computation was not the primary
  remaining tail cost.
- A regression graph with two routes that reconverge before a shared outgoing
  fiberlet records dominated labels and selects the same committed route as the
  exact oracle. The complete path test still has only its 298 pre-existing
  float-bitwise and Q4 extraction checks.
- Rebuilding the retained top-K vector for every exact completion was removed;
  a multiset now maintains the exact stopping cutoff and candidates are ranked
  once after expansion. Local search limits now request one candidate per
  represented stable prefix plus only globally required fill slots. On the
  full hot-cache radius-768 run this moved the dense-tail entry from 94.7% to
  95.5% almost immediately, but one decision near 95.9% remained expensive.
- Retesting the relaxed cost-to-go with the corrected local quota reached 96.0%
  in about 75 seconds, effectively the same dense-decision rate as uniform-cost
  ordering. Bounded mode therefore keeps the simpler uniform-cost labels.
- Rebuilt `vc_fiberlets` and `test_fiberlet_paths` with `-j32` after adding
  state dominance and local quotas. The full test binary still reports exactly
  the 298 pre-existing float-bitwise and Q4 extraction checks; the new tests
  add no failures. They verify bounded/exact route agreement, reconvergent
  label elimination, one/four-thread diagnostic equality, a full-width first
  front, and a one-candidate later-front quota when every global slot already
  has a stable-prefix representative.

## Performance finding

- The old edge-depth beam pruned to 16 after every fiberlet layer. Plain
  intermediate search instead enumerates all valid route combinations until
  an equal-distance front and only then prunes to `K`. Several short fiberlets
  fit inside one 24/48-base-voxel front in the dense region, so the temporary
  population remains combinatorial even though the retained population is
  bounded.
- The active refinement uses uniform-cost distance labels within each short
  front. Its stopping condition covers both best-per-stable-prefix
  representatives and global fill candidates. Reconvergent labels are rejected
  before successor generation; queued labels replaced by a better history are
  rejected as stale when popped.
- The completed full hot-cache radius-768 run over 46,147.996 base voxels took
  about 19 minutes 13 seconds, used 2,029,020 KiB peak RSS, and produced seven
  fiberlet failures versus fourteen greedy failures. The earlier fast
  three-fiberlet-depth beam result over the identical reference interval stored
  one fiberlet failure and the same fourteen greedy failures. This is a
  negative result: the new search is both much slower and less accurate. The
  focused exact-oracle fixtures did not predict this full-fiber regression.
- Compact replay had also accidentally gated immediate failure lines on
  `--stats`. Failure callbacks now always print `fiber_replay_failure`; compact
  mode interrupts and redraws its progress bar. A 1,700-base-voxel normal-mode
  replay verified that the greedy failure at arc 6,297.991 was printed before
  the final summary without `--stats`.
