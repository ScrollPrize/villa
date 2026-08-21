# Task log: exact cost-bounded fiberlet lookahead

## Initial findings

- The current whole-fiberlet implementation incorrectly scores the complete
  terminal edge and divides by the overshooting route length. Whole-fiberlet
  state/commitment does not imply that geometry beyond the logical horizon
  should contribute cost.
- The interrupted exhaustive full-fiber radius-768 run reached 94.4% in 5m10s
  without a search error, but interruption occurred before atomic replay
  publication. It therefore has no recoverable partial failure or quality
  summary.
- The completed focused interval remains the only validated result: the
  600-base-voxel run reproduced the older baseline's physical failure at
  reference arc `43205.62206902002` and evaluator XYZ
  `(25457.0546875, 17248.955078125, 55132.734375)`.
- A hot-cache run of the previous implementation reached 90.9% in 18 seconds
  and 94.2% in 19 seconds, then stopped advancing at the pathological search
  decision. It was interrupted at 32 seconds. The shortened `--length 41533`
  run cannot reuse the full cache because cache chunk headers intentionally
  bind the selected reference interval, so it would have regenerated hours of
  data rather than isolate the search.
- The first exact uniform-cost implementation retained zero failures through
  96.0% of the full radius-768 replay, confirming that the graph supports the
  reference in the region where the previous beam failed. It was interrupted
  after 2m25s at 96.0% with 150.58 user seconds, 1.66 system seconds, 104% CPU,
  and 907708 KiB peak RSS. The mixed-prefix queue was therefore exact but not
  practical: its zero-future-cost lower bound left one late branching decision
  effectively serial and weakly pruned.
- The implementation is being reorganized into exact independent continuation
  searches for each distinct prefix crossing the next checkpoint. This does not
  alter candidates, costs, the exact horizon, or the final top-16 selection; it
  exposes the independent prefix work to `--threads` and gives each prefix a
  useful stop condition after its first best completion is established.
- Parallel fixed-prefix uniform-cost search completed the focused 2500-base-
  voxel tail interval in 7.94 seconds with zero failures and unchanged exact
  route/reference metrics. The full run again slowed at 95.8% and was
  interrupted after 1m43.93s. It used 191.54 user seconds and 820.00 system
  seconds with 33,997,182 voluntary context switches, showing both a weak
  remaining-cost bound and severe cached-graph contention during excessive
  parallel expansion. The next iteration adds a memoized admissible relaxed
  cost-to-go heuristic to reduce graph calls rather than accepting this result.
- Sharing one admissible 0.5-prediction-voxel relaxed cost-to-go table across
  prefix workers reduced the same focused interval to 4.61 seconds with the
  same zero failures and route metrics. The uninterrupted full replay still
  remained at 95.8% after 1m22s and was interrupted. A deterministic greedy
  incumbent rollout did not improve that decision and was removed. Exact
  simple-path certification therefore remains impractical in this dense late
  region; the next measured candidate should be the documented wider
  intermediate lookahead prune, compared to exact focused windows.

## Independent review

- Horizon-crossing successors must be exact-scored immediately; their complete
  overshooting edge cost must never enter a queue cutoff.
- Exact-anchor boundaries charge neither an outgoing edge nor its join.
- Lower-bound pruning requires finite nonnegative edge and join components,
  including cached values, and equality remains eligible for deterministic
  ties.
- Scoring is cumulative from the segment seed to `T`. Decision diagnostics
  retain full terminal geometry and expose the full terminal length separately.
- Tests require malformed-cost cases and an independent exhaustive oracle, not
  an oracle implemented by the production queue helpers.

## Deviations

- None.

## Validation

- Pending.
