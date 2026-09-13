# Improve embedding tracing beyond 69 restarts

First implement a bounded multi-anchor fiber appearance model: representative of as much observed path as possible and diverse, optionally combined with EMA. Measure against frozen EMA99 update6500 winning EMA512/.25 (69 restarts). Continue promising additional methods/refinements on own initiative, not stopping after an intermediate result. Preserve training and manager grants, same dataset/snapshot/non-embedding settings. Commit prior state first.

User additionally requests LRU anchors next: fixed capacity, nearest-anchor use
tracking, least-recently-used eviction, bounded admission frequency; score via
nearest/average or EMA combination. Continue experiments, do not stop at report.

After LRU, correct the earlier rolling-window restriction: descriptive/diverse
anchors must cover the whole path, not just a recent anchor window. Bounded
memory is fine; distance-based forgetting is not the requested appearance model.

After initial implementations/sweeps, refine every approach's own best settings,
not only the global winner: weight, capacity, admission cadence, EMA blend and
mode-specific parameters. Repeat winners, retain evidence of explored bounds;
do not claim a global optimum from a finite parameter search.

Refinement is not only parameter sweeps: inspect weaknesses and test algorithmic
variants/new methods too (admission, replacement, aggregation, stable/recent
appearance combinations). Keep evidence distinguishing mechanism from tuning.
