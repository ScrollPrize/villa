# Task: write-back memory cache for temporary Fiberlet reduction layers

Accelerate staged `vc_fiberlets chunk-route-stats` reductions by replacing
eager temporary overlay publication with a bounded asynchronous write-back
cache.

- Temporary anchor and Fiberlet overlay chunks must remain immediately visible
  to later boxes and stages without first writing and rereading files.
- Retain dirty chunks in an LRU memory cache. Spill exact existing bytes with
  asynchronous atomic writes only when the shared RAM budget requires
  eviction.
- Prefix and route chunks are one consistency unit and may never be observed or
  spilled as a partial pair.
- Account write-back memory against the existing `--cache-gib` decoded-cache
  budget; do not silently add a second unbounded RAM allowance.
- Preserve exact retained IDs, payload bytes, stage ordering, fallback-layer
  behavior, reporting, and one-thread/multi-thread determinism.
- Measure the established four-stage Paris4 Release workload before and after.
