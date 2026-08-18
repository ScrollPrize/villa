# Task: staged fiberlet anchor acceleration

Continue accelerating anchor and fiberlet extraction from commit `73fe64e09`.
Implement the remaining optimization options one at a time. After every option,
run the canonical 5,000-base-voxel replay, report wall/CPU phase measurements,
anchor and fiberlet populations, DP work, and replay failures, then stop for
user review before beginning the next option.

The ordered options are:

1. Build compact normalized observations once per extraction tile and let each
   cell reference them instead of rebuilding expanded observation records.
2. Reuse robust-proposal assignments, Gaussian/alignment state, baseline
   objective, and final membership across adjacent fitting phases.
3. Batch peak candidates or use a contiguous spatial index so repeated peak
   responses share observation loads without the locality regression of the
   rejected linked-bin experiment.
4. Evaluate reducing robust refinement from two passes to one, with explicit
   visual and replay-quality review.
5. Reduce duplicated prediction sampling across overlapping anchor-tile halos
   while preserving enough independent jobs for 32-worker load balance.

Small floating-point differences are acceptable. Persistent geometry and file
formats must remain valid, and each checkpoint must retain acceptable replay
quality before it can become the next baseline.

The current continuation targets the remaining fiberlet dynamic-programming
cost. Measure separately: caching decoded static data for every retained node,
reusing each reached node's outgoing-edge descriptors across incoming states,
fully pre-generating all retained-node transitions, and compacting DP state by
deriving incoming edges and predecessor nodes from the node/transition key.
Retain the fastest acceptable composition and record rejected variants.
