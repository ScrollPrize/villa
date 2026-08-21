# Task: bounded intermediate fiberlet lookahead

Accelerate the expensive fiberlet replay decisions while retaining the improved
route quality demonstrated by long lookahead.

Record and evaluate three search options in this order:

1. First implement wider intermediate pruning. Keep the 192-base-voxel logical
   lookahead, but prune a wider working population at equal 48-base-voxel
   fronts before the final 16-prefix selection. Within each front use a
   distance-binned uniform-cost label search and close reconvergent states by
   keeping only their best history, so the implementation does not enumerate
   every complete route combination before applying the width bound.
2. Later evaluate a deterministic adaptive horizon that shortens lookahead when
   generated-state counts expose combinatorial growth.
3. Keep exact A* over the complete horizon as the oracle. Evaluate the existing
   relaxed distance-to-go DP against uniform-cost ordering, and retain the
   simpler ordering unless the heuristic provides a measured benefit.

Intermediate scoring must use the same exact physical horizon, proportional
terminal-fiberlet cost, complete terminal geometry, join ownership, graph
validity, cycle rejection, deterministic ordering, failure handling, and cache
semantics as the exact implementation. The approximation must be explicit and
diagnosed. Preserve the exact implementation as a benchmark oracle selected by
`--search-width 0`.

Start with working width 128 and prune distance 48 base voxels. Preserve route
diversity by retaining the best continuation for each currently represented
committed prefix before filling the remaining working slots globally.
