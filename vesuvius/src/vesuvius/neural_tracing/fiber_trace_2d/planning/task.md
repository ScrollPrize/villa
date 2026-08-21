# Task: exact cost-bounded fiberlet lookahead

Replace exhaustive whole-fiberlet lookahead enumeration with an exact
uniform-cost lookahead search.

At checkpoint `C`, every candidate is scored over the exact common interval
ending at `C+H`. The route state retains the complete fiberlet crossing the
horizon, but only the fraction of that fiberlet inside the interval contributes
path cost. Its entering join contributes fully. Candidate scoring must not
penalize geometry beyond the horizon.

Search partial routes in nondecreasing accumulated nonnegative cost. Maintain
the best completed continuation for each distinct whole-fiberlet prefix through
`C+D`. Once 16 distinct prefixes have completions, reject or leave unexpanded
only partial routes whose admissible cost lower bound is strictly worse than
the current 16th completion. Preserve valid joins, cycle rejection,
deterministic tie ordering, whole-fiberlet commitment, failure/reseed behavior,
active float or quantized costs, cache behavior, and cache identities.

Record two alternatives for later comparison without implementing them now:

- approximate speculative-beam pruning at equal 48-base-voxel fronts;
- a stronger A* heuristic based on a relaxed minimum future-cost problem.

The interrupted exhaustive full-fiber run reached 94.4% without a search error
but did not publish a partial result. Validate the exact search first on focused
fixtures and the completed 600-base-voxel radius-768 interval, then retry the
full fiber only if those results are correct and practical.
