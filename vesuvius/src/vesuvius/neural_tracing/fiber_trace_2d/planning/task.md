# Task: make fiberlet replay prefix processing incremental

Remove replay work whose cost grows with the already committed segment prefix.
The fixed-distance lookahead must do work proportional to the new search and
newly selected suffix, not repeatedly rebuild route geometry, logical route
keys, reference matches, normals, or visited-node state from the segment seed.

Preserve route selection, exact cycle rejection, failure locations, costs,
serialized replay results, cache behavior, and numerical evaluation order.
Final result materialization may make one linear pass because the result itself
contains the complete route.

Measure the change on a hot-cache replay and retain the existing focused replay
tests. No approximation, probabilistic identity, or relaxed cycle handling is
acceptable.
