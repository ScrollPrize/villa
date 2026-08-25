# Task: compact Fiberlet crop lookahead state

Accelerate the sustained `vc_fiber_trace_chunk` tracing phase without changing
its numerical work or deterministic output.

- Stop copying the complete committed visited-anchor set and complete rollout
  arc vector for every accepted lookahead branch.
- Represent rollout states in an indexed parent arena.
- Preserve expansion order, cycle rejection, limits, loss accumulation,
  density ordering, lexicographic route tie-breaking, and selected routes
  exactly.
- Measure the same Paris4 1024-base-voxel crop and compare all generated OBJ
  files byte-for-byte with the current Release result.
