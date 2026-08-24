# Task: finish staged Fiberlet reduction performance and reporting

Fix the staged `vc_fiberlets chunk-route-stats` implementation so the normal
local workflow is fast and its reports remain complete.

- Actually remove the serial hot path in cache-backed local graph
  materialization. Use chunk-granular reads and parallel cache-free graph work,
  while preserving exact numerical order, retained IDs, and serialized output.
- Remove redundant reporting materializations from the reduction path.
- Per-stage reporting must cover the union of that stage's processed boxes and
  show all three scopes: inside anchors, all incident Fiberlets, and interior
  Fiberlets. Offset stages must not count untouched selected-region geometry.
- Keep the joint whole-selected-region report with the same three scopes.
- Configure and rebuild the ordinary `volume-cartographer/build/` directory as
  an optimized Release build so the documented command does not accidentally
  benchmark a Debug CI binary.
- Measure the exact hot Paris4 workload before and after, including actual CPU
  use and deterministic result hashes.
