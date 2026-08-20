# Task: corridor-filter on-demand fiberlet preprocessing

Make cached fiberlet replay retain the chunk caches as bounded processing swap
storage without extracting every anchor cell in each touched spatial chunk.

- Select anchor cells by their exact distance from the active reference-fiber
  interval and replay radius.
- Build that selection with a spatially bounded/indexed method, never an
  all-cells-by-all-reference-segments scan.
- Generate and persist only selected anchors; fiberlet chunks must consequently
  contain only paths whose endpoint anchors belong to the selected corridor.
- Make persisted cache identity include the corridor selection so incompatible
  filtered chunks cannot be reused by another replay.
- Actually submit the already-computed reference chunk schedule for background
  anchor/fiberlet preprocessing, while keeping cache misses authoritative and
  blocking only when traversal reaches unavailable data.
- Preserve deterministic anchor, fiberlet, and replay results relative to the
  existing eager tube extraction for the same reference interval and radius.
