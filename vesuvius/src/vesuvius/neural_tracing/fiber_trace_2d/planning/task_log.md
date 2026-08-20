# Task log: corridor-filter on-demand fiberlet preprocessing

## Findings

- `prefetchScheduled()` was implemented but has no caller. Replay computes its
  complete ordered schedule only to associate chunk callbacks with reference
  positions, so traversal still discovers work through blocking graph queries.
- On-demand anchor generation currently enumerates every owned cell in a
  touched 512-base-voxel chunk: normally 4,096 cells. The canonical eager path
  instead uses `fiberAnchorCellsNearPolyline()` to enumerate bounded per-segment
  cell neighborhoods and exact-test segment/AABB distance.
- A generated fiberlet chunk subsequently sees roughly 100,000 halo anchors in
  observed runs and takes 15-23 seconds. The resulting full-cube amplification,
  repeated across the reference, explains the >120-second 5,000-voxel run even
  though each individual extraction uses parallel workers.
- Filtering persisted chunks requires the selected corridor to participate in
  dataset identity; otherwise rerunning a different fiber or radius could reuse
  incomplete chunks silently.

## Deviations

- None currently.

## Independent plan review

- Require the same exact-tube fitted-anchor and DP-point predicates as eager
  extraction; selected cells alone are only a broad phase.
- Bind clipped reference geometry and radius, not only the selected cell set,
  into persisted dataset identity.
- Derive scheduling and generation from one immutable selection and promote a
  foreground demand that was already queued at background priority.
- Extend boundary correctness cases and use a same-checkout repeated/profiled
  performance comparison rather than only the historical timing range.
- Preserve external NMS suppressor context across chunk ownership and submit the
  complete nonblocking schedule after progress bookkeeping and before graph
  evaluation.
