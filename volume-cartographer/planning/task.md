# VC3D control-point collapse rollback fixes

Correct two regressions in generated-view control-point editing introduced by
the 32-base-voxel multi-control collapse behavior.

Requirements:

- Automatic multi-control collapse must use the same local line-update sequence
  as insertion and single-control replacement: reconstruct the adjacent spans
  around the replacement's authoritative `linePosition`, then start fiber-mode
  optimization from that updated line.
- Do not start multi-control optimization directly against the unchanged old
  line. In particular, a replacement on a self-approaching fiber must not be
  associated with another winding by nearest 3-D position.
- Preserve the existing 32-base-voxel inclusive collapse selection, control
  metadata ownership, branch-index remapping, dirty-span scope, seed/focus
  behavior, no-reoptimization behavior, and asynchronous failure rollback.
- If synchronous local update preparation throws, leave the pre-edit controls,
  line, branches, seed/focus, and optimization state unchanged.
- Keep the broader persisted-control/legacy nearest-3-D reconstruction issue out
  of scope; this task fixes the PR 1484 regression without changing the fiber
  file format.
