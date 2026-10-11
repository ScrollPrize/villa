# Remove pairwise wrapped-phase term

- User selected squared as working candidate and requested removing the
  redundant pairwise wrapped-phase penalty, not vertical alignment.
- Independently approved plan: uv_remove_pair_phase_plan.md.
- Shared pair_geometry now returns only vertical metric; both fused and eager
  and all objective variants use the same helper. No compatibility toggle added.
- Winding-distance observed phase correction and H/V layer offset unchanged.
  Individual angular priors, offset bounds, sign and vertical terms unchanged.
- Regression tests isolate q/phase independence of metric, preserved distance
  correction, vertical value/gradient/clamp and all four objective modes.
- No new GPU solve or moment acceleration in this narrowly scoped change.
- Validation:54passed using approved pytest entrypoint with --noconftest and
  pythonpath=/home/hendrik/villa2/vesuvius/src, covering test_fiber_uv_distance_objectives,
  pair_backend, layout, sign_charge, full_pass and global_accumulation modules.
  Existing sparse-invariant warning only. git diff --check passed.
