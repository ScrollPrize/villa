# Fixed-crop test refinement — 2026-09-15

- Corrected previously missed test constraint: removed border taper AND endpoint
  pinning for independent test observations, retaining uniform1/8-ellipse cap.
  Shared scalar/batched limit function preserves training taper and endpoint locks.
  Reviewer caught artificial outside joins: dense test labels now omit them.
  Independent test quadrature outside model support contributes no presence force.
- Stopped replacement root3545749 before changes. Restart still targets original
  non-refinement update11800 with fresh geometry. No manager protocol changes.
- Approved runner20260915_120124_866724:16 CPU refinement passes/2CUDA skips,
  57 managed passes/2skips,2 manager-exclusive CUDA passes. Final extra join test pending.
- Final CPU suite20260915_120258_615398:17 passed/2CUDA skips, including join
  removal, uniform cap/end movement and scalar/batched parity. Independent review
  found no blocker for current non-embedding configuration. Adjacent follow-up:
  embedding negative-exclusion geometry does not yet honor label_breaks; that
  combination is not used by this run. No embedding code changed in this correction.
- Replacement launched root3826686 from original update11800, fresh geometry:
  /ephemeral/hendrik/fiber3d_train2/runs/.launches/5scroll_json_eligible_refinement_128_sd1_16x8_ema-20260915_120323/train.log

- User corrected restart origin: replace update13300 refinement continuation
  with original non-refinement run 5scroll_json_eligible_128_sd1_16x8_ema_20260914_105436.
  CPU checkpoint inspection: update11800, optimizer present, no refinement_state
  or refinement config. Independent review confirms fresh geometry on launch.
- Stopped root3228885 via approved production-stop. Stop required built-in
  termination escalation after a stalled shutdown; final process state dead.
  Config corrected to original current.pt; corrected test protocol retained.
- Replacement launched root3545749 via approved production-start; adaptive
  manager request and loader startup observed, using existing measured calibration.
  Log: /ephemeral/hendrik/fiber3d_train2/runs/.launches/5scroll_json_eligible_refinement_128_sd1_16x8_ema-20260915_113518/train.log

- Saved prior implementation as 5ba084bfd before changes.
- Independent plan/implementation review; corrected selected tangent lookup after
  material-knot resampling without changing original crop transforms.
- Existing store/solver/queues reused; crop-local test state keyed by original
  transform, source, sample and shape. Other visible fiber visits retain baseline
  supervision; nearest connected visit refined as before. Cap/taper unchanged.
- Legacy whole-test refined state intentionally not imported into crop state.
  Baseline source exports are labeled immutable_source_only; train state retained.
- Shared raw/EMA scalar helper exposes missing direction/presence and branch tags.
- Clean managed stop confirmed. Latest raw snapshot update13300 contains optimizer,
  both EMA tracks and fibers_step_00013300.zip. Config resumes that checkpoint.
- Initial CPU regression15 passed/1 CUDA skip. A subsequent archive test needed
  its fixture split changed to train because whole-test exports now show baseline.
- Approved cooperative runner final folder20260915_111948_357598:
  refinement15 passed/1 CUDA skip; unified-managed57 passed/2 skipped;
  manager-exclusive CUDA2 passed, including fixed-crop feedback/reload.
  git diff --check clean. Restart uses approved production-start launcher.
- Final independent review found no blockers. New launch root3228885, manager
  task e0f8fc97-1699-4094-86ce-33908962cffd, granted8GPUs/128CPU workers.
  Log: /ephemeral/hendrik/fiber3d_train2/runs/.launches/5scroll_json_eligible_refinement_128_sd1_16x8_ema-20260915_112130/train.log
- Active run: 5scroll_json_eligible_refinement_128_sd1_16x8_ema_20260915_112135.
  First actual test at update13300 completed in30.749s. Event file confirms
  test_ema_0p99/loss_direction=.00322634, loss_presence=.37071338;
  test_ema_0p999/loss_direction=.00307072, loss_presence=.31341600.
  Launch remains managed with automatic real-update memory calibration.
