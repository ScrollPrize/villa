# Resume full-resolution training with refinement — 2026-09-16

- Stopped managed full-resolution run
  `5scroll_json_eligible_fullres_128_sd1_16x8_ema_20260916_192344` after its
  completed update4100/test. Manager allocation released cleanly.
- Durable source checkpoint exists at that run's `snapshots/current.pt` (750MiB).
- Existing reviewed refinement delta adds per-scroll normal manifests,
  `bounded_gradient_accumulation:true`, `_config_dir`, refinement weights
  normal0.1/tangent10/direction0.1, a resume source and distinct run name.
- Planned new config is the existing refinement config with only output
  stride4->1, resume source->the stopped fullres current checkpoint, and a
  distinct fullres-refinement run name.
- Independent review approved that three-field derivation and requested direct
  checkpoint-state verification.
- Verified the checkpoint itself: update4100, 524,800 samples, raw snapshot,
  optimizer and EMA present, and no `refinement_state`.
- Added the full-resolution refinement continuation config and an exact-delta
  regression.
- Approved CPU validation passed: training/config spans 41 passed, 2 skipped;
  refinement 23 passed, 2 skipped. Logs are under
  `/ephemeral/hendrik/fiber3d_train2/cooperative-tests/20260916_213539_062927`.
- Committed the reproducible continuation state as `39b7277cd`.
- Launched managed continuation root1628019 at
  `/ephemeral/hendrik/fiber3d_train2/runs/.launches/5scroll_json_eligible_fullres_refinement_128_sd1_16x8_ema-20260916_213749`.
  Active run directory is
  `/ephemeral/hendrik/fiber3d_train2/runs/5scroll_json_eligible_fullres_refinement_128_sd1_16x8_ema_20260916_213753`.
- Verified exact raw resume at checkpoint step4100/524,800 samples and fresh
  refinement progress through updates4101-4103. Full-resolution refinement
  calibration measured microbatch1 and is advancing to microbatch2 before the
  adaptive continuation.
