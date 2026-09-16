# Fresh full-resolution five-scroll training — 2026-09-16

- Committed preceding direction-weight trial as `597af5c93`.
- Stopped refinement launch `5scroll_json_eligible_refinement_128_sd1_16x8_ema-20260916_062132`; launcher reports dead and manager task finished/interrupted with zero resources.
- Baseline is `train_5scroll_json_eligible_ema_16x8.json`: eligible five-scroll JSON-only, no embeddings/refinement/resume, LR0.01, global batch128, 16/GPU cap, BF16/GroupNorm, EMA0.99/0.999.
- Planned experimental delta: output stride4->1 plus distinct run name only.
- Independent review approved; specifically require a new stride-1 calibration.
  Calibration fingerprint hashes the full numerical/data config (excluding only
  cooperative controls), so output stride1 cannot reuse the stride4 report.
- Added `train_5scroll_json_eligible_fullres_ema_16x8.json`; exact-diff test
  permits only stride4->1 and the distinct fullres run name. Fresh/no
  refinement/no embedding/no resume assertions included.
- Approved `training-spans` suite passed: 40 passed,2 skipped in33.02s;
  `/ephemeral/hendrik/fiber3d_train2/cooperative-tests/20260916_191912_825700`.
- Independent final review approved the exact config delta, regression, fresh
  semantics, calibration separation and managed launcher selection.
- Committed config/test/docs/launcher as `a456f1dc5`.
- Approved production launch root1436973, launch dir
  `5scroll_json_eligible_fullres_128_sd1_16x8_ema-20260916_192340`.
- Initially queued behind exclusive `refinement062136-best`; it then acquired
  8GPU/128CPU. Fresh real-update calibration measured microbatch8 at step12 and
  saved fingerprint `7fa1c121dd3ca12a56efeda8046f3725d81bdd565c281961e115ca977d6305e5`
  (adaptive minimum20,373,831,680 bytes/GPU).
- Active run is `5scroll_json_eligible_fullres_128_sd1_16x8_ema_20260916_192344`.
  Verified fresh progress through update200 and raw/EMA tests at100 and200.
