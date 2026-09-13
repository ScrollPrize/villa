# Local contrastive competition

- [x] Commit prior state and write plan.
- [x] Review plan.
- [x] Objective and test diagnostics.
- [x] CPU/managed GPU validation:54 CPU passes,5 CUDA passes (FP32/BF16 and cost).
- [x] Stop old run; verify dead and manager release.
- [x] Launch fresh managed run (root3343490; no resume; first failed launch retained for diagnosis).
- [x] Verify all 24 real raw/EMA similarity tags and three finite optimizer updates.

Active run: `combined_5scroll_s1a_contrastive_128_sd1_16x8_ema_20260913_121417`.
Automatic real-update startup calibration continues; adaptive handoff has not
yet been observed. The requested fresh training is running, not an evaluation-only probe.
