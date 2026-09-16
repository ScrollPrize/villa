# Anisotropic/direction refinement — 2026-09-15

- 2026-09-16: stopped follow-up launch212754 using approved production-stop at
  update37144; launcher confirmed dead and manager released CPU/GPU allocation.

- Follow-up: user requested10x tangent and direction, NOT normal. Config now
  normal0.1/tangent10/direction1, original11800 resume, training LR/losses unchanged.
- Follow-up review confirmed exact config/resume. Prior launch185706 confirmed
  dead; CPU regression23 passed/2 CUDA-gated skips via approved runner212724.
  GPU suite not repeated for this configuration-only change (previously passed).
- New approved managed launch212754 root748805, original11800 checkpoint.
  Queued behind nonpausable task376de7b9-03ce-4f0e-a2de-bad9474ae664; no new batch yet.

- Prior run stopped at24454; preserve its uncommitted implementation.
- New production weights normal0.1/tangent1.0/direction0.1; restart original11800.
- Independent plan/final code reviews approved. Full17band solve, signed-axis-safe
  interpolation, analytic material-normalized direction force with full warp pullback.
- CPU23 passed: dense anisotropic reference, direction finite-difference gradient,
  nonlinear-warp transform, old constraints and raw/EMA same-target regression.
- First synthetic10x energy assertion used a curve with longitudinal curvature;
  corrected fixture to pure normal curvature (no solver change).
- Guided solve+losses: 16 synthetic patches/24nodes/4iters/10repeats, one CPU core,
  min/median/max107.12/107.39/111.66ms. No end-to-end training speed claim.
- Direction term is lagged-field (no spatial derivative), single branch only;
  other branch counts fail explicitly. Existing production model is single branch.
- Approved cooperative launcher 20260915_185529_021647: 23 CPU refinement,
  57 managed, 2 manager-exclusive GPU tests passed; four gated CPU-suite skips.
- Approved production-start: root370938, launch20260915_185706; original config
  resume path unchanged, LR0.01; source diff recorded by launcher.
- Confirmed checkpoint_step11800/next11801 from original20260914_105436.
  New run5scroll_json_eligible_refinement_128_sd1_16x8_ema_20260915_185710.
  Initial test completed, including EMA99/999 and normal/tangent/direction loss
  prints. Example128patch wave: presence.12174, normal.00001593, tangent.00004179,
  localization.001369, direction.001402. Startup calibration still active.
