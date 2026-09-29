# Task log: anisotropic Gaussian fiber-presence supervision (2026-09-29)

- Preserved prior work as commits `5eee7e0bb`, `afb7152d8`, and `ae58b079e`.
- Selected source checkpoint: high-resolution refinement run
  `5scroll_json_eligible_fullres_refinement_128_sd1_16x8_ema_20260916_213753/snapshots/best_ema_0p999.pt`
  (checkpoint update 17,000).
- Design assumption: existing `presence_radius_voxels=2` is thickness sigma;
  width sigma is 4x and support is truncated at 3 sigma. Local axes come from
  the fiber tangent and native Lasagna surface normal after augmentation.
- Independent review required EMA weights-only initialization (not optimizer
  resume), inverse-transpose local normal transforms, exact soft-tail loss and
  prepass counts, complete metadata transport, stride coverage, and bounded
  target benchmarking. These requirements were incorporated into the plan.
- Implemented bounded per-segment Gaussian tube rasterization. The Lasagna
  surface normal is transformed as a covector through the augmentation map;
  the orthogonal in-sheet axis receives four times the thickness sigma.
- Soft target voxels now contribute BCE against their Gaussian probability.
  Only exact zero-support voxels contribute to the separately balanced
  background term; direction supervision remains on the centerline.
- Configured weights-only initialization from the high-resolution refinement
  run's best EMA-0.999 snapshot. The associated refined fiber state is restored,
  while optimizer/update state intentionally starts fresh.
- Validation: focused Gaussian geometry/loss tests passed (6/6); combined arc,
  shared-slot, and EMA-refinement restore regression selection passed (33 passed,
  2 CUDA-only skips). Python compilation and `git diff --check` passed.
- Broader test files reach unrelated pre-existing distributed-test failures in
  this host environment (`Gloo` cannot resolve `127.0.0.1`); no Gaussian-path
  assertion failed before those infrastructure failures.
