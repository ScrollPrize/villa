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
- Follow-up: stopped the first Gaussian run at update 1,827. Expanded direction
  supervision from the centerline to every nonzero Gaussian-support voxel. The
  winning segment supplies the tangent; both direction loss and angle metrics
  use the Gaussian falloff and normalize by its summed weight.
- Kept soft-target BCE: its pointwise optimum is the requested Gaussian, while
  separate support/background normalization prevents background domination.
- Restricted the production launcher to physical GPUs 0-5. Startup calibration
  now accepts an explicit visible subset of managed GPUs rather than requiring
  the entire node.
- Follow-up validation: arc/target and cooperative setup suites passed (44
  passed, 2 CUDA-only skips).
- The direction-weighted sigma-2 run was stopped cleanly at update 18,098; its
  manager grant and workers were released. The continuation uses the latest
  full checkpoint, halves normal sigma 2->1 voxel (width sigma 8->4 voxels),
  and expands the allowed managed device subset from physical GPUs 0-5 to 0-6.
- Orientation-refinement plan review required a modulo-pi doubled-angle state,
  immutable baseline-material normal references for both refinement and target
  generation, exact nonlinear probe transforms, periodic tethering, joint
  geometry/orientation CAS, resampling-aware interpolation, and explicit legacy
  archive migration. The plan now includes these corrections.
- Implemented persistent `(cos(2 theta), sin(2 theta))` orientation state,
  joint geometry/orientation optimistic commits, independent-test-patch state,
  legacy zero-angle migration, baseline-material normal lookup, transformed
  Gaussian targets, exact nonlinear quadrature probes, periodic tethering, and
  capped/tapered angular updates.
- CPU validation: refinement + Gaussian target suites excluding the known host
  Gloo test passed (60 passed, 4 CUDA-only skipped, 1 Gloo test deselected);
  shared-slot/config selection passed 12 tests; non-distributed single-backward
  selection passed 16 with 1 skip. Compilation and diff checks passed.
- One-thread synthetic timing, 16 patches x 24 nodes x four iterations x ten:
  geometry-only 38.94/39.04/39.15 ms and orientation-enabled
  111.62/111.75/115.99 ms min/median/max (143.2 patches/s). This is solver-only.
- Implementation review rejected the first equal-Mahalanobis contour score
  because it favored the 90-degree swapped axes. Replaced it before launch with
  equal-physical-radius `mean(width probes)-mean(normal probes)` and a wider
  five-degree derivative stencil. Synthetic perfect-Gaussian tests converge
  from both signs at 0.01, 0.1, 0.3, and 1.2 radians.
- Resume source verified: `current.pt` is full checkpoint step 18,000 with
  optimizer and EMA state and references `fibers_step_00018000.zip`; its actual
  SHA256 exactly matches checkpoint metadata. The 98 post-checkpoint updates
  before the requested stop are intentionally replayed.
- Final CPU validation after review fixes: 67 passed, 4 CUDA-only skipped, one
  host-Gloo test deselected; shared-slot/config 12 passed; non-distributed
  single-backward 16 passed and 1 skipped; `git diff --check` passed.
