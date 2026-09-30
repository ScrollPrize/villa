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
- First half-sigma/orientation launch restored update 18,000 but incorrectly
  included reserved physical GPU 6. Its initial 121-batch test also deadlocked
  on the final 2-batch wave:
  ranks 0-1 ran an extra DDP forward while ranks 2-6 entered the next refinement
  collective. NCCL reported sequence 599 `ALLGATHER` timeout after ten minutes.
- Managed evaluation now pads ranks without a real final-wave batch with a
  deterministic duplicate forward, disables their stateful observer, and drops
  the padded result. This preserves identical DDP sequencing without changing
  test rows or refinement proposals. Focused partition/gather tests: 6 passed.
- Corrected the production allowlist to physical GPUs 0-5; GPUs 6-7 remain
  reserved.
- The corrected six-rank launch completed ordinary test waves through 114:119,
  then exposed a broker ownership bug in the padded 120:120 wave: multiple idle
  ranks borrowed sample 1920 concurrently, but the first release deleted the
  sole ticket and the next release raised `KeyError`. Test sample leases are now
  reference-counted; the queue ticket is returned only after the final reader.
  Focused broker/evaluation tests pass.
- Replaced uniform soft-target BCE with explicit Gaussian components. Ridge
  positive (`y`) is normalized by Gaussian mass, distribution negative
  (`1-y`) by nonzero-support tail mass, and external background independently;
  coefficients 1/0.5/0.5 retain the historical total scale. The fourth target
  count and numerator are now carried through the single-backward accumulator,
  not merely reported as metrics. Focused prepass, gradient, loss, and broker
  tests pass (4 selected, 308 deselected).
- A six-GPU relaunch proved that reference counting alone was insufficient:
  rank 0 reached refinement `gather_object` after final batch 120 while ranks
  1-5 remained behind the shared-ticket transport, producing NCCL sequence 632
  timeout. Padding now assigns distinct deterministic existing test batches to
  idle ranks whenever the test set has enough batches (121 here); duplicate
  shared leases remain only as the bounded fallback for test sets smaller than
  the world size. Focused wave/broker tests pass (2 selected, 290 deselected).
- The distinct-batch relaunch completed every full wave through 114:119, then
  exposed a second parity condition: real batch 120 has 15 samples while the
  distinct padding batches have 16, so ranks issued 15 versus 16 DDP forwards.
  Managed evaluation now performs one metric-free local parity forward for a
  partial logical batch. Its loss and refinement observer effects are discarded.
  Focused parity tests pass (2 selected, 289 deselected); the broader affected
  CPU suite passes (60 passed, 14 CUDA-only skips).
