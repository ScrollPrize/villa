# Local sideband supervision

- Plan self-review: the sideband is opt-in; does not replace the Gaussian tube
  or global background; uses target-only masks so gradient accumulation remains
  equivalent; synchronizes sums/counts before the single backward pass; and
  reuses the shared cross-section implementation for test visualization.
- Process deviation: no independent sub-agent review because the active agent
  policy disallows delegation without an explicit user request.
- Implemented opt-in segment-frame sidebands with 64-base half-width,
  16-base half-height, Gaussian-support exclusion, separate left/right masks,
  and 0.1 total independently normalized BCE-to-zero weight.
- Integrated raw-model cross-section accumulation into the existing full
  deterministic test forwards. Every 100 updates the derived run writes float
  mean and q0.1 TIFFs, logs both images, fits the central 51x25 mean crop, and
  logs absolute sigmas, supervised-target multiples, center/amplitude and RMSE.
- Validation: focused target/loss/cross-section/single-backward suite: 55 passed,
  3 skipped, with the environment-only Gloo loopback case excluded. Added
  explicit sideband geometry, scale, normalization-count and gradient-weight
  regression tests. A broader CPU run reached 153 passing cases before its
  long-running remainder exited without a retained final pytest summary.
- Stopped old managed launch
  `5scroll_json_eligible_fullres_refinement_gaussian_halfsigma_128_sd1_16x8_ema-20260930_155812`
  cleanly while it was paused in update-34900 evaluation. The first replacement
  attempt exposed a shared-memory conflict with a detached broad pytest process;
  that process was stopped and the committed implementation was relaunched as
  `5scroll_json_eligible_fullres_refinement_gaussian_halfsigma_sideband_128_sd1_16x8_ema-20261001_114603`.
- The managed resume is running on GPUs 0-5 in run directory
  `5scroll_json_eligible_fullres_refinement_gaussian_halfsigma_sideband_128_sd1_16x8_ema_20261001_114609`.
  Startup calibration resumed from update 34809, completed updates 34810-34812,
  saved its measured memory profile, re-registered adaptively, and continued at
  update 34813. It reached update 34840 without a training error, then released
  its allocation and remained alive in `waiting_for_resources`; GPUs 6-7 were
  never requested. The first scheduled integrated cross-section evaluation is
  update 34900 and therefore remains pending until the cooperative grant resumes.
