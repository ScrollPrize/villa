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
  cleanly while it was paused in update-34900 evaluation. Started replacement
  launch
  `5scroll_json_eligible_fullres_refinement_gaussian_halfsigma_sideband_128_sd1_16x8_ema-20261001_113301`
  from the old run's `snapshots/current.pt`, restricted to GPUs 0-5. It is alive
  and cooperatively queued behind non-pausable task
  `766aee14-4751-48b2-86af-ffdef378611c`.
