# Task log

- 2026-09-17: Repository was clean at `1079e727d`; no pre-task changes remained to
  commit. Stopped full-resolution refinement run has durable `current.pt` at update
  17800 (2,278,400 samples) referencing `fibers_step_00017800.zip`, SHA256
  `a9120cd0c21c968f94c3ec802339e5bc678712c72dd7f183b0f7f32dd345ed53`.
- 2026-09-17: Interpreted “from scratch” as resetting model/optimizer/EMA/progress
  while importing only the saved refined fiber geometry. Total embedding width
  remains16: short channels0:8, long channels8:16.
- 2026-09-17: Independent review found archived test-patch identities are from the
  unpaired run and cannot be reused by paired evaluation. Scope frozen geometry to
  training fibers; use immutable original test GT and forbid frozen mode from
  creating new state. Review also requires separate bank statistics/normalization,
  read-only archive provenance, downstream bank selection and matched-budget timing.
- 2026-09-17: Implemented explicit frozen geometry restore, separate from model
  resume. The training split reads and validates update17800 geometry without
  proposals/commits/archive copies; paired tests retain original GT.
- 2026-09-17: Implemented independently normalized8-channel short and long banks.
  Short positives/negatives stay within a patch; long positives cross paired
  patches and negatives cover the pair. The existing65536 negative budget is
  split evenly and the bank means have equal loss weight. Inference requires an
  explicit bank for split checkpoints.
- 2026-09-17: Focused CPU functionality tests passed; the broad refinement subset
  had45passes/5skips plus one sandbox-only Gloo loopback failure, to be rerun with
  an explicit loopback interface. Managed CUDA validation and timing remain.
- 2026-09-17: Approved validation results: split contract16passed/6expected skips;
  broader paired suite78passed/12expected skips; managed CUDA10passed/47deselected;
  real frozen archive/data preflight1passed in92.91s, validating1524train and90test
  fibers plus actual train/test pairs. The earlier broad sandbox Gloo failure was
  environmental; managed two-GPU NCCL coverage passed.
- 2026-09-17: Matched-budget one-GPU timing (two16-channel32-cube patches,
  65536 negatives/patch,3warmups+10samples): legacy mean/p50/p95
  6.867/6.882/6.907ms; split8x8 9.899/9.895/9.968ms. The split objective adds
  about3.03ms in this isolated loss benchmark; no model/data path is included.
- 2026-09-17: Committed implementation/config as `6068cd822` and launched managed
  run `5scroll_json_eligible_fullres_frozen_refined_split_embedding_128_sd1_16x8_ema_20260917_121854`.
  Update0 evaluation restored the exact frozen archive and reported finite short
  (`5.165938`) and long (`5.276968`) embedding losses from fresh model/EMA state.
- 2026-09-17: Startup exposed an existing native remote-open bug: 128 loader
  workers eagerly ran `aws configure export-credentials` even though the public
  S3 objects accepted anonymous access. Added an anonymous-first, deferred-auth
  fix for future runs; the active run remains uninterrupted.
- 2026-09-17: Rebuilt the editable VC extension with GCC13. Focused native auth
  coverage passes 18/18, all changed production translation units compile, and
  the approved `run_fiber3d_experiment remote-auth-probe` opened the public
  Paris4 pyramid (levels0-5) with a sentinel AWS helper that was never invoked.
