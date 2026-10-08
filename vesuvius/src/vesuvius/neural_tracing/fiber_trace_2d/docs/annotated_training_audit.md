# Annotated-training audit — 2026-10-05

No training restarted. Comparison baseline is the exact frozen launch config
`/ephemeral/hendrik/fiber3d_train2/runs/.launches/5scroll_json_eligible_fullres_refinement_gaussian_halfsigma_sideband1_reviewed32_128_sd1_16x8_ema-20261003_222618/config.json`.
Current candidate: `fiber_trace_3d/configs/train_reviewed_annotations_16x8.json`.

## Requested changes retained

- Full newly downloaded reviewed training collection:92 pieces (0139:38,1299:52,
  1451:2), replacing the older five-scroll training globs. All92 original prepared
  files remain under `2026-10-05_reviewed_annotators_124643/train/`; no new holdout.
- Six normal output channels with shared sign-ambiguous encoding, full-width
  scalar in input voxels and binary parent H/V output. Normals/width transform
  with geometry. Dense normal labels baked via shared native VC3D sampler and
  CP correction field, with original source provenance and explicit validity.
- Fast shared loading path, compact line targets, target-only global counts,
  one backward, raw/EMA metrics and checkpoint support. Old direction/presence
  output positions unchanged; auxiliary output explicit. No auxiliary volume
  exporter added; conditioned decoder explicitly unsupported for these heads.
- Fresh training instead of resume; refinement disabled (all prior refinement
  weights removed from these configs, inactive). Embeddings remain disabled.
- GPU0–5 only through existing manager; no GPU6–7 use. Prior limit2M updates,
  logical batch128, target microbatch16, LR0.01,100-step warmup retained.
- New requested trace gate: raw globally aggregated dense-test mean angle must
  first be strictly below10deg with valid supervision; latch persists through
  worsening, checkpoint resume and cooperative handoff. Dense/EMA/cross-section
  tests stay active. Same trace cadence once unlocked; no protocol change.

## Unrequested changes, now removed/restored by explicit instruction

- I invented `split_by_parent` and `annotation-prepare`, holding out16 pieces
  and reducing training to76. Both implementation and split-only regression test,
  runner settings and documentation instructions removed. Derived
  `2026-10-05_reviewed_annotations_smoke` deleted only after verifying each of its
  92 files against the intact original collection's SHA256.
- I substituted those16 pieces for the existing32-piece failure set, chose512
  validation patches and disabled resident traces/cross-section visualization.
  These were not authorized. Both annotation configs now copy exact original
  test datasets and all test-related settings from the frozen baseline:
  test_control_points=0, interval100, cross-section interval100, resident reviewed
  trace interval100 and original template/manifests/paths. Original test JSONs
  were never changed.
- The wrong long run was stopped by the previously issued stop command;
  PID3980909 is gone. Its logs/checkpoints and the initial smoke artifacts are
  historical evidence, not active configs. Nothing restarted.

## Agent-selected choices and subsequent explicit reversions

| Setting | Previous production | Current candidate | Verdict |
|---|---|---|---|
| prefetch_sampler_workers |8|8|Restored on user instruction|
| prefetch_workers |64|64|Restored on user instruction|
| volume_cache_memory_mib |null (512MiB default)|null|Restored; removed unrequested64GiB-per-process cap|
| volume_io_threads |unset|unset|Removed explicit override|
| volume_decode_threads |unset|unset|Removed explicit override|
| volume_sample_threads |unset|unset|Removed explicit override|
| scalar_log_interval |100|100|Unrequested change to10 reverted on user instruction|
| normal/width/HV weights |absent|10/0.01/1|User-selected replacement of initial agent defaults1/0.1/1|
| auxiliary losses/support |absent|normal uses direction-style projection-weighted MSE across Gaussian tube, divided by6*sum(falloff); width/HV remain on line with global element means|Corrected on user instruction: normal was wrongly centerline-only and lacked projection weights; normal logging now unweighted|

The new auxiliary losses contribute to total loss, so the existing raw/EMA
best-by-total-loss selection would include them. That follows my implementation;
it was not a separately requested best-selection policy. Original direction and
presence weights, Gaussian/sideband settings, model backbone/output resolution,
augmentations, EMA decays, optimizer settings, sample/kept snapshot cadence,
sample visualization cadence and production pipeline128 workers/2048 slots are
unchanged from the baseline. Run name changes for identification. Training uses
baked corrected normals instead of live Lasagna, as requested. The short config
still has its intentionally small batch/step limits; its worker/cache settings now
also match production. It too no longer uses
the invented test split.

## Test-label compatibility — resolved by requested baked copy

The user requested baking the existing test set with normal labels and full-parent
H/V. Both configs now use
`/ephemeral/hendrik/fiber3d_train/2026-10-05_reviewed_failure_union_baked_v1`.
All32 pieces retain their geometry, widths, corrections and selection. All71,165
normal labels are valid; H/V is23H/9V. One Paris4 piece changes H to its parent's V.
Original files are unchanged. No test loss is disabled. Test settings differ from
the frozen baseline only in the new data/manifest paths (trace split_root included).
Training remains stopped; all other previously agreed weights/settings remain.

## Verification

-62 CPU tests passed,1 optionalGPU test skipped across gate/checkpoint/config,
  reviewed-trace selection/failure, collector, single-backward and annotation
  tests. No GPU job launched.
-Exact test-config equality checked against the frozen baseline for BOTH configs;
  training count92 and test count32; gate strict10/NaN/missing support/latching
  and checkpoint roundtrip tested. `git diff --check` clean.
-Removed split dataset absent; original92 file hashes verified. No changes to
  other datasets, historical run artifacts or unrelated working-tree files.
