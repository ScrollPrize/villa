# Historical observation and EMA task log

- Reproduction closed in cffe3cde9; user accepted evaluation-only explanation.
  New task is regular2M-step training, not further ablations.
- Plan: fixed EMA0.99/.999, both test definitions, unchanged current metrics/raw
  training, full disk/RAM state. Historical total-loss compatibility requires
  source audit (reproduction did not reproduce archived total loss).
- Independent review approved: preserve optimizer parameter identity, restore raw
  state before every manager yield, include EMA in RAM replica verification,
  distinguish test sources, account for skip/update/calibration semantics.
  Prefer evaluating raw+EMA from one loaded batch to avoid repeated reads.
- Implemented fixed-decay FP32 EMA with storage-preserving eval swaps, raw
  resume/RAM state and separate inference exports; each EMA best uses current
  dense loss. Shared queue now supports historical source transitions without
  duplicate model-track reads. Protocol unchanged.
- Historical presence loss restores old central patch restriction. Deviation:
  modern native source loading has no historical availability mask; old hole
  handling is not reproduced. Explicit startup/config/docs warning, no masks
  or other supervision changes to training.
- First CPU run35passed/2failed: stale tests expected14-channel loss support,
  disabled by prior target-only-normalization task. Updated them to assert the
  explicit unsupported error. Final CPU run20260912_110855_949178:
  39passed/3CUDA-skipped65.85s, including raw/EMA snapshot roundtrip, RNG/raw
  metric isolation and historical broker transition/resume.
- GPU suite20260912_110553_431829 through approved manager-exclusive launcher:
  one-H100,B4,128cube,BF16:10measured paired updates per case after6warmups;
  raw98.947ms vs twoEMA99.586ms (+0.65%), EMA CUDA0.617ms, shadows440128056bytes.
  Two-GPU→one-GPU train/test-interruption state tests2passed45.85s (EMA included).
- Real-data8H100 smoke through approved experiment start:
  reproduction/ema_smoke-20260912_110704,3updates384samples,exit0,85.42s.
  All6evaluation variants:initial14.30s,subsequent2.817/2.982/2.906s; emitted
  current/best/numbered snapshots for raw and bothEMA tracks. No training
  architecture/augmentation change. Extra evaluation cost reported separately.
- Snapshot/TB audit of smoke passed: both exports exactly equal their raw
  checkpoint EMA states,3updates each; all12requested raw/EMA angle/loss tags
  plus existing detailed current tags present. Implementation committed c4ef954c4.
- Approved production-start launched root1359679 with automatic manager handling.
  Run: /ephemeral/hendrik/fiber3d_train2/runs/
  s1a_nml_128_sd1_16x8_ema_20260912_111221.
  Log: runs/.launches/s1a_nml_128_sd1_16x8_ema-20260912_111218/train.log.
  Startup measured real microbatches1/2/4/8/16, retained all15updates, saved
  profile f5673d079f07bbaa38a4467fc475fe11b36645083b3c4308bbafb87fc891bec7.json.
  Adaptive task26f08c32-91c1-4c6e-bd5b-e0dca525db5e running on8GPUs/128CPU,
  calibrated choice B8/GPU with2microbatches/global128 (config cap16).
- Verified update400/51200samples at LR.01, all6eval variants. Warm managed tests
  5.067/5.058/5.131s at updates200/300/400; recent fw_bw_step~0.5s includes
  two physical microbatches, not isolated GPU-kernel timing. Run stays active
  toward2000000updates; no further reproduction trials or training changes.
