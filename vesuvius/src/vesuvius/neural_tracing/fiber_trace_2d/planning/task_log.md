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
