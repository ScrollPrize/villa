# Task log

- User follow-up: set new S1a test_interval and checkpoint_interval to100;
  five-scroll config and kept_snapshot_interval10000 unchanged. Updated contract
  test and documentation to record this deliberate difference.
- Follow-up validation: approved CPU caller run20260912_062413_349604,
  17passed/2CUDA skips in18.02s; existing NML branch warnings unchanged.

- 2026-09-12: New S1a-only config requested. Existing S1a train source is NML with
  inverse affine into Paris4 scan; held-out tests are JSON, not NML. Use those
  exact sources, current collection settings and LR0.01. No running job changes.
- Independent plan review approved by review_dense_sampling. New config is
  train_s1a_nml_16x8.json; source NML transform preserved exactly, no transform
  added to held-out JSON. Local files:8NML plus1held-out JSON.
- Approved run_fiber3d_cooperative_tests, arc-sampling suite:
  20260912_062018_129264,17passed/2CUDA-onlyskips in19.50s. Exact config equality
  excluding sources/run name verified, local annotation parser audit passed.
  Existing parser warns/skips18branched NML components, unchanged from baseline.
  No GPU tests warranted: config-only data selection with unchanged model/augs
  and target code already CUDA-validated. No training launched. Diff check clean.
