# Refinement direction-weight ablation — 2026-09-16

- Config-only change refinement.direction_weight1->0.1; normal0.1/tangent10,
  training direction10, LR0.01, original non-refinement11800 resume unchanged.
- Independent review approved. Approved run_fiber3d_cooperative_tests:
  23 passed,2 GPU-gated skips in9.11s; output20260916_062115_299762.
  No GPU regression rerun for this weight-only edit; prior solver CUDA tests passed.
- Approved run_fiber3d_experiment production-start launched root1231471:
  /ephemeral/hendrik/fiber3d_train2/runs/.launches/5scroll_json_eligible_refinement_128_sd1_16x8_ema-20260916_062132/train.log
- Prior trial remains stopped. New run uses manager allocation; no unmanaged work.
