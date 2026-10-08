# Cross-section alignment

Prior log archived in docs/benchmarks/eligible_ema99_u5200_vis_task_log_20260914.md.
Cause: independent longest-offset-row arc calculations stretch top and side differently; sections use only top columns. No tracing/inference semantics change intended.

Independent plan/code review passed. Added single top-derived centerline longitudinal sample count reused for side. Existing native-group sampling density unchanged for legacy APIs; own native failure markers now use endpoint instead of ambiguous arc inversion (also fixes plateau matches).
Native regression reads actual cyan marker pixels below section centers for curved three-component ribbons, with reference-progress and plateau cases. 23 native tests pass. Initial synthetic plateau test needed matched reference XYZ updated consistently; corrected fixture.
Approved workspace build succeeds; approved inspection suite: 23 native cases and 138 Python tests passed (Python44.35s).
Full rerender started through run_classic_fiber_evaluation.py full with configs/eval_eligible_ema99_u5200_vis_20260914_v2.json. Output /ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema99_u5200_vis_20260914_v2; unchanged model SHA/namespace/trace policy. Manager task6ad22c12-5b3a-4208-b403-b80efe9d983f, all8GPUs exclusive priority100.

Completed exit0,416.814738s: all180 galleries/JPEGs,32,041 sections,50-fiber index/per_fiber.csv/json. Every component has equal top/side bounds and every section center equals its strip column. Real cases19/60 visually inspected; case60 JPEG pixel checks confirm R0-/R1- cyan bands at section center in all4traced modality/view rows.
All180 trace JSON documents exactly match v1; summary metrics unchanged:148 restarts,6867.627253mm,mean46.091458mm,median8.202461mm,P05.143040mm,P9588.437892mm.
Cache reused18,346warm chunks;939tiles inferred/96chunks published for newly touched visualization sampling coordinates, no cache reset. Prior gallery preserved. No deferred work; reference-vs-traced paths retain independent arc axes intentionally (different paths).
Durable record: docs/benchmarks/inspection_alignment_20260914.md.
