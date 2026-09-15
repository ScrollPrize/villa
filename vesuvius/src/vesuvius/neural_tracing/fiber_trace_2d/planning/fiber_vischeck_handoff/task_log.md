# EMA999 u10200 visualization

Same frozen snapshot from preceding full eval; no new snapshot selection. Reuse inference cache and aligned renderer. Previous evaluation log archived in docs/benchmarks/eligible_ema999_u10200_task_log_20260914.md.

User additionally requested worst restarts/mm per scroll. Ranked all50paired fibers from complete saved traces, aggregating all eligible runs/both directions; denominator is total directed mm. Results saved in docs/benchmarks/eligible_ema999_u10200_worst_fibers_20260914.md with fiber IDs/cases/image links. All180replay payloads exactly match prior evaluation; case60 visually inspected with aligned endpoint sections.

Prepared/validated through approved runner. Config comparison and independent review confirm only output root,render=true,manager label differ. SHA4753ee9bee448fe3d5be1914aefc88ca1cc33799cc69195408ea6a8bc1be7d1b; model cache keye01285ed6edb280b1aa5832990e359c965467fd82049caca9037e58f941cd4bc,existing namespace unchanged.
Full launched with configs/eval_eligible_ema999_u10200_vis_20260914_v1.json; output /ephemeral/hendrik/fiber3d_train2/evaluations/eligible_ema999_u10200_vis_20260914_v1. All8GPU manager exclusive priority100. No implementation changes; tested shared renderer reused.

User requested STOP visualization. Sent SIGTERM to exact managed launcher1912366; session exited130 and no matching launcher/child remains. Completed images/cache preserved; no further render started. Final complete gallery/index intentionally not generated for partial run.

Final retained render count145/180. User requested commit of current state; checkpointing changes by subproject (Lasagna cache/publication, native replay/renderer, Vesuvius evaluation/configs/results/docs). External model/data/cache/images remain outside Git; repository includes provenance and result reports.

Pre-commit regression rerun:23native cases and138Python tests passed(47.80s); git diff --check clean.
User asked about restart filtering case42: metric uses reference/GT arc gap to previous actual failure, threshold80basevoxels. Case42 has8visual events but3reported(R0,R6,R7); R1-R5 suppressed(8.564,4.848,8.872,2.280,8.000vx gaps). R6 gap179.642vx,R7gap2355.792vx. Fiber total10 is cases42/43/44/45=3/4/2/1, not10in case42.
Read-only image diagnosis: case42 traced top/side are not duplicated. Components1-5 CT mean absolute RGB differences41-47/255; presence54-66/255. Shared LineViewBuilder constructs side from normalized(meshNormal cross tangent), with meshNormal projected perpendicular to tangent, hence orthogonal planes by construction. Failure local_normal_valid=true for all8events; this rules out recorded missing/invalid sample fallback, not physically incorrect normal orientation. No normal-volume corruption proven, no code/data modifications for this question.
