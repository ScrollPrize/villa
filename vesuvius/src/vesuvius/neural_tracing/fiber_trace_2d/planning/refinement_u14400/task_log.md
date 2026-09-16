# Refinement EMA999 evaluation — 2026-09-15

Latest/current requested, not best. Reused paired no-embedding workload, disabled embedding emission. Full original locked50-fiber cohort, no render. Prior merge task archived in commit0577d1d8f.

Frozen update14400 / 1843200 samples, SHA256 e5c6d57b534e0b22c4c004be483a47058deaf4f69cb1ae17b2292949b03f976c. Output: /ephemeral/hendrik/fiber3d_train2/evaluations/refinement_ema999_current_off_20260915_v1. Independent review approved; prepare and validate-full passed. Original fixed GT is deliberately retained; dense refinement test loss uses evolving targets and is not the metric here.

Reusable command: `/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py MODE --config /home/hendrik/villa2/vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/configs/eval_refinement_ema999_current_off_20260915_v1.json`, with MODE freeze-snapshot, prepare, validate-full, then full. Full launched via exclusive manager, devices0-7, priority100. No code changes needed.

Completed exit0, all50fibers/90eligible spans/180directions,6867.627253directed mm. Manager task46ae18f0-5c75-4820-a358-e1c5518e4b60 received all8GPUs. New inference cache; source fetches reused. Inference wall343.647s, overlapping native replay299.814s (not additive, excludes outer setup/cleanup).

111filtered restarts, mean61.318100mm, median10.511990mm,P05 0.210585mm,P95 109.668420mm. Compared previous non-embedding EMA999u11800:113restarts,60.242344mm mean; Classic292restarts23.439001mm. Per-scroll restarts0139=52,0332=2,1299=5,1451=0,Paris4=52. Small overall improvement, not uniformly better and no statistical significance claim.

Saved full historical comparison, per-scroll metrics and immutable provenance in docs/benchmarks/refinement_ema999_u14400_20260915.{md,json}. Metrics recomputed through shared complete_report, which verifies completion, event totals and directed reference distance. No rendering requested or performed; no refinement archives used as ground truth.

Independent final review recomputed all metrics exactly; approved. Report JSON/cohort distance checks and git diff --check passed; no incomplete markers. Additional2600training updates versus u11800 confound causal attribution to refinement alone.

User requested visualization follow-up. New workload eval_refinement_ema999_u14400_vis_20260915_v1.json keeps immutable checkpoint/cache and protocol; only output directory, render=true and manager label differ. Existing renderer generates combined CT/presence reference/traced strips, aligned cross sections and restart metadata. Uses manager for optional additional coverage, not a forced fresh inference run.
