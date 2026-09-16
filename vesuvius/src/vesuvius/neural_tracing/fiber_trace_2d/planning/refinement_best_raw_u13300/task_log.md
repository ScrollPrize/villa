# Best raw refinement evaluation

User requests best raw snapshot from run185710. Using best.pt, not best_ema or current. Same original fixedGT cohort/protocol; no embeddings or rendering. Freeze pending; previous task archived under refinement_u24200.

Frozen rawu13300/1702400samples, test/loss_total0.24048426168516648. SHAe1cbb432388598f59d29c88e31eaf6e1db4c710bc53e7cc396963890eb16fcef. Independent reviewer verified payload model tensors loaded via payload[model]; included EMA bank ignored. Config/protocol review approved, prepare passed; full launched via approved run_classic_fiber_evaluation.py full --config eval_refinement_185710_best_raw_off_20260915_v1.json (absolute paths as previous evaluations). No model or numerical code changes.

Completed exit0,180directions/50fibers/6867.627253directedmm.126filtered restarts,54.075805mean,9.653438median,0.218733P05,89.519676P95. Scroll failures59/3/6/0/58. Best EMA999u16500 remains103/66.034877; raw+23restarts, mainly0139+14 andParis4+8. Different runs mean not controlled raw-vsEMA ablation.

Inference/coordinator345.703s/native299.659s overlapping; all8GPUs exclusively allocated managerfb8d2f6c-21ea-45ee-844e-164a540c99c5. Fresh inference, source cache reused. Shared complete_report validates totals/distance. Full historical/per-scroll table and exact command/provenance saved docs/benchmarks/refinement_185710_best_raw_u13300_20260915.{md,json}. No rendering requested/performed.
