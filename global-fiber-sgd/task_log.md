# Corrected fresh/reference solves and GT constraint score

Committed corrected GT implementation/results e2ba3415a before new work.
Plan gt_constraint_score_plan.md independently reviewed; configs approved.
Matched sixGPU exclusive managed suite, canonical subset2199pieces/24771events,
same best weights and10080updates, shared-phase correction enabled; fresh vs
reference winding+width initialization. GT solution v3 frozen and not supervised.
Added shared hard_assignment helper for disabled nonbroken argmax placement.
Native cross extraction uses full original curves, filters evaluated piece arcs
after extraction. GT endpoint normal signs are intentionally used (not bilateral
normal agreement with tested fibers); tangent/invalid exclusions reported.
29CPUtests passed. Distributed sixGPU gradient and coordinate preflights passed.
Both solve suite and CPU cross-constraint preparation launched via approved runner.

Completed fresh173.23s/reference173.99s.30CPUtests passed. Final reviewed score
3342/6735=49.6214% fresh,3709/6735=55.0705% reference; both additive offset-2.
Includes1316/351disabled-piece events respectively.5970parallel/765perpendicular;
1119tangent connectors and7outside-span cross events excluded identically.
Review caught event-dependent normal sweep gauge: exported exact original GT
sweep (native verifies old endpoints), then aligned new endpoint normal signs
against it using shared CurveBank. Zero actual gauge corrections on this set.
Scoring exact repeat SHA256 c0662bac95b67a106c2f23d84720092eedf40c1793c112d5c57bd4f7b053e7bf.
Full results/commands/limitations in gt_constraint_score_results.md; all per-event
votes in data/gt_constraint_score_v1. No tasks left running, suite released GPUs.
