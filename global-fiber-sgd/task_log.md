# Expanded GT and constraint-vote evaluation

Current follow-up: user confirmed shared phase must enter coarse comparisons.
Plan bbox_gt_phase_fix_plan.md independently approved. Training and diagnostic
coarse delta now add learned shared phase B-A to existing fixed shift (original
phase already included). Fine offsets/width excluded. Versioned opt-in true
in run_bbox_gt_v3.json; prior outputs/configs retained. Same seven whole fibers.

V3 complete128.33s/10080updates: all7/all46retained,0coarse and0physical sign
violations. Coarse distance mismatch0.206087 vs0.983133; final online loss0.214366
vs256.371918 (formula corrected). Zero violations from1000 onward.25tests passed,
independent implementation review approved; results bbox_gt_phase_fix_results.md.
Visualization uses existing TIFF/SVG/PDF renderer. Previous v2 preserved.

Previous results preserved in bbox_gt_results.md and data/bbox_gt_solve_v1.
Seven JSONs now present. Use fresh v2 artifacts and shared pipeline with whole,
non-rejectable fibers. Independent review requested for bbox_gt_eval_plan.md.
Asked once for comparison artifact identities and disabled-piece vote convention.
No tested solutions will be optimized or changed by evaluation.

Plan review approved rerun; metric must distinguish signs from exact winding
votes, and avoid hard_geometry's disabled-label fallback to final winding bucket.
First import found missing shape fields on files6/7; verified scan manifests and
added bounds validation with optional shape metadata (mismatch still fails).
12 tests passed. Native extraction and orientation complete (<0.25s each),40
perpendicular events, seven non-isolated fibers. Managed GPU queue waiting for
existing training allocation to release; no bypass or kill of other work.

Completed seven-fiber solve:127.48s/10080updates, all7/all46 retained. Final
bucket violations5/46 despite positional violations0/46. Shared phase moves
physical layout independently of bucket-only ordering. Not accepted as validated
GT yet. Results/provenance in bbox_gt_v2_results.md; visualization launched.
Cross-solution metric not implemented/run pending clarified artifact identities,
vote convention and resolution of this target discrepancy. No false accuracy claim.
