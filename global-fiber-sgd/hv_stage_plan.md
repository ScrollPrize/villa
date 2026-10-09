# Stage2 whole-fiber H/V and rejection

Cache per-original-fiber arc-weighted 2(tangent dot local umbilicus-axis tangent)^2-1
in native extraction as optional feature sidecar. Reuse native AFV loader/frame
check and add feature-only backfill for existing data (no constraint rerun). Sample
segments at <=16basevoxels, local axis secant across16zvoxels; document endpoints.
Store IDs, lengths, score, AFV UUID and umbilicus provenance once. No splitting.

Reuse completed stage1 penalty0.1 inward solution (2402excluded fibers). Stage2
loads all parallel/perpendicular events, fixes those exclusions, learns H/V and
additional broken logits. Fixed event normalization, parallel same/perp opposite,
very weak axial prior. Compare uniform vs half-uniform/half normalized-active-degree
rejection costs at penalties0.1/1.0, with and without late three-state entropy
ramp (start50%, reach0.01at80%).10000Adamsteps seed0 lr0.05, one managed GPU4GiB.
Ternary argmax hard readout; report H/V/broken, event retention and violations per
relation, ambiguity and elapsed time. More constraints can reveal family-switching
fibers, but rejection is not ground-truth proof of a crossover.

Shared manager launch helper must be extracted from stage1, not duplicated. Native
tests on alignment; Python scalar/gradient/entropy/degree/frozen exclusion tests;
independent saved-output metrics. Approved build and crop-benchmark runners only.
Plan review before implementation. Specs additive cached features/stage2, no edits
to original AFV/constraint DB or previous solution. Docs log/summary commands and
limits. Outputs under data/hv_stage_v1. Preserve unrelated dirty files.

## Reviewed mathematical contract

Active graph removes stage1broken endpoints before optimization; these IDs have
no trainable parameters or prior/entropy costs. p=sigmoid(h),b=sigmoid(v),
P=(H,V,B)=((1-b)(1-p),(1-b)p,b). Pair loss is mean over fixed active events of
(1-bi)(1-bj)(1-r*(2pi-1)*(2pj-1))/2, r=+1parallel/-1perpendicular.
Ties omitted/reported. Axial prior .001mean((1-b)(1-a*(2p-1))/2).
Rejection mean(c*b),c=penalty or penalty*(.5+.5degree/meanDegree), degree fixed on
active graph including isolated fibers. Entropy +weight*mean(-sum(PlogP)), ramp
0until50%, linear to full at80%. Argmax(H,V,B), ties first index. Report inherited
vs new broken, retained arc, per-relation errors and pre-stage1retention. All-broken
has unavailable retained-error metric. Source AFV/umbilicus hashes included.
Native feature-only backfill and regular extraction use the same function before
constraint search; existing database remains immutable. Axis uses16zvoxel secant
of frame-checked dense umbilicus centers, clamped to valid scan z at endpoints.
