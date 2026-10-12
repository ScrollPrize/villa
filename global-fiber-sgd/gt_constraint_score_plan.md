# Corrected fresh/reference benchmark and reviewed-fiber score

Commit shared-phase fix first (e2ba3415a). Run canonical subset IDs14–20,
Z30k–50k, unchanged best weights and10080updates, fresh and spiral-reference
initialized (including reference widths), with bucket_shared_phase enabled.
Use existing sixGPU managed suite, no full corpus rerun.

## GT constraint metric
Freeze bbox_gt_solve_v3. Extract shared native local constraints on a combined
AFV containing test source curves and seven GT curves, then keep cross events
whose test endpoint falls inside an evaluated piece. Preserve original arcs,
never cut curves before extracting local evidence. Ignore test-test/GT-GT events.
Use oriented GT normals to sign connectors, not guessed tested-fiber normals.
Each cross event predicts GT coarse winding from the test piece's most-probable
nonbroken hypothesis plus the signed measured winding separation, with original
angle/along-fiber and H/V layer correction. Compare with fixed GT coarse winding.
Round required global offsets to nearest integer (explicit deterministic tie
rule); select one modal integer offset over all events and report exact-vote
agreement percentage. Also continuous residuals, support perGT, parallel/perp,
enabled/disabled counts and scores; no disabled-piece exclusion or refitting.

## Implementation and validation
Reuse AFV writer, native extractor/orientation, existing arc/angle geometry.
Extract reusable nonbroken-hypothesis readout rather than duplicating geometry.
Validate offset invariance, wrong votes, signs, half-layer correction, arcs at
piece boundaries and disabled argmax. Hash all source/config/solution artifacts.
Reference initialization is only an initialization; GT labels never enter solves.

## Spec/docs
Add reproducible CPU scoring runner and saved event-level votes. This is an
inferred-GT constraint-consistency metric, not proof of absolute winding truth.
Update task/status/log/results with both old spiral metrics and new percentages.
