# H direction breakage implementation plan

## Scope and model

Reuse cached whole-H tangents, shared sign pair loss and managed GPU launcher.
Add a sigmoid broken variable per eligible H fiber. Gate pair loss by both keeps;
normalize by the ORIGINAL total event weight. Gate weak prior by keep; charge a
positive per-fiber breakage cost (uniform mean, swept), and anneal joint ternary
entropy. Warm-start the saved sign-only solution, without freezing signs.
No per-event deletion, geometry cuts, new H/V decisions or global U/V solve.

Sweep breakage weights 0.1, 0.03, 0.01, 0.003 (extend lower only if necessary).
After v1 initialization audit, preserve exact warm logits (no confidence reset)
and refine penalty sweep to0.1/0.07/0.05/0.03 in v2. v1 retained as diagnostic only.
Report retained conflicts, disabled fibers/arc, retained events and uncertain
states. Prefer zero retained conflicts with least rejected arc/fibers; never hide
remaining conflicts or apply unreported post-hoc pruning. Keep old run intact.
Report disagreement against both original and retained denominators.

## Visualization

Reuse original AFV reader, spatial_subset, OBJ segment writer and arrow_mesh.
Export all newly disabled curves in red; select ~1000 long, spatially distributed
surviving H curves in blue. Separate OBJ contains these same curves plus tangent
arrow meshes oriented by saved CCW signs (not outward-normal display convention).
Include umbilicus, source IDs, selection criteria, signs and file hashes in index.
Earlier-stage rejections remain distinct from new H direction rejections.

## Tests and review

Independent plan/code review required. Small CPU synthetic tests: gate removes
incident losses, positive penalty preserves isolated consistent fibers, all-keep
equivalence, global gauge invariance, prior gating, conflicting graph rejection.
Regression previous stage tests. Audit full saved signs/labels, original identities,
conflicts and exclusion masks independently; validate OBJ indices/counts/arrows.
Use existing approved evaluation runner and GPU manager, no installs/new approvals.

## Spec update

Stage3 gains optional broken state and combined labels; disabled H direction sign
is zero, NOT a valid direction. Conditional sign probabilities remain distinguishable
from joint kept-forward/kept-reverse/broken probabilities. Earlier stages immutable.

## Docs updates

Update h_direction.md, results, README, task/status/log and changelog with penalty
sweep, selected parameters, artifacts, run commands and limitations. Archive prior
sign-only results unchanged. Global assignment stays paused.
