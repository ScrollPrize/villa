# V direction implementation plan

1. Commit H work and pinned recommendation before V changes (done3f5b443e8).
2. Generalize the existing H direction implementation, not duplicate it:
   select H or V family; identical constraints/loss/optimizer/sweep/exports.
   H uses circulation, V uses signed agreement with local positive-Z umbilicus
   tangent. Arc-weighted quadrature, same native L0 frame and tangent sampling.
3. Preserve recommended H labels/rejections/signs via hash-bound inherited
   solution. Normal/HV exclusions remain fixed. Only retained V fibers participate
   in new V–V pair terms. H and V have no direct tangent-equality constraints.
4. Run V sign-only baseline, then preserve fitted logits in breakage warm-start.
   Sweep .1/.07/.05/.03; extend lower only if conflicts remain. Fixed original
   event normalization, keep-gated weak prior, positive broken cost, joint entropy.
   Use approved runner and one managed GPU grant for the sweep.
5. Audit counts, retained arc/events, conflicts, previous-stage identity, exact
   up/down gauge negation, isolated/unrelated rejections and individual restoration.
   Prefer zero conflicts with maximal retained arc; no unreported pruning.
6. Export all newly disabled V curves plus1000long spatially spread survivors,
   normal view and separate UP tangent-arrow OBJ. Same established renderer.

## Spec update

Whole V fibers receive one sign relative to stored point order, positive means
forward. Canonical positive world convention is along positive-Z umbilicus axis,
not a hard per-point Z monotonicity constraint. Save explicit up/down aliases,
family-neutral positive/negative fields and combined H/V labels/broken mask.
Zero sign denotes non-family/excluded, not an orientation. Preserve H sign outputs
separately in V solution; H CW reversal does not reverse V up. No global layout.

## Tests and review

Independent review before implementation. Test axial prior on straight/curved axes,
reversing stored point order negates prior; undefined/zero-length and on-axis cases.
H regression bit-identical; family selectors, inheritance and aliases validated.
Existing normal/HV/H regression tests and full-data saved-result/OBJ audit.

## Docs updates

New v_direction.md/results/config/recommendation, current task/log/status and
changelog. Existing H documentation notes shared implementation with no H changes.
Keep full run manifests and source hashes; datasets/artifacts stay ignored data/.
