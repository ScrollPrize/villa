# Plan

## Follow-up: cosine histograms
Use saved results only (CPU, no GPU interruption). Plot equal-scroll aggregate
and per-scroll positive/negative cosine histograms. Positives are all distinct
GT point pairs in each saved blended window; negatives are the saved unblended
same-tile GT/off-GT pairs. Label this distinction and do not manufacture
matched-anchor gap distributions. Show all negatives and presence>=0.5 subset,
normalize each distribution separately, report counts and quantiles. Add small
CPU tests for pair construction and histogram normalization. Save PNG, SVG,
numeric summary and a short reproducible documentation note. No loss changes.

1. Use existing approved embedding-consistency runner, current raw snapshot of the contrastive run, new output directory; retain other settings.
2. Independently review this plan. Freeze snapshot via existing evaluator and record exact step, samples and SHA. Exclusive manager allocation pauses training.
3. Compare same-point overlap, along-GT within/blended/boundary cosines, high-presence off-GT similarity and cancellation against previous report.
4. Record results in a new report and brief task log; verify manager release and training continuation. No training or tracer changes.

## Spec update
None. Preserve ten-window pilot, conservative >80-base-voxel off-GT controls (not exact training ellipse distribution), FP32 vector blending and hypothetical storage boundaries (no serialization test).

## Docs updates
New comparison report with provenance and limitations; retain previous report.

## Validation and changelog
Existing runner unchanged; verify all ten cases finish and frozen snapshot identity. Document results, not a claimed full 50-fiber trace benchmark or matched-age training ablation.
