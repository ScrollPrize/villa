# Task: confidence-weighted winding evidence refinement

Extend the current winding-BP hyperparameter search with neutral-by-default
variants for evidence confidence and sign handling:

- after choosing the dominant parallel/perpendicular hypothesis, optionally
  remap its score from `[0.5, 1]` to `[0, 1]` with linear and cosine variants;
- optionally weight winding evidence by Lasagna-normal alignment, with
  linear-in-angle and cosine/dot-product variants;
- optionally replace enabled hard winding-sign rejection with a finite
  sign-infringement cost; and
- test the new controls individually and in combinations against the fixed
  baseline, then refine the combined hyperparameters.

Variants that do not improve immediately must remain available but disabled or
behaviorally neutral by default.

Follow-up: promote the selected fixed-crop result to the standard defaults and
commit the completed implementation. The promoted row uses both sign classes,
finite sign cost `44`, winding Defect cost `100`, and orientation BP
temperature `1.25`; decision confidence remains `legacy` and normal confidence
remains `none`. The old hard-sign behavior must remain explicitly selectable.
