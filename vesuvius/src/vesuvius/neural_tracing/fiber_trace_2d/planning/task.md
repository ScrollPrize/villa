# Task: parallel-separate-winding labeling ablation

Rerun the centered 384-base-voxel crop LP relaxation while excluding measured
constraints classified as parallel with separate winding from the labeling
solve. Keep hard continuity constraints and retain all extracted constraint
visualizations. Report the excluded population and compare thresholded label
counts against the committed full-constraint baseline.

All generated visualization names must use only the crop size and individual
label, such as `384_perpendicular_same_winding.obj`, `384_h_even.obj`, and
`384_values.csv`.

Split perpendicular visualization links at winding distance `0.5`: same
winding owns `[0,0.5)`, while separate winding owns `[0.5,1.5)`.
