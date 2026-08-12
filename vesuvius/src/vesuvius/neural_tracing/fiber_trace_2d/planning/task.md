# Task: reduce fiber-anchor NMS radii

Reduce duplicate suppression to a transverse radius of 2 and a longitudinal
radius of 1 stored fiber-prediction voxels.

The transverse NMS radius must no longer reuse the refinement local-window
radius. Refinement retains its current default one-cell-side window; only NMS
becomes narrower. The fiberlet format remains experimental, so update its
strict schema directly without compatibility or repair behavior.
