# Task: Shift/Scale Augmentation Order

Fix the shift/scale augmentation composition so shift happens in scaled output
space, not in original source space. If a patch is strongly shifted and then
zoomed, the control point should not be scaled away from the intended shifted
location. The expected implementation is to move shift after scale in the
coordinate transform while keeping image sampling, line-coordinate generation,
and control-point coordinate generation aligned.
