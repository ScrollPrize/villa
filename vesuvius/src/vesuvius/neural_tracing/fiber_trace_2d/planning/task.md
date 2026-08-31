# Task: perpendicular parallel-fiber correspondence search

Replace the closest-distance phase refinement used while walking a parallel
fiber pair with a small, deterministic two-dimensional arc-offset search.

For each paired step, independently vary the advance on both fibers around the
target step and minimize:

- deviation of both advances from the target step; and
- non-perpendicularity of the connector to both local fiber tangents.

Do not add a local regression/Gauss-Newton refinement and do not minimize
connector length. Keep the existing closest sampled pair as the initial seed.
Use a grid resolution of 5% of the target step and allow independent corrections
up to 25% of the target step on each fiber.

Preserve the grid as an explicit optional CLI variant and restore the original
closest-distance phase walk as the default before real-data experimentation.
