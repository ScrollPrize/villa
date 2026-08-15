# Task: preserve control-point bends in generated line ribbons

Build generated line-annotation ribbons directly through every non-duplicate
control point. Subdivide each control-point segment independently so its
interval length is as close as possible to the configured target spacing.

- Every geometric control-point bend must be a ribbon support column.
- Intermediate supports must lie on their control-point segment.
- Strip/original-position mapping must remain exact with nonuniform support
  arclengths.
- Consecutive duplicate points must retain canonical mapping without creating
  zero-width ribbon quads.
