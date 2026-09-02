# Current task: iterative ordered-winding offender removal

Extend the experimental sign-only continuous winding ordering with an optional
diagnostic that repeatedly removes the worst offending original traced fiber.

- Score a fiber by the percentage of its incident admitted sign constraints
  violated by the current continuous ordering.
- Remove the complete source fiber, including every split piece.
- Re-solve the continuous ordering after each removal.
- Continue until the remaining ordering has no violated sign constraint.
- Report every removal with the fiber's incident/violated counts and the global
  ordering violations before and after the re-solve.
- Preserve existing behavior unless the diagnostic is explicitly requested.
