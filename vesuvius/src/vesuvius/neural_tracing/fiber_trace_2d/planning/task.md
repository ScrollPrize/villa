# Task: weighted reference winding diagnostics

Before the existing per-reference winding accuracy table, print an additional
diagnostic that explains how each constraint group scores the calibrated true
winding and which winding that group alone prefers.

Split evidence into five groups per reference fiber:

- perpendicular next-winding (`0.5`)
- perpendicular larger-distance (`1.5+`)
- parallel same-winding (`0`)
- parallel one-winding (`1`)
- parallel two-or-more-windings (`2+`)

Use the winding solver's admitted dominant-hypothesis winding objective and report
both aggregate and normalized disagreement so constraint-count majorities can
be distinguished from the weighted objective. Group rows and the displayed
per-reference estimate must use the same inference. Contradictory hard signed
ordering constraints are ranked by fewest violations before finite energy.
