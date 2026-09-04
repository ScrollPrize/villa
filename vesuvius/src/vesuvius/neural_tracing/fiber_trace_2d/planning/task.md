# Task: benchmark experiment-step plots

Change the fiber benchmark plots to place every recorded experiment at one
integer x-axis step while preserving stable chronological ordering. Label only
strict best-so-far measured results, rotate those labels 30 degrees upward and
left, give every other result its own distinct marker and named legend entry,
and place the legend below the plot.

Correct the reference replay reliability metric to mean segment length divided
by total tested length: `100 / (failures + 1)`, so only zero failures yields
100 percent.

Use consistent labels for the same method across every benchmark plot. Only
append the BP stage where applicable; do not reorder or otherwise rename the
underlying method label.
