# Task: direct winding-factor weight search

Add independent multipliers for the five canonical winding-factor classes:
`perp_0.5`, `perp_1.5+`, `parallel_0`, `parallel_1`, and `parallel_2+`.
Allow a fixed tuple to be run normally and an exhaustive direct grid search to
reuse the already extracted BP graph, topology, fixed orientation prepass, and
reference cross-constraints. Report and rank each setting from the calibrated
reference-fiber winding benchmark so robust next-winding perpendicular and
same-winding parallel evidence can be tested at larger relative weights.

Extend the search with multiplicative coordinate descent. Starting from an
explicit tuple, evaluate each single-coordinate `/2` and `*2` neighbor, move to
the best strict improvement, and repeat until the current tuple is a local
optimum (minimum benchmark error) under that neighborhood.

Promote the resulting `8,1,2,2,1` tuple to the standard winding-factor default
for CLI and shared-library runs, while retaining explicit override and search
behavior.
