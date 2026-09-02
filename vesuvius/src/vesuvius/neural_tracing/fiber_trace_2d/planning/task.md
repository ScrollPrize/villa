# Current task: oracle construction of a correct winding working set

The sign-consistent conditioned inlier selector improves the 1024 reference
benchmark from 16/26 to 20/26 exact windings but is still insufficient. Build a
reference-supervised oracle pruning mode that finds a retained ordinary
fiber-piece set whose conditioned solve has no wrong identifiable reference
winding. References that lose all correct support must be reported as missing,
not retained under a known false label.

Reference sign constraints are authoritative. Winding-magnitude constraints
may be biased and may be weighted or disabled while ranking removals. Preserve
as much retained fiber arc as possible after first maximizing exact references
and eliminating wrong labels.
Re-solve after pruning rounds rather than assuming one conditioned assignment
remains valid. Report the trajectory, final retained geometry, direct
conditioned benchmark, and fresh reference-free stability benchmark.
