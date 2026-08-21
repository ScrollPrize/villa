# Task: persistent equal-arc fiberlet beam search

Replace fiberlet replay's per-fiberlet collapse to one locally selected route
with a persistent beam of up to 16 route histories. Beam comparison and pruning
must use absolute traveled arc length from the original seed of the current
uninterrupted trace segment.

Use separate base-voxel distances for the beam-front step and lookahead. At one
iteration, a beam may finish zero, one, or multiple fiberlets depending on its
current sub-fiberlet position and the lengths of the following fiberlets. Beam
fronts and lookahead endpoints may therefore lie inside fiberlets.

Before pruning, every candidate must cover the same total distance from the
segment seed, including the same lookahead distance. Keep the best 16 beam
prefixes instead of selecting one fiberlet and restarting the search. Select a
single route only when the uninterrupted trace terminates or fails and is
reseeded.

Preserve the existing on-demand anchor/fiberlet caches and active float or
quantized graph cost view. This search change must not regenerate cache data or
change cache identity.
