# Task

Get the skeleton backend's ports and refined lines close to the refined
ground truth, in particular at the worst interior deviation (case31, around
reference arc 1837, 57%).

User instruction (2026-09-19):

- Experiment with ways to construct the problem so that ports and line
  refinement land near the refined ground truth.
- The likely fix is to change how the problem is constructed rather than to
  tune weights: use the skeleton for **topology only**, and construct the
  route geometry and the connection points from the **prediction** attached to
  the skeleton, not from the skeleton's own geometry.
- Iterate until there is a good solution.
- Track task and progress under the `fiber_trace_2d/AGENTS.md` workflow.

## Why, from the measurements already taken

- The thresholded presence contains the reference to within 2.00 base voxels,
  at every threshold from 128 to 224.
- The skeleton degrades that to 17.62 for the nearest point and 46.18 for the
  best connected route, because the medial axis of two touching fibers runs
  between them rather than along either.
- Every refined line whose deviation exceeds 8 base voxels has its worst point
  in the outer 20% of the line, at a port. None has it in the middle.
- Ports sit a median of 7.52 base voxels from the reference at their skeleton
  position, before any port optimization.
- Disabling port optimization changes nothing (25.71 either way), and changing
  its Gaussian sigma from 6 to 2 changes nothing (25.71).
- An oracle over the full candidate network does find a good route: median
  2.85, p95 17.94, built from three long lines and the connections between
  them. So the topology is adequate and the geometry is not.

The conclusion these support: the skeleton's geometry is the defect, the
topology is not, and the port optimizer is not the lever.

## Outcome (2026-09-19)

The prediction was measured rather than assumed. The thresholded fibre section
is a ribbon: 1.3 base voxels across the sheet and 3.6 along it, and the
reference sits within 0.3 of the ribbon's centre across the sheet but up to
17.6 from it along the sheet. Presence places the fibre in one direction only,
so where two fibres merge the thinning voxels run between them and nothing
local says which side is the fibre being traced.

That choice is therefore left to the tracing. Each port gets the centre and
the two extremes of its section as sub-ports, each centred by a small local
presence step; every line variant and region connection between sub-ports is
refined from a path search through the thresholded prediction and kept; and
each sub-port carries its own cheapest continuation on either side, so the
tracer walks sub-port to sub-port with nothing chosen ahead of it. Case31
traces as one segment from start to end, maximum 21.17 to 8.69 base voxels,
p95 13.75 to 5.77.
