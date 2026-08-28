# Task: retain the main BP constraint component

For `vc_fiber_trace_chunk direction-ablation` BP runs:

- remove disconnected BP constraint subgraphs before solving and retain only
  the largest main component;
- keep component selection independent of optional reference fibers;
- move the reference/reference constraint tables and the reference-to-BP
  winding benchmark to the end of CLI output.
- exclude final Mixed/Defect or otherwise winding-invalid BP pieces from both
  reference offset calibration and benchmark totals.
