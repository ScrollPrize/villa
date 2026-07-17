# Native 3D Trace2CP All-Pairs Direction Product

Update native 3D Trace2CP candidate scoring to consider all four relevant
directions:

- last step direction: last accepted point to current point;
- last sampled direction: model direction sampled at the current point;
- current candidate step direction: current point to candidate point;
- candidate sampled direction: model direction sampled at the candidate point.

The new scoring mode should multiply all pairwise aligned dot products so any
individual outlier direction penalizes the candidate. Put this behind a switch,
but enable it by default for testing.
