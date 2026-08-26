# Task: mixed-integer crop-fiber labeling

Discard measured crop-trace constraints whose normal-modulated winding
distance is greater than or equal to `1.5`, then solve a HiGHS MILP assigning
each fiber piece exactly one of five states:

- H/even;
- H/odd;
- V/even;
- V/odd;
- broken.

For an active pair, H/V mismatch costs `parallel_score` and H/V agreement
costs `1 - parallel_score`. Parity agreement costs `winding_distance`, while
parity mismatch costs `abs(1 - winding_distance)`. A broken piece disables all
incident pair terms and costs `0.5` times its incident constraint count.

Write the five resulting piece classes as separate OBJ polyline files.
