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

## LP-relaxation tightening follow-up

Retain the same binary objective, but add an explicit diagnostic LP relaxation
whose three per-piece values remain continuous on `[0,1]`. Represent H/V and
parity edge differences consistently as actual gated XOR values and add
triangle cycle inequalities so edges cannot independently select mutually
inconsistent cheapest relations. Remove the global H/V and parity flip gauges
per connected component, then compare the relaxed solution on centered 256 and
1024 base-voxel crops.

Visualize the raw relaxation by splitting H/V and parity at `0.5` and splitting
active/broken at the mean active value, while keeping these threshold layers
explicitly distinct from the optimized MILP labels.

Expose HiGHS' LP algorithm and parallel-mode selection explicitly, then compare
the tightened centered 384-base-voxel relaxation with parallel automatic
selection and the HiPO solver. Keep the existing HiGHS automatic selection as
the default and do not change the MILP path or mathematical model.
