# Task

Add an independent Ceres least-squares mode for crop-fiber H/V, Defect, and
winding inference. All three quantities must remain continuous: horizontalness
and activity are bounded to `[0,1]`, while winding is real-valued.

The Ceres path must reuse the existing extracted constraint graph, scale-first
canonical targets, confidence attenuation, winding class weights, sign
weights, Defect cost, piece-break cost, and thread default. It must not alter
the existing BP solvers or their defaults.

When reference fibers are supplied, benchmark the continuous solution by
solving the same least-squares model once per reference source. Only that
reference source is free; every connected crop-piece source state is fixed to
the main Ceres result. Report the raw per-reference solution and a single
global half-step sign/offset calibration against the filename-ordered virtual
reference ladder.
