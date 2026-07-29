# Manifest-Scale Native Fiber Metric

Adjust the VC C++ fiber metric/tracer command so it uses the precomputed fiber
inference manifest scale as the tracer working scale.

Desired behavior:

- `vc_fiber_trace_metric` infers the working-to-base scale from the fiber
  inference manifest's persisted prediction channels;
- the manifest inference/output size and scale define the tracer coordinate
  system;
- the JSON fiber coordinates are assumed to already be in the base coordinate
  system of that manifest;
- `step_voxels`, candidate tracing, and restart thresholds remain expressed in
  inferred working-grid voxels;
- local and remote manifests continue to use the shared Lasagna opener and
  remote cache behavior;
- optional `--normal-manifest` sampling uses the same inferred working scale.

Out of scope for this task:

- do not add a command-line argument for scaling the fiber JSON into the
  manifest base coordinate system;
- do not change GUI segment tracing scale handling in this task unless needed
  by the metric command;
- do not change the generic Lasagna dataset runtime-scale API used by other
  VC tools.
