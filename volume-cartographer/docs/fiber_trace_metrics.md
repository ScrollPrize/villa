# Fiber replay error metrics

`vc/fiber_tracer/FiberReplayMetric.hpp` provides the shared geometric error
metric for direct and fiberlet reference replay. It does not perform tracing
or reference matching: callers supply an evaluated point and its matched
reference point, both in base-volume XYZ voxels.

The normal is sampled at the reference point in normal-volume working
coordinates (`reference / normalWorkingToBaseScale`). With a valid unit normal
`n`, displacement `d`, and tangential factor `f`, the threshold error is
`hypot(abs(dot(d,n)), sqrt(max(0,dot(d,d)-dot(d,n)^2))/f)`.
The default tangential factor is 4; callers may explicitly choose another
positive factor. The acceptance boundary is inclusive: failure means error
**strictly greater** than the supplied normal radius. Normal sign is irrelevant.
If the normal is invalid, the metric falls back to Euclidean distance.

JSON helpers record the radii, units, comparison and invalid-normal policy,
and serialize missing components as null. Validation rejects inconsistent
measurements. No physical-unit conversion is implicit.

`FiberReplayExcursion` optionally delays failure until sufficient distance has
been travelled outside the acceptance region. The first outside sample starts
at zero distance; subsequent outside steps accumulate. Reentry resets the
excursion, including at the grace boundary. A zero grace fails immediately.
The caller remains responsible for handling an unresolved excursion when a
trace terminates.

## Validation

With a configured native build:

```sh
cmake --build build --target test_fiber_replay_metric -j 8
ctest --test-dir build -R '^test_fiber_replay_metric$' --output-on-failure
```

The test covers strict boundaries, normal sign/scale, missing normals,
zero-radius handling, JSON metadata, invalid inputs, and excursion reentry.
An optional read-only real-data check samples normals at anchor positions and
compares the metric with the explicit component formula:

```sh
export VC_TEST_METRIC_NORMAL_MANIFEST=/path/to/normals.lasagna.json
export VC_TEST_METRIC_NORMAL_SCALE=4
export VC_TEST_METRIC_ANCHORS=/path/to/anchors.json
build/bin/test_fiber_replay_metric
```

The test opens the normal sampler at the specified working-to-base scale;
this is not the stored channel's pyramid level. The anchor JSON is the output
of `vc_fiberlets anchors`. Real datasets and generated outputs are not bundled.
