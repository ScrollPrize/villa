# Native C++ Trace2CP Inference Scaledown Argument

The native C++ `vc_fiber_trace_metric` path must consume existing persisted
fiber inference `.lasagna.json` manifests whose prediction groups store only
the persisted prediction scale relative to base.

The current Python fiber inference writer records the persisted prediction
level in each group `scaledown`. It does not serialize the tracing/input scale
used before inference-output downsampling. Add a native command-line argument
for that missing value:

- `--inference-scaledown-power`, default `2`
- literal inference scaledown factor is `2**power`
- derive `prediction_to_base = source_to_base * 2**group.scaledown`
- derive `trace_to_base = prediction_to_base / 2**power`
- derive `prediction_spacing_in_trace_voxels = 2**power`

For the current `base_volume_scale=2` Python tracing setup with persisted
prediction level `4`, this gives `trace_to_base=4`,
`prediction_to_base=16`, and `prediction_spacing_in_trace_voxels=4`.

Do not add or require new manifest fields. The JSON fiber is assumed to already
be in the manifest base coordinate system.
