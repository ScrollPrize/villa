# Task: Native 3D Trace2CP Scaled Inference Field

Add an opt-in native 3D Trace2CP switch that runs normal model inference at the
configured patch size, then box-downsamples the raw inferred output before the
tracer samples direction and presence from it.

The scaledown argument is an integer power of two:

- `0` means no scaledown, factor `1`, scale `1.0`.
- `1` means factor `2`, scale `0.5`.
- `2` means factor `4`, scale `0.25`.

The tracer must fail loudly before tracing if the inference patch shape or core
margin is not evenly divisible by the scaledown factor. The trusted core margin
must scale with the inferred field so point routing still avoids model-output
edge artifacts.
