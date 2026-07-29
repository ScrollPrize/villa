# Native C++ Trace2CP Inference Scaledown Argument Task Log

## Notes

- Re-checked the Python code path:
  - `fiber_trace_3d/infer.py` computes `input_sd`, `output_sd_input`, and
    `effective_output_sd`.
  - It passes `level=log2(effective_output_sd)`,
    `scaledown=effective_output_sd`, and `inference_scaledown=output_sd_input`
    into `FiberTrace3DPredictAdapter`.
  - `write_lasagna_product_manifest(...)` is called without `source_to_base`,
    so the writer default `source_to_base=1.0` is used for new manifests.
  - The manifest writer serializes each group `scaledown` as `product.level`,
    so current factor-16 persisted products are recorded as group
    `scaledown=4`.
- Re-checked the Python tracer coordinate code:
  - `FiberTrace3DLoader` traces in selected input-level voxels using
    `volume_spacing_base = 1 << base_volume_scale`.
  - For the current config, `base_volume_scale=2`, so trace coordinates are
    base/4.
  - Python `--inference-scaledown-power 2` makes prediction samples spaced 4
    trace voxels apart, or 16 base voxels.
- Native C++ now derives:
  - `prediction_to_base = source_to_base * 2**group.scaledown`
  - `trace_to_base = prediction_to_base / 2**inference_scaledown_power`
  - `prediction_spacing_in_trace_voxels = 2**inference_scaledown_power`
- `vc_fiber_trace_metric` now accepts `--inference-scaledown-power`, default
  `2`, and prints that value with the derived scale diagnostics.
- No manifest fields were added or required.

## Deviations

- The previous task text/spec said `trace_to_base = source_to_base`; that was
  inconsistent with the current Python writer and has been replaced.

## Validation

- `cmake --build volume-cartographer/build --target test_fiber_trace3d vc_fiber_trace_metric VC3D -j 4`
  passed.
- `volume-cartographer/build/bin/test_fiber_trace3d` passed: 18 test cases.
- `volume-cartographer/build/bin/vc_fiber_trace_metric --help` shows
  `--inference-scaledown-power`.
- Short interrupted sanity run on the S3 manifest and local fiber confirmed:
  `derived_trace_to_base=4`, `derived_prediction_to_base=16`,
  `derived_prediction_spacing_trace_voxels=4`,
  `inference_scaledown_power=2`, and first-segment max steps dropped from the
  previously observed `242` to `61`.
- `ctest --test-dir volume-cartographer/build -R test_fiber_trace3d --output-on-failure`
  passed.
- `git diff --check` passed.
