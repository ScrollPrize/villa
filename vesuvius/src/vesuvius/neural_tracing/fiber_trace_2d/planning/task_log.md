# Task Log: Metric-Only 3D Trace2CP Config

## Implementation Notes

- Replaced `metric_sd2_s1.json`, which was a full training config clone, with a
  compact native 3D Trace2CP metric config.
- Kept a single `datasets` entry only as a volume/scale/manifest template.
- Removed the config-local JSON fiber glob because this metric config expects
  the concrete fiber to be supplied through `--fiber-json`.
- Removed the NML training glob, affine transform, transform inversion,
  `test_datasets`, augmentations, prefetch settings, training-loop settings,
  loss weights, TensorBoard settings, and run/checkpoint settings.
- Kept `lasagna_manifest_path` because the Trace2CP geometry path still uses
  Lasagna normals for strip geometry; this is not NML-specific.
- Kept `training.mixed_precision` because native 3D model inference reads it.
- Added a spec note that dedicated native 3D Trace2CP metric configs may be
  `--fiber-json` only and should not carry unrelated full training config
  fields or config-local fiber sources.

## Deviations / Deferred Items

- None.

## Validation

- `python -m json.tool vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/configs/metric_sd2_s1.json`
  passed.
- `PYTHONPATH=vesuvius/src:lasagna:. python -c "from vesuvius.neural_tracing.fiber_trace_3d.loader import load_config; cfg=load_config('vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/configs/metric_sd2_s1.json'); print(len(cfg.datasets), sorted(cfg.datasets[0]))"`
  passed and printed only `base_volume_path`, `base_volume_scale`, and
  `lasagna_manifest_path` in the dataset template.
- `PYTHONPATH=vesuvius/src:lasagna:. python -c "from pathlib import Path; from vesuvius.neural_tracing.fiber_trace_3d.trace2cp_tool import _load_raw_config, _tool_raw_config; raw=_load_raw_config('vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/configs/metric_sd2_s1.json'); patched=_tool_raw_config(raw, fiber_json=Path('/tmp/example.json')); print(patched['datasets'][0]['fiber_paths'])"`
  passed and confirmed the CLI fiber path is injected as `fiber_paths`.
