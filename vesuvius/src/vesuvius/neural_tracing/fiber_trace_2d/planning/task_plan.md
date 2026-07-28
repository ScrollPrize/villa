# Plan: Metric-Only 3D Trace2CP Config

## Goals

- Keep `metric_sd2_s1.json` focused on native 3D Trace2CP metric execution.
- Support JSON fibers supplied via `--fiber-json` only.
- Remove training-only, augmentation-only, prefetch-only, and NML-only config
  keys.
- Preserve the fields the metric path still needs: model construction,
  inference normalization/precision, VC3D volume cache, base volume path/scale,
  and Lasagna manifest for geometry normals.

## Implementation Steps

1. Replace the full training-style config clone with a compact metric config.
2. Use a single `datasets` entry as the volume/scale/manifest template.
3. Remove `test_datasets`; the trace tool already patches the single dataset
   when `--fiber-json` is provided.
4. Remove NML-specific dataset fields: NML glob, affine transform, and transform
   inversion.
5. Remove augmentation, prefetch, loss, TensorBoard, checkpoint, run, loader, and
   training-loop settings that the metric CLI does not need.
6. Keep only minimal `training.mixed_precision` because model inference reads it.
7. Remove the config-local fiber glob because the fiber is supplied by CLI.
8. Validate that the JSON parses, the 3D config loader accepts it, and the
   native Trace2CP CLI patch path injects `fiber_paths` from `--fiber-json`.

## Spec Update

- Add/confirm that dedicated native 3D Trace2CP metric configs may be
  `--fiber-json` only and should not carry unrelated training/NML settings or
  config-local fiber globs.

## Docs Updates

- No public docs update needed for this config-only cleanup.

## Validation Commands

- `python -m json.tool vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/configs/metric_sd2_s1.json`
- `PYTHONPATH=vesuvius/src:lasagna:. python -c "from vesuvius.neural_tracing.fiber_trace_3d.loader import load_config; cfg=load_config('vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/configs/metric_sd2_s1.json'); print(len(cfg.datasets), sorted(cfg.datasets[0]))"`
- `PYTHONPATH=vesuvius/src:lasagna:. python -c "from pathlib import Path; from vesuvius.neural_tracing.fiber_trace_3d.trace2cp_tool import _load_raw_config, _tool_raw_config; raw=_load_raw_config('vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/configs/metric_sd2_s1.json'); patched=_tool_raw_config(raw, fiber_json=Path('/tmp/example.json')); print(patched['datasets'][0]['fiber_paths'])"`

## Changelog Update

- Not needed; this is a small config cleanup.
