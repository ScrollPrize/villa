# Plan: CUDA conversion and checkpoint AMP inference

## Implementation

1. Change both shared spawned and serial CUDA paths to construct a tensor view over the
   compact shared uint8/uint16 tile, transfer that dtype to its CUDA device, and
   normalize on CUDA to float32 with exactly the existing `/255` semantics
   (`uint16 -> int32`, floor divide by 257, then float32 normalization). Keep
   the CPU-device fallback's current NumPy semantics exact.
2. Keep AMP owned by the Fiber adapter, which already reuses training's
   `_mixed_precision_config_from_training` and `_autocast_context`; do not add a
   nested shared autocast scope. Normalization and adapter preprocessing remain
   CUDA FP32, autocast covers model forward only, and adapter output is restored
   to FP32 before product transformation, weighting, downsampling, and D2H.
3. In Fiber inference, add `--inference-precision {auto,fp32,fp16,bf16}`.
   `auto` reads `checkpoint.config.training.mixed_precision`, normalizes known
   aliases, and falls back to fp32 when absent/disabled. Validate CUDA/device
   support through the existing training helper, inject the resolved setting
   into the adapter's effective config, and print its source and mode. Auto mode
   treats missing/off/invalid/legacy `auto` metadata as FP32 (warning for
   invalid/ambiguous values); unsupported checkpoint-derived AMP warns and
   falls back to FP32, while unsupported explicit overrides fail. Validate all
   selected devices, not only the first.
4. Update profiling stage names so compact H2D and CUDA conversion are measured
   separately; preserve the disabled profiling path.

## Tests

- Exact uint8 and uint16 conversion equivalence, including uint16 boundaries
  0/256/257/65534/65535, on CPU and CUDA when available.
- Fiber checkpoint precision resolution for bf16/fp16/disabled/missing/invalid.
- Fiber CLI forwarding and shared CUDA conversion coverage through both
  frontends' existing shared-runner tests. Verify preprocessing and product
  arithmetic remain FP32 under AMP.
- Existing exact serial/multi-device regression suite, compilation, and diff
  hygiene. Record that AMP output equality is not expected to be bitwise FP32.

## Spec update

Specify compact integer H2D, on-device normalization, checkpoint-derived Fiber
AMP defaults, explicit overrides, existing adapter AMP ownership, FP32
accumulation, and fallback/error rules.

## Documentation update

Document precision resolution, the inspected checkpoint's BF16 result, compact
transfer, profiling interpretation, and numerical implications.

## Changelog and workflow records

Update changelog, status, and task log with review, tests, measurements, and
limitations. Do not claim performance improvement without a comparable run.
