# Task log: CUDA conversion and checkpoint AMP inference

## Initial findings

- The measured checkpoint contains `config.training.mixed_precision = "bf16"`;
  it does not request fp16.
- The current spawned worker expands every uint8 tile to float32 on one CPU
  core, then transfers four times as many bytes. Profiling measured about
  358 ms CPU conversion and 62 ms float32 H2D per 512-cubed tile.
- Existing `gpuN_sum` and the new profiler cover this path; the profiler must be
  updated to distinguish compact H2D from CUDA conversion.
- Fiber's inference adapter already calls training's mixed-precision resolver
  and autocast context, but uses the inference JSON config. The implementation
  will resolve checkpoint/CLI precision into that adapter config rather than
  nesting another autocast scope in the shared runner.

## Plan review

Independent review required compact H2D in serial and spawned paths, FP32
normalization/preprocessing/output arithmetic, explicit uint16 CUDA floor
semantics, all-device precision validation, safe checkpoint-auto fallback,
explicit-override errors, boundary tests, and accurate profiling. The plan was
updated accordingly. Checkpoint metadata describes training policy rather than
the state-dict dtype; a saved `auto` value is historically ambiguous.

## Implementation

- Shared serial CUDA and persistent multi-GPU workers now transfer source
  uint8/uint16 tensors before FP32 expansion. UInt16 uses CUDA int32 floor
  division by 257, matching the old mapping; CPU fallback is unchanged.
- Profiling separates `compact_h2d_*` and `cuda_convert_*` from adapter/model
  stages. Input slots are released after the synchronous pageable compact H2D,
  before CUDA conversion and model work.
- Fiber `--inference-precision` defaults to `auto`, reads checkpoint
  `config.training.mixed_precision`, validates every selected device using the
  training resolver, and injects the resolved policy into the adapter config.
  The measured checkpoint resolves to BF16. Missing/off metadata uses FP32;
  invalid, ambiguous saved `auto`, or unsupported derived modes warn and use
  FP32; unsupported explicit modes fail.
- Training mixed-precision alias parsing was factored into
  `_normalize_mixed_precision_mode`, retaining existing behavior and allowing
  inference resolution to share it. Existing Fiber adapter autocast remains
  model-only and returns FP32 before shared output arithmetic.

## Validation and limitations

- Compilation and `git diff --check` pass.
- Four focused shared-runner tests passed in 6.84 seconds: uint8/uint16 CPU
  boundary mapping, serial/parallel exactness, TensorStore/Python-Zarr
  exactness, and profiling schema.
- Direct precision-resolution smoke checks passed for checkpoint BF16 and CLI
  FP32. Fiber pytest coverage was added for checkpoint BF16, ambiguous `auto`,
  unsupported derived FP16, explicit unsupported FP16, and CLI forwarding, but
  pytest is not installed in the active venv.
- CUDA is hidden in the sandbox (`torch.cuda.is_available() == False`), so the
  CUDA uint16 boundary test and real BF16 inference/performance measurement
  must be exercised by the user's next GPU run. No speedup is claimed yet.
- AMP inference is intentionally not bitwise equivalent to FP32. Normalization,
  adapter preprocessing, output conversion, weighting, filtering, D2H, and
  accumulation remain FP32; only the Fiber model forward uses checkpoint AMP.
