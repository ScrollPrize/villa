# Task: CUDA input conversion and checkpoint-derived inference precision

Remove CPU uint8/uint16-to-float32 expansion from shared multi-device inference:
transfer compact source dtype to each GPU and normalize there. For Fiber
inference, resolve AMP precision from checkpoint training metadata by default,
with an explicit CLI override and safe FP32 fallback. The inspected checkpoint
stores `training.mixed_precision = "bf16"`.
