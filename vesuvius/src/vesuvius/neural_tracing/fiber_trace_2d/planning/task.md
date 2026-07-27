# 3D Fiber Training Mixed Precision

Add automatic mixed precision support to
`vesuvius.neural_tracing.fiber_trace_3d.train` so the current conditioned 3D
fiber training can run with lower activation memory.

Requirements:

- Support BF16 autocast first, with FP16 AMP/scaler support available as an
  explicit config mode.
- Preserve the configured batch as one real model batch so BatchNorm statistics
  still come from the full configured batch.
- Apply the same precision mode to training loss, dense test loss, benchmark
  forward loss, and TensorBoard sample-sheet inference.
- Keep old configs valid; active S1A conditioned configs should opt into BF16.
- Store and resume GradScaler state when FP16 is used.
