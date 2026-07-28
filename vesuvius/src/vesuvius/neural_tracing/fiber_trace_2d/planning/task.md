Refactor native 3D Trace2CP live inference to use the current shared fiber
3D inference infrastructure instead of carrying separate model loading,
normalization, recurrent/multibranch output splitting, and direction/presence
decode logic inside the tracer.

Scope clarifications:

- Do not remove or lazy-hide `FiberTrace3DPredictAdapter` from
  `fiber_trace_3d.__init__` for this task. The import path must be fixed
  directly by making the shared helpers package-safe.
- Do not implement tracing from precomputed fiber inference output yet. The
  tracer should get a clean prediction-field boundary so that a future
  precomputed `.lasagna.json` provider can be added without changing tracing
  logic, but this task only implements the live-checkpoint provider.
- Share common functions instead of reimplementing Lasagna/fiber inference
  behavior. In particular, normal estimation / compact `nx/ny` encoding and
  raw 3x2 direction + presence decoding must live in shared helpers.
