# Task: Distributed Prefetched Fiber 3D Dense Tests

Replace rank-0 synchronous dense-test loading with deterministic DDP-sharded
evaluation using the same process-worker preloading mechanism as training.
Preserve test sample selection and metric semantics. Measure each complete test
routine, print its elapsed time, and log it as a TensorBoard time scalar.
