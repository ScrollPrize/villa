# Parallel CUDA Training Pipeline

Improve `fiber_trace_2d` training overlap after observing:

- `load_ms` around 900 ms,
- `prep_ms` around 970 ms,
- `prep_submit_ms` around 950 ms,
- `prep_wait_ms` at 0 ms,
- low CPU and GPU utilization.

This means preparation is ready when consumed, but the main training thread is
still spending nearly a second submitting/preparing future work. Add concurrent
whole-batch loading and move CUDA preparation submission off the main training
thread while preserving deterministic step-order consumption.
