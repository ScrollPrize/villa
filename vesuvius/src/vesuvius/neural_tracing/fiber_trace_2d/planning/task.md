# Task: remove unconditional Fiber3D scalar synchronization

Avoid reducing and materializing training metrics on every DDP step when they
are not logged or otherwise consumed. Preserve logged values and training
semantics, then measure batch-28 two-GPU throughput against 108.09 samples/s.
