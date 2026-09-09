# Task: tune Fiber3D training batch size

Measure two-GPU Fiber3D training throughput across per-GPU batch sizes while
keeping the model, data path, augmentation, optimizer, precision, and two H100
devices fixed. Select the fastest safe batch size and report aggregate and
per-GPU throughput, iteration distribution, and peak device memory.
