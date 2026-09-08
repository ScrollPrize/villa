# Task: overlap Fiber3D batch preparation with training

The native quarter-resolution Fiber3D model reaches 95.7 global samples/s when
replaying a GPU-resident batch, but the shared real-sample trainer reaches only
83.8 samples/s even though its CPU sample pipeline sustains over 220 samples/s.

Move ordered batch collation/staging off the training critical path and keep two
future batches prepared. Use pinned host staging and a dedicated CUDA stream so
H2D transfer and GPU target materialization for batch N+2 overlap computation of
batch N. Preserve deterministic samples, bounded memory, and existing behavior
when the optimization is disabled or CUDA is unavailable. Measure against both
the 83.8 samples/s real-pipeline result and 95.7 samples/s replay ceiling on GPUs
0 and 1.
