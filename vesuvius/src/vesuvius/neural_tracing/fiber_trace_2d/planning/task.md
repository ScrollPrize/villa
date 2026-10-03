# Resident training trace evaluation

Call the optimized trace evaluation as a library from training. Keep training
models and optimizer state resident through ordinary trace tests; reuse the
existing GPU replicas for inference. Preserve manager pause/resize, trace
numerics, restart-selected snapshots, and optimized CPU processing.
Commit the existing evaluation/config changes first.
