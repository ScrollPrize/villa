# Task: explain the native quarter-resolution end-to-end throughput gap

Profile why Fiber3D native stride-4 GroupNorm training improves replay/model-only
throughput by about 73%, but improves the full loading and augmentation pipeline
by only about 18%. Use the S1a 128-cube two-GPU reference on physical GPUs 0 and
1, `/home/hendrik/.venv_las`, and `/ephemeral/hendrik/fiber3d_train2`.

Attribute time and stalls precisely enough to select and implement the next
highest-impact optimization. Keep the overarching target of at least 221
samples/s for the eventual 8-GPU training run.
