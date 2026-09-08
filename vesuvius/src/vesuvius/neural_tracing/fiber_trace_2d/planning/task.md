# Native lower-resolution Fiber3D U-Net output

Accelerate Fiber3D training by making the shared 3D U-Net emit predictions
directly at the lower resolution already used by Fiber inference. The initial
experiment uses output stride 4 (quarter resolution along each axis), stops the
decoder before its full- and half-resolution stages, and retains GroupNorm as
the fastest measured normalization.

Keep the existing full-resolution architecture as the compatibility default.
Measure the native-stride architecture against the matched GroupNorm replay and
real-sample baselines on physical GPUs 0 and 1. This phase optimizes speed; any
effect on convergence or prediction quality will be ablated later.
