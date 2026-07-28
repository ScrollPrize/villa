# Task: Native 3D Trace2CP Pyramid Scaledown

Switch native 3D Trace2CP scaled inference from box/mean pooling to the same
Gaussian pyramid downscale used by Lasagna predict3d.

The scaling should still run after model inference and before optional
inference-field blur and trusted-core cache cropping.
