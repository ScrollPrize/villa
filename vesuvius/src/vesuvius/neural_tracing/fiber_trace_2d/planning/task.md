# Task: eight-GPU S1a retraining configuration

Make Fiber3D console and TensorBoard progress use cumulative globally processed training samples by default, so runs with different batch sizes have comparable axes. Preserve internal optimizer-update scheduling, and preserve cumulative counts on checkpoint resume, including changed batch size. Do not launch training.
