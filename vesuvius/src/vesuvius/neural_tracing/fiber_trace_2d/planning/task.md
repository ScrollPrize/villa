# Task: make Fiber3D losses invariant to DDP batch partitioning

Change every Fiber3D training-loss reduction so an optimizer batch produces the
same objective and gradients regardless of how sparse targets and samples are
partitioned across DDP ranks. Synchronize the sufficient statistics required by
the loss, preserve the existing single-process objective, and measure the
per-step slowdown caused by the added collectives.
