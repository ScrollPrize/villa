# Task: Pipeline Training Batch Loading And GPU Work

The actual 2D fiber-strip training run leaves the GPU idle for much of the
step because loading, coordinate generation/augmentation, image augmentation,
and model training currently execute mostly as a sequential chain.

Investigate and implement a feasible pipelined training mode so upcoming
batches are loaded and coordinate-augmented while the current batch is being
image-augmented and trained. The goal is to keep the GPU busier without
changing deterministic sample order, augmentation semantics, label targets, or
the shared augment-vis/training loader path.
