# Task: move post-sampling Fiber3D augmentation to CUDA

Move every production value augmentation after CPU volume sampling to the
asynchronous per-GPU preparation stream. Preserve deterministic parameter
selection and operation order, measure the exact host producer with 128 workers,
and measure two-GPU end-to-end training using only GPUs 0 and 1.

Coordinate-lattice generation remains CPU-side because it is the direct input
to the CPU VC3D sampler; moving it to CUDA alone would require transferring the
same dense lattice back to the CPU.
