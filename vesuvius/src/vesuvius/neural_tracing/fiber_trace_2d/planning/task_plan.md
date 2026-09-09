# Plan: CUDA post-sampling augmentation

1. Generalize the deferred augmentation payload from blur kernels to the full
   immutable augmentation parameters plus prebuilt blur kernels.
2. Implement a shared batched CUDA helper preserving normalization, contrast,
   brightness, gamma, deterministic noise, isotropic blur, and directional blur
   order. Keep the ordinary CPU path using the same parameter semantics.
3. Make asynchronous training defer the entire post-sampling chain; keep one
   H2D transfer and no device-to-host round trip.
4. Extend the reusable CPU benchmark to stop exactly at the production H2D
   boundary, then benchmark 128 workers. Run the approved two-GPU benchmark on
   GPUs 0 and 1 and report per-GPU and aggregate throughput.
5. Add focused CPU/GPU equivalence and determinism tests and run the approved
   test callers.

Independent-review constraints: reductions are per sample over CZYX; CUDA work
runs explicitly in FP32 and in canonical operation order on the preparation
stream. CUDA noise uses one canonical-stream-index-keyed generator per sample,
so it is invariant to worker/rank/batch partition and repeatable on the same
CUDA stack, but it is intentionally not bit-identical to PyTorch's CPU RNG.
Equivalence tests therefore disable noise, while permutation/partition tests
cover CUDA noise. Deferred compact metadata remains alive through enqueue and
is cleared only from the returned prepared batch.

## Spec update

Change the CUDA augmentation contract from blur-only deferral to complete
post-sampling value-augmentation deferral. Document that coordinate generation
remains colocated with CPU sampling and that `all` is distinct from legacy
`blur` rather than silently changing that mode.

## Documentation update

Record exact commands, artifacts, host production rate, GPU augmentation time,
and two-GPU end-to-end throughput in the training profile document.

## Changelog

Add one entry for complete asynchronous CUDA value augmentation.
