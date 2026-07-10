# CUDA Training Preparation Pipeline

Implement CUDA-side image/value augmentation and image preparation overlap for
`fiber_trace_2d` training.

- Keep deterministic CPU/VC3D batch loading ahead of training.
- Run deferred image/value augmentation and image normalization on a separate
  CUDA stream where possible.
- Keep prepared tensors on CUDA for model forward/backward instead of
  round-tripping augmented images through NumPy.
- Add timing measurements for work outside the critical forward/backward/step
  path, including preparation wait and preparation work.
- Preserve deterministic sample order, target semantics, and existing runner
  loader APIs.
