# Task log: distribution-invariant Fiber3D losses

- 2026-09-09: Audited current loss structure. Direction and diagnostic terms
  are local sparse-point means; positive and negative presence terms are means
  over nonempty per-sample/per-branch group means. DDP currently averages those
  rank-local means, making the optimizer objective depend on rank partitioning.
- 2026-09-09: Wrote plan to synchronize sufficient statistics while retaining
  the established balanced presence-loss semantics.
- 2026-09-09: Independent review identified partition-dependent two-branch
  quota routing, unequal virtual-microbatch collective ordering, DDP gradient
  scaling, and non-lockstep test constraints. Updated the plan to address all
  four explicitly.
- 2026-09-09: Implemented update-scoped `_LossStatistics`. Direction uses a
  global sparse-element ratio; each presence component reduces the sum/count of
  established nonempty group means. Diagnostics share the same packed
  collective. The autograd surrogate multiplies each local differentiable sum
  by physical world size before DDP/manual gradient averaging.
- 2026-09-09: Removed the old sample-count microbatch loss scaling from the
  optimizer path. Virtual batches now retain local logical-forward graphs,
  combine statistics, perform one collective and one backward, then use the
  existing manual gradient average. Manual mode executes the raw model rather
  than stacking multiple outstanding DDP graphs.
- 2026-09-09: Made legacy quota routing per stable stream/sample. Distributed
  dense tests deliberately keep their non-lockstep local loss and row-gather
  protocol.
- 2026-09-09: Full focused suites passed: 292 tests. A real two-rank NCCL run
  and an unequal six-physical/eight-logical rank smoke completed successfully.
- 2026-09-09: Matched two-H100 batch-16/GPU trials reached 100.937, 100.909,
  and 100.879 global samples/s (mean 100.908) versus the prior 100.948
  samples/s artifact. Packed loss sync averaged 1.44--1.47ms/update; observed
  throughput change was -0.04%, within noise.
- Limitation: virtual ranks assigned multiple logical batches now retain
  multiple activation graphs until the update-level backward, increasing peak
  memory for emulation. Normal one-batch-per-GPU production training does not.
