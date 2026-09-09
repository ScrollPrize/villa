# Task log: remove unconditional Fiber3D scalar synchronization

- Baseline batch 28/GPU: 108.088 samples/s globally (54.044/GPU), DDP
  critical-path iteration mean/median/p95 517.79/517.42/530.51ms, peak reserved
  70.48GiB/GPU.
- Inspection found `_distributed_mean_loss_tensors` followed by conversion to
  Python floats on every step. The first-100-step console policy also causes
  every measured benchmark step through step 100 to consume metrics.
- Independent review required separate console, TensorBoard, and collective-due
  predicates and an output-suppressed forced-reduction control. Those schedules
  remain rank-invariant; default console/TensorBoard behavior is unchanged.
- Forced every-step reduction control, batch 28/GPU: 107.672 samples/s globally
  (53.836/GPU). Artifact suffix: `scalar_sync_control_20260909_062357_589452`.
- On-demand repeats: 108.949 and 108.808 samples/s globally (54.474 and
  54.404/GPU), averaging 1.12% above the isolated control. This confirms the
  synchronization was mostly charging preceding asynchronous GPU work; the
  metric collective/materialization itself was only a small bottleneck.
