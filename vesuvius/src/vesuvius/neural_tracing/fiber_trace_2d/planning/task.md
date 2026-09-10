# Task: cooperative node resource manager and elastic Fiber3D training

Current authorization: implementation approved on 2026-09-09. Follow all phases
and validation gates. Protocol remains a draft until reference validation passes.

Build a reusable Python package in a new `gpu_manager/` directory at the Villa
root. A central server in a user's process coordinates cooperating clients on
one node through a filesystem-named socket, including explicit multi-user access.
Communication is metadata only; clients promise compliance, with no enforcement.

Requirements:

- Register/version-check, identify tasks by root PID and descendants, request
  maximum resources with priority, replace requests, poll each work step, receive
  new allocations and acknowledge the resources actually applied.
- Integer physical CPU cores, GPU count and per-GPU VRAM; initially proportional
  resource scaling. Partial GPU allocation means VRAM sharing. Approximate CPU
  accounting permits brief/light I/O activity.
- Exclusive tasks reserve all managed resources. Otherwise give each feasible
  task one GPU allocation and give spare capacity to highest priority.
- Manual splits within available limits, optionally bound to a task's lifetime,
  returning to automatic allocation when that task ends.
- Explicit finish and periodic PID-tree checks (proposed 60s); conservative
  release-before-reassignment and no scheduling dependency/priority inversions.
- Pause/unpause releases all GPU memory; CPU state/communication can remain.
  Bounded preloader queues may fill, then become idle.
- Simple protocol covering the entire design from the outset. Document the
  protocol and reference Python client. Lock the protocol after full reference
  validation; changes thereafter need explicit user instructions.

Fiber3D requirements:

- Standard PyTorch launch primitives, one worker per allocated GPU.
- Measured per-sample VRAM config, including fixed/preparation/workspace costs.
- Constant effective optimizer batch (reference 128) despite GPU count, physical
  microbatch and VRAM changes; support single-sample gradient accumulation.
- No BatchNorm/SyncBatchNorm or other cross-sample dependence; training statistics
  and gradient communication only at the end of the effective batch.
- Preserve objective, canonical samples/augmentation RNG, optimizer, LR/scaler,
  test-driven snapshots and logging through reconfiguration.
- CPU supervisor preserves state; fully unload/exit GPU workers at an update
  boundary and relaunch only on granted devices. Support zero-GPU pause.
- Reuse existing loading, augmentation, loss, state and test code.
- Test with the reference training, measure correctness/performance/memory, then
  document the validated locked protocol. All phases are in scope, in order.
